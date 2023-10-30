from LP_solver import Solver
import argparse
import random
import numpy as np
import torch
import copy
import time
from tqdm import tqdm
import math
import wandb
from datetime import datetime
from tabular_MDP import TabularMDP, GridWorld
from visualizer import Plotter
import subprocess

from hyperparams import ini_hpp
import time

# ini_hpp("params/params_LobsDICE_cvxpy_arbitrary.txt")

class LobsDICE_Solver:
    def __init__(self, real_MDP, MDP, TA_dataset, time=None):
        self.MDP = MDP
        # print(self.MDP.T[self.MDP.ed, self.MDP.ed, :])
        
        self.MDP.T = self.MDP.T.transpose(1, 2, 0) # p(s'|s,a) -> p(s,a->s')
        self.MDP.R = -0.01 * np.ones((self.MDP.n, self.MDP.m))
        self.MDP.R[self.MDP.ed, :] = 1
        
        self.time = time if time is not None else time.time()
        self.real_MDP = real_MDP
        self.TA_dataset = TA_dataset
        
    def compute_marginal_distribution(self, mdp, pi, regularizer=0):
        """
        d: |S||A|
        """
        p0_s = mdp.p0
        p0 = (p0_s[:, None] * pi).reshape(mdp.n * mdp.m)
        # print(p0)
        # print("T-shape:", mdp.T.shape, "checker:", mdp.T[0, 0])
        P_pi = (mdp.T.reshape(mdp.n * mdp.m, mdp.n)[:, :, None] * pi).reshape(mdp.n * mdp.m, mdp.n * mdp.m)
        
        d = np.ones(mdp.n * mdp.m); d /= np.sum(d)
        D = np.diag(d)
        E = np.sqrt(D) @ (np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi)
        #print(P_pi[0], P_pi.shape)
        #exit(0)
        Q = np.linalg.solve(E.T @ E + regularizer * np.eye(mdp.n * mdp.m), (1 - mdp.gamma) * p0)
        # Q_hat = np.linalg.inv(E.T @ E + regularizer * np.eye(mdp.n * mdp.m)) @ ((1 - mdp.gamma) * p0)# for debugging
        # print("ERROR:", np.linalg.norm(Q_hat - Q)) this error is very small (6e-13). 
        w = Q - mdp.gamma * P_pi @ Q
        # print("gamma:", mdp.gamma)#, P_pi @ np.ones((mdp.n * mdp.m, 1)))
        # print(np.ones((1, mdp.n * mdp.m)) @ w - mdp.gamma * np.ones((1, mdp.n * mdp.m)) @ P_pi.T @ w - (1 - mdp.gamma) * mdp.n * mdp.m * np.ones((1, mdp.n * mdp.m)) @ p0, np.ones((1, mdp.n * mdp.m)) @ P_pi.T)
        # This is simply (I - \gamma P_\pi)^T y = (1-\gamma) p_0, where p_0 and y are (MDP.n * MDP.m)-dimensional vectors.
        # For positive definiteness they write like this and add regularizer. But is it necessary for a linear programming?
        # print((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - mdp.n * mdp.m * (1-mdp.gamma) * p0) almost 0
        # print("sum:", w.sum())# P_pi.shape, np.linalg.norm((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - (1 - mdp.gamma) * p0))
        assert np.all(w > -1e-3), w
        d_pi = w * d
        d_pi[w < 0] = 0
        d_pi /= np.sum(d_pi)
        return d_pi
    
    def solve(self, TS_dataset, args, extra_param=None):
        # strangely, the SMODICE author in their code assumes that they have access to the random policy besides TA-dataset generated by the random policy.
        
        # get expert and TA marginal distribution
        
        pi_b = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(len(self.TA_dataset)):
            pi_b[self.TA_dataset[i]["state"], self.TA_dataset[i]["action"]] += 1
        for i in range(self.MDP.n):
            if pi_b[i].sum() == 0: pi_b[i] = np.ones(self.MDP.m) / self.MDP.m
            else: pi_b[i] /= pi_b[i].sum()
        # print("pi_b:", pi_b)
        
        if extra_param is not None:
            if extra_param["GT_rho_I"]: 
                pi_b = np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m
        # pi_b = np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m 
        
        d = self.compute_marginal_distribution(self.MDP, pi_b)  # |S||A|
        d_s = d.reshape(self.MDP.n, self.MDP.m).sum(axis=1) # |S| (task-agnostic dataset)
        d_ss_E = np.zeros((self.MDP.n, self.MDP.n))
        N_expert_traj = 1 #NOT args.N_expert_traj for arbitrary!!! there is only one trajectory with multiple steps!!!
        
        self.mode = args.TS_type
        N = 1 / len(TS_dataset)
        rho_E = np.zeros(self.MDP.n) 
        # print("mode:", self.mode)
        if self.mode == "full":
            # full expert dataset
            for i in range(len(TS_dataset)):
                # rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
                rho_E[TS_dataset[i]["state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TS_dataset[i]["step"]) / N_expert_traj 
                
                d_ss_E[TS_dataset[i]["state"], TS_dataset[i]["next_state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TS_dataset[i]["step"]) / N_expert_traj 
                # print(TS_dataset[i]["next_state"], self.MDP.ed)
                if TS_dataset[i]["next_state"] == self.MDP.ed:
                    rho_E[self.MDP.ed] += self.MDP.gamma ** (TS_dataset[i]["step"] + 1) / N_expert_traj
                    d_ss_E[self.MDP.ed, self.MDP.ed] += self.MDP.gamma ** (TS_dataset[i]["step"] + 1) / N_expert_traj 
                    # print("!!", self.MDP.gamma ** (TS_dataset[i]["step"] + 1) / N_expert_traj)
        elif self.mode == "perfect_full":
            rho_E = self.compute_marginal_distribution(self.MDP, self.get_expert_policy()).reshape(self.MDP.n, self.MDP.m).sum(axis=1) # mdp_expert for mismatch
        elif self.mode == "goal":
            # goal-based
            rho_E[self.MDP.ed] = 1
        else:
            for i in range(len(TS_dataset)):
                rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)

        rho_E /= rho_E.sum()
        d_ss_E /= d_ss_E.sum()
        
        if extra_param is not None:
            d_ss_E = np.zeros((self.MDP.n, self.MDP.n))
            if extra_param["GT_rho_E"]: 
                expert_pi = self.MDP.expert_pi if extra_param["argmax"] == 0 else self.MDP.expert_pi_argmax
                rho_E = self.compute_marginal_distribution(self.MDP, expert_pi).reshape(self.MDP.n, self.MDP.m).sum(axis=1)
                for i in range(self.MDP.n):
                    for j in range(self.MDP.n):
                        d_ss_E[i, j] = rho_E[i] * (self.MDP.T[i, :, j] * self.MDP.expert_pi[i]).sum()
        
        d_expert_s = rho_E
        d0 = d.reshape(self.MDP.n, self.MDP.m)
        d_sas_matrix = torch.zeros(self.MDP.n, self.MDP.m, self.MDP.n).to('cuda:0')
        # print("d:", d.shape, "d_sas:", d_sas_matrix.shape, "T:", self.MDP.T.shape)
        for i in range(self.MDP.n):
            for j in range(self.MDP.m):
                d_sas_matrix[i, j] = d0[i, j] * torch.from_numpy(self.MDP.T[i, j]).to('cuda:0') # vector = scalar * vector
        
        d_ss_I = d_sas_matrix.sum(dim=1) # summing over action axis
        
        
        
        # train expert discriminator ...
        
        delta = 0.0001
        
        d_ss_E = torch.from_numpy(d_ss_E).to('cuda:0')
        # print("d_ss_E:", d_ss_E[0], "d_ss_I:", d_ss_I[0])
        # print((d_ss_E + delta).shape, (d_ss_E + d_ss_I + delta).shape)
        # C = torch.from_numpy((d_expert_s + delta) / (d_s + d_expert_s + delta))
        # C = (d_ss_E + delta) / (d_ss_E + d_ss_I + delta)
        # C(s, s') = d_E(s, s') / [d_E(s, s') + d_I(s, s')]
        
        
        
        # print("d_expert_s:", d_expert_s)
        # print("C:", C[0])
        # exit(0)
        # R = -torch.log(1 / C - 1 + delta).to('cuda:0')
        R = torch.log((d_ss_E + delta) / (d_ss_I + delta))
        # print("R:", R[0], R[80])
        # exit(0)
        initials = []
        
        terminal = np.zeros(len(self.TA_dataset))
        
        for i in range(len(self.TA_dataset)):
            if self.TA_dataset[i]["step"] == 0:
                initials.append(self.TA_dataset[i]["state"])
            if i < len(self.TA_dataset) - 1:
                if self.TA_dataset[i + 1]["step"] != self.TA_dataset[i]["step"] + 1:
                    terminal[i] = 1
            else: 
                terminal[i] = 1
            # if terminal[i] == 1: print("terminal:", i)
        terminal = torch.from_numpy(terminal).to('cuda:0')
        
        
        
        alpha = 0.01
        
        # perfect policy
        d_sa_matrix = torch.from_numpy(d.reshape(self.MDP.n, self.MDP.m)).to('cuda:0')
        # calculating d_sas_matrix
        d_sas_matrix = torch.zeros(self.MDP.n, self.MDP.m, self.MDP.n).to('cuda:0')
        
        for i in range(self.MDP.n):
            for j in range(self.MDP.m):
                d_sas_matrix[i, j] = d_sa_matrix[i, j] * torch.from_numpy(self.MDP.T[i, j]).to('cuda:0') # vector = scalar * vector
        d_ss_matrix = d_sas_matrix.sum(dim=1)
        
        # cvxpy solver
        """
        import cvxpy as cp
        x = cp.Variable(self.MDP.n)
        A0 = cp.Variable((self.MDP.n, self.MDP.n))
        constraints = []
        for i in range(self.MDP.n):
            for j in range(self.MDP.n):
                constraints.append(A0[i, j] == R[i, j].detach().cpu().numpy() + self.MDP.gamma * x[j] - x[i])
        objective = cp.Minimize((1 - self.MDP.gamma) * cp.sum(cp.multiply(self.MDP.p0, x)) + (1 + alpha) * cp.log_sum_exp(np.log(1e-20 + d_ss_matrix.detach().cpu().numpy()) + A0 / (1 + alpha) - 1))
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        print("retreived V:", x.value)
        # print("A:", A0.value)
        
        exit(0)
        """
        import cvxpy as cp
        p0 = self.MDP.p0.reshape(-1, 1)
        one = np.ones((self.MDP.n, 1))
        x = cp.Variable((self.MDP.n, 1))
        objective = cp.Minimize((1 - self.MDP.gamma) * cp.sum(cp.multiply(p0, x)) + (1 + alpha) * cp.sum(cp.multiply(d_ss_matrix.cpu().detach().numpy(), cp.exp((R.cpu().detach().numpy() + self.MDP.gamma * one @ x.T - x @ one.T) / (1 + alpha) - 1))))
        prob = cp.Problem(objective)
        result = prob.solve()
        # print("solved V:", x.value)
        V = torch.from_numpy(x.value).to('cuda')
        
        w = np.zeros((self.MDP.n, self.MDP.m))
        self.pi = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(self.MDP.n):
            for j in range(self.MDP.m):
                A = R[i] + self.MDP.gamma * V - V[i]
                w[i, j] = np.sum(self.MDP.T[i, j] * torch.exp(A / (1 + alpha)).cpu().detach().numpy())
            self.pi[i] = w[i] * d.reshape(self.MDP.n, self.MDP.m)[i]
            if self.pi[i].sum() < 1e-12: self.pi[i] = np.ones(self.MDP.m) / self.MDP.m
            else: self.pi[i] /= self.pi[i].sum()
        
        
        # w = w.detach().cpu().numpy()
       
        #self.pi = d.reshape(self.MDP.n, self.MDP.m) * w / (d.reshape(self.MDP.n, self.MDP.m) * w).sum(axis=1).reshape(-1, 1)        
        
        # 1 / alpha is too big
        """
        self.pi = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(self.MDP.n):
            if i != self.MDP.ed: self.pi[i, e[i].argmax()] = 1
        """
        
    def evaluation(self, eval_use_argmax):
        T = 10
        avg_r, avg_suc, tot_l = 0, 0, 0    
        for i in range(T):
            agent_buffer, r, l = self.real_MDP.evaluation(self.pi, log=True, return_reward=True, collect=True, deterministic=(eval_use_argmax == "yes")) 
            avg_r += r / T
            if r > 0: 
                avg_suc += 1
                tot_l += l 
        return avg_r, avg_suc / T, 999999 if avg_suc == 0 else tot_l / avg_suc


def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--N_expert_traj", type=int)
    parser.add_argument("--TA_expert_traj", type=int)
    parser.add_argument("--TA_optimality", type=float, default=0)
    parser.add_argument("--TS_type", type=str, default="full") # "full" or "goal"
    parser.add_argument("--noise_level", type=float)
    args = parser.parse_args()
    return args

def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()
    
def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()

def compute_marginal_distribution(mdp_input, pi, regularizer=0):
        """
        d: |S||A|
        """
        mdp = copy.deepcopy(mdp_input)
        mdp.T = mdp.T.transpose(1, 2, 0)
        
        p0_s = mdp.p0
        p0 = (p0_s[:, None] * pi).reshape(mdp.n * mdp.m)
        # print(p0)
        P_pi = (mdp.T.reshape(mdp.n * mdp.m, mdp.n)[:, :, None] * pi).reshape(mdp.n * mdp.m, mdp.n * mdp.m)
        
        d = np.ones(mdp.n * mdp.m); d /= np.sum(d)
        D = np.diag(d)
        E = np.sqrt(D) @ (np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi)
        #print(P_pi[0], P_pi.shape)
        #exit(0)
        Q = np.linalg.solve(E.T @ E + regularizer * np.eye(mdp.n * mdp.m), (1 - mdp.gamma) * p0)
        # Q_hat = np.linalg.inv(E.T @ E + regularizer * np.eye(mdp.n * mdp.m)) @ ((1 - mdp.gamma) * p0)# for debugging
        # print("ERROR:", np.linalg.norm(Q_hat - Q)) this error is very small (6e-13). 
        w = Q - mdp.gamma * P_pi @ Q
        # print("gamma:", mdp.gamma)#, P_pi @ np.ones((mdp.n * mdp.m, 1)))
        # print(np.ones((1, mdp.n * mdp.m)) @ w - mdp.gamma * np.ones((1, mdp.n * mdp.m)) @ P_pi.T @ w - (1 - mdp.gamma) * mdp.n * mdp.m * np.ones((1, mdp.n * mdp.m)) @ p0, np.ones((1, mdp.n * mdp.m)) @ P_pi.T)
        # This is simply (I - \gamma P_\pi)^T y = (1-\gamma) p_0, where p_0 and y are (MDP.n * MDP.m)-dimensional vectors.
        # For positive definiteness they write like this and add regularizer. But is it necessary for a linear programming?
        # print((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - mdp.n * mdp.m * (1-mdp.gamma) * p0) almost 0
        # print("sum:", w.sum())# P_pi.shape, np.linalg.norm((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - (1 - mdp.gamma) * p0))
        assert np.all(w > -1e-3), w
        d_pi = w * d
        d_pi[w < 0] = 0
        d_pi /= np.sum(d_pi)
        # torch.save(d_pi, "retrieved_d_pi.pt")
        return d_pi.reshape(mdp.n, mdp.m)
    

def compute_ss(MDP, pi):
    d_sa = compute_marginal_distribution(MDP, pi)# .sum(axis=1)
    d_ss = np.zeros((MDP.n, MDP.n))
    for i in range(MDP.n):
        for j in range(MDP.n):
            d_ss[i, j] = (d_sa[i] * MDP.T[j, i, :]).sum()
    print("dss:", d_ss.sum())
    return d_ss

if __name__ == "__main__":
    args = get_args()
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime)
    
    TS_type = args.TS_type
    seed = args.seed
    
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing. 
    
    f = open("res/LDexperiment/LobsDICE/LobsDICE-"+str(args.TA_expert_traj)+"-"+str(args.N_expert_traj)+"-"+str(args.noise_level)+".txt", 'w')
    
    for data_index in range(10):
        MDP = torch.load("data/LobsDICE_"+str(args.TA_expert_traj)+"_"+str(args.N_expert_traj)+"_"+("1" if args.noise_level == 1 else str(args.noise_level))+"/"+str(data_index)+"/MDP.pt")# GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)
        # MDP = GridWorld(2, 0, 0, 1, 1)
        
        TS_dataset = torch.load("data/LobsDICE_"+str(args.TA_expert_traj)+"_"+str(args.N_expert_traj)+"_"+("1" if args.noise_level == 1 else str(args.noise_level))+"/"+str(data_index)+"/TS.pt") # MDP.generate_expert_traj(N_expert_traj)
        TS2_dataset = torch.load("data/LobsDICE_"+str(args.TA_expert_traj)+"_"+str(args.N_expert_traj)+"_"+("1" if args.noise_level == 1 else str(args.noise_level))+"/"+str(data_index)+"/TS_argmax.pt")
        TA_dataset = torch.load("data/LobsDICE_"+str(args.TA_expert_traj)+"_"+str(args.N_expert_traj)+"_"+("1" if args.noise_level == 1 else str(args.noise_level))+"/"+str(data_index)+"/TA.pt") # MDP.generate_random_traj(TA_expert_traj) # 1000 traj * 25 steps / traj (s,a,s') a list of length 25000
        
        # print('TS_dataset:', TS_dataset)
        
        # print('TA_dataset:', TA_dataset)
        
        MDP_estimate = copy.deepcopy(MDP)
        MDP_estimate.T = np.zeros((MDP_estimate.n, MDP_estimate.n, MDP_estimate.m)) # (s' | s, a)
        
        for i in range(len(TA_dataset)):
            MDP_estimate.T[TA_dataset[i]["next_state"], TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
        s = MDP_estimate.T.sum(axis=0)
        tag = (s == 0)
        # MDP_estimate.T = np.nan_to_num(np.ones_like(MDP_estimate.T) / MDP_estimate.n) * tag + np.nan_to_num(MDP_estimate.T / s.reshape([1]+list(s.shape))) * (1 - tag)
        random_estimation = np.ones((MDP_estimate.n, MDP_estimate.n, MDP_estimate.m)) / MDP_estimate.n
        
        MDP_estimate.T = random_estimation * tag + np.nan_to_num(MDP_estimate.T / s.reshape([1]+list(s.shape))) * (1 - tag)
    
        MDP_estimate_exact = copy.deepcopy(MDP)
    
        # exit(0)
        """
        hyperparams["TS_type"] = args.TS_type
        f.write(str(runtime)+" ")
        for key in hyperparams.keys():
           f.write(str(hyperparams[key])+" ")
        """
        for i, solver in enumerate([LobsDICE_Solver(MDP, MDP_estimate, TA_dataset, runtime), LobsDICE_Solver(MDP, MDP_estimate_exact, TA_dataset, runtime)]):
            for j, TS in enumerate([TS_dataset, TS2_dataset]):
                for GT_rho_E in [False, True]:
                    for GT_rho_I in [False, True]:
                        solver.solve(TS, args, extra_param={"GT_rho_E": GT_rho_E, "GT_rho_I": GT_rho_I, "argmax": j})
                        A0 = np.linalg.norm(compute_ss(MDP, solver.pi).reshape(-1) - compute_ss(MDP, MDP.expert_pi).reshape(-1), 1) if j == 0 else np.linalg.norm(compute_ss(MDP, solver.pi).reshape(-1) - compute_ss(MDP, MDP.expert_pi_argmax).reshape(-1), 1)
                        DASS0, EASS0 = compute_ss(MDP, solver.pi),  compute_ss(MDP, MDP.expert_pi) if j == 0 else compute_ss(MDP, MDP.expert_pi_argmax)
                        #print("occupancy-diff-ss:", np.linalg.norm(DASS0.reshape(-1) - EASS0.reshape(-1), 1))
                        DASS0, EASS0 = DASS0.sum(axis=1), EASS0.sum(axis=1)
                        #print("occupancy-diff-sumss:", np.linalg.norm(DASS0 - EASS0, 1))
                        DA0 = compute_marginal_distribution(MDP, solver.pi).sum(axis=1)
                        EA0 = compute_marginal_distribution(MDP, MDP.expert_pi).sum(axis=1) if j == 0 else compute_marginal_distribution(MDP, MDP.expert_pi_argmax).sum(axis=1)
                        # there is still bug; the result of EASS0 sum and EA0 does not match!
                        assert np.abs(DA0 - DASS0).sum() < 1e-10 and np.abs(EA0 - EASS0).sum() < 1e-10, "Mismatch!"
                        assert A0 >= np.linalg.norm(DA0 - EA0, 1), "Wrong Discrepancy!"
                        #print("margdistdiff:", DA0 - DASS0, EA0 - EASS0, DA0.shape, DASS0.shape, EA0.shape, EASS0.shape)
                        #print("total state-pair occupancy difference:", A0)
                        # print("PI:", solver.pi, MDP.expert_pi)
                        #print("total state occupancy difference:", np.linalg.norm(DA0 - EA0, 1))
                        # 
                        f.write(str(A0)+" "+str(np.linalg.norm(DA0 - EA0, 1))+" "+str(EA0[MDP.target] - DA0[MDP.target])+" ")
        f.write("\n")
    f.close() 
    print("running ends!!!")