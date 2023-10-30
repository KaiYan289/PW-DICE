import argparse
import random
import numpy as np
import torch
import copy
from tqdm import tqdm
import math
# import wandb
import subprocess
from datetime import datetime
from tabular_MDP import TabularMDP, GridWorld
from visualizer import Plotter
from hyperparams import ini_hpp
import time
hyperparams = ini_hpp("params/params_SMODICE_KL_nosample.txt")

N_expert_traj = hyperparams["N_expert_traj"]
TA_expert_traj = hyperparams["TA_expert_traj"]
grid_size = hyperparams["grid_size"]
max_step = hyperparams["max_step"]
noise_level = hyperparams["noise_level"]
TA_optimality = hyperparams["TA_optimality"]

class SMODICE_Solver:
    def __init__(self, real_MDP, MDP, TA_dataset, time, visualize=True):
        self.MDP = MDP
        self.time = time
        # print(self.MDP.T[self.MDP.ed, self.MDP.ed, :])
        if self.MDP.ed >= 0:
            self.MDP.T[:, self.MDP.ed, :] = 0
            self.MDP.T[self.MDP.ed, self.MDP.ed, :] = 1 # stay at the same location; effectively "absorbing state"
            self.MDP.p0 = np.zeros_like(self.MDP.p0) 
            self.MDP.p0[self.MDP.st] = 1
        self.visualize = visualize
        self.MDP.T = self.MDP.T.transpose(1, 2, 0) # p(s'|s,a) -> p(s,a->s')
        self.MDP.R = -0.01 * np.ones((self.MDP.n, self.MDP.m))
        self.MDP.R[self.MDP.ed, :] = 1
        
        self.real_MDP = real_MDP
        self.TA_dataset = TA_dataset
        if self.visualize: 
            self.visualizer = Plotter(int(math.sqrt(self.MDP.n)), self.MDP.st, self.MDP.ed, time, directory="res/SMODICE_KL_CVXPY/fig")
    
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
        print("gamma:", mdp.gamma)#, P_pi @ np.ones((mdp.n * mdp.m, 1)))
        print(np.ones((1, mdp.n * mdp.m)) @ w - mdp.gamma * np.ones((1, mdp.n * mdp.m)) @ P_pi.T @ w - (1 - mdp.gamma) * mdp.n * mdp.m * np.ones((1, mdp.n * mdp.m)) @ p0, np.ones((1, mdp.n * mdp.m)) @ P_pi.T)
        # This is simply (I - \gamma P_\pi)^T y = (1-\gamma) p_0, where p_0 and y are (MDP.n * MDP.m)-dimensional vectors.
        # For positive definiteness they write like this and add regularizer. But is it necessary for a linear programming?
        # print((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - mdp.n * mdp.m * (1-mdp.gamma) * p0) almost 0
        print("sum:", w.sum())# P_pi.shape, np.linalg.norm((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - (1 - mdp.gamma) * p0))
        assert np.all(w > -1e-3), w
        d_pi = w * d
        d_pi[w < 0] = 0
        d_pi /= np.sum(d_pi)
        return d_pi
    
    def policy_evaluation(self, mdp, pi):
        r = np.sum(mdp.R * pi, axis=-1)
        P = np.sum(pi[:, :, None] * mdp.T, axis=1)
    
        if len(mdp.R.shape) == 3:
            V = np.tensordot(np.linalg.inv(np.eye(mdp.n) - mdp.gamma * P), r, axes=[-1, -1]).T
            Q = mdp.R + mdp.gamma * np.tensordot(mdp.T, V, axes=[-1, -1]).transpose([2, 0, 1])
        else:
            V = np.linalg.inv(np.eye(mdp.n) - mdp.gamma * P).dot(r)
            Q = mdp.R + mdp.gamma * mdp.T.dot(V)
        return V, Q
    
    def solve_MDP(self, method='PI'):
        if method == 'PI':
            # policy iteration
            pi = np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m
            V_old = np.zeros(self.MDP.n)
    
            for _ in range(1000000):
                V, Q = self.policy_evaluation(self.MDP, pi)
                pi_new = np.zeros((self.MDP.n, self.MDP.m))
                pi_new[np.arange(self.MDP.n), np.argmax(Q, axis=1)] = 1.
    
                if np.all(pi == pi_new) or np.max(np.abs(V - V_old)) < 1e-8:
                    break
                V_old = V
                pi = pi_new
    
            return pi, V, Q
            
        elif method == 'VI':
            # perform value iteration
            V, Q = np.zeros(self.MDP.n), np.zeros((self.MDP.n, self.MDP.m))
            for _ in range(100000):
                Q_new = mdp.R + mdp.gamma * mdp.T.dot(V)
                V_new = np.max(Q_new, axis=1)
    
                if np.max(np.abs(V - V_new)) < 1e-8:
                    break
    
                V, Q = V_new, Q_new
    
            pi = np.zeros((self.MDP.n, self.MDP.m))
            pi[np.arange(self.MDP.n), np.argmax(Q, axis=1)] = 1.
    
            return pi, V, Q
        else:
            raise NotImplementedError('Undefined method: %s' % method)
    
    def solve(self, TS_dataset, args, extra_param=None):
        # strangely, the SMODICE author in their code assumes that they have access to the random policy besides TA-dataset generated by the random policy.
        
        # get expert and TA marginal distribution

        pi_b = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(len(self.TA_dataset)):
            pi_b[self.TA_dataset[i]["state"], self.TA_dataset[i]["action"]] += 1
        # print("TA policy:")
        #for i in range(self.MDP.n):
        #    print(i // self.MDP.S, i % self.MDP.S, "pi_b:", pi_b[i]) 
        for i in range(self.MDP.n):
            if pi_b[i].sum() == 0: pi_b[i] = np.ones(self.MDP.m) / self.MDP.m
            else: pi_b[i] /= pi_b[i].sum()
        d = self.compute_marginal_distribution(self.MDP, pi_b)  # |S||A|
        d_s = d.reshape(self.MDP.n, self.MDP.m).sum(axis=1) # |S| (task-agnostic dataset)

        # print("TS_dataset:", TS_dataset)
        
        self.mode = args.TS_type
        N = 1 / len(TS_dataset)
        rho_E = np.zeros(self.MDP.n) 
        print("mode:", self.mode)
        if self.mode == "full":
            # full expert dataset
            for i in range(len(TS_dataset)):
                # rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
                rho_E[TS_dataset[i]["state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TS_dataset[i]["step"]) / N_expert_traj 
                # print(TS_dataset[i]["next_state"], self.MDP.ed)
                if TS_dataset[i]["next_state"] == self.MDP.ed:
                    rho_E[self.MDP.ed] += self.MDP.gamma ** (TS_dataset[i]["step"] + 1) / N_expert_traj
                    # print("!!", self.MDP.gamma ** (TS_dataset[i]["step"] + 1) / N_expert_traj)
            rho_E /= rho_E.sum()
            
            if extra_param is not None:
                if extra_param["GT_rho_E"]: 
                    expert_pi = self.MDP.expert_pi if extra_param["argmax"] == 0 else self.MDP.expert_pi_argmax
                    rho_E = self.compute_marginal_distribution(self.MDP, expert_pi).reshape(self.MDP.n, self.MDP.m).sum(axis=1)
                if extra_param["GT_rho_I"]: 
                    d = self.compute_marginal_distribution(self.MDP, np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m)
                    d_s = d.reshape(self.MDP.n, self.MDP.m).sum(axis=1)
            
            d_expert_s = rho_E
        elif self.mode == "perfect_full":
            rho_E = self.compute_marginal_distribution(self.MDP, self.get_expert_policy()).reshape(self.MDP.n, self.MDP.m).sum(axis=1) # mdp_expert for mismatch
            d_expert_s = rho_E
        elif self.mode == "goal":
            # goal-based
            rho_E[self.MDP.ed] = 1
            d_expert_s = rho_E
        else:
            for i in range(len(TS_dataset)):
                rho_E[TS_dataset[i]["state"]] += 1 / (len(TS_dataset) + 100)
            rho_E[self.MDP.ed] = 100 / (len(TS_dataset) + 100)
            d_expert_s = rho_E
        
        
        # train expert discriminator ...
        
        delta = 0.00001
        
        # C = torch.from_numpy((d_expert_s + delta) / (d_s + d_expert_s + delta))
        # C_nodelta = torch.from_numpy(d_expert_s / (d_s + d_expert_s))
        # print("d_s:", d_s)
        # print("d_expert_s:", d_expert_s)
        # print("C:", C)
        # print("C_nodelta:", C_nodelta)
        # R = -torch.log(1 / C - 1 + delta).to('cuda:0')
        
        R = np.log((d_expert_s+delta)/(d_s+delta)) # |S|
        
        """
        r1, r2 = np.zeros(self.MDP.n), np.zeros(self.MDP.n)
        for i in range(len(TS_dataset)):
            r1[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
        for i in range(len(TA_dataset)):
            r2[TA_dataset[i]["state"]] += 1 / len(TA_dataset)
        np.set_printoptions(linewidth=1000)
        print("from dataset:", np.log(r1 / r2).reshape(self.MDP.S, self.MDP.S)) 
        
        print("from policy:", R.reshape(self.MDP.S, self.MDP.S))
        exit(0)
        """
        # R_nodelta = -torch.log(1 / C_nodelta - 1).to('cuda:0')
        # print("R:", R, "GT-R:", np.log(d_expert_s / d_s), "nodelta-R:", R_nodelta)
        # exit(0)
        initials = []
        """ 
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
        """
        initials = np.array(initials)
        # exit(0)
        
        # perfect policy
        d_matrix = d.reshape(self.MDP.n, self.MDP.m)
        
        # cvxpy
        import cvxpy as cp
        p0 = self.MDP.p0
        B = np.repeat(np.eye(self.MDP.n), self.MDP.m, axis=0)  # |S||A| x |S|
        P = self.MDP.T.reshape(self.MDP.n * self.MDP.m, self.MDP.n)
        """
        x = cp.Variable(self.MDP.n)
        
        # print(B @ R)
        # print(cp.multiply(d, (B @ R - (self.MDP.gamma * P - B) @ x + 1) ** 2)) 
        # print(p0.shape) # cp.Minimize(cp.sum((x - 1) ** 2))# 
        objective = cp.Minimize((1 - self.MDP.gamma) * cp.sum(cp.multiply(p0, x)) + cp.log_sum_exp(np.log(d + 1e-100) + (B @ R + (self.MDP.gamma * P - B) @ x))) 
        # + cp.log(cp.sum(cp.multiply(d, cp.exp(B @ R.detach().cpu().numpy() + (self.MDP.gamma * P - B) @ x)))))
        # Rv = B @ R - (self.MDP.gamma * P - B) @ x 
        # objective = cp.Minimize((1 - self.MDP.gamma) * cp.sum(cp.multiply(p0, x)) + 0.5 * (Rv + 1).T @ D @ (Rv + 1))
        
        constraint = [x >= 0] # , cp.sum(x) == 1 the result is the same with or without this cp.sum(x) == 1
        
        prob = cp.Problem(objective, constraint)
        # prob = cp.Problem(objective)
        result = prob.solve()
        
        print("solve result:", result)
        #print("R.shape:", d.shape)
        
        ###############################################################################################################################
        
        x = cp.Variable(self.MDP.n)
        lag_dual_objective = cp.Minimize((1 - self.MDP.gamma) * cp.sum(cp.multiply(p0, x)) + cp.sum(cp.multiply(d, R + self.MDP.gamma * P @ x - B @ x)) - (cp.sum(cp.multiply(x, -np.log(d + 1e-10))) - cp.sum(cp.entr(x))))
        lag_dual_constraint = [x >= 0]
        lag_dual_prob = cp.Problem(lag_dual_objectie, lag_dual_constraint)
        lag_dual_res = lag_dual_prob.solve()
        print("lag_dual result:", lag_dual_res)
        """
        y = cp.Variable(self.MDP.n * self.MDP.m)
        primal_objective = cp.Minimize(cp.sum(cp.multiply(-B @ R, y)) + cp.sum(cp.multiply(y, -np.log(d + 1e-10))) - cp.sum(cp.entr(y))) #  
        primal_constraint = [y >= 0, (1 - self.MDP.gamma) * p0 + self.MDP.gamma * P.T @ y == B.T @ y]
        primal_prob = cp.Problem(primal_objective, primal_constraint)
        primal_res = primal_prob.solve()
        result = primal_res
        print("primal result:", primal_res)
        print("p0:", p0)
        
        cvxpy_solved_rhosa = y.value
        print("cvxpy_solved_rhosa:", cvxpy_solved_rhosa, "primal solution:", primal_prob.value)
        cvxpy_solved_rhosa = np.maximum(cvxpy_solved_rhosa, 0)
        
        cvxpy_solved_rhosa = cvxpy_solved_rhosa.reshape(self.MDP.n, self.MDP.m)
        
        cvxpy_solved_pi = np.zeros_like(cvxpy_solved_rhosa)
       
        for i in range(self.MDP.n):
            if cvxpy_solved_rhosa[i].sum() < 1e-12: cvxpy_solved_pi[i] = np.ones(self.MDP.m) / self.MDP.m 
            else: cvxpy_solved_pi[i] = cvxpy_solved_rhosa[i] / cvxpy_solved_rhosa[i].sum()
        
        # cvxpy_solved_pi = else cvxpy_solved_rhosa / (np.sum(cvxpy_solved_rhosa, axis=1, keepdims=True))
        
        if self.visualize: self.draw_policy(cvxpy_solved_pi, "SMODICE_CVXPY")
        self.pi = cvxpy_solved_pi
        
        ###############################################################################################################################
        # np.set_printoptions(precision=2, linewidth=300)
        """
        V = x.value
        # future_V = (torch.from_numpy(self.MDP.T).to('cuda:0') * torch.from_numpy(x.value).to('cuda:0').view(1, 1, -1)).sum(dim=-1) 
        future_V = (self.MDP.T * V.reshape(1, 1, -1)).sum(axis=-1) 
        
        def softmax(x):
            y = np.exp(x - np.max(x))
            f_x = y / np.sum(np.exp(x - np.max(x)))
            return f_x
        # (s, a, s') * V(s')

        # print("V:", V.reshape(self.MDP.S, self.MDP.S))
        # print("future value:", (R.view(-1, 1) + self.MDP.gamma * future_V - V.view(-1, 1)))
        # vv0 = R.reshape(-1, 1) + self.MDP.gamma * future_V - V.reshape(-1, 1)
        # vv = softmax(R.reshape(-1, 1) + self.MDP.gamma * future_V - V.reshape(-1, 1))
        # print("d:", d[self.MDP.ed])
        # exit(0)
        # print("vv0:", vv0) 
        # print("vv:", vv)
        
        d_optimal = d.reshape(self.MDP.n, self.MDP.m) * softmax(R.reshape(-1, 1) + self.MDP.gamma * future_V - V.reshape(-1, 1))
         
        
        self.pi = (d_optimal / d_optimal.sum(axis=1).reshape(-1, 1))
        #print("self.pi:", self.pi)
        #exit(0)
        for i in range(self.MDP.n):
            if np.count_nonzero(np.isnan(self.pi[i])) > 0:
                self.pi[i] = np.ones(self.MDP.m) / self.MDP.m
        print("pi:", self.pi)
        
        if self.visualize: self.draw_policy(self.pi)  
        """
        
        
        

    def draw_policy(self, pi):
        self.visualizer.clear()
        self.visualizer.draw_grid()
        self.visualizer.draw_policy(pi)
        self.visualizer.save("SMODICE_policy")

    def get_expert_policy(self):
        pi = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(self.MDP.n):
            x, y = self.MDP.get_pos(i)
            if self.MDP.edy < y: pi[i, 1] = 1 # left
            elif self.MDP.edy > y: pi[i, 0] = 1 # right
            elif self.MDP.edx < x: pi[i, 3] = 1 # up
            elif self.MDP.edx > x: pi[i, 2] = 1 # down  
            else: pi[i] = np.ones(self.MDP.m) / self.MDP.m # at optimal point
        return pi

    def draw_expert_policy(self):
        pi = self.get_expert_policy()
        self.visualizer.clear()
        self.visualizer.draw_grid()
        self.visualizer.draw_policy(pi)
        self.visualizer.save("expert_policy")

    def evaluation(self, eval_use_argmax):
        T = 10
        self.visualizer.clear()
        self.visualizer.draw_grid()
        avg_r, avg_suc, tot_l = 0, 0, 0    
        for i in range(T):
            agent_buffer, r, l = self.real_MDP.evaluation(self.pi, log=True, return_reward=True, collect=True, deterministic=(eval_use_argmax == "yes")) 
            avg_r += r / T
            if r > 0: 
                avg_suc += 1
                tot_l += l 
            if self.visualize: self.visualizer.draw_traj(agent_buffer, "orange")
        if self.visualize: self.visualizer.save("agent_traj")
        return avg_r, avg_suc / T, 999999 if avg_suc == 0 else tot_l / avg_suc


def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--data_index", help="data_index", type=int, default=0)
    parser.add_argument("--TS_type", type=str, default="full") # "full" or "goal"
    args = parser.parse_args()
    return args

def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()
    
def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()

if __name__ == "__main__":
    args = get_args()
    TS_type = args.TS_type
    seed = args.seed
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime)

    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing. 
    
    MDP = GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)
    # MDP = GridWorld(2, 0, 0, 1, 1)
    TS_dataset = torch.load("data/"+str(N_expert_traj)+"_"+str(TA_expert_traj)+"_"+str(grid_size)+"_"+str(noise_level)+"_"+str(max_step)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TS.pt") # MDP.generate_expert_traj(N_expert_traj)
    
    TA_dataset = torch.load("data/"+str(N_expert_traj)+"_"+str(TA_expert_traj)+"_"+str(grid_size)+"_"+str(noise_level)+"_"+str(max_step)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TA.pt") # MDP.generate_random_traj(TA_expert_traj) # 1000 traj * 25 steps / traj (s,a,s') a list of length 25000

    
    MDP_estimate = copy.deepcopy(MDP)

    MDP_estimate.T = np.zeros((MDP_estimate.n, MDP_estimate.n, 4)) # (s' | s, a)
    
    for i in range(len(TA_dataset)):
        MDP_estimate.T[TA_dataset[i]["next_state"], TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
    s = MDP_estimate.T.sum(axis=0)
    tag = (s == 0)
    
    random_estimation = np.zeros((MDP_estimate.n, MDP_estimate.n, 4))
    dx, dy = [0, 0, 1, -1], [1, -1, 0, 0] # 0 = right, 1 = left, 2 = down, 3 = up
    for i in range(MDP_estimate.n):
        x, y = i // MDP_estimate.S, i % MDP_estimate.S
        for j in range(4):
            p = 1
            for k in range(4):
                eks, wai = x + dx[k], y + dy[k]
                i_new = eks * MDP_estimate.S + wai
                if eks >= 0 and eks < MDP_estimate.S and wai >= 0 and wai < MDP_estimate.S:
                    p -= 0.25
                    random_estimation[i_new, i, j] = 0.25
            random_estimation[i, i, j] = p 
        
    MDP_estimate.T = random_estimation * tag + np.nan_to_num(MDP_estimate.T / s.reshape([1]+list(s.shape))) * (1 - tag)    
    
    # MDP_estimate.T = np.nan_to_num(np.ones_like(MDP_estimate.T) / MDP_estimate.n) * tag + np.nan_to_num(MDP_estimate.T / s.reshape([1]+list(s.shape))) * (1 - tag)
    """
    print("MDP_estimate transition:")
    for i in range(MDP_estimate.n):
        for j in range(MDP_estimate.m):
            for k in range(MDP_estimate.n):
               if MDP_estimate.T[k, i, j] > 0:
                   print("state:", i, "action:", j, "transition to", k, ":", MDP_estimate.T[k, i, j])
    """
    """
    for i in range(MDP.n):
        for j in range(MDP.m):
            if MDP_estimate[:, i, j] == 0:
                MDP_estimate[:, i, j] = np.ones(MDP.n) / MDP.n
            else:
                MDP_estimate[:, i, j] /= MDP_estimate[:, i, j].sum()
    """
    MDP_estimate_exact = copy.deepcopy(MDP)
    
    solver = SMODICE_Solver(MDP, MDP_estimate, TA_dataset, runtime)
    
    solver.visualizer.clear()
    solver.visualizer.draw_grid()
    solver.visualizer.draw_traj(TS_dataset, "orange")
    solver.visualizer.save("expert_dataset")
    t0 = time.time()
    solver.solve(TS_dataset, args)
    t1 = time.time()
    # print("V_star:", V_star, "f_div:", f_div)
    avg_r, suc_rate, avg_len = solver.evaluation(eval_use_argmax="yes")
    print("avg_rew:", avg_r, "suc_rate:", suc_rate, "avg_len:", avg_len, "runtime:", t1 - t0)
    
    f = open("res/SMODICE_KL_CVXPY/"+runtime.replace("/", "-").replace(" ", "_")+"aka"+str(time.time())+".txt", "w")
        
    hyperparams["TS_type"] = args.TS_type
    
    for key in hyperparams.keys():
        f.write(key+" "+str(hyperparams[key])+"\n")
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")
    
    print("avg_rew:", avg_r, "suc_rate:", suc_rate, "avg_len:", avg_len, "runtime:", t1 - t0)

    avg_r, suc_rate, avg_len = solver.evaluation(eval_use_argmax="no")
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")    
    f.close()
