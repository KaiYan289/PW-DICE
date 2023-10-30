from LP_solver import Solver
import argparse
import random
import numpy as np
import torch 
import time
import copy
from hyperparams import ini_hpp
import math
from tabular_MDP import TabularMDP, GridWorld
from visualizer import Plotter
from datetime import datetime
hyperparams = ini_hpp("params/params_SMODICE_chi.txt")

N_expert_traj = hyperparams["N_expert_traj"]
TA_expert_traj = hyperparams["TA_expert_traj"]
grid_size = hyperparams["grid_size"]
max_step = hyperparams["max_step"]
noise_level = hyperparams["noise_level"]
TA_optimality = hyperparams["TA_optimality"]

class SMODICE_Solver:
    def __init__(self, real_MDP, MDP, time, visualize=True):
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
        
        if self.visualize: self.visualizer = Plotter(int(math.sqrt(self.MDP.n)), self.MDP.st, self.MDP.ed, time, directory="res/SMODICE_CHI_CVXPY/fig")
    
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
        # print("w:", w)# P_pi.shape, np.linalg.norm((np.eye(mdp.n * mdp.m) - mdp.gamma * P_pi).T @ w - (1 - mdp.gamma) * p0))
        # print("d:", d)
        assert np.all(w > -1e-3), w
        d_pi = w * d
        d_pi[w < 0] = 0
        d_pi /= np.sum(d_pi)
        # torch.save(d_pi, "retrieved_d_pi.pt")
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
    
    def solve(self, TS_dataset, TA_dataset, args, extra_param=None):
        # strangely, the SMODICE author in their code assumes that they have access to the random policy besides TA-dataset generated by the random policy.
        
        pi_b = np.zeros((self.MDP.n, self.MDP.m)) # np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m
        d = np.zeros((self.MDP.n * self.MDP.m))
        
        
        for i in range(len(TA_dataset)):
            pi_b[TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
        for i in range(self.MDP.n):
            if pi_b[i].sum() == 0: pi_b[i] = np.ones(self.MDP.m) / self.MDP.m
            else: pi_b[i] /= pi_b[i].sum()
        # print("pi_b:", pi_b)
        # print("self.MDP.T:", self.MDP.T)
        
        """
        for i in range(len(TA_dataset)):
                # rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
                d[TA_dataset[i]["state"] * self.MDP.m + TA_dataset[i]["action"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TA_dataset[i]["step"]) / TA_expert_traj
        """
        d = self.compute_marginal_distribution(self.MDP, pi_b)  # |S||A|
        # d = self.compute_marginal_distribution(self.MDP, np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m)
        d_s = d.reshape(self.MDP.n, self.MDP.m).sum(axis=1) # |S| (task-agnostic dataset)
        
        self.mode = args.TS_type
        N = 1 / len(TS_dataset)
        rho_E = np.zeros(self.MDP.n) 
        # print("mode:", self.mode)
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
        elif self.mode == "goal":
            # goal-based
            rho_E[self.MDP.ed] = 1
        
        else:
            for i in range(len(TS_dataset)):
                rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
        
        d_expert_s = rho_E
        # print("rho_E:", d_expert_s)
        """
        # d_s = d.reshape(mdp.S, mdp.A).sum(axis=1).repeat(mdp.A)
        # different expert distribution depending on whether exampleMDP:
        if not example:
            d_expert = compute_marginal_distribution(mdp_expert, pi_star)
            d_expert_s = d_expert.reshape(mdp_expert.S, mdp_expert.A).sum(axis=1)
    
            # d_expert_s = d_expert.reshape(mdp_expert.S, mdp_expert.A).sum(axis=1).repeat(mdp_expert.A)
        else:
            # p_U(s|e)
            d_expert_s = np.zeros((mdp.S)) # |S|
            d_expert_s[mdp_expert.absorbing_state] = 1
        """
        # d_expert 
        delta = 1e-12
        p0 = self.MDP.p0
        
        P = self.MDP.T.reshape(self.MDP.n * self.MDP.m, self.MDP.n)  # |S||A| x |S|
        R = np.log((d_expert_s+delta)/(d_s+delta)) # |S||A|
        
        # print("R:", R.reshape(self.MDP.S, self.MDP.S))
    
        B = np.repeat(np.eye(self.MDP.n), self.MDP.m, axis=0)  # |S||A| x |S|
        
        # exit(0)
        I = np.ones(self.MDP.n * self.MDP.m)
        # alpha = alpha
        
        # cvxpy-primal 
        import cvxpy as cp
        x = cp.Variable(self.MDP.n * self.MDP.m)
        # print("d:", d)
        if d.min() == 0:
            objective = cp.Maximize(cp.sum(cp.multiply(x, B @ R)) - 0.5 * cp.sum(cp.multiply((d + 1e-10), (x / (d + 1e-10) - 1) ** 2)))
        else:
            objective = cp.Maximize(cp.sum(cp.multiply(x, B @ R)) - 0.5 * cp.sum(cp.multiply(d, (x / d - 1) ** 2)))
        constraint = [B.T @ x == (1 - self.MDP.gamma) * p0 + self.MDP.gamma * P.T @ x, x>=0]
        prob = cp.Problem(objective, constraint)
        result = prob.solve(solver='MOSEK', verbose=True)
        cvxpy_solved_rhosa = x.value
        print("cvxpy_solved_rhosa:", cvxpy_solved_rhosa, "primal solution:", prob.value)
        cvxpy_solved_rhosa = np.maximum(cvxpy_solved_rhosa, 0)
        
        cvxpy_solved_rhosa = cvxpy_solved_rhosa.reshape(self.MDP.n, self.MDP.m)
        
        cvxpy_solved_pi = np.zeros_like(cvxpy_solved_rhosa)
       
        for i in range(self.MDP.n):
            if cvxpy_solved_rhosa[i].sum() < 1e-12: cvxpy_solved_pi[i] = np.ones(self.MDP.m) / self.MDP.m 
            else: cvxpy_solved_pi[i] = cvxpy_solved_rhosa[i] / cvxpy_solved_rhosa[i].sum()
        
        # cvxpy_solved_pi = else cvxpy_solved_rhosa / (np.sum(cvxpy_solved_rhosa, axis=1, keepdims=True))
        
        if self.visualize: self.draw_policy(cvxpy_solved_pi, "SMODICE_CVXPY")
        self.pi = cvxpy_solved_pi
        # self.pi /= self.pi.sum(axis=1, keepdims=True)
        """
        print("sum:", self.pi.sum(axis=1))
        if abs((self.pi.sum() - 1).max()) > 1e-8: 
            print("not1!")
            exit(0)
        """
        return cvxpy_solved_pi

    def draw_policy(self, pi, name="SMODICE_policy"):
        self.visualizer.clear()
        self.visualizer.draw_grid()
        self.visualizer.draw_policy(pi)
        self.visualizer.save(name)

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
    
if __name__ == "__main__":
    args = get_args()
    TS_type = args.TS_type
    seed = args.seed
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing. 
    
    MDP = GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)

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
    
    solver = SMODICE_Solver(MDP, MDP_estimate, runtime)

    
    solver.visualizer.clear()
    solver.visualizer.draw_grid()
    solver.visualizer.draw_traj(TS_dataset, "orange")
    solver.visualizer.save("expert_dataset")
    t0 = time.time()
    pi_star = solver.solve(TS_dataset, TA_dataset, args)
    t1 = time.time()
    # solver.pi = pi_star
    
    f = open("res/SMODICE_CHI_CVXPY/"+runtime.replace("/", "-").replace(" ", "_")+"aka"+str(time.time())+".txt", "w")
    avg_r, suc_rate, avg_len = solver.evaluation(eval_use_argmax="yes")
    
    hyperparams["TS_type"] = args.TS_type
    
    for key in hyperparams.keys():
        f.write(key+" "+str(hyperparams[key])+"\n")
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")
    
    print("avg_rew:", avg_r, "suc_rate:", suc_rate, "avg_len:", avg_len, "runtime:", t1 - t0)

    avg_r, suc_rate, avg_len = solver.evaluation(eval_use_argmax="no")
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")    
    f.close()
    
    # for i in range(10):
    #    path = solver.policy_execution(solver.real_MDP, solver.pi)
    #    print(path, len(path))
    solver.draw_expert_policy()
    
    print('TS-S:', [TS_dataset[i]["state"] for i in range(len(TS_dataset))])
    print("TS-A:", [TS_dataset[i]["action"] for i in range(len(TS_dataset))])
    
