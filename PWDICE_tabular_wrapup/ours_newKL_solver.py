from gurobipy import GRB
import gurobipy as gp
import torch
from torch.optim import Adam, Adagrad, SGD, AdamW, LBFGS
import torch.nn as nn
import copy
import random
import time
from tabular_MDP import *
from ortools.graph import pywrapgraph
from NN import Onehot_Predictor
import argparse
from tqdm import tqdm
import math
import networkx as nx
# import wandb
import subprocess
from datetime import datetime
from tqdm import tqdm
from hyperparams import ini_hpp
import cvxpy as cp
device = torch.device('cuda:0')

def concatlist(lst): # return a concatenation of list
    return sum(lst, [])

def list2gen(lst):
    return (_ for _ in lst)

hyperparams = ini_hpp("params/params_ours_ent2KL_solver.txt")

N_expert_traj = hyperparams["N_expert_traj"]
TA_expert_traj = hyperparams["TA_expert_traj"]
grid_size = hyperparams["grid_size"]
max_step = hyperparams["max_step"]
noise_level = hyperparams["noise_level"]
TA_optimality = hyperparams["TA_optimality"]



class Solver:
    def __init__(self, MDP, real_MDP, time, visualize=True):
        self.MDP = MDP
        self.real_MDP = real_MDP
        if self.MDP.ed >= 0:
            self.MDP.T[:, self.MDP.ed, :] = 0
            self.MDP.T[self.MDP.ed, self.MDP.ed, :] = 1 # stay at the same location; effectively "absorbing state"
        
        self.time = time
        self.dist = self.get_distance_matrix('dirac')
        self.visualize = visualize
        if visualize:
            self.visualizer = Plotter(int(math.sqrt(self.MDP.n)), self.MDP.st, self.MDP.ed, self.time, directory='res/ours_ent2KL_solver/fig')
        self.pi = np.zeros((self.MDP.n, self.MDP.m))

    def get_distance_matrix(self, typ):
        if typ == 'dirac':
            dist = np.ones((self.MDP.n, self.MDP.n)) 
            for i in range(self.MDP.n): dist[i, i] = 0
        elif typ == 'manhattan':
            assert isinstance(self.MDP, GridWorld), "Cannot apply manhattan!"
            dist = np.zeros((self.MDP.n, self.MDP.n))
            for i in range(self.MDP.n):
                for j in range(self.MDP.n):
                    posx = self.MDP.get_pos(i)
                    posy = self.MDP.get_pos(j)
                    dist[i, j] = abs(np.array(posx) - np.array(posy)).sum()
        elif typ == 'euclidean':
            assert isinstance(self.MDP, GridWorld), "Cannot apply manhattan!"
            dist = np.zeros((self.MDP.n, self.MDP.n))
            for i in range(self.MDP.n):
                for j in range(self.MDP.n):
                    posx = self.MDP.get_pos(i)
                    posy = self.MDP.get_pos(j)
                    dist[i, j] = ((np.array(posx) - np.array(posy)) ** 2).sum()
        return dist
    
    def compute_marginal_distribution(self, mdp_input, pi, regularizer=0):
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
    
    def retrieve_policy(self, r_sa, r_s): # r_sa is np array shape n * m (state * action), r_s is np array shape n.
        for i in range(self.MDP.n):  
            # if r_s[i] < 1e-15: self.pi[i] = np.ones(self.MDP.m) / self.MDP.m 
            # else: 
            self.pi[i] = r_sa[i] / r_s[i]
            # print(i, self.pi[i], np.count_nonzero(np.isnan(self.pi[i])))
            if np.count_nonzero(np.isnan(self.pi[i])) > 0:
                self.pi[i] = np.ones_like(self.pi[i]) / self.MDP.m
            self.pi[i] /= self.pi[i].sum() # precision issue; this is unnecessary theoretically.
            # print("pi["+str(i)+"]", self.pi[i])
    
    def solve(self, TS_dataset):
        raise NotImplementedError("Error!")
        
    def self_check(self):
        assert self.MDP.check_p0(), "MDP p0 error!"
        assert self.MDP.check_transition(), "MDP transition error!"
        
    def evaluation(self):
        T = 10
        for i in range(T):
            self.real_MDP.evaluation(self.pi, log=True)


class Direct_Convex_Solver(Solver):
    def __init__(self, real_MDP, MDP, time, visualize=True):
        super().__init__(MDP, real_MDP, time, visualize)
        
    def solve(self, TS_dataset, TA_dataset, args, extra_param=None):
    
        epsilon_1, epsilon_2 = args.epsilon1, args.epsilon2
        self.self_check()

        self.mode = args.TS_type
        # self variables: rho_sa, rho_s
        rho_E = np.zeros(self.MDP.n)
        rho_I = np.zeros((self.MDP.n, self.MDP.m))
        
        N = 1 / len(TS_dataset)

        pi_b = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(len(TA_dataset)):
            pi_b[TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
        
        for i in range(self.MDP.n):
            if pi_b[i].sum() == 0: pi_b[i] = np.ones(self.MDP.m) / self.MDP.m
            else: pi_b[i] /= pi_b[i].sum()
        #print("pi_b:", pi_b)
        
        for i in range(len(TS_dataset)):
            if TS_dataset[i]["step"] == 0: print("")
            #print(TS_dataset[i]["state"], end = " ")
            
        # exit(0)
        rho_I = self.compute_marginal_distribution(self.MDP, pi_b)
        
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
                    rho_I = d.reshape(self.MDP.n, self.MDP.m)
        elif self.mode == "perfect_full":
            rho_E = self.compute_marginal_distribution(self.MDP, self.get_expert_policy()).reshape(self.MDP.n, self.MDP.m).sum(axis=1) # mdp_expert for mismatch
        elif self.mode == "goal":
            # goal-based
            rho_E[self.MDP.ed] = 1
        else:
            for i in range(len(TS_dataset)):
                rho_E[TS_dataset[i]["state"]] += 1 / (len(TS_dataset) + 100)
            rho_E[self.MDP.ed] = 100 / (len(TS_dataset) + 100)
        
        ############################################################################### 
        delta = 0.00001         
        R0 = np.zeros((self.MDP.n, self.MDP.n))
        for i in range(self.MDP.n):
            R0[i] = -np.log((rho_E+delta)/(rho_I.sum(axis=1)+delta))[i] # |S|
        #print(-np.log((rho_E+delta)/(rho_I.sum(axis=1)+delta)).reshape(self.MDP.S, self.MDP.S))
        ###############################################################################
        if args.distance == "smodice": dist = R0.reshape(-1)
        else: dist = self.get_distance_matrix(args.distance)
        d0 = dist.reshape(-1)
        d = np.concatenate([d0, np.zeros((self.MDP.n * self.MDP.m))]).reshape(1, -1)
        
        np.set_printoptions(linewidth=1000)
        
        tot_vis = np.zeros((self.MDP.S, self.MDP.S))
        for i in range(len(TS_dataset)):
            tot_vis[TS_dataset[i]["state"] // self.MDP.S , TS_dataset[i]["state"] % self.MDP.S] += 1 
        #print("tot_vis:", tot_vis)
        #print("rho_E:", rho_E.reshape(self.MDP.S, self.MDP.S))
        #print("rho_I:", rho_I.sum(axis=1))
        # construction of matrix A and B
        
        A, b = np.zeros((3 * self.MDP.n + 1, self.MDP.n * (self.MDP.n + self.MDP.m))), np.zeros((3 * self.MDP.n + 1))
        # marginal of rho_sa
        for i in range(self.MDP.n):
            for j in range(self.MDP.n):
                A[i, i * self.MDP.n + j] = 1
            for j in range(self.MDP.m):
                A[i, self.MDP.n * self.MDP.n + i * self.MDP.m + j] = -1
        # marginal of rho_E
        for i in range(self.MDP.n):
            for j in range(self.MDP.n):
                A[i + self.MDP.n, j * self.MDP.n + i] = 1
            b[i + self.MDP.n] = rho_E[i]
        # Bellman constraint
        for i in range(self.MDP.n):
            A[i + 2 * self.MDP.n, self.MDP.n * self.MDP.n + i * self.MDP.m: self.MDP.n * self.MDP.n + (i + 1) * self.MDP.m] += 1
            for j in range(self.MDP.n):
                for k in range(self.MDP.m):
                    A[i + 2 * self.MDP.n, self.MDP.n * self.MDP.n + j * self.MDP.m + k] -= self.MDP.gamma * self.MDP.T[i, j, k]
                    # print("p({}|{}, {}):{} actual {}".format(i, j, k, self.MDP.T[i, j, k], self.real_MDP.T[i, j, k]))
            b[i + 2 * self.MDP.n] = (1 - self.MDP.gamma) * self.MDP.p0[i]
        A[3 * self.MDP.n, self.MDP.n * self.MDP.n:] = 1
        b[3 * self.MDP.n] = 1
        
        
        A_tmp, b_tmp = A.copy(), b.copy()
        
        A_tmp[self.MDP.n:3 * self.MDP.n], A_tmp[:self.MDP.n] = A[:2 * self.MDP.n], A[2 * self.MDP.n: 3*self.MDP.n] # to fit the deep learning version 
        b_tmp[self.MDP.n:3 * self.MDP.n], b_tmp[:self.MDP.n] = b[:2 * self.MDP.n], b[2 * self.MDP.n: 3*self.MDP.n]
        
        A, b = A_tmp.copy(), b_tmp.copy()
        
        
        
        x = cp.Variable(self.MDP.n * (self.MDP.n + self.MDP.m))
        #print("d:", d[0])
        # print(x[self.MDP.n * self.MDP.n:], rho_I.shape)
        #print("A:", A)
        #print("b:", b)
        
        rho_E_sampled = np.zeros(self.MDP.n)
        
        for i in range(len(TS_dataset)):
                rho_E_sampled[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
        
        U = 1e-10 + (rho_I.sum(axis=1).reshape(-1, 1) * rho_E.reshape(1, -1)).reshape(-1)
        # 
        objective = cp.Minimize(d @ x - epsilon_1 * cp.sum(cp.entr(x[:self.MDP.n * self.MDP.n])) + epsilon_1 * cp.sum(cp.multiply(x[:self.MDP.n * self.MDP.n], -np.log(U)))  - epsilon_2 * cp.sum(cp.entr(x[self.MDP.n * self.MDP.n:])) + epsilon_2 * cp.sum(cp.multiply(x[self.MDP.n * self.MDP.n:], - np.log(1e-10 + rho_I.reshape(-1)))))
        """
        y = cp.Variable(self.MDP.n * self.MDP.m)
        primal_objective = cp.Minimize(cp.sum(cp.multiply(-B @ R, y)) + cp.sum(cp.multiply(y, -np.log(d + 1e-10))) - cp.sum(cp.entr(y))) #  
        primal_constraint = [y >= 0, (1 - self.MDP.gamma) * p0 + self.MDP.gamma * P.T @ y == B.T @ y]
        primal_prob = cp.Problem(primal_objective, primal_constraint)
        primal_res = primal_prob.solve()
        print("primal result:", primal_res)
        """
        constraints = [x >= 0, A[:self.MDP.n*3] @ x == b[:self.MDP.n*3]]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        #print("result:", result)
        #print("dual - nonzero:", prob.constraints[0].dual_value.max())
        #print("dual - bellman:", prob.constraints[1].dual_value[:self.MDP.n])
        #print("dual - marginal with pi: ", prob.constraints[1].dual_value[self.MDP.n:self.MDP.n*2])
        #print("R:", -R0[:, 0])
        #print("dual - marginal with E:", prob.constraints[1].dual_value[self.MDP.n*2:self.MDP.n*3])
        #exit(0)
        """
        #####################
        A1, A2 = A[:, self.MDP.n ** 2], A[:, self.MDP.n ** 2:]
        dual_lambda = cp.Variable(3 * self.MDP.n + 1) 
        dual_objective = cp.Minimize(cp.max(A1.T @ dual_lambda - d) + epsilon_2 * cp.log(cp.sum(cp.multiply(cp.exp(A2.T @ dual_lambda), cp.maximum(rho_I.reshape(-1), 1e-10)))) - b.reshape(1, -1) @ dual_lambda)  # d^I & d^E are not in the limit! 
        dual_prob = cp.Problem(dual_objective)
        dual_result = dual_prob.solve()
        print("dual result:", dual_result)
        exit(0)
        #####################
        """
        
        Pi = np.maximum(x.value[:self.MDP.n * self.MDP.n], 1e-10)
        r_sa, r_s = np.zeros((self.MDP.n, self.MDP.m)), np.zeros(self.MDP.n)
        for i in range(self.MDP.n):
            for j in range(self.MDP.m):
                r_sa[i, j] = x.value[self.MDP.n * self.MDP.n + i * self.MDP.m + j]
                # print(i, j, r_sa[i, j])

                r_s[i] += x.value[self.MDP.n * self.MDP.n + i * self.MDP.m + j]
          
        #print("Pi:", Pi.reshape(self.MDP.n, self.MDP.n).sum(axis=1))
        print("d^pi:", r_s)      
        U2 = 1e-10 + (r_s.reshape(-1, 1) * rho_E.reshape(1, -1)).reshape(-1)
        #print("big_pi:", Pi.reshape(self.MDP.n, self.MDP.n) - U2.reshape(self.MDP.n, self.MDP.n))
        #print("KL of big pi and d^I cross d^E:", (Pi * np.log(Pi / U)).sum())
        #print("KL of big pi and d^pi cross d^E:", (Pi * np.log(Pi / U2)).sum())
        #print("Pi sum:", Pi.sum())
        #print("KL of d^pi and d_I:", (r_sa * np.log( (r_sa + 1e-10) / (rho_I + 1e-10) )).sum())

        np.set_printoptions(edgeitems=20, linewidth=5000)

        print("solved rho_s:", r_s.reshape(self.MDP.S, self.MDP.S)) 
        #exit(0)
        self.retrieve_policy(r_sa, r_s)
        print("retreived policy:", self.pi)
        exit(0)
        print("primal value:", prob.value)
        
        # print(d0 @ Pi)
        
        # ent = -Pi * np.log(Pi)
        """
        for i in range(self.MDP.n):
            print("Pi[", i, "] =", Pi[i * self.MDP.n:(i + 1) * self.MDP.n])
            print("ent[", i, "] =", ent[i * self.MDP.n:(i + 1) * self.MDP.n])
        """
        # print(np.sum(-Pi * np.log(Pi)))
        
        # print(np.sum(-r_sa.reshape(-1) * np.log(r_sa.reshape(-1))))
        
        # print(np.sum(r_sa.reshape(-1) * -np.log(1e-8 + rho_I.reshape(-1))))
        
        # mpv = d0 @ Pi - epsilon_1 * np.sum(-Pi * np.log(Pi+ 1e-8)) - epsilon_2 * np.sum(-r_sa * np.log(r_sa + 1e-8)) + epsilon_2 * np.sum(r_sa.reshape(-1) * -np.log(1e-8 + rho_I.reshape(-1)))
        
        # print("manual primal value:", mpv)
        # if args.seed == 114514: exit(0)
        # torch.save([Pi, r_sa, mpv, -prob.constraints[1].dual_value], "optimal.pt")
        # wandb.log({"primal": mpv}, commit=False)
        # for i in range(len(prob.constraints)):
        #    print("dual solution of constraint #", i, ":", prob.constraints[i].dual_value)
        
        lmbda = prob.constraints[1].dual_value
        
        # M2 = rho_I.reshape(-1) * np.exp(((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2)
        
        # print("retrieved Pi from lambda:", M1, M1.sum())
        #print("retreived rho_sa from lambda:", M2)
        # print("lmbda:", lmbda)
        lmbda = -lmbda
        
        print("AT1:", (A.T @ lmbda))
        
        print("lmbda:", lmbda, "value:", prob.value)

        print("-lmbda_|S|:2|S|\n", -lmbda[self.MDP.n:self.MDP.n*2].reshape(self.MDP.S, self.MDP.S))
        print("-gamma * lmbda_:|S|\n", -self.MDP.gamma * lmbda[:self.MDP.n].reshape(self.MDP.S, self.MDP.S))
        print("lmbda_:|S|-lmbda_|S|:2|S|\n", (lmbda[:self.MDP.n]-lmbda[self.MDP.n:self.MDP.n*2]).reshape(self.MDP.S, self.MDP.S))
        print("coeff:")
        
        coeff = np.zeros((self.MDP.S, self.MDP.S, 4))
        
        for i in range(self.MDP.S):
            for j in range(self.MDP.S):
                now = i * self.MDP.S + j
                for k in range(4):
                    future_lmbda = lmbda[:self.MDP.n].reshape(-1) @ self.MDP.T[:, now, k]
                    coeff[i, j, k] = ((-self.MDP.gamma * future_lmbda + lmbda[now] - lmbda[now + self.MDP.n]) / epsilon_2 - 1).item()
                    print(round(coeff[i, j, k], 2), end="")
                    print(end="/")
                print(end="   ")
            print()
        # print("d_I * coeff:")
        
        calculated_rho_sa = np.zeros((self.MDP.n, 4))
        
        for i in range(self.MDP.S):
            for j in range(self.MDP.S):
                now = i * self.MDP.S + j
                for k in range(4):
                    calculated_rho_sa[i*self.MDP.S+j, k] = np.exp(coeff[i, j, k]) * rho_I[i*self.MDP.S+j, k]
        
        # print("rho_I:", rho_I)
        
        # print("difference between rho_sa and calculated_rho_sa:", r_sa - calculated_rho_sa)
        
        # print("rho_I", rho_I[0, 0], rho_I[71, 2], np.exp(coeff[0, 0, 0]), np.exp(coeff[7, 8, 2]), r_sa[0, 0], r_sa[71, 2])
        # exit(0)
        
        # print("lmbda:", (((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2).max())
        M2 = rho_I.reshape(-1) * np.exp(((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2) + 1e-20
        
        # print("primal Pi:", Pi)
        # print("primal rho_sa:", r_sa)
        # print("retrieved Pi from minus lambda:", M1, M1.sum())
        # print("retreived rho_sa from minus lambda:", M2, M2.sum())
        
        
        # use retreived policy
        
        old_pi = self.pi.copy()
        
        self.pi = M2.reshape(self.MDP.n, self.MDP.m) / M2.reshape(self.MDP.n, self.MDP.m).sum(axis=1).reshape(-1, 1)
        # print((((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2).max())
        exit(0)
        print("pi:", self.pi)
        # print(np.exp(((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2))
        for i in range(self.MDP.S):
            for j in range(self.MDP.S):
                if np.count_nonzero(np.isnan(self.pi)) > 0: self.pi[i * self.MDP.S + j] = np.ones(self.MDP.m) / self.MDP.m # this condition is buggy, but let it remains like this and be a canary
                print( i, j, "newpi:", self.pi[i * self.MDP.S + j], "oldpi:", self.pi[i * self.MDP.S + j], "rsa:", r_sa[i * self.MDP.S + j])
        print("sum:", self.pi.sum())
        if self.visualize:
            self.visualizer.clear()
            self.visualizer.draw_grid()
            self.visualizer.draw_policy(self.pi)
            self.visualizer.fig.canvas.draw()
            data = np.frombuffer(self.visualizer.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.visualizer.fig.canvas.get_width_height()[::-1] + (3, ))
            # wandb.log({"direct_sovler_policy": wandb.Image(data, caption="direct solver policy")}, commit=False)
            self.visualizer.save("direct_solver_policy")
 
        """
        self.visualizer.clear()
        self.visualizer.draw_grid()
        self.visualizer.draw_traj(TS_dataset, "orange")
        self.visualizer.save("expert_dataset")
        """

    def draw_expert_policy(self):
        pi = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(self.MDP.n):
            x, y = self.MDP.get_pos(i)
            if self.MDP.edy < y: pi[i, 1] = 1 # left
            elif self.MDP.edy > y: pi[i, 0] = 1 # right
            elif self.MDP.edx < x: pi[i, 3] = 1 # up
            elif self.MDP.edx > x: pi[i, 2] = 1 # down  
            else: pi[i] = np.ones(self.MDP.m) / self.MDP.m # at optimal point
        if self.visualize:
            self.visualizer.clear()
            self.visualizer.draw_grid()
            self.visualizer.draw_policy(pi)
            self.visualizer.save("expert_policy")

    def evaluation(self, eval_use_argmax):
        T = 10
        if self.visualize:
            self.visualizer.clear()
            self.visualizer.draw_grid()
        avg_r, avg_suc, tot_l = 0, 0, 0    
        print("self.pi:", self.pi)
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
    parser.add_argument("--transition", type=str, default="estimated") # "ideal" or "estimated"
    parser.add_argument("--distance", type=str, default="dirac")
    parser.add_argument("--epsilon1", type=float, default=0.02)
    parser.add_argument("--epsilon2", type=float, default=0.02)
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
    
    MDP = GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)
    # MDP = GridWorld(2, 0, 0, 1, 1)
    
    TS_dataset = torch.load("data/"+str(N_expert_traj)+"_"+str(TA_expert_traj)+"_"+str(grid_size)+"_"+str(noise_level)+"_"+str(max_step)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TS.pt") # MDP.generate_expert_traj(N_expert_traj)
    
    TA_dataset = torch.load("data/"+str(N_expert_traj)+"_"+str(TA_expert_traj)+"_"+str(grid_size)+"_"+str(noise_level)+"_"+str(max_step)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TA.pt") # MDP.generate_random_traj(TA_expert_traj) # 1000 traj * 25 steps / traj (s,a,s') a list of length 25000
    
    MDP_estimate = copy.deepcopy(MDP)
    if args.transition == "estimated":
        MDP_estimate.T = np.zeros((MDP_estimate.n, MDP_estimate.n, 4)) # (s' | s, a)
        
        for i in range(len(TA_dataset)):
            MDP_estimate.T[TA_dataset[i]["next_state"], TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
        s = MDP_estimate.T.sum(axis=0)
        tag = (s == 0)
        # MDP_estimate.T = np.nan_to_num(np.ones_like(MDP_estimate.T) / MDP_estimate.n) * tag + np.nan_to_num(MDP_estimate.T / s.reshape([1]+list(s.shape))) * (1 - tag)
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
        
        # MDP_estimate.T = MDP.T
        
        # This is problematic: how could unseen state transport to arbitrary state with equal probability?
        
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
    
    # print(MDP.T - MDP_estimate.T)
    
    solver_sanity = Direct_Convex_Solver(MDP, MDP_estimate, runtime)
    t0 = time.time()
    solver_sanity.solve(TS_dataset, TA_dataset, args)
    t1 = time.time()
    # exit(0)
    """
    solver = Lagrangian_Solver(MDP, MDP_estimate, TA_dataset, runtime) # Direct_Convex_Solver(MDP, MDP_estimate, runtime)
    
    solver.solve(TS_dataset, args)
    
    """
    avg_r, suc_rate, avg_len = solver_sanity.evaluation(eval_use_argmax="yes")
    solver_sanity.draw_expert_policy()
    if suc_rate == 0: avg_len = 999999
    print("avg_rew:", avg_r, "suc_rate:", suc_rate, "avg_len:", avg_len, "runtime:", t1 - t0)
    f = open("res/ours_ent2KL_solver/"+runtime.replace("/", "-").replace(" ", "_")+"aka"+str(time.time())+"_"+str(epsilon_1)+"_"+str(epsilon_2)+".txt", "w")
    
    hyperparams["TS_type"] = args.TS_type
    hyperparams["distance"] = args.distance
    
    for key in hyperparams.keys():
        f.write(key+" "+str(hyperparams[key])+"\n")
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")
    
    avg_r, suc_rate, avg_len = solver_sanity.evaluation(eval_use_argmax="no")
        
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")
    
    f.close()

   
