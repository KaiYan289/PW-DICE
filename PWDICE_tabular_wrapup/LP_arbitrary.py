from gurobipy import GRB
import gurobipy as gp
import torch
from torch.optim import Adam
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
import subprocess
from datetime import datetime
from tqdm import tqdm
from hyperparams import ini_hpp

device = torch.device('cuda:0')

def concatlist(lst): # return a concatenation of list
    return sum(lst, [])

def list2gen(lst):
    return (_ for _ in lst)

class Solver:
    def __init__(self, MDP, real_MDP, time):
        self.MDP = MDP
        
        if self.MDP.ed >= 0:       
            self.MDP.T[:, self.MDP.ed, :] = 0
            self.MDP.T[self.MDP.ed, self.MDP.ed, :] = 1 # stay at the same location; effectively "absorbing state"
        
        self.real_MDP = real_MDP
        self.time = time
        self.dist = self.get_distance_matrix('dirac')

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
        return dist
    
    def retrieve_policy(self, r_sa, r_s): # r_sa is np array shape n * m (state * action), r_s is np array shape n.
        for i in range(self.MDP.n):  
            print("rsa:", r_sa[i], "rs:", r_s[i])
            if r_sa[i].sum() < 1e-12: self.pi[i] = np.ones(self.MDP.m) / self.MDP.m 
            else: 
                #assert np.abs(r_sa[i].sum() - r_s[i]) < 1e-10, "Error "+str(r_sa[i].sum() - r_s[i])+" "+str(r_sa[i])+" "+str(r_s[i])
                
                #r_sa[i] += (r_s[i] - r_sa[i].sum()) / self.MDP.m
                
                #self.pi[i] = r_sa[i] / r_s[i]
                self.pi[i] = r_sa[i] / r_sa[i].sum() # precisionissue; the sum of r_sa[i] is not necessarily r_s[i]
            print("pi["+str(i)+"]", self.pi[i])
    
    def solve(self, TS_dataset):
        raise NotImplementedError("Error!")
        
    def self_check(self):
        assert self.MDP.check_p0(), "MDP p0 error!"
        assert self.MDP.check_transition(), "MDP transition error!"
        
    def evaluation(self):
        T = 10
        for i in range(T):
            self.real_MDP.evaluation(self.pi, log=True)

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

class Direct_LP_Solver(Solver):
    def __init__(self, real_MDP, MDP, time):
        super().__init__(MDP, real_MDP, time)
        
    def solve(self, TS_dataset, args, extra_param=None):
        
        self.self_check()
        ev = gp.Env(empty=True)
        # ev.setParam('OutputFlag', 0)
        ev.start()
        dist = self.get_distance_matrix(args.distance)
        # print("dist:", dist[20].reshape(7, 7))
        # exit(0) 
        m = gp.Model("matrix1", env=ev)
        self.mode = args.TS_type
        # self variables: rho_sa, rho_s
        rho_E = np.zeros(self.MDP.n)
        rho_sa = m.addVars(self.MDP.n, self.MDP.m, vtype=GRB.CONTINUOUS, lb=0, name='rho_sa')
        rho_s =  m.addVars(self.MDP.n, vtype = GRB.CONTINUOUS, lb=0, name='rho_s')
        Pi_ss =  m.addVars(self.MDP.n, self.MDP.n, vtype=GRB.CONTINUOUS, lb=0, name='Pi_ss')
        
        # print(Pi_ss)
        m.setObjective(gp.quicksum(concatlist([[dist[i, j] * Pi_ss[i, j] for j in range(self.MDP.n)] for i in range(self.MDP.n)])))
        # m.addConstrs(list2gen(concatlist([[rho_sa[i, j] >= 0 for i in range(self.MDP.n)] for j in range(self.MDP.m)])))
        m.addConstrs(list2gen([rho_s[i] == gp.quicksum([rho_sa[i, j] for j in range(self.MDP.m)]) for i in range(self.MDP.n)]))
        
        m.addConstr(gp.quicksum([rho_s[i] for i in range(self.MDP.n)]) == 1)
        m.addConstrs(list2gen([rho_s[state] == (1 - self.MDP.gamma) * self.MDP.p0[state] + self.MDP.gamma * gp.quicksum(concatlist([[self.MDP.T[state, last_state, last_action] * rho_sa[last_state, last_action] for last_state in range(self.MDP.n)] for last_action in range(self.MDP.m)])) for state in range(self.MDP.n)])) 
        
        
        # TS_datase
        # for i in range(len(TS_dataset)): # only one perfect trajectory
        
        
        print("states in TS:", [TS_dataset[i]["state"] for i in range(len(TS_dataset))])
        
        # m.addConstrs(list2gen())
        N_expert_traj = 1 # / len(TS_dataset)
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
        elif self.mode == "goal":
            # goal-based
            rho_E[self.MDP.ed] = 1
        
        else:
            for i in range(len(TS_dataset)):
                rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)

        for i in range(self.MDP.n):
            print("rho_E[", i, "] =", rho_E[i])
        print("rho_E:", rho_E.sum())
        
        if extra_param is not None:
            if extra_param["GT_rho_E"]: 
                print("accurate!")
                pi = self.MDP.expert_pi if extra_param["argmax"] == 0 else self.MDP.expert_pi_argmax
                rho_E = compute_ss(self.MDP, pi).sum(axis=1)
            rho_E_sa = compute_marginal_distribution(self.MDP, self.MDP.expert_pi)
            # print("difference:", rho_E - rho_E_sa.sum(axis=1))
            # Empirically proved that the rho_E == rho_E_sa.sum(axis=1).
            # if extra_param["GT_rho_I"]: rho_I = self.compute_marginal_distribution(self.MDP, np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m)
        
        # exit(0)
        m.addConstrs((gp.quicksum([Pi_ss[i, j] for i in range(self.MDP.n)]) == rho_E[j] for j in range(self.MDP.n)))
        m.addConstrs((gp.quicksum([Pi_ss[i, j] for j in range(self.MDP.n)]) == rho_s[i] for i in range(self.MDP.n)))

        m.optimize()
        
        
        
        r_sa, r_s = np.zeros((self.MDP.n, self.MDP.m)), np.zeros(self.MDP.n)
        print("rho_E:", rho_E)
        for i in range(self.MDP.n):
            # print("solved_rhoS:", sum([rho_sa[i, j].X for j in range(self.MDP.m)]), rho_E[i])
            for j in range(self.MDP.m):
                # print(i, j, rho_sa[i, j].X, rho_E_sa[i, j])
                assert rho_sa[i, j].X > -1e-5, "Negative answer! {} {} {}".format(i, j, rho_sa[i, j].X)
                r_sa[i, j] = max(rho_sa[i, j].X, 0)
        
        for i in range(self.MDP.n): 
            r_s[i] = rho_s[i].X
            # print("rho_s[", i, "]:", r_s[i], "rho_sa[", i, "]:", r_sa[i])
        # print("rho_s:", r_s.sum())
        
        # print('TS-S:', [TS_dataset[i]["state"] for i in range(len(TS_dataset))])
        # print("TS-A:", [TS_dataset[i]["action"] for i in range(len(TS_dataset))])
        self.retrieve_policy(r_sa, r_s)


    def draw_expert_policy(self):
        pi = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(self.MDP.n):
            x, y = self.MDP.get_pos(i)
            if self.MDP.edy < y: pi[i, 1] = 1 # left
            elif self.MDP.edy > y: pi[i, 0] = 1 # right
            elif self.MDP.edx < x: pi[i, 3] = 1 # up
            elif self.MDP.edx > x: pi[i, 2] = 1 # down  
            else: pi[i] = np.ones(self.MDP.m) / self.MDP.m # at optimal point


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
        return avg_r, avg_suc / T, 999999 if avg_suc == 0 else tot_l / avg_suc


    
def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--N_expert_traj", type=int)
    parser.add_argument("--TA_expert_traj", type=int)
    parser.add_argument("--TA_optimality", type=float, default=0)
    parser.add_argument("--TS_type", type=str, default="full") # "full" or "goal"
    parser.add_argument("--noise_level", type=float)
    parser.add_argument("--distance", type=str, default="dirac")
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
    
    f = open("res/LDexperiment/LP/LP-"+str(args.TA_expert_traj)+"-"+str(args.N_expert_traj)+"-"+str(args.noise_level)+".txt", 'w')
    
    for data_index in range(10):
        MDP = torch.load("data/LobsDICE_"+str(args.TA_expert_traj)+"_"+str(args.N_expert_traj)+"_"+("1" if args.noise_level == 1 else str(args.noise_level))+"/"+str(data_index)+"/MDP.pt")# GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)
        # MDP = GridWorld(2, 0, 0, 1, 1)
        print(MDP.target)
        # exit(0)
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
            
            # MDP_estimate.T = MDP.T
            
            # This is problematic: how could unseen state transport to arbitrary state with equal probability?
    
        
        # print(MDP.T - MDP_estimate.T)
        MDP_estimate_exact = copy.deepcopy(MDP)
        #print("T before:", np.linalg.norm(MDP.T - MDP_estimate_exact.T))
        
        # print("expert_pi:", MDP.expert_pi, MDP_estimate_exact.expert_pi)
        #print("T after:", np.linalg.norm(MDP.T - MDP_estimate_exact.T))
        #print("estimated MDP:", np.linalg.norm(compute_ss(MDP, solver.pi) - compute_ss(MDP, MDP.expert_pi), 1))
        #print(compute_marginal_distribution(MDP_estimate_exact, MDP.expert_pi).sum(axis=1))
        #print("single state error:", np.linalg.norm(compute_marginal_distribution(MDP, solver.pi).sum(axis=1) - compute_marginal_distribution(MDP, MDP.expert_pi).sum(axis=1)))
        # print("solver.pi:", solver.pi, "expert_pi:", MDP.expert_pi)
           
        for i, solver in enumerate([Direct_LP_Solver(MDP, MDP_estimate, runtime), Direct_LP_Solver(MDP, MDP_estimate_exact, runtime)]):
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
                        f.write(str(A0)+" "+str(np.linalg.norm(DA0 - EA0, 1))+" "+str(EA0[MDP.target] - DA0[MDP.target])+" ")
        f.write("\n")
    f.close()
    print("running ends!!!")
    