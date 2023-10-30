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
#import wandb
import subprocess
from datetime import datetime
from tqdm import tqdm
from hyperparams import ini_hpp

device = torch.device('cuda:0')

def concatlist(lst): # return a concatenation of list
    return sum(lst, [])

def list2gen(lst):
    return (_ for _ in lst)

hyperparams = ini_hpp("params/params_LP.txt")

N_expert_traj = hyperparams["N_expert_traj"]
TA_expert_traj = hyperparams["TA_expert_traj"]
grid_size = hyperparams["grid_size"]
max_step = hyperparams["max_step"]
noise_level = hyperparams["noise_level"]
TA_optimality = hyperparams["TA_optimality"]


class Solver:
    def __init__(self, MDP, real_MDP, time):
        self.MDP = MDP
        
               
        self.MDP.T[:, self.MDP.ed, :] = 0
        self.MDP.T[self.MDP.ed, self.MDP.ed, :] = 1 # stay at the same location; effectively "absorbing state"
        
        self.real_MDP = real_MDP
        self.time = time
        self.dist = self.get_distance_matrix('dirac')
        self.visualizer = Plotter(int(math.sqrt(self.MDP.n)), self.MDP.st, self.MDP.ed, self.time, directory='res/ours/fig')
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
        elif typ == "euclidean":
            assert isinstance(self.MDP, GridWorld), "Cannot Apply Euclidean!"
            dist = np.zeros((self.MDP.n, self.MDP.n))
            for i in range(self.MDP.n):
                for j in range(self.MDP.n):
                    posx = self.MDP.get_pos(i)
                    posy = self.MDP.get_pos(j)
                    dist[i, j] = ((np.array(posx) - np.array(posy)) ** 2).sum()
        return dist * 10
    
    def retrieve_policy(self, r_sa, r_s): # r_sa is np array shape n * m (state * action), r_s is np array shape n.
        for i in range(self.MDP.n):  
            if r_s[i] < 1e-12: self.pi[i] = np.ones(self.MDP.m) / self.MDP.m 
            else: 
                self.pi[i] = r_sa[i] / r_s[i]
                self.pi[i] /= self.pi[i].sum() # precisionissue; the sum of r_sa[i] is not necessarily r_s[i]
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

class Direct_LP_Solver(Solver):
    def __init__(self, real_MDP, MDP, time):
        super().__init__(MDP, real_MDP, time)
        
    def solve(self, TS_dataset, args):
        
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
        N = 1 / len(TS_dataset)
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

        # exit(0)
        m.addConstrs((gp.quicksum([Pi_ss[i, j] for i in range(self.MDP.n)]) == rho_E[j] for j in range(self.MDP.n)))
        m.addConstrs((gp.quicksum([Pi_ss[i, j] for j in range(self.MDP.n)]) == rho_s[i] for i in range(self.MDP.n)))

        m.optimize()
        print(m.ObjVal)
        exit(0)
        r_sa, r_s = np.zeros((self.MDP.n, self.MDP.m)), np.zeros(self.MDP.n)
        for i in range(self.MDP.n):
            for j in range(self.MDP.m):
                assert rho_sa[i, j].X > -1e-5, "Negative answer! {} {} {}".format(i, j, rho_sa[i, j].X)
                r_sa[i, j] = max(rho_sa[i, j].X, 0)
        
        for i in range(self.MDP.n): 
            r_s[i] = rho_s[i].X
            print("rho_s[", i, "]:", r_s[i], "rho_sa[", i, "]:", r_sa[i])
        print("rho_s:", r_s.sum())
        print('TS-S:', [TS_dataset[i]["state"] for i in range(len(TS_dataset))])
        print("TS-A:", [TS_dataset[i]["action"] for i in range(len(TS_dataset))])
        self.retrieve_policy(r_sa, r_s)
        self.visualizer.clear()
        self.visualizer.draw_grid()
        self.visualizer.draw_policy(self.pi)
        self.visualizer.save("policy")
        self.visualizer.clear()
        self.visualizer.draw_grid()
        self.visualizer.draw_traj(TS_dataset, "orange")
        self.visualizer.save("expert_dataset")

    def draw_expert_policy(self):
        pi = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(self.MDP.n):
            x, y = self.MDP.get_pos(i)
            if self.MDP.edy < y: pi[i, 1] = 1 # left
            elif self.MDP.edy > y: pi[i, 0] = 1 # right
            elif self.MDP.edx < x: pi[i, 3] = 1 # up
            elif self.MDP.edx > x: pi[i, 2] = 1 # down  
            else: pi[i] = np.ones(self.MDP.m) / self.MDP.m # at optimal point
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
            self.visualizer.draw_traj(agent_buffer, "orange")
        self.visualizer.save("agent_traj")
        return avg_r, avg_suc / T, 999999 if avg_suc == 0 else tot_l / avg_suc

    
def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--data_index", help="data_index", type=int, default=0)
    parser.add_argument("--solver_type", type=str, default="LP") # LP (linear programming) or FN (flow network)
    parser.add_argument("--TS_type", type=str, default="full") # "full" or "goal"
    parser.add_argument("--transition", type=str, default="estimated") # "ideal" or "estimated"
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
    solver_type = args.solver_type
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
        MDP_estimate.T = np.zeros((MDP_estimate.n, MDP_estimate.n, MDP_estimate.m)) # (s' | s, a)
        N = np.zeros((MDP.n, MDP.m))
        for i in range(len(TA_dataset)):
            N[TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
            if TA_dataset[i]["next_state"] == MDP.ed:
                print("the last step to arrive:", TA_dataset[i]["state"], TA_dataset[i]["action"])
            MDP_estimate.T[TA_dataset[i]["next_state"], TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
        s = MDP_estimate.T.sum(axis=0)
        tag = (s == 0)
        #for i in range(MDP.n):
        #    print("MDP data at(",i // MDP.S,",", i % MDP.S ,"):", s[i].astype('int'))
        #exit(0)
        for i in range(MDP.n):
            for j in range(MDP.m):
                print("state {} action {}: {}".format(i, j, N[i, j]))
        # exit(0)
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
    
    print(MDP.T - MDP_estimate.T)
    

    solver = Direct_LP_Solver(MDP, MDP_estimate, runtime)
    t0 = time.time()
    solver.solve(TS_dataset, args)
    t1 = time.time()
    avg_r, suc_rate, avg_len = solver.evaluation(eval_use_argmax="yes")
    solver.draw_expert_policy()
    if suc_rate == 0: avg_len = 999999
    print("avg_rew:", avg_r, "suc_rate:", suc_rate, "avg_len:", avg_len, "runtime:", t1 - t0)
    f = open("res/ours/"+runtime.replace("/", "-").replace(" ", "_")+"aka"+str(time.time())+".txt", "w")
    
    hyperparams["TS_type"] = args.TS_type
    hyperparams["distance"] = args.distance
    
    for key in hyperparams.keys():
        f.write(key+" "+str(hyperparams[key])+"\n")
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")
    
    avg_r, suc_rate, avg_len = solver.evaluation(eval_use_argmax="no")
    
    f.write(str(avg_r)+" "+str(suc_rate)+" "+str(avg_len)+" "+str(t1 - t0)+"\n")
    
    f.close()
    # np.set_printoptions(precision=2)
    """
    for i in range(9):
        for j in range(9):
            print("{:.2f}".format(np.linalg.norm((MDP.T - MDP_estimate.T)[:, i * 9 + j, :])), end=" ")
        print("")
    """
    
    
