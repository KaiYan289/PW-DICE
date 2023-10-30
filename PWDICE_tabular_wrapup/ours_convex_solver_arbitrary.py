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
import subprocess
# from ours_convex_solver import Solver, Direct_Convex_Solver
from datetime import datetime
from tqdm import tqdm
from hyperparams import ini_hpp
import cvxpy as cp
device = torch.device('cuda:0')

def concatlist(lst): # return a concatenation of list
    return sum(lst, [])

def list2gen(lst):
    return (_ for _ in lst)

hyperparams = ini_hpp("params/params_ours_convex_solver_arbitrary.txt")

N_expert_traj = hyperparams["N_expert_traj"]
TA_expert_traj = hyperparams["TA_expert_traj"]
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
            self.visualizer = Plotter(int(math.sqrt(self.MDP.n)), self.MDP.st, self.MDP.ed, self.time, directory='res/ours_convex_solver/fig')
        self.pi = np.zeros((self.MDP.n, self.MDP.m))

    def get_distance_matrix(self, typ):
        if typ == 'dirac':
            dist = np.ones((self.MDP.n, self.MDP.n)) 
            for i in range(self.MDP.n): dist[i, i] = 0.01
        elif typ == 'manhattan':
            assert isinstance(self.MDP, GridWorld), "Cannot apply manhattan!"
            dist = np.zeros((self.MDP.n, self.MDP.n))
            for i in range(self.MDP.n):
                for j in range(self.MDP.n):
                    posx = self.MDP.get_pos(i)
                    posy = self.MDP.get_pos(j)
                    dist[i, j] = abs(np.array(posx) - np.array(posy)).sum() + 0.01
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
        
        self.self_check()
        # ev = gp.Env(empty=True)
        # ev.setParam('OutputFlag', 0)
        # ev.start()
        dist = self.get_distance_matrix(args.distance)
        # print("dist:", dist[20].reshape(7, 7))
        # exit(0) 
        # m = gp.Model("matrix1", env=ev)
        self.mode = args.TS_type
        # self variables: rho_sa, rho_s
        rho_E = np.zeros(self.MDP.n)
        rho_I = np.zeros((self.MDP.n, self.MDP.m))
        """
        
        rho_sa = m.addVars(self.MDP.n, self.MDP.m, vtype=GRB.CONTINUOUS, lb=0, name='rho_sa')
        rho_s =  m.addVars(self.MDP.n, vtype = GRB.CONTINUOUS, lb=0, name='rho_s')
        Pi_ss =  m.addVars(self.MDP.n, self.MDP.n, vtype=GRB.CONTINUOUS, lb=0, name='Pi_ss')
        """
        
        x = cp.Variable(self.MDP.n * (self.MDP.n + self.MDP.m))
        
        d0 = dist.reshape(-1)
        # print("shape:", d0.shape)
        d = np.concatenate([d0, np.zeros((self.MDP.n * self.MDP.m))]).reshape(1, -1)
        # print("shape:", d.shape)
        epsilon_1, epsilon_2 =  0.02, 0.02
        
        N = 1 / len(TS_dataset)
        """
        for i in range(len(TA_dataset)):
                # rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
                # rho_I[TA_dataset[i]["state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TA_dataset[i]["step"]) / TA_expert_traj
                rho_I[TA_dataset[i]["state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TA_dataset[i]["step"]) / TA_expert_traj
                # print(TS_dataset[i]["next_state"], self.MDP.ed)
                #if TA_dataset[i]["next_state"] == self.MDP.ed:
                #    rho_I[self.MDP.ed, :] += self.MDP.gamma ** (TA_dataset[i]["step"] + 1) / (TA_expert_traj * self.MDP.m)
        """
        pi_b = np.zeros((self.MDP.n, self.MDP.m))
        for i in range(len(TA_dataset)):
            pi_b[TA_dataset[i]["state"], TA_dataset[i]["action"]] += 1
        for i in range(self.MDP.n):
            if pi_b[i].sum() == 0: pi_b[i] = np.ones(self.MDP.m) / self.MDP.m
            else: pi_b[i] /= pi_b[i].sum()
        rho_I = self.compute_marginal_distribution(self.MDP, pi_b)
        if self.mode == "full":
            # full expert dataset
            for i in range(len(TS_dataset)):
                # rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
                rho_E[TS_dataset[i]["state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TS_dataset[i]["step"])
        elif self.mode == "goal":
            # goal-based
            rho_E[self.MDP.ed] = 1
        
        else:
            for i in range(len(TS_dataset)):
                rho_E[TS_dataset[i]["state"]] += 1 / len(TS_dataset)
        
        print("sum:", rho_E.sum())
        rho_E /= rho_E.sum()
        if extra_param is not None:
            old_rho_E = rho_E.copy()
            expert_pi = self.MDP.expert_pi if extra_param['argmax'] == 0 else self.MDP.expert_pi_argmax
            if extra_param["GT_rho_E"]: rho_E = self.compute_marginal_distribution(self.MDP, expert_pi).sum(axis=1)
            print("subE:", old_rho_E - self.compute_marginal_distribution(self.MDP, expert_pi).sum(axis=1))
            
            p1, p2 = old_rho_E, self.compute_marginal_distribution(self.MDP, expert_pi).sum(axis=1)
            
            old_rho_I = rho_I.copy()
            if extra_param["GT_rho_I"]: rho_I = self.compute_marginal_distribution(self.MDP, np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m)
            print("subI:", old_rho_I - self.compute_marginal_distribution(self.MDP, np.ones((self.MDP.n, self.MDP.m)) / self.MDP.m))
            
            v1, v2 = np.zeros((self.MDP.n, self.MDP.n)), np.zeros((self.MDP.n, self.MDP.n))
            
            for i in range(len(TS_dataset)):
                v1[TS_dataset[i]["state"], TS_dataset[i]["next_state"]] += (1 - self.MDP.gamma) * (self.MDP.gamma ** TS_dataset[i]["step"])
            v1 /= v1.sum()
            for i in range(self.MDP.n):
                for j in range(self.MDP.n):
                    v2[i, j] = p2[i] * (self.MDP.T[j, i, :] * expert_pi[i]).sum()
            print(v1 - v2, (v1 - v2).sum())
            # compute SS
            # exit(0)
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
        #print("d:", d[0])
        # print(x[self.MDP.n * self.MDP.n:], rho_I.shape)
        #print("A:", A)
        #print("b:", b)
        
        objective = cp.Minimize(d @ x - epsilon_1 * cp.sum(cp.entr(x[:self.MDP.n * self.MDP.n])) - epsilon_2 * cp.sum(cp.entr(x[self.MDP.n * self.MDP.n:])) + epsilon_2 * cp.sum(cp.multiply(x[self.MDP.n * self.MDP.n:], - np.log(1e-8 + rho_I.reshape(-1)))))
        constraints = [x >= 0, A @ x == b]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        # m.optimize()
        Pi = np.maximum(x.value[:self.MDP.n * self.MDP.n], 1e-14)
        r_sa, r_s = np.zeros((self.MDP.n, self.MDP.m)), np.zeros(self.MDP.n)
        for i in range(self.MDP.n):
            for j in range(self.MDP.m):
                r_sa[i, j] = x.value[self.MDP.n * self.MDP.n + i * self.MDP.m + j]
                r_s[i] += x.value[self.MDP.n * self.MDP.n + i * self.MDP.m + j]
        np.set_printoptions(edgeitems=20, linewidth=5000, precision=4)
        
        self.retrieve_policy(r_sa, r_s)
        
        # print("retreived policy:", self.pi)
        
        # print("primal value:", prob.value)
        
        # print(d0 @ Pi)
        
        ent = -Pi * np.log(Pi)
        """
        for i in range(self.MDP.n):
            print("Pi[", i, "] =", Pi[i * self.MDP.n:(i + 1) * self.MDP.n])
            print("ent[", i, "] =", ent[i * self.MDP.n:(i + 1) * self.MDP.n])
        """
        # print(np.sum(-Pi * np.log(Pi)))
        
        # print(np.sum(-r_sa.reshape(-1) * np.log(r_sa.reshape(-1))))
        
        # print(np.sum(r_sa.reshape(-1) * -np.log(1e-8 + rho_I.reshape(-1))))
        
        mpv = d0 @ Pi - epsilon_1 * np.sum(-Pi * np.log(Pi+ 1e-8)) - epsilon_2 * np.sum(-r_sa * np.log(r_sa + 1e-8)) + epsilon_2 * np.sum(r_sa.reshape(-1) * -np.log(1e-8 + rho_I.reshape(-1)))
        
        print("manual primal value:", mpv)
        # if args.seed == 114514: exit(0)
        torch.save([Pi, r_sa, mpv, -prob.constraints[1].dual_value], "optimal.pt")
        # wandb.log({"primal": mpv}, commit=False)
        # for i in range(len(prob.constraints)):
        #    print("dual solution of constraint #", i, ":", prob.constraints[i].dual_value)
        
        lmbda = prob.constraints[1].dual_value
        
        M1, M2 = (np.exp(((A.T @ lmbda)[:self.MDP.n * self.MDP.n] - d0 - epsilon_1) / epsilon_1)), rho_I.reshape(-1) * np.exp(((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2)
        
        # print("retrieved Pi from lambda:", M1, M1.sum())
        #print("retreived rho_sa from lambda:", M2)
        
        lmbda = -lmbda
        
        M1, M2 = (np.exp(((A.T @ lmbda)[:self.MDP.n * self.MDP.n] - d0 - epsilon_1) / epsilon_1)), rho_I.reshape(-1) * np.exp(((A.T @ lmbda)[self.MDP.n * self.MDP.n:] - epsilon_2) / epsilon_2)
        
        # print("primal Pi:", Pi)
        # print("primal rho_sa:", r_sa)
        # print("retrieved Pi from minus lambda:", M1, M1.sum())
        # print("retreived rho_sa from minus lambda:", M2, M2.sum())
        # exit(0)
        
        # use retreived policy
        
        self.pi = M2.reshape(self.MDP.n, self.MDP.m) / M2.reshape(self.MDP.n, self.MDP.m).sum(axis=1).reshape(-1, 1)
        print("sum:", self.pi.sum())
        if self.visualize:
            self.visualizer.clear()
            self.visualizer.draw_grid()
            self.visualizer.draw_policy(self.pi)
            self.visualizer.fig.canvas.draw()
            data = np.frombuffer(self.visualizer.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.visualizer.fig.canvas.get_width_height()[::-1] + (3, ))
        # wandb.log({"direct_sovler_policy": wandb.Image(data, caption="direct solver policy")}, commit=False)
        # self.visualizer.save("direct_solver_policy")
 
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
    
    MDP = torch.load("data/LobsDICE_"+str(TA_expert_traj)+"_"+str(N_expert_traj)+"_"+str(noise_level)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/MDP.pt")# GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)
    # MDP = GridWorld(2, 0, 0, 1, 1)
    
    TS_dataset = torch.load("data/LobsDICE_"+str(TA_expert_traj)+"_"+str(N_expert_traj)+"_"+str(noise_level)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TS.pt") # MDP.generate_expert_traj(N_expert_traj)
    TS2_dataset = torch.load("data/LobsDICE_"+str(TA_expert_traj)+"_"+str(N_expert_traj)+"_"+str(noise_level)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TS_argmax.pt")
    TA_dataset = torch.load("data/LobsDICE_"+str(TA_expert_traj)+"_"+str(N_expert_traj)+"_"+str(noise_level)+"_"+str(TA_optimality)+"/"+str(args.data_index)+"/TA.pt") # MDP.generate_random_traj(TA_expert_traj) # 1000 traj * 25 steps / traj (s,a,s') a list of length 25000
    
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
    hyperparams["TS_type"] = args.TS_type
    hyperparams["distance"] = args.distance
    
    
    f = open("res/LDexperiment/ours_Entreg.txt", 'a')
    f.write(str(runtime)+" ")
    for key in hyperparams.keys():
       f.write(str(hyperparams[key])+" ")
       
    for i, solver in enumerate([Direct_Convex_Solver(MDP, MDP_estimate, runtime, visualize=False), Direct_Convex_Solver(MDP, MDP_estimate_exact, runtime, visualize=False)]):
        for j, TS in enumerate([TS_dataset, TS2_dataset]):
            for GT_rho_E in [False, True]:
                for GT_rho_I in [False, True]:
                    solver.solve(TS, TA_dataset, args, extra_param={"GT_rho_E": GT_rho_E, "GT_rho_I": GT_rho_I, "argmax": j})
                    A0 = np.linalg.norm(compute_ss(MDP, solver.pi) - compute_ss(MDP, MDP.expert_pi), 1)
                    DA0 = compute_marginal_distribution(MDP, solver.pi).sum(axis=1)
                    EA0 = compute_marginal_distribution(MDP, MDP.expert_pi).sum(axis=1) if j == 0 else compute_marginal_distribution(MDP, MDP.expert_pi_argmax).sum(axis=1)
                    print("total state occupancy difference:", np.linalg.norm(DA0 - EA0, 1))
                    print("regret:", EA0[MDP.target] - DA0[MDP.target])
                    f.write(str(A0)+" "+str(np.linalg.norm(DA0 - EA0, 1))+" "+str(EA0[MDP.target] - DA0[MDP.target])+" ")
    f.write("\n")
    f.close() 
   
