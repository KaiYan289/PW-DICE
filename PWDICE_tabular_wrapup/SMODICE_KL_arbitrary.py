import argparse
import random
import numpy as np
import torch
import copy
from tqdm import tqdm
import math
import subprocess
from datetime import datetime
from tabular_MDP import TabularMDP, GridWorld
from visualizer import Plotter
from SMODICE_KL_cvxpy import SMODICE_Solver
from hyperparams import ini_hpp
import time

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
    
    f = open("res/LDexperiment/SMODICE-KL/SMODICE-KL-"+str(args.TA_expert_traj)+"-"+str(args.N_expert_traj)+"-"+str(args.noise_level)+".txt", 'w')
    
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
            
            # MDP_estimate.T = MDP.T
            
            # This is problematic: how could unseen state transport to arbitrary state with equal probability?
    
        
        # print(MDP.T - MDP_estimate.T)
        MDP_estimate_exact = copy.deepcopy(MDP)
    
        # exit(0)
        """
        solver = Lagrangian_Solver(MDP, MDP_estimate, TA_dataset, runtime) # SMODICE_Solver(MDP, MDP_estimate, runtime)
        
        solver.solve(TS_dataset, args) 
        
        """
        
        
        # f = open("res/ours_convex_solver/"+runtime.replace("/", "-").replace(" ", "_")+"aka"+str(time.time())+".txt", "w")

        for i, solver in enumerate([SMODICE_Solver(MDP, MDP_estimate, TA_dataset, runtime, visualize=False), SMODICE_Solver(MDP, MDP_estimate_exact, TA_dataset, runtime, visualize=False)]):
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