import os
import torch
import sys
import numpy as np
import argparse
import random
sys.path.append("..")
from tabular_MDP import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    args = parser.parse_args()
    return args

args = get_args()

N_expert_traj = 5
N_TA_traj = 1000
grid_size = 11
noise_level = 0
max_step = 40
TA_optimality = -10

name = str(N_expert_traj) + "_" + str(N_TA_traj) + "_" + str(grid_size) + "_" + str(noise_level) + "_" + str(max_step) + "_" + str(TA_optimality)

if not os.path.exists(name): os.mkdir(name)
idx, TA_cnt = 0, N_TA_traj
while True:
    if not os.path.exists(name + "/" + str(idx)):
        os.mkdir(name + "/" + str(idx))
        break
    idx += 1
seed = args.seed
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
np.random.seed(seed) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

TS_suc_rate, TA_suc_count = 0, 0
# MDP = GridWorld(grid_size, 0, 0, grid_size - 1, grid_size - 1, noise=noise_level, max_step=max_step)
MDP = GridWorld(grid_size, 1, 2, 6, 5, noise=noise_level, max_step=max_step)

TA = []
TS = MDP.generate_expert_traj(N_expert_traj, balance=(TA_optimality == -2))

assert grid_size ** 2 < N_TA_traj

vis = np.zeros((MDP.n))

for i in range(len(TS)):
    vis[TS[i]["next_state"]] = 1 
    if TS[i]["next_state"] == MDP.ed: TS_suc_rate += 1 / N_expert_traj
print("TS_suc_rate:", TS_suc_rate)
# 2 is old way of generating data
# -1 is randomly selecting start and end
# -2 is let optimal policy vary
# -3 is randomly selecting start and end, plus some data banging on the wall (30% or so?)
# -4 is randomly selecting start and end, then use random policy
if TA_optimality in [-1, -2]:
    for i in range(N_TA_traj):
        st, ed = np.random.randint(0, MDP.n), np.random.randint(0, MDP.n)
        while st == ed: ed = np.random.randint(0, MDP.n) 
        MDP_alter = GridWorld(grid_size, st // grid_size, st % grid_size, ed // grid_size, ed % grid_size, noise=noise_level, max_step=max_step)
        ex = MDP_alter.generate_expert_traj(1)
        TA.extend(ex)
elif TA_optimality <= -10 and TA_optimality >= -20: #TA_optimality in [-11, -10]:
    part = int(N_TA_traj * (TA_optimality + 20) * 0.1) # -10 is fully normal; -11 is fully poisoned
    for i in range(part):
        st, ed = np.random.randint(0, MDP.n), np.random.randint(0, MDP.n)
        while st == ed: ed = np.random.randint(0, MDP.n)
        MDP_alter = GridWorld(grid_size, st // grid_size, st % grid_size, ed // grid_size, ed % grid_size, noise=noise_level, max_step=max_step)
        ex = MDP_alter.generate_expert_traj(1)
        TA.extend(ex)
    for i in range(part, N_TA_traj):
        _ = np.random.random()
        # policy = np.ones((MDP.n, MDP.m)) / MDP.m
        # print(MDP.S, "_:", _)
        
        if _ < 0.5: # (x, S)
            st, ed = np.random.randint(0, MDP.S), MDP.S ** 2 - 1
            # policy = np.array([[0, 1, 0, 0] for _ in range(MDP.n)])
            policy = np.array([[0, 0, 0, 1] for _ in range(MDP.n)])
        else: # (S, x)
            st, ed = np.random.randint(0, MDP.S) * MDP.S + MDP.S - 1, 0
            # policy = np.array([[0, 0, 0, 1] for _ in range(MDP.n)])
            policy = np.array([[0, 1, 0, 0] for _ in range(MDP.n)])
        
        """
        st, ed = MDP.n - 1, 0
        policy = np.array([[0, 0, 0, 1] for _ in range(MDP.n)])
        """
        MDP_alter = GridWorld(grid_size, st // grid_size, st % grid_size, ed // grid_size, ed % grid_size, noise=noise_level, max_step=max_step)
        ex = MDP_alter.evaluation(policy, collect=True)
        # print([_["state"] for _ in ex])
        TA.extend(ex)
        # exit(0)
elif TA_optimality == -4:
    for i in range(N_TA_traj):
        st, ed = np.random.randint(0, MDP.n), np.random.randint(0, MDP.n)
        while st == ed: ed = np.random.randint(0, MDP.n) 
        MDP_alter = GridWorld(grid_size, st // grid_size, st % grid_size, ed // grid_size, ed % grid_size, noise=noise_level, max_step=max_step)
        ex = MDP_alter.generate_random_traj(1)
        TA.extend(ex)
else:
    for i in range(MDP.n):
        # print(i, vis[i], MDP.st)
        if vis[i] == 1 and i != MDP.st:
            # print(i, MDP.ed)
            
            MDP_alter = GridWorld(grid_size, 0, 0, i // grid_size, i % grid_size, noise=noise_level, max_step=max_step)
            
            while True:
                ex = MDP_alter.generate_expert_traj(1)
                # print(ex)
                if ex[-1]["next_state"] == i: break
                print("regen...", ex[-1]["next_state"], i)
            
            # print('ex:', ex)
            TA.extend(ex)
            TA_cnt -= 1
    
    TA.extend(MDP.generate_random_traj(TA_cnt, optimality=0))

# TA = MDP.generate_random_traj(TA_expert_traj)

cnt = np.zeros((MDP.n, MDP.m)) 
for i in range(len(TA)):
    cnt[TA[i]["state"], TA[i]["action"]] += 1
    if TA[i]["next_state"] == MDP.ed: TA_suc_count += 1

assert TA_suc_count > 0, "Error!"

torch.save(TA, name + "/" + str(idx) + "/TA.pt")
torch.save(TS, name + "/" + str(idx) + "/TS.pt")
f = open(name + "/" + str(idx) + "/summary.txt", "w")
f.write("seed: "+str(seed)+"\n")
f.write("state-action pairs that reaches the goal in TA: " + str(TA_suc_count) + "\n")
f.write("success rate of TS: " + str(TS_suc_rate) + "\n")
f.write("state-action pairs count in TA:\n")
for i in range(MDP.n):
    for j in range(MDP.m):
        f.write(str(cnt[i, j])+" ")
    f.write("\n")
f.close()
