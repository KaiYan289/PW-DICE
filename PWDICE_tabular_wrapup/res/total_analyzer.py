import os
import argparse
import torch
import copy
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_expert_traj", type=int, default=5)
    parser.add_argument("--TA_expert_traj", type=int, default=1000)
    parser.add_argument("--grid_size", type=int, default=9)
    parser.add_argument("--max_step", type=int, default=40)
    parser.add_argument("--noise_level", type=int, default=0.1)
    parser.add_argument("--TS_type", type=str, default="full") # "full" or "goal"
    # parser.add_argument("--distance", type=str, default="dirac")
    parser.add_argument("--TA_optimality", type=str, default="-10")
    #parser.add_argument("--earliest", type=str, default="")
    args = parser.parse_args()
    return args
    
args_old = get_args()

if os.path.exists("restotal.txt"):
    os.system("rm restotal.txt")


# for name in ["ours_dirac", "ours_manhattan",  "ours_convex_solver_dirac", "ours_convex_solver_manhattan", "ours_Lagrangian_dirac", "ours_Lagrangian_manhattan","ours_exact_penalty_solver", "SMODICE_CHI", "SMODICE_CHI_CVXPY", "SMODICE_KL", "SMODICE_KL_nosample", "SMODICE_KL_CVXPY", "LobsDICE",  "LobsDICE_nosample", "LobsDICE_cvxpy"]:
# ["ours_convex_solver_dirac", "ours_convex_solver_manhattan"]: 
for name in ["ours_convex_solver_dirac", "ours_convex_solver_manhattan"]: # ["ours_dirac", "ours_manhattan",  "ours_convex_solver_dirac", "ours_convex_solver_manhattan", "ours_ent2KL_solver_dirac", "ours_ent2KL_solver_manhattan", "SMODICE_CHI_CVXPY","SMODICE_KL_CVXPY", "LobsDICE_cvxpy"]:
    args = copy.deepcopy(args_old)
    if name.find("dirac") != -1: args.distance = 'dirac'
    elif name.find('manhattan') != -1: args.distance = 'manhattan'
    if args.TS_type == "goal" and name.find("LobsDICE") != -1: continue
    g = open("resparam.txt", "w")
    for arg in vars(args):
        print(arg, str(getattr(args, arg)))
        g.write(arg + " " + str(getattr(args, arg)) +"\n")
        g.flush()
    g.close()
    name = name.replace("_dirac", "").replace("_manhattan", "")
    os.system('cd ~/LP_RL_test/res/'+name+' && python analyzer.py')

"""
hpp = {
"N_expert_traj": "5",
"TA_expert_traj": "250",
"grid_size": "9",
"max_step": "30",
"noise_level": "0.05",
"TS_type": "goal",
"distance": "manhattan",
"TA_optimality": "0"
}
"""
