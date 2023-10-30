from gurobipy import GRB
import gurobipy as gp
import torch
from torch.optim import Adam, Adagrad, SGD, AdamW, LBFGS
import torch.nn as nn
from env import *
import os
os.environ['MUJOCO_GL'] = 'osmesa'
import copy
import gym
import d4rl
from discriminator import train_discriminator
from get_args import *
import os
import random
from matplotlib import cm
import time
import seaborn as sns
import numpy as np
from contrastive_learning import train_contrastive_model
from torch.utils.data import DataLoader, Dataset
from IDM import *
from ortools.graph import pywrapgraph
from NN import *
from advance_NN import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
import math
from ema_pytorch import EMA
import networkx as nx
import wandb
from datetime import datetime
from tqdm import tqdm
from dist_metric import get_dist, soft_linear_piecewise_loss
from dataset import Testdataset, RepeatedDataset, add_terminals, construct_dataset, concatlist, list2gen, get_dataset
import cvxpy as cp

import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

class Lagrangian_Solver:
    def __init__(self, runtime, desc):
        self.time = runtime
        self.WBC_step, self.WBC_test_step = 0, 0
        self.main_step = 0

        self.RECORD_TIME = time.time() 
        wandb.log({"record_time": self.RECORD_TIME}, commit=False)
        self.RECORD_TIME = str(self.RECORD_TIME) + str(desc)
        
    def evaluation(self, env, eval_use_argmax, mean_state=None, std_state=None, ep=None, visualization=False):
        T = 10
        
        points_x, points_y = [], []
        tot_r, tot_l, tot_cnt = 0, 0, 0
        import imageio
        imgs = []
        Rs = []
        for i in range(T):
            # TODO: an evaluation of a whole episode in the environment.
            state = env.reset()
            print("state:", state)
            while True:
                # print("state:", state)
                if self.args.absorb == 1:
                    state = np.concatenate([state, np.array([0])]) 
                    
                if mean_state is not None and std_state is not None: state = (state - mean_state.cpu().numpy()) / std_state.cpu().numpy()
                
                if not eval_use_argmax:
                    action = self.policy.sample(torch.from_numpy(state).to(device).double())
                else:
                    action = self.policy.deterministic_action(torch.from_numpy(state).to(device).double())
                if isinstance(self.policy, ActorDiscrete): 
                    action = action.item()
                else: action = action.detach().cpu().numpy().reshape(-1)
                state, reward, done, _ = env.step(action)
                #print("new_state:", state)
                tot_r += reward
                if visualization:
                    try:
                        imgs.append(env.render(mode='rgb_array'))
                    except:
                        print("cannot render!")
                #print("step!")
                if done: # do not use normalization with visualization together on Navigation!
                    Rs.append(tot_r)
                    tot_r = 0
                    if isinstance(env, Navigation) or isinstance(env, Navigation_long): 
                        points_x.append(state[0])
                        points_y.append(state[1])
                        plt.plot(points_x, points_y)
                        points_x, points_y = [], []
                    break
        #draw_grid(ax)
        Rs = np.array(Rs)
        if visualization:
            try:
                imageio.mimsave("video/"+self.RECORD_TIME+"-"+str(args.absorb)+"-"+str(args.absorb_distance)+"-"+str(args.distance)+".mp4", imgs, fps=25)
                try: 
                    env.close()
                except:
                    print("cannot close!")
            except:
                print("no video!")
        wandb.log({"average_reward": Rs.mean(), "max_reward": Rs.max(), 'min_reward': Rs.min(), "std_reward": Rs.std()})
        return

    def solve(self, env, IDM, TS_dataset, TA_dataset, args, dist_model=None):    
    
        if args.use_s2 == 0 or args.use_s3 == 0: assert args.use_s1 == 1, "Error!"
    
        epsilon_1, epsilon_2 = args.epsilon_1, args.epsilon_2
        USE_BN = (args.use_bn == "yes")
        self.args = args
        self.IDM = IDM
        self.gamma = 0.999
        print(env.observation_space, env.action_space)
        n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
        # print(n_state, n_action)
        self.args = args
        assert args.absorb in [0, 1], "absorb must be 0 or 1!"
        self.net = Normal_Predictor(n_state + args.absorb, use_bn=USE_BN).to(device).double()
        if args.EMA > 1e-10:
            self.ema = EMA(self.net, beta=args.EMA, update_after_step=100, update_every=10)
        if args.env_name in ["navigation", "navigation_long"]: self.the_terminal, self.SAVE_CONST = torch.tensor([[0.6, 0.5, 1]]).double(), args.save_const# 
        elif args.env_name in ["navigation_1d", "navigation_1d_long"]: self.the_terminal, self.SAVE_CONST = torch.tensor([[0.99, 1]]).double(), args.save_const
        else: self.the_terminal, self.SAVE_CONST = torch.cat([torch.zeros_like(torch.from_numpy(TA_dataset[0]["state"])).double(), torch.ones(1).double()]).view(1, -1), 400 if args.env_name == "lunarlander" else args.save_const #
        # real_action = action_in_actor * scale + bias. So it is [real_lb, real_ub] = [-scale+bias, scale+bias]

        if args.env_name in ["lunarlander-discrete", "cartpole", "pendulum-discrete", "lunarlander-doublediscrete"]:
            self.policy = ActorDiscrete(n_state+args.absorb, n_action).to(device).double()
        else:
            action_low, action_high = env.action_space.low, env.action_space.high
            if args._use_policy_entropy_constraint == 1: self.policy = TanhNormalPolicy(n_state + args.absorb, n_action, action_space=(action_low, action_high)).to(device).double()
            else: self.policy = ActorTanh(n_state + args.absorb, n_action, scale=(action_high - action_low) / 2, bias=(action_low + action_high) / 2).to(device).double()

        
        self.policy_optim = Adam(self.policy.parameters(), lr=args.wbc_lr, weight_decay=args.wbc_weight_decay)
        if args._use_policy_entropy_constraint:
            self._log_ent_coeff = torch.zeros(1, requires_grad=True, device=device)
            self.ent_coeff_optim = torch.optim.Adam([self._log_ent_coeff], args.wbc_lr)
        self.distance = args.distance
        # exit(0)
        states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA = get_dataset(TA_dataset)
        states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, rewards_TS = get_dataset(TS_dataset)

        print("original mean:", states_TA.mean(dim=0))
        print("original std:", states_TA.std(dim=0) + 1e-10)
        print('original length:', states_TA.shape)
        

        states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA = add_terminals(states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA, args.absorb, self.the_terminal)
        states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, rewards_TS = add_terminals(states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, rewards_TS, args.absorb, self.the_terminal)
        
        # rewards is either empty list or normal
        
        TA_mean, TA_std = None, None
        self.data_TA_mean, self.data_TA_std = None, None
        context_TA = [] 
        
        print("TS before normalize:", states_TS[0])
        print("TA before normalize:", states_TA[0])
        if args.normalize_obs == 1:
            TA_mean, TA_std = states_TA.mean(dim=0), states_TA.std(dim=0) + (1e-4 if args.env_name != "kitchen" else 1e-3) # 1e-10
            print("mean:", TA_mean, "std:", TA_std)
            states_TA = (states_TA - TA_mean.view(1, -1)) / TA_std.view(1, -1)
            states_TS = (states_TS - TA_mean.view(1, -1)) / TA_std.view(1, -1)
            next_states_TA = (next_states_TA - TA_mean.view(1, -1)) / TA_std.view(1, -1)
            next_states_TS = (next_states_TS - TA_mean.view(1, -1)) / TA_std.view(1, -1)
            self.data_TA_mean, self.data_TA_std = TA_mean, TA_std
        
        print("mean:", self.data_TA_mean)
        print("std:", self.data_TA_std)
        print("TS after normalize:", states_TS[0])
        # exit(0)
        
        if args.distance.find("learned") != -1 and isinstance(dist_model, str):
            print("The contrastive model is not trained. Let us train from scratch...")
            non_terminals = torch.nonzero(terminals_TA == 0).view(-1)
            if args.distance.find("twinpd") != -1: 
                initials = torch.nonzero(steps_TA.view(-1) == 0)
                belong_TA = torch.zeros(steps_TA.shape[0])
                for i in range(len(initials)):
                    if i < len(initials) - 1:
                        belong_TA[initials[i]:initials[i+1]] = i
                    else: 
                        belong_TA[initials[i]:] = i
                dist_model = train_contrastive_model_twin(args.env_name, dist_model, states_TA[non_terminals], next_states_TA[non_terminals], belong_TA[non_terminals])
            else: dist_model = train_contrastive_model(args.env_name, dist_model, states_TA[non_terminals], next_states_TA[non_terminals]) # train a model with state being states_TA and save at model/env_name/dist_model
        
        if args.scale_dist_with_occupancy > 1e-10 or args.scale_dist_with_context > 1e-10 or args.scale_dist_with_occupancy_product > 1e-10:
            print("The R model is not trained. Let us train from scratch...")
            disc_hyperparam = {"batch_size": min(512, states_TS.shape[0]), "lr": 3e-4, "N": 40000, "lipschitz": args.lipschitz, "suffix": args.env_name + "_" + args.skip_suffix_TA + "_" + args.skip_suffix_TS, "EMA": args.EMA, 'mixed_ratio': args.mixed_ratio} 
            if args.load_smodice_reward == 0 or args.normalize_obs == 1: disc = train_discriminator(states_TA, states_TS, disc_hyperparam)
            else: 
                print("loading smodice disc...")
                disc = torch.load("external_model/"+args.env_name+"/discriminator_from_smodice.pt").double().to('cuda:0')
                
            if args.scale_dist_with_context > 1e-10:
                
                hyperparam = {"lr": 3e-4, "N": 1000, "lipschitz": args.lipschitz, "batch_size": min(512, states_TS.shape[0]), "suffix": args.env_name + "_" + args.skip_suffix_TA + "_" + args.skip_suffix_TS, "EMA": args.EMA}  
                initials = torch.nonzero(steps_TA.view(-1) == 0)
                belong_TA = torch.zeros(steps_TA.shape[0])
                belong_TS = torch.zeros(steps_TS.shape[0])
                
                for i in range(len(initials)):
                    if i < len(initials) - 1:
                        belong_TA[initials[i]:initials[i+1]] = i
                    else: 
                        belong_TA[initials[i]:] = i
                        
                disc2 = train_wandering_discriminator(states_TA, states_TS, belong_TA, belong_TS, steps_TA, steps_TS, hyperparam, initials, N_states=2)
                N_TA, N_states, new_R = len(initials), 2, []
                for i in tqdm(range(len(initials))):
                    if i < len(initials) - 1: 
                        st, ed = initials[i].item(), initials[i+1].item()
                    else: 
                        st, ed = initials[i].item(), states_TA.shape[0]
                    if ed - st < N_states: v = 0 # too short!
                    else:
                        states = []
                        for j in range(100):
                            idx_TA_now = np.sort(np.random.choice(np.arange(st, ed), size=N_states, replace=False))
                            states.append(states_TA[idx_TA_now].view(-1).unsqueeze(0))
                        states = torch.cat(states, dim=0)
                        v = torch.exp(disc2.predict_reward(states).mean() * args.scale_dist_with_context)
                        new_R.append(v * torch.ones(ed-st).double().to(device))
                    #f.write(str(v.item())+"\n")
                new_R = torch.cat(new_R, dim=0)
                context_TA = (new_R / new_R.max() * 3)
                
                """
                R = []
                for i in range(states_TA.shape[0] // 4096 + 1): 
                    R.append(disc.predict_reward(states_TA[i*4096:(i+1)*4096]))
                R = torch.exp(torch.cat(R, dim=0) * args.scale_dist_with_context) 
                initials = torch.nonzero(steps_TA.view(-1) == 0)
                #f = open("trajcoeff.txt", "w")

                for i in range(len(initials)):
                    if i < len(initials) - 1: 
                        v, s = R[initials[i]:initials[i+1]].mean(), initials[i+1]-initials[i]
                    else: 
                        v, s = R[initials[i]:].mean(), R.shape[0]-initials[i] 
                    #f.write(str(v.item())+"\n")
                    context_TA.append(v * torch.ones(s).double().to(device))
                context_TA = torch.cat(context_TA, dim=0)
                context_TA = (context_TA / context_TA.max() * 3) # 3 to find the distance
                print(context_TA, context_TA.max(), context_TA.min())
                #f.close()
                #exit(0)
                """
        rho_E = torch.zeros(states_TS.shape[0])
        batch_size2 = args.batch_size2
        
        train_loader = RepeatedDataset([states_TA.double(), actions_TA.double(), terminals_TA.double(), next_states_TA.double()], batch_size2)
          
        initials, N_TS = [], 0
        
        print(states_TS.shape, steps_TS.shape, terminals_TS.shape)
        
        num_TS_traj = torch.count_nonzero(steps_TS == 0)
        
        print("num_TS_traj:", num_TS_traj)
        j = 0
        for i in range(states_TS.shape[0]):
            if terminals_TS[i] == 1:
                print("terminal:", i)   
            if args.d_e == "uniform":
                rho_E[i] = 1 / states_TS.shape[0] 
            elif args.d_e == "exp":
                rho_E[i] = (1 - self.gamma) * (self.gamma ** j)
                print(rho_E[i], 1 - self.gamma, (self.gamma ** j))
                if i == states_TS.shape[0] - 1: 
                    rho_E[i] = self.gamma ** j # (1 - self.gamma) / (1 - self.gamma)
            elif args.d_e == "uniform-goal":
                rho_E[i] = 0.5 / states_TS.shape[0] + 0.5 * terminals_TS[i]
            elif args.d_e == "exp-goal":
                rho_E[i] = (1 - self.gamma) * (self.gamma ** j) * 0.5
                if terminals_TS[i] == 1:
                    rho_E[i] = 0.5 * self.gamma ** j + 0.5
            # TODO: add the absorbing state!
            # rho_E[i] = 1 / states_TS.shape[0] long navigation
            j += 1
            if steps_TS[i] == 0:
                initials.append(states_TS[i])
                N_TS += 1   
                j = 0
        # rho_E /= N_TS
        #for i in range(rho_E.shape[0]):
        #   print(i, rho_E[i]) 
        print("rho_E_sum:", rho_E.sum())
        # if args.env_name in ["navigation", "navigation_discretegrid", "navigation_1d", "navigation_1d_long", "navigation_long"]:
        rho_E /= rho_E.sum() 

        # assert np.abs(rho_E.sum() - 1) < 1e-6, "error!"+str(rho_E.sum())
        assert args.initial in ["TS", "TA", "TATS"], "initial error!"
        if args.initial == "TS": initials = torch.cat([initials[i].view(1, -1) for i in range(len(initials))], dim=0) # torch.tensor(initials).double()        
        elif args.initial == "TA": 
            initials = states_TA[torch.nonzero(steps_TA == 0).view(-1)]
        elif args.initial == "TATS": 
            initials = torch.cat([torch.cat([initials[i].view(1, -1) for i in range(len(initials))], dim=0), states_TA[torch.nonzero(steps_TA == 0).view(-1)]], dim=0)

        optimizer = Adam(self.net.parameters(), lr=args.lr,weight_decay=args.weight_decay) # 0.0001
        scheduler = ExponentialLR(optimizer, gamma=args.decay)# ReduceLROnPlateau(optimizer, 'min')
        print("initials:", initials.shape)
        # exit(0)
        if args.wbc_scheduler == 1:
            self.wbc_scheduler = ExponentialLR(self.policy_optim, gamma=0.99)
        elif args.wbc_scheduler == 2:
            self.wbc_scheduler = CosineAnnealingLR(self.policy_optim, T_max=10, eta_min=1e-6)
        
        self.dist_normalizing_factor = args.normalizing_factor
        
        if USE_BN: self.net.train()
        batch_size1 = args.batch_size1

        # absorb-distance is deprecated!
        if args.cheat != 0:
            if args.scale_dist_with_context <= 1e-10: uniform_sampler1 = RepeatedDataset([states_TA, next_states_TA, rewards_TA], batch_size1)
            else: uniform_sampler1 = RepeatedDataset([states_TA, next_states_TA, rewards_TA, context_TA], batch_size1)
        else: 
            if args.scale_dist_with_context <= 1e-10: uniform_sampler1 = RepeatedDataset([states_TA, next_states_TA], batch_size1)
            else: uniform_sampler1 = RepeatedDataset([states_TA, next_states_TA, context_TA], batch_size1)
        print(states_TS.shape, next_states_TS.shape)

        # loop 10 to allow batch size up to 1000 * 10 = 10000
        
        tot_batches = 0    
        
        for __ in tqdm(range(args.N)):
            
            print("Start training at epoch ", __, "...")
            if args.BCdebug != 1:
              for ___ in tqdm(range(len(train_loader))):
                t0 = time.time()
                tot_batches += 1
                states_I, action_I, terminal_I, next_states_I = train_loader.getitem()
                states_I, action_I, terminal_I, next_states_I = states_I.to(device), action_I.to(device), terminal_I.to(device), next_states_I.to(device) 
                # part 1: epsilon_1 / e * E_{(s_i, s_j)\sim U} exp([\lambda_{i+|S|}+\lambda_{j+2|S|}-d(s_i,s_j)] / epsilon_1)
                if args.uniform == 0:
                    if args.cheat != 0:
                        if args.scale_dist_with_context <= 1e-10: 
                            states1, next_states1, rewards1 = uniform_sampler1.getitem()
                            states1, next_states1, rewards1 = states1.to(device), next_states1.to(device), rewards1.to(device)
                        else: 
                            states1, next_states1, rewards1, context1 = uniform_sampler1.getitem()
                            states1, next_states1, rewards1, context1 = states1.to(device), next_states1.to(device), rewards1.to(device), context1.to(device)
                    else: 
                        if args.scale_dist_with_context <= 1e-10: 
                            states1, next_states1 = uniform_sampler1.getitem() # torch.randint(low=0, high=self.MDP.n, size=(batch_size,))
                            states1, next_states1 = states1.to(device), next_states1.to(device)
                        else: 
                            states1, next_states1, context1 = uniform_sampler1.getitem()
                            states1, next_states1, context1 = states1.to(device), next_states1.to(device), context1.to(device)
                    # states2, next_states2, is_terminal2, next_is_terminal2, is_others2 = uniform_sampler2.getitem() # torch.randint(low=0, high=self.MDP.n, size=(batch_size,))
                else:
                   raise NotImplementedError("Error!")
                    
                idx_E = torch.multinomial(rho_E, args.batch_size1, replacement=True) #np.random.randint(low=0, high=len(TS_dataset), size=batch_size5)
                states_E = states_TS[idx_E]
                states2 = states_E.to(device)
                
                US1 = self.net(states1, 1) if args.use_s1 == 1 else disc.predict_reward(states_I).view(-1, 1)
                US2, DIST = self.net(states2, 2) * args.use_s2, get_dist(states1, states2, args.dist_scale, args.distance, None if dist_model is None else dist_model) # distance needs normalization!

                if args.scale_dist_with_occupancy > 1e-10:
                    v = disc.predict_reward(states1).view(-1, 1) 
                    DIST -= args.scale_dist_with_occupancy * v
                
                if args.scale_dist_with_occupancy_product > 1e-10:
                    v = -torch.exp(disc.predict_reward(states1)).view(-1, 1) * torch.exp(disc.predict_reward(states2)).view(-1, 1)
                    DIST -= args.scale_dist_with_occupancy_product * v
                
                if args.scale_dist_with_context > 1e-10:
                    DIST -= context1.view(-1, 1)
                
                if args.cheat != 0:
                    DIST -= rewards1.view(-1, 1)
                
                DIST = DIST / self.dist_normalizing_factor
                #assert batch_size1 == batch_size2, "Error!"
                #US1, US2, DIST = self.net(states_I, 1), self.net(states2, 2), get_dist(states_I, states2)
                
                loss_part1 = (US1 + US2 - DIST) / epsilon_1 - math.log(batch_size1) 

                # part 2: epsilon_2 / e * E_{(s_i, a_j, s_k)\sim I} exp([-\gamma * \lambda_k + lambda_i - \lambda_{i+|S|} + \lambda_{3|S|}] / epsilon_2)
                batch_size2 = args.batch_size2
                batch_size3 = args.batch_size3
                t1 = time.time()
                  
                groundtruth_flag1 = args.GT1
                
                IS0, NS0, S3 = self.net(states_I, 0), self.net(next_states_I, 0), self.net(None, 3) 
                IS1 = self.net(states_I, 1) if args.use_s1 == 1 else disc.predict_reward(states_I).view(-1, 1) 
                if self.main_step % args.log_interval == 0: 
                    if args.scale_dist_with_occupancy > 1e-10: res = {"R_min": v.min(), "R_max": v.max()}
                    else: res = {}
                    res.update({"IS0_mean": IS0.mean(), "IS0_min": IS0.min(), "IS0_max":IS0.max(),"IS1_mean":IS1.mean(), "IS1_min":IS1.min(), "IS1_max":IS1.max(), \
                     "US1_mean":US1.mean(), "US1_min": US1.min(), "US1_max":US1.max(),"US2_mean":US2.mean(), "US2_min": US2.min(), "US2_max":US2.max(), "S3":S3.item(), "avg_dist": DIST.mean(), "dist": DIST})
                    wandb.log(res, commit=False)
                    with torch.no_grad():
                        st = initials[0].to(device)
                        wandb.log({"INS_example": self.net(st, 0)}, commit=False)
                # estimated
                if groundtruth_flag1 == 0:
                    # logs_part2 = (-self.gamma * lmbda[next_states_I] + lmbda[states_I] - lmbda[states_I + self.MDP.n] + lmbda[3 * self.MDP.n]) / epsilon_2 + (math.log(epsilon_2) - 1) - math.log(batch_size2)
                    if args.train_with_terminal == 1: 
                        loss_part2 = (-self.gamma * NS0 + IS0 - IS1 + S3 * args.use_s3) / epsilon_2 - math.log(batch_size2)
                    else: 
                        loss_part2 = (-self.gamma * NS0 * (1 - terminal_I).reshape(-1, 1) + IS0 - IS1 + S3 * args.use_s3) / epsilon_2 - math.log(batch_size2)  
                # part 3: (1 - \gamma) E_{s\sim p_0} \lambda (minus)
                
                batch_size4 = args.batch_size4
                t2 = time.time()
                idx_init = np.random.randint(low=0, high=len(initials), size=batch_size4)
                init_states = initials[idx_init].to(device) # now only single state; sample all
                #print(initials.shape, len(initials), initials)
                #exit(0)
                INS0 = self.net(init_states, 0)
                             
                loss_part3 = (-(1 - self.gamma) * INS0).mean()
                
                # part 4: E_{s\sim d^E(s)} (minus)
                t3 = time.time()
                
                # TODO: add step-based weights on states??
                batch_size5 = args.batch_size5
                idx_E = torch.multinomial(rho_E, batch_size5, replacement=True)
                states_E = states_TS[idx_E].to(device)
                ES2 = self.net(states_E, 2) * args.use_s2
                # print("ES2:", ES2)
                loss_part4 = -ES2.mean()
                loss_part5 = -S3 * args.use_s3
                if self.main_step % args.log_interval == 0: 
                    e_v = (-self.gamma * NS0 * (1 - terminal_I).reshape(-1, 1) + IS0 - IS1 + S3 * args.use_s3) / epsilon_2
                    wandb.log({"e_v": e_v, "e_v_mean": e_v.mean(), "e_v_min": e_v.min(), "e_v_max": e_v.max(), \
                      "ES2_mean": ES2.mean(),"ES2_min": ES2.min(),"ES2_max": ES2.max(),  "INS0_min": INS0.min(), "INS0_max": INS0.max(), "INS0_mean": INS0.mean(),  "NS0_min": NS0.min(), "NS0_max": NS0.max(), "NS0_mean": NS0.mean(), \
                      'main_step':self.main_step, "logs_part1": epsilon_1 * torch.logsumexp(loss_part1, dim=0), "logs_part2": epsilon_2 * torch.logsumexp(loss_part2, dim=0), "coeff": torch.exp(loss_part2).mean(), "logs_part3": loss_part3, "logs_part4": loss_part4, "logs_part5": loss_part5, "terminal_I": terminal_I.sum()}, commit=False)
                self.main_step += 1
                if self.main_step >= 1050000: exit(0)
                # part 5: \lambda_{3|S|} (minus)
                t4 = time.time()
                
                if self.main_step % args.log_interval == 0:
                    wandb.log({"std-lmbda-E": ES2.std(), "lmbda-E": ES2, "max-lmbda-E": ES2.max(), "min-lmbda-E": ES2.min()}, commit=False)
                    wandb.log({"std-lmbda-I": IS0.std(), "lmbda-I": IS0, "max-lmbda-I": IS0.max(), "min-lmbda-I": IS0.min()}, commit=False)
                

                indirect_loss = epsilon_1 * torch.logsumexp(loss_part1, dim=0) + epsilon_2 * torch.logsumexp(loss_part2, dim=0) + (loss_part3 + loss_part4 + loss_part5) + args.lambda2_reg * (torch.mean(ES2) ** 2 + torch.mean(IS0) ** 2 + torch.mean(IS1) ** 2) #torch.logsumexp(torch.cat([logs_part1.view(-1), logs_part2.view(-1), logs_part3.view(-1), logs_part4.view(-1), logs_part5.view(-1), logs_part6.view(-1)]), dim=0)
                if self.main_step % args.log_interval == 0:
                    wandb.log({"loss": indirect_loss, "terminal_NS_value": (terminal_I.reshape(-1, 1) * NS0).sum() / terminal_I.sum() if terminal_I.sum() > 0 else 0, "terminal_S_value": (terminal_I.reshape(-1, 1) * IS0).sum() / terminal_I.sum() if terminal_I.sum() > 0 else 0, "ES2-mean-square": torch.mean(ES2) ** 2, "IS0-mean-square": torch.mean(IS0) ** 2, "IS1-mean-square": torch.mean(IS1) ** 2})
                # 0.0001 comes from (1-self.MDP.gamma) ** 2.
                t5 = time.time()      
                # gradient descent
                optimizer.zero_grad()
                indirect_loss.backward() 
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
                optimizer.step() # (closure)
                if args.EMA > 1e-10: self.ema.update()
                t6 = time.time()

              scheduler_n = args.scheduler_n if args.scheduler_n > 0 else args.N * len(train_loader)           
              if tot_batches % scheduler_n == scheduler_n - 1: 
                  scheduler.step()
                
            # print("__:", __)
            
            if __ % self.SAVE_CONST == self.SAVE_CONST - 1:
                torch.save(self.net, "model/net-"+args.env_name+"-"+self.RECORD_TIME+"-"+str(args.absorb)+"-"+str(args.absorb_distance)+"-"+str(args.distance)+"-ep"+str(__)+".pt")    
                torch.save(self.policy, "model/actor-"+args.env_name+"-"+self.RECORD_TIME+"-"+str(args.absorb)+"-"+str(args.absorb_distance)+"-"+str(args.distance)+"-ep"+str(__)+".pt")
            print("current epoch:", __)
            
            if args.load_net == 1:
                if args.env_name == "hopper": self.net = torch.load("model/hopper-1670138121.9077141hopper-A-0-0-manhattan-ep399.pt")
                elif args.env_name == "pendulum": self.net = torch.load("model/net-pendulum-1671561882.6203885pendulum-continuous-nf2-twinhead-0-0-manhattan-ep519.pt")# torch.load("model/pendulum-1671320369.360351pendulum-continuous-ep0.05-0-0-manhattan-ep799.pt") 
                elif args.env_name == "pendulum-discrete": self.net = torch.load("model/pendulum-discrete-1671319055.8258448pendulum-discrete-ep0.05-0-0-manhattan-ep799.pt") #torch.load("model/pendulum-discrete-1671319055.8258448pendulum-discrete-ep0.05-0-0-manhattan-ep799.pt")
                elif args.env_name == "antmaze": self.net = torch.load("model/net-antmaze-1672903344.5298474antmaze-lottery2-0-0-learned_pd-ep399.pt")
                elif args.env_name == "halfcheetah": self.net = torch.load("model/net-halfcheetah-1678348138.2071187halfcheetah-expert40-10xdist-multistep-1.25-threeparts-average-0-0-learned_twinpd-ep2399.pt") 
                try:
                    os.mkdir("res/"+self.RECORD_TIME) 
                except:
                    print("cannot create!")
                    exit(0)
                
            CONST = args.ratio_start_BC_training
            # CONST = 10 if args.env_name == "navigation" else 50    
                  
            if args.BCdebug == 1: 
                self.Train_WeightedBC(states_TA, actions_TA, terminals_TA, next_states_TA, torch.nonzero(steps_TA.view(-1) == 0), args)
                return 
            elif args.joint == 1 and __ >= args.N // CONST and __ % args.BC_stage_per_weight_epoch == 0: # we don't need TATSdebug == 1 here for now.
                self.Train_WeightedBC(states_TA, actions_TA, terminals_TA, next_states_TA, torch.nonzero(steps_TA.view(-1) == 0), args)
                
                if __ % args.BC_stage_per_eval == 0:
                    print("Start evaluation at epoch ", __, "...") 
                    print("mean:", self.data_TA_mean, "std:", self.data_TA_std)
                    self.evaluation(env, eval_use_argmax=(args.eval_deter == 1), mean_state=(None if args.normalize_obs == 0 else self.data_TA_mean), std_state=(None if args.normalize_obs == 0 else self.data_TA_std), ep=__, visualization=(__ % 100 == 99)) 
            elif args.joint == 0 and __ == args.N - 1: 
                self.Train_WeightedBC(states_TA, actions_TA, terminals_TA, next_states_TA, torch.nonzero(steps_TA.view(-1) == 0), args) # self.Train_WeightedBC(states_TS, actions_TS, terminals_TS, next_states_TS, args) # DEBUGGING!
                if __ % args.BC_stage_per_eval == 0:
                    print("Start evaluation at epoch ", __, "...")
                    self.evaluation(env, eval_use_argmax=(args.eval_deter == 1), mean_state=(None if args.normalize_obs == 0 else self.data_TA_mean), std_state=(None if args.normalize_obs == 0 else self.data_TA_std), visualization=False) 
            
            if __ == args.N - 1:
                torch.save(self.net, "model/net_"+self.time.replace("/","-").replace(" ", "")+".pt")
                torch.save(self.policy, "model/policy_"+self.time.replace("/", "-").replace(" ", "")+".pt")

       
    def Train_WeightedBC(self, s_TA, a_TA, t_TA, ns_TA, initials, args, is_TS=None):
        wandb.log({"shape": s_TA.shape[0]}, commit=False)
        states_TA, actions_TA, terminals_TA, next_states_TA = s_TA.clone(), a_TA.clone(), t_TA.clone(), ns_TA.clone()
        epsilon_1, epsilon_2 = args.epsilon_1, args.epsilon_2
        GT1 = args.GT1
        # behavior cloning
        coeff_max = 0
        batch_size3 = args.batch_size3
            
        if args.BCdebug == 0:
            coeff_lst = []
            BS = 4096
            for i in tqdm(range((states_TA.shape[0] - 1) // BS + 1)): # first do an empty epoch to find the biggest coefficient
                state, acton, terminal, next_state = states_TA[i*BS:i*BS+BS], actions_TA[i*BS:i*BS+BS], terminals_TA[i*BS:i*BS+BS], next_states_TA[i*BS:i*BS+BS] 
                state, acton, terminal, next_state = state.to(device), acton.to(device), terminal.to(device), next_state.to(device)
                # e_v = (-self.gamma * NS0 * (1 - terminal_I).reshape(-1, 1) + IS0 - IS1 + S3 * args.use_s3) / epsilon_2
                assert GT1 == 0, "Error!"          
                if args.EMA > 1e-10:
                    coeff = torch.exp((-self.gamma * self.ema(next_state, 0) * (1 - terminal).reshape(-1, 1) + self.ema(state, 0) - self.ema(state, 1) + args.use_s3 * self.ema(None, 3)) / epsilon_2).detach() 
                else: 
                    coeff = torch.exp((-self.gamma * self.net(next_state, 0) * (1 - terminal).reshape(-1, 1) + self.net(state, 0) - self.net(state, 1) + args.use_s3 * self.net(None, 3)) / epsilon_2).detach()
                
                coeff_lst.append(coeff)
        else: 
            coeff_lst = [torch.ones(states_TA.shape[0]).double().to(device)]
        coeff_lst = torch.cat(coeff_lst, dim=0).view(-1)
        coeff_max = coeff_lst.max()
        # print(coeff.shape)
        wandb.log({"coeff_max": coeff_max}, commit=False)
        
        # warning: notice this!
        print("shape:", initials.shape)
        if args.smooth != "no":
            new_coeff_lst = torch.zeros_like(coeff_lst)
            assert args.smooth in ["laplacian", "exponential", "average"], "Error!"
            
            def laplacian_smoothing(v):
                    v2 = v / 2
                    v2[0] += v[1] / 2
                    v2[-1] += v[-1] / 2
                    v2[1:] += v[:-1] / 4
                    v2[:-1] += v[1:] / 4
                    return v2
                
            def exponential_smoothing(array, factor): # adopted from wandb https://docs.wandb.ai/v/zh-hans/dashboard/features/standard-panels/line-plot/smoothing
                res = torch.zeros_like(array)
                last = 0
                for i in range(array.shape[0]):
                    last = last * factor + (1 - factor) * array[i]
                    debias_weight = 1 - factor ** (i + 1)
                    res[i] = last / debias_weight
                return res
            
            for i in tqdm(range(len(initials))):
                if i < len(initials) - 1:
                    st, ed = initials[i].item(), initials[i+1].item()
                else:
                    st, ed = initials[i].item(), states_TA.shape[0]
                if args.smooth == "laplacian":
                    new_coeff_lst[st:ed] = laplacian_smoothing(coeff_lst[st:ed])
                elif args.smooth == "exponential":
                    new_coeff_lst[st:ed] = exponential_smoothing(coeff_lst[st:ed], args.smooth_coeff)
                elif args.smooth == "average":
                    new_coeff_lst[st:ed] = coeff_lst[st:ed].mean()
            coeff_lst = new_coeff_lst
        
        coeff_max = coeff_lst.max()
        coeff_lst /= coeff_max
        
        
        quant = torch.quantile(coeff_lst, torch.tensor([0.5, 0.9, 0.99, 0.999]).double().to(device), interpolation='midpoint')
        quant_50 = quant[0]
        quant_90 = quant[1]
        quant_99 = quant[2]
        quant_999 = quant[3]
        print("!!!", coeff_lst.shape[0])    
        if args.BC_optimize >= 1:
            idx1, idx2 = torch.nonzero(coeff_lst >= 1e-6).cpu().numpy(), torch.nonzero(terminals_TA == 0).cpu().numpy()    
            if args.absorb == 1: idx = np.intersect1d(idx1, idx2)
            else: idx = idx1
            ratio = len(idx) / states_TA.shape[0]
            idx = idx.reshape(-1)
            print(idx.shape, states_TA.shape[0])
            # exit(0)
            states_TA, actions_TA, terminals_TA, next_states_TA, coeff_lst = states_TA[idx], actions_TA[idx], terminals_TA[idx], next_states_TA[idx],  coeff_lst[idx] 
            # wandb.log({}, commit=False)
            if is_TS is not None: is_TS = is_TS[idx]
            wandb.log({"coeff_top_0.1percent_mean": quant_999, "coeff_top_1percent_mean": quant_99, "coeff_top_10percent_mean": quant_90, "coeff_top_50percent_mean": quant_50, "maintained_WBC_ratio": ratio})
        
        else: wandb.log({"coeff_top_0.1percent_mean": quant_999, "coeff_top_1percent_mean": quant_99, "coeff_top_10percent_mean": quant_90, "coeff_top_50percent_mean": quant_50})
        
        # train_loader, test_loader = construct_dataset(states_TA, actions_TA, terminals_TA, next_states_TA, args, args.batch_size0, coeff_lst)
    
        train_loader = RepeatedDataset([states_TA, actions_TA, terminals_TA, next_states_TA, coeff_lst], batch_size=args.batch_size0)
        print("???", states_TA.shape[0])
        self.policy.train()
        self._target_entropy = -actions_TA.shape[-1]
        N = args.BCdebug_N if (args.BCdebug == 1 or args.joint == 0 or args.load_net == 1) else args.BC_epoch_per_stage # if (args.BCdebug == 1 or args.TATSdebug == 1) else 1
        for _ in tqdm(range(N)):
            # for batch_idx, sample_batched in enumerate(train_loader):
            
            if N > 1 and _ % 5 == 4 and (args.load_net == 1 or args.BCdebug == 1):
                self.evaluation(env, eval_use_argmax=(args.eval_deter == 1), ep=_,mean_state=(None if args.normalize_obs == 0 else self.data_TA_mean), std_state=(None if args.normalize_obs == 0 else self.data_TA_std), visualization=False)  
            
            debug_actions, debug_coeffs = [], []
            print("size:", states_TA.shape, len(train_loader))
            for __ in tqdm(range(len(train_loader))):
                state, action, terminal, next_state, coeff = train_loader.getitem()
                state, action, terminal, next_state, coeff = state.to(device), action.to(device), terminal.to(device), next_state.to(device), coeff.to(device)
                if args._use_policy_entropy_constraint == 1:
                    sampled_action, sampled_pretanh_action, sampled_action_log_prob, sampled_pretanh_action_log_prob, pretanh_action_dist = self.policy(state)
                    # Entropy is estimated on newly sampled action.
                    negative_entropy_loss = torch.mean(sampled_action_log_prob)
                    action_log_prob, _ = self.policy.logprob(pretanh_action_dist, action, is_pretanh_action=False)
                    loss = - torch.mean(coeff.view(-1) * action_log_prob.view(-1))
                    
                    ent_coeff = torch.exp(self._log_ent_coeff).squeeze(0)
                    loss += ent_coeff * negative_entropy_loss
                    ent_coeff_loss = torch.mean(- self._log_ent_coeff * (sampled_action_log_prob + self._target_entropy).detach())
                    log_prob = action_log_prob
                else:
                    log_prob, entropy, var = self.policy.logprob(state, action)
                    if args.BC_with_terminal == 0:
                        loss = -((1 - terminal).view(-1, 1) * log_prob.view(-1, 1) * coeff.view(-1, 1)).mean()#.sum()
                    else: # with terminal data
                        loss = -(log_prob.view(-1, 1) * coeff.view(-1, 1)).mean()

                self.policy_optim.zero_grad()
                loss.backward()

                g = 0
                for param in self.policy.parameters():
                    g += torch.norm(param.grad, 2)

                if args.clip_wbc > 1e-10:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), args.clip_wbc)
                
                logprob_50 = ((coeff >= quant_50).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_50).sum()
                logprob_90 = ((coeff >= quant_90).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_90).sum()
                logprob_99 = ((coeff >= quant_99).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_99).sum()
                logprob_999 = ((coeff >= quant_999).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_999).sum()
                if self.WBC_step % args.log_interval == 0:
                    wandb.log({"WBC gradient norm": g}, commit=False) 
                    if args._use_policy_entropy_constraint == 1: wandb.log({"ent_coeff_loss": ent_coeff_loss, "ent_coeff": ent_coeff}, commit=False)
                    else: wandb.log({"entropy_train":entropy.mean(), "logvar_mean": var.mean(), "weighted_entropy_train": (entropy.view(-1) * coeff.view(-1)).mean(), "weighted_logvar_train": (var.mean(dim=-1) * coeff.view(-1)).mean()}, commit=False)
                    wandb.log({"WBC train loss": loss, 'WBC_steps': self.WBC_step, "negative_logprob_50percent_top": logprob_50, "negative_logprob_10percent_top": logprob_90, "negative_logprob_1percent_top": logprob_99, "negative_logprob_0.1percent_top": logprob_999})    
                self.WBC_step += 1
                self.policy_optim.step()
                
                if args._use_policy_entropy_constraint == 1:
                    self.ent_coeff_optim.zero_grad()
                    ent_coeff_loss.backward()
                    self.ent_coeff_optim.step()
                
                if args.wbc_scheduler != 0 and self.WBC_step % 50 == 49: 
                    print("scheduler in effect!")
                    self.wbc_scheduler.step()
            
            """
            self.policy.eval()
            
            for __ in range(len(test_loader)):
                state, action, terminal, next_state, coeff = train_loader.getitem() 
                log_prob, entropy, var = self.policy.logprob(state, action)
                if args.BC_with_terminal == 0:
                    loss = -((1 - terminal).view(-1, 1) * log_prob.view(-1, 1) * coeff.view(-1, 1).detach()).mean() # double detach on coeff now
                else: # with terminal data
                    loss = -(log_prob.view(-1, 1) * coeff.view(-1, 1).detach()).mean()
                self.WBC_test_step += 1
                
                logprob_50 = ((coeff >= quant_50).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_50).sum()
                logprob_90 = ((coeff >= quant_90).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_90).sum()
                logprob_99 = ((coeff >= quant_99).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_99).sum()
                logprob_999 = ((coeff >= quant_999).view(-1) * (1 - terminal).view(-1) * -log_prob.view(-1)).sum() / (coeff >= quant_999).sum()
                if self.WBC_test_step % args.log_interval == 0:
                    wandb.log({"entropy_test":entropy.mean(), "logvar_test": var.mean(), "weighted_entropy_test": (entropy * coeff.view(-1)).mean(), "weighted_logvar_test": (var.mean(dim=-1) * coeff.view(-1)).mean()}, commit=False)
                    wandb.log({'WBC_test_steps': self.WBC_test_step, "test_loss_50percent_top": logprob_50, "test_loss_10percent_top": logprob_90, "test_loss_1percent_top": logprob_99, "test_loss_0.1percent_top": logprob_999, "WBC test loss": loss}) 
            self.policy.train()
            """
            
        
if __name__ == "__main__":
    args = get_args()
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime) 

    if args.auto == 1: a = args.env_name + "-" + args.skip_suffix_TA + "-" + args.skip_suffix_TS + '-' + ("threepart-mixed-ratio-"+str(args.mixed_ratio) if args.scale_dist_with_context >= 1e-10 else ("twopart-mixed-ratio-"+str(args.mixed_ratio) if args.scale_dist_with_occupancy >= 1e-10 else "onepart")) + "-" + args.distance + "-auto-epsilonablation-"+str(args.epsilon_1)+"-"+str(args.epsilon_2) 
    else: 
        print("please input description:")
        a = input()
    # a = "hopper-lowlr-bclv2" 
    wandb.init(entity="XXXX",project="project2-smodiceized", name=str(runtime)+"_"+str(args.seed)+"_ours_smodiceized-deep_"+args.distance+"_"+a+"-testfinale")
            
    
    seed = args.seed
    
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False # CUDNN will try different methods and use an optimal one if this is set to true. This could be harmful if your input size / architecture is changing. 
    
    # train a inverse dynamic model
    
    if args.GT1  == -1:
        IDM = get_IDM(TA_dataset, args)
        torch.save(IDM, args.env_name+".pt")
    else: IDM = None
    # get environment
    
    dist_model = None
    
    if args.env_name in ["hopper", "hopper_150step", "halfcheetah", "ant", "ant_150step", "walker2d", "cartpole", "pendulum", "pendulum-discrete", "pendulum-statediscrete", "lunarlander-discrete", "lunarlander-doublediscrete", "kitchen", 'antmaze']:
        if args.env_name not in ["cartpole", "pendulum", "pendulum-discrete", "lunarlander-discrete", 'pendulum-statediscrete', "lunarlander-doublediscrete"]:
            
            suffix = "random"+str(args.max_random_traj)+"_expert"+str(args.max_expert_traj) if (args.max_random_traj != 9999999 or args.max_expert_traj != 9999999) else ""
            
            TS_dataset = torch.load("data/"+args.env_name+"/TS-read-again-unnormalized"+suffix+args.skip_suffix_TS+".pt") # MDP.generate_expert_traj(N_expert_traj)
            TA_dataset = torch.load("data/"+args.env_name+"/TA-read-again-unnormalized"+suffix+args.skip_suffix_TA+".pt") 
            
            if args.distance.find("learned") != -1:
                name = "_contrastive" + args.skip_suffix_TS + args.skip_suffix_TA
                #print(name,":",name)
                #exit(0)
                if args.normalize_obs == 0:
                    name = "read-again-unnormalized" + name
                elif args.normalize_obs == 1:
                    name = "read-again-normalized" + name
                if args.distance == "learned_pd": name += "_pd"
                elif args.distance == "learned_spd": name += "_spd_1_0"
                elif args.distance == "learned_sphere": name += "_sphere"
                elif args.distance == "learned_twinpd": name += "_twinpd"
                if args.max_random_traj != 9999999 or args.max_expert_traj != 9999999:
                    name += "random"+str(args.max_random_traj)+"_expert"+str(args.max_expert_traj)
                
                try:
                    dist_model = torch.load("model/"+args.env_name+"/"+name+".pt")
                except:
                    dist_model = name
                """
                if args.distance == "learned_pd": 
                    dist_model = torch.load("model/"+args.env_name+"/read-again-unnormalized_contrastive_pd.pt") 
                elif args.distance == "learned": dist_model = torch.load("model/"+args.env_name+"/read-again-unnormalized_contrastive.pt") 
                elif args.distance == "learned_sphere": dist_model = torch.load("model/"+args.env_name+"read-again-normalized_contrastive.pt")
                """
            print(len(TS_dataset), len(TA_dataset))
            env = gym.make(args.env_name+"-"+(("random" if args.env_name != "antmaze" else "umaze") if args.env_name != "kitchen" else "mixed")+"-"+("v2" if args.env_name != "kitchen" else "v0"))
        else:
            TS_dataset = torch.load("data/"+str(args.env_name)+"/TS"+args.data_suffix+".pt")
            TA_dataset = torch.load("data/"+str(args.env_name)+"/TA"+args.data_suffix+".pt")

            if args.env_name == "cartpole":
                env = gym.make('CartPole-v1')
                try:
                    data_suffix = int(args.data_suffix)
                except:
                    data_suffix = int(args.data_suffix[:-1])
                if data_suffix <= 10: env._max_episode_steps = 500
                elif data_suffix <= 20: env._max_episode_steps = 200
                elif data_suffix <= 30: env._max_episode_steps = 100
                elif data_suffix <= 40: env._max_episode_steps = 50 
                elif data_suffix <= 50: env._max_episode_steps = 1000
                else: raise NotImplementedError("No such data!")
            elif args.env_name == "pendulum-discrete":
                if int(args.data_suffix) < 10: env = pendulum_wrapper(gym.make('Pendulum-v1'))
                else: env = pendulum_wrapper_big(gym.make('Pendulum-v1'))
            elif args.env_name == "pendulum":
                env = gym.make('Pendulum-v1')
                try:
                    dist_model = torch.load("model/"+args.env_name+"/"+args.env_name+".pt")
                except:
                    dist_model = args.env_name
            elif args.env_name == "pendulum-statediscrete":
                env = pendulum_wrapper2(gym.make('Pendulum-v1'))
            elif args.env_name == "lunarlander-discrete":
                env = gym.make('LunarLander-v2')
            elif args.env_name == "lunarlander-doublediscrete":
                env = pendulum_wrapper2(gym.make('LunarLander-v2')) # universal wrapper for discretization of states
            
    elif args.env_name == "lunarlander":
        TS_dataset = torch.load("data/"+args.env_name+"/TS-read-again"+args.data_suffix+".pt") 
        TA_dataset = torch.load("data/"+args.env_name+"/TA-read-again"+args.data_suffix+".pt")
        if args.distance.find("learned") != -1:
            if args.distance == "learned_pd": 
                dist_model = torch.load("model/"+args.env_name+"/read-again"+args.data_suffix+"_contrastive_pd.pt") 
            elif args.distance == "learned": dist_model = torch.load("model/"+args.env_name+"/read-again"+args.data_suffix+"_contrastive.pt")
        env = gym.make("LunarLanderContinuous-v2")
    
    env.seed(seed)
    
    solver = Lagrangian_Solver(runtime, a) # Direct_Convex_Solver(MDP, MDP_estimate, runtime)
    solver.solve(env, IDM, TS_dataset, TA_dataset, args, dist_model)
