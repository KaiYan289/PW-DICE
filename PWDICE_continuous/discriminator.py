import torch
import wandb
import numpy as np
import random
from torch import autograd
import torch.nn.functional as F
from NN import Discriminator
from tqdm import tqdm
import torch.nn as nn
import time
from EMA import EMA
from torch.optim import Adam
from dataset import RepeatedDataset
device = torch.device('cuda:0')
def train_discriminator(states_TA, states_TS, train_hyperparams, no_log=False, is_expert_traj=None):
    device = torch.device('cuda:0')
    Disc = Discriminator(states_TA.shape[-1], mixed_ratio=(0 if 'mixed_ratio' not in train_hyperparams else train_hyperparams['mixed_ratio'])).to(device).double()
    # we assume that |states_TA| >> states_TS, so we can sample states_TS while iterating states_TA.
    optimizer = torch.optim.Adam(Disc.net.parameters(), lr=train_hyperparams['lr'])
    dataset_TS = RepeatedDataset([states_TS], train_hyperparams['batch_size'])  
    dataset_TSforN = RepeatedDataset([states_TS], train_hyperparams['batch_size']) 
    # flag = train_hyperparams["suffix"].find("antmaze") == -1 and train_hyperparams["suffix"].find("kitchen") == - 1 and is_expert_traj is not None
    
    
    # if flag: dataset_TA = RepeatedDataset([states_TA, is_expert_traj], train_hyperparams['batch_size'])
    # else: 
    dataset_TA = RepeatedDataset([states_TA], train_hyperparams['batch_size'])
    """
    # VALIDATION DATASET #####################################
    idx = torch.randperm(states_TA.shape[0])
    valid_idx, train_idx = idx[:states_TA.shape[0]//10], idx[states_TA.shape[0]//10:] # 10% data
    
    if flag: dataset_TA, dataset_TAvalid = RepeatedDataset([states_TA[train_idx], is_expert_traj[train_idx]], train_hyperparams['batch_size']), RepeatedDataset([states_TA[valid_idx], is_expert_traj[valid_idx]], len(valid_idx))
    else: dataset_TA, dataset_TAvalid = RepeatedDataset([states_TA[train_idx]], train_hyperparams['batch_size']), RepeatedDataset([states_TA[valid_idx]], len(valid_idx))
    
    ##########################################################
    """
    
    USE_EMA = train_hyperparams["EMA"]
    
    
    # WARNING: ONLY VALID IN OPENAI GYM ENVIRONMENTS! and walker2d is 100
    #if flag:
    #    expert_vs, policy_vs = [], []
    
    ema = None
    if USE_EMA:
        ema = EMA(Disc, 0.999)
        ema.register()
    
    
    def compute_grad_pen(expert_state, offline_state, lambda_, target_grad=1):
        if lambda_ == 0: return torch.zeros(1).double().to(device)
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = Disc(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - target_grad).pow(2).mean()
        return grad_pen
    
    N = train_hyperparams['N'] if 'N' in train_hyperparams else len(dataset_TA) * train_hyperparams['N_epoch']
    
    for i in tqdm(range(N)):
        # sample
        # if not flag: 
        seed = np.random.random()
        if seed < (0 if 'mixed_ratio' not in train_hyperparams else train_hyperparams['mixed_ratio']):
            states_TA, states_TS = dataset_TSforN.getitem(), dataset_TS.getitem()
        else:
            states_TA, states_TS = dataset_TA.getitem(), dataset_TS.getitem()
        states_TA, states_TS = states_TA.to(device), states_TS.to(device)
        # else: (states_TA, is_expert_traj), states_TS = dataset_TA.getitem(), dataset_TS.getitem()
        # training loss
        #print(states_TA.shape, states_TS.shape)
        policy_d = Disc(states_TA)
        expert_d = Disc(states_TS)
        
        #if flag: 
        #    expert_vs.append(policy_d[torch.nonzero(is_expert_traj == 1)].cpu().detach().numpy())
        #    policy_vs.append(policy_d[torch.nonzero(is_expert_traj == 0)].cpu().detach().numpy())
        
        if "PU" in train_hyperparams and train_hyperparams["PU"].find("rebalance") != -1:  # positive-unlabeled learning, "mixed distribution" alpha = 0.7: the negative label consists of 70% expert
            """
            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(device))
            policy_actually_expert = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.ones(policy_d.size()).to(device))
            policy_actually_nonexpert = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.zeros(policy_d.size()).to(device)) 
            gail_loss = expert_loss + train_hyperparams["PU_alpha"] * policy_actually_expert + (1 - train_hyperparams["PU_alpha"]) * policy_actually_nonexpert
            """
            expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(device))
            policy_nonexpert = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.zeros(policy_d.size()).to(device))
            expert_nonexpert = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.zeros(policy_d.size()).to(device))             
            # https://arxiv.org/pdf/1703.00593.pdf
            # gail_loss = train_hyperparams["PU_alpha"] * expert_loss + policy_nonexpert + (1 - train_hyperparams["PU_alpha"]) * policy_actually_nonexpert
            
            def QuadZero(x):
                if x < -0.01: return torch.zeros(1).double().to('cuda:0')
                elif -0.01 < x < 0.01: return 25 * (x + 0.01) ** 2
                else: return x
            
            if train_hyperparams["no_max"] == 1:
                    gail_loss = train_hyperparams["PU_alpha"] * expert_loss + torch.maximum(policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert, torch.zeros(1).double().to('cuda:0')) 
            elif train_hyperparams["no_max"] == 3: # softmax
                    gail_loss = train_hyperparams["PU_alpha"] * expert_loss + QuadZero(policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert) # linear + quadratic + linear
            elif train_hyperparams["no_max"] == 4:
                    gail_loss = train_hyperparams["PU_alpha"] * expert_loss + torch.nn.functional.softplus(policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert) 
            elif train_hyperparams['no_max'] == 2:     
                if policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert >= 0 or train_hyperparams["no_max"] == 0:
                    gail_loss = train_hyperparams["PU_alpha"] * expert_loss + policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert 
                else:
                    gail_loss = train_hyperparams["PU_alpha"] * expert_nonexpert - policy_nonexpert
            else:
                gail_loss = train_hyperparams["PU_alpha"] * expert_loss + policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert
            """
            if policy_nonexpert >= train_hyperparams["PU_alpha"] * expert_nonexpert:
                gail_loss = train_hyperparams["PU_alpha"] * expert_loss - train_hyperparams["PU_alpha"] * expert_nonexpert + policy_nonexpert
            else:
                gail_loss = train_hyperparams["PU_alpha"] * expert_nonexpert - policy_nonexpert
            """
        else: 
            expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(device))
            policy_loss = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.zeros(policy_d.size()).to(device))
            gail_loss = expert_loss + policy_loss
        #if "PU" in train_hyperparams and train_hyperparams["PU"] == "pretrain": grad_pen = compute_grad_pen(states_TS, states_TA, train_hyperparams['lipschitz'], target_grad=0.1)
        #else:
        grad_pen = compute_grad_pen(states_TS, states_TA, train_hyperparams['lipschitz'])
 
        loss = gail_loss + grad_pen
        if not no_log:
            
            #print("PU:", train_hyperparams["PU"])
            #exit(0)
            if "PU" in train_hyperparams and train_hyperparams["PU"].find("rebalance") != -1 and train_hyperparams["PU"].find("pretrain") != -1:
                wandb.log({'expert_d_pretrain': expert_d.mean(), 'policy_nonexpert_pretrain': policy_nonexpert, "expert_nonexpert_pretrain": expert_nonexpert, 'difference_pretrain': policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert, 'expert_output_pretrain': torch.sigmoid(expert_d).mean(), 'offline_output_pretrain': torch.sigmoid(policy_d).mean(), "expert_loss_pretrain": expert_loss, "grad_pen_pretrain": grad_pen, "loss_pretrain": loss})
            elif "PU" in train_hyperparams and train_hyperparams["PU"].find("rebalance") != -1:
                wandb.log({'expert_d': expert_d.mean(), 'policy_nonexpert': policy_nonexpert, "expert_nonexpert": expert_nonexpert, 'difference': policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert, 'expert_output': torch.sigmoid(expert_d).mean(), 'offline_output': torch.sigmoid(policy_d).mean(), "expert_loss": expert_loss, "grad_pen": grad_pen, "loss": loss})
            elif "PU" in train_hyperparams and train_hyperparams["PU"].find("pretrain") != -1:
               wandb.log({'expert_d_pretrain': expert_d.mean(), 'offline_d_pretrain': policy_d.mean(), 'expert_output_pretrain': torch.sigmoid(expert_d).mean(), 'offline_output_pretrain': torch.sigmoid(policy_d).mean(), "expert_loss_pretrain": expert_loss, "policy_loss_pretrain": policy_loss, "grad_pen_pretrain": grad_pen, "loss_pretrain": loss})
            else:
                wandb.log({'expert_d': expert_d.mean(), 'offline_d': policy_d.mean(), 'expert_output': torch.sigmoid(expert_d).mean(), 'offline_output': torch.sigmoid(policy_d).mean(), "expert_loss": expert_loss, "policy_loss": policy_loss, "grad_pen": grad_pen, "loss_disc": loss})
            """
            if flag:
                if train_hyperparams["PU"].find("pretrain") != -1:
                    wandb.log({"expert_in_TA_d_pretrain": np.concatenate(expert_vs).mean(), "offline_in_TA_d_pretrain": np.concatenate(policy_vs).mean(), "diff_in_TA_pretrain": np.concatenate(expert_vs).mean() - np.concatenate(policy_vs).mean()})
                else: 
                    wandb.log({"expert_in_TA_d": np.concatenate(expert_vs).mean(), "offline_in_TA_d": np.concatenate(policy_vs).mean(), "diff_in_TA": np.concatenate(expert_vs).mean() - np.concatenate(policy_vs).mean()})
                expert_vs, policy_vs = [], []
            """
        
        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if USE_EMA: ema.update()
        """
        ## VALIDATION #################
        if i % 10 == 0:
            if USE_EMA: ema.apply_shadow()
            if not flag: states_TA_valid = dataset_TAvalid.getitem()
            else: states_TA_valid, is_expert_traj_valid = dataset_TAvalid.getitem()
            
            policy_d_valid = Disc(states_TA_valid)
            
            policy_nonexpert = F.binary_cross_entropy_with_logits(
                    policy_d_valid,
                    torch.zeros(policy_d_valid.size()).to(device))
            if train_hyperparams["PU"].find("pretrain") != -1:
                wandb.log({'policy_nonexpert_validation_pretrain': policy_nonexpert})
            else:
                wandb.log({'policy_nonexpert_validation': policy_nonexpert})
            
            if USE_EMA: ema.restore()
        ############################
        """
    if USE_EMA: return Disc, ema
    else: return Disc
if __name__ == "__main__":
    env_name = "hopper"
    seed = 100
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    runtime = time.time()
    wandb.init(entity="XXXXXX",project="project2_contrastive", name=str(runtime)+"_"+str(seed)+"_"+env_name+"_discriminator")
    hyperparam = {"lr": 3e-4, "N": 2000, "lipschitz": 10}
    TA = torch.load("data/"+env_name+"/TA-read-again-unnormalized.pt")
    TS = torch.load("data/"+env_name+"/TS-read-again-unnormalized.pt")
    states_TA, states_TS = torch.from_numpy(np.array([TA[i]["state"] for i in range(len(TA))])).double().to(device), torch.from_numpy(np.array([TS[i]["state"] for i in range(len(TS))])).double().to(device)
    hyperparam["batch_size"] = min(500, len(TS))
    model, _ = train_discriminator(states_TA, states_TS, hyperparam)
    
    print(states_TA, states_TS)    
    e2 = model.embedding(states_TS)
    d = torch.zeros(states_TS.shape[0], states_TS.shape[0])
    f=open("seeembedding.txt", "w")
    for i in tqdm(range(states_TS.shape[0])): 
        d[i] = ((e2[i].view(1, -1).repeat(states_TS.shape[0], 1) - e2) ** 2).sum(axis=1)
        for j in range(states_TS.shape[0]):
            f.write(str(d[i, j].item())+" ")
        f.write("\n")
        f.flush()
    f.close()
    #v1 = model.predict_reward(states_TA[-500:])
    #v2 = model.predict_reward(states_TS)
    # print("v1min:", v1.min(),"v2min:", v2.min(),"v1max:", v1.max(), "v2max:", v2.max())
    #torch.save(model, "model/"+env_name+"/discriminator.pt")
