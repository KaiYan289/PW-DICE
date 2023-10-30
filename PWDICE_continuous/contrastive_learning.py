import torch
import argparse
from dataset import get_dataset, RepeatedDataset
from NN import Contrastive, Contrastive_Sphere, Contrastive_PD, Contrastive_SPD
from get_args import get_git_diff, git_commit
from advance_NN import *
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import wandb
from torch.optim import Adam
device = torch.device('cuda:0')
def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--data_name", help="data_name", type=str)
    parser.add_argument("--data_path", help="data_path", type=str)
    parser.add_argument("--batch_size", help="BS", type=int, default=4096)
    parser.add_argument("--N", help="N", type=int, default=100)
    parser.add_argument("--eval", help="eval", type=int, default=0)
    parser.add_argument("--type", help="type", type=str, default="normal")
    parser.add_argument("--first_normalization", help="norm", type=int, default=1)
    parser.add_argument("--second_normalization", help="norm", type=int, default=0)
    args = parser.parse_args()
    return args

def train_contrastive_model(env_name, model_name, states_TA, next_states_TA):
    if model_name.find("_pd") != -1:
        model = Contrastive_PD(states_TA.shape[-1]).to(device).double()
    elif model_name.find("_spd") != -1:
        model = Contrastive_SPD(states_TA.shape[-1], first_normalization=int(model_name[-3]), second_normalization=int(model_name[-1])).to(device).double() # ...spd_0_1
    elif model_name.find("_sphere") != -1:
        model = Contrastive_Sphere(states_TA.shape[-1]).to(device).double()
    else: 
        model = Contrastive(states_TA.shape[-1]).to(device).double()
    
    N = 200
    
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    train_loader = RepeatedDataset([states_TA.to(device).double(), next_states_TA.to(device).double()], 4096)
    for i in tqdm(range(N)):
        for batch in tqdm(range(len(train_loader))):
            states, next_states = train_loader.getitem()
            contrastive_logits = model(states, next_states)
            labels = torch.arange(contrastive_logits.shape[0]).long().to(device) # as similar as matrix I as possible
            loss = cross_entropy_loss(contrastive_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            wandb.log({"CEloss": loss})
    torch.save(model, "model/"+env_name+"/"+model_name+".pt")
        
    return model
if __name__ == "__main__":
    args = get_args()    
    data = torch.load("data/"+args.data_path+"/TA-"+args.data_name+".pt")
    
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime+"_contrastive") 
        
    seed = args.seed
    if args.type == "normal": suffix = ""
    elif args.type == "sphere": suffix = "_sphere" 
    elif args.type == "pd": suffix = "_pd"
    elif args.type == "spd": suffix = "_spd_" + str(args.first_normalization)+"_"+str(args.second_normalization)
    elif args.type == "twinpd": suffix = "_twinpd" 
    else: raise NotImplementedError("Error!")
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    if args.eval == 0:
        runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if len(get_git_diff()) > 0:
            git_commit(runtime) 
        print("please input description:")
        a = input()
        wandb.init(entity="XXXX",project="project2_contrastive", name=str(runtime)+"_"+str(args.seed)+a+"_contrastive")
        states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, is_TS_TA = get_dataset(data)
        non_terminal = torch.nonzero(terminals_TA == 0).view(-1)
        states_TA, next_states_TA = states_TA[non_terminal], next_states_TA[non_terminal]
        print("shape:", states_TA.shape)
        train_loader = RepeatedDataset([states_TA.to(device).double(), next_states_TA.to(device).double()], args.batch_size)
        if args.type == "normal": model = Contrastive(states_TA.shape[-1]).to(device).double()
        elif args.type == "sphere": model = Contrastive_Sphere(states_TA.shape[-1]).to(device).double()
        elif args.type == "pd": model = Contrastive_PD(states_TA.shape[-1]).to(device).double()
        elif args.type == "spd": model = Contrastive_SPD(states_TA.shape[-1], first_normalization=args.first_normalization, second_normalization=args.second_normalization).to(device).double()
        # elif args.type == "twinpd": model = Contrastive_TwinPD(states_TA.shape[-1])
        if args.type != "twinpd":
            cross_entropy_loss = torch.nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(),)
            for i in tqdm(range(args.N)):
                for batch in tqdm(range(len(train_loader))):
                    states, next_states = train_loader.getitem()
                    contrastive_logits = model(states, next_states)
                    labels = torch.arange(contrastive_logits.shape[0]).long().to(device) # as similar as matrix I as possible
                    loss = cross_entropy_loss(contrastive_logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    wandb.log({"CEloss": loss})
                # print(states.shape)
        
        else:
            belong_TA = torch.zeros(steps_TA.shape[0])
            initials = torch.nonzero(steps_TA == 0)
            for i in range(len(initials)):
                if i < len(initials) - 1:
                    belong_TA[initials[i]:initials[i+1]] = i
                else: 
                    belong_TA[initials[i]:] = i
                    
            model = train_contrastive_model_twin(args.data_path, args.data_name+"_contrastive"+str(suffix), states_TA, next_states_TA, belong_TA)
           
        torch.save(model, "model/"+args.data_path+"/"+args.data_name+"_contrastive"+str(suffix)+".pt")
    print("evaluation!")
    model = torch.load("model/"+args.data_path+"/"+args.data_name+"_contrastive"+str(suffix)+".pt")
    
    if args.eval == 1:
        states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, is_TS_TA = get_dataset(data)
    for i in range(100):
        # print("next state:", ((model.encode(states_TA[i].view(1, -1)) - model.encode(next_states_TA[i].view(1, -1))) ** 2).sum(dim=1))
        j = np.random.randint(states_TA.shape[0])
        
        if args.type in ["normal", "pd", "spd", "twinpd"]: d = lambda X, Y: ((model.encode(X) - model.encode(Y)) ** 2).sum(dim=1)
        elif args.type == "sphere": d = lambda X, Y: torch.arccos(model.encode(X) - model.encode(Y))
        else: raise NotImplementedError("Error!")
        print("next state:", d(states_TA[i].view(1, -1), next_states_TA[i].view(1, -1)))
        print("i", i, "j:", j, "random state:", d(states_TA[i].view(1, -1), states_TA[j].view(1, -1)))
