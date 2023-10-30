import torch 
import numpy as np
from dataset import get_dataset
env_names = ["hopper", "walker2d", "ant", "halfcheetah"]
def normalize(env_name):
    TS = torch.load("data/"+env_name+"/TS-read-again-unnormalized.pt")
    TA = torch.load("data/"+env_name+"/TA-read-again-unnormalized.pt")
    torch.save(TS, 'data/'+env_name+"/TS-unnormalized.pt")
    torch.save(TA, "data/"+env_name+"/TA-unnormalized.pt")
    
    states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, is_TS_TA = get_dataset(TA)
    states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, is_TS_TS = get_dataset(TS)
    
    TA_mean, TA_std = states_TA.mean(dim=0), states_TA.std(dim=0) + 1e-10
    print("mean:", TA_mean, "std:", TA_std)
    states_TA = (states_TA - TA_mean.view(1, -1)) / TA_std.view(1, -1)
    states_TS = (states_TS - TA_mean.view(1, -1)) / TA_std.view(1, -1)
    next_states_TA = (next_states_TA - TA_mean.view(1, -1)) / TA_std.view(1, -1)
    next_states_TS = (next_states_TS - TA_mean.view(1, -1)) / TA_std.view(1, -1)
    
    assert len(TS) == states_TS.shape[0] and len(TA) == states_TA.shape[0], "Error!"
    #print(states_TS[0], TS[0]["state"])
    #exit(0)
    for i in range(len(TS)):
        TS[i]["state"], TS[i]["next_state"] = states_TS[i].cpu().numpy(), next_states_TS[i].cpu().numpy()
    for i in range(len(TA)):
        TA[i]["state"], TA[i]["next_state"] = states_TA[i].cpu().numpy(), next_states_TA[i].cpu().numpy()
    
    torch.save(TS, 'data/'+env_name+"/TS-normalized.pt")
    torch.save(TA, "data/"+env_name+"/TA-normalized.pt")
    
for env_name in env_names:
    normalize(env_name)