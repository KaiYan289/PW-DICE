import torch
from tqdm import tqdm
device = torch.device('cuda:0')
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
class Testdataset(Dataset):
    def __init__(self, feature1, feature2, feature3, label):
        self.n = feature1.shape[0]
        self.dim_feature1, self.dim_feature2, self.dim_label = feature1.shape[1], feature2.shape[1], label.shape[1]
        self.feature1, self.feature2, self.feature3, self.label = feature1, feature2, feature3, label   

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {"state": self.feature1[idx], "action": self.feature2[idx], "next_state": self.label[idx], "terminal": self.feature3[idx]}
        
class RepeatedDataset:
    def __init__(self, datas, batch_size, start_with_random=True):
        self.datas = []
        for data in datas: # list of arrays with the same first dimension.
            self.datas.append(data.clone())
        self.counter, self.idx, self.batch_size = 0, torch.randperm(self.datas[0].shape[0]), batch_size
        if start_with_random:
            for _ in range(len(self.datas)):
                print("shape:", self.datas[_].shape)
                self.datas[_] = self.datas[_][self.idx]
    
    def __len__(self):
        return self.datas[0].shape[0] // self.batch_size    
    
    def getitem(self):
        if self.counter + self.batch_size > len(self.idx):
            self.counter, self.idx = 0, torch.randperm(self.datas[0].shape[0])
            for _ in range(len(self.datas)):
                self.datas[_] = self.datas[_][self.idx]
        ret = []
        for _ in range(len(self.datas)):
            ret.append(self.datas[_][self.counter:self.counter+self.batch_size])
        self.counter += self.batch_size
        """
        print(self.counter, self.counter+self.batch_size)
        
        for _ in range(len(self.datas)):
            print(self.datas[_][self.counter:self.counter+self.batch_size])
        """
        if len(self.datas) == 1: return ret[0]
        else: return ret
        
def add_terminals(states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA, absorb, THE_TERMINAL):
    new_states_TA, new_actions_TA, new_next_states_TA, new_terminals_TA, new_steps_TA, new_rewards_TA = [], [], [], [], [], []
    have_reward = len(rewards_TA) > 0
    
    for i in tqdm(range(states_TA.shape[0])):
        # print(steps_TA[i].shape, terminals_TA[i].shape, is_TS_TA[i].shape)
        if absorb == 1 and terminals_TA[i] == 1: # remember to debug this! is true/false or 1/0?
            the_terminal = THE_TERMINAL
            
            next_state = torch.cat([next_states_TA[i], torch.zeros(1).double()]).view(1, -1)
            
            # type 1: add two steps
            new_actions_TA.append(actions_TA[i].view(1, -1))
            new_steps_TA.append(steps_TA[i].view(1))
            new_states_TA.append(torch.cat([states_TA[i], torch.zeros(1).double()]).view(1, -1))
            new_next_states_TA.append(next_state)
            new_terminals_TA.append(torch.zeros(1).double())
            if have_reward: new_rewards_TA.append(rewards_TA[i].view(-1))
            
            new_actions_TA.append(actions_TA[i].view(1, -1)) # slightly different from SMODICE.
            new_steps_TA.append(steps_TA[i].view(1) + 1)
            new_states_TA.append(next_state)
            new_next_states_TA.append(the_terminal)
            new_terminals_TA.append(torch.zeros(1).double())
            if have_reward: new_rewards_TA.append(rewards_TA[i].view(-1))
            
            new_actions_TA.append(actions_TA[i].view(1, -1))
            new_steps_TA.append(steps_TA[i].view(1) + 2)
            new_states_TA.append(the_terminal)
            new_next_states_TA.append(the_terminal)
            new_terminals_TA.append(terminals_TA[i].view(1))
            if have_reward: new_rewards_TA.append(rewards_TA[i].view(-1))
            
            """
            # type 2: add one step
            
            new_actions_TA.append(actions_TA[i].view(1, -1))
            new_steps_TA.append(steps_TA[i].view(1))
            new_states_TA.append(torch.cat([states_TA[i], torch.zeros(1).double()]).view(1, -1))
            new_next_states_TA.append(next_state)
            new_terminals_TA.append(torch.zeros(1).double())
            
            new_actions_TA.append(actions_TA[i].view(1, -1)) # slightly different from SMODICE.
            new_steps_TA.append(steps_TA[i].view(1) + 1)
            new_states_TA.append(next_state)
            new_next_states_TA.append(the_terminal)
            new_terminals_TA.append(terminals_TA[i].view(1))
            """
        elif absorb == 1:
            next_state = torch.cat([next_states_TA[i], torch.zeros(1).double()]).view(1, -1)
            
            new_actions_TA.append(actions_TA[i].view(1, -1))
            new_steps_TA.append(steps_TA[i].view(1))
            new_states_TA.append(torch.cat([states_TA[i], torch.zeros(1).double()]).view(1, -1))
            new_next_states_TA.append(next_state)
            new_terminals_TA.append(terminals_TA[i].view(1))        
            if have_reward: new_rewards_TA.append(rewards_TA[i].view(-1))
        else:
            new_actions_TA.append(actions_TA[i].view(1, -1))
            new_steps_TA.append(steps_TA[i].view(1))
            new_states_TA.append(states_TA[i].view(1, -1))
            new_next_states_TA.append(next_states_TA[i].view(1, -1)) 
            new_terminals_TA.append(terminals_TA[i].view(1)) 
            if have_reward: new_rewards_TA.append(rewards_TA[i].view(-1))
            """
            if terminals_TA[i] == 1:
                new_actions_TA.append(actions_TA[i].view(1, -1)) # slightly different from SMODICE.
                new_steps_TA.append(steps_TA[i].view(1) + 1)
                new_states_TA.append(next_states_TA[i].view(1, -1))
                new_next_states_TA.append(next_states_TA[i].view(1, -1))
                new_terminals_TA.append(torch.ones(1).double())
           """
    
    if have_reward: new_rewards_TA = torch.cat(new_rewards_TA)
    new_states_TA, new_actions_TA, new_next_states_TA = torch.cat(new_states_TA), torch.cat(new_actions_TA), torch.cat(new_next_states_TA)
    new_terminals_TA, new_steps_TA =  torch.cat(new_terminals_TA), torch.cat(new_steps_TA)
    return new_states_TA, new_actions_TA, new_next_states_TA, new_terminals_TA, new_steps_TA, new_rewards_TA
    
      
def construct_dataset(states_TA, actions_TA, terminals_TA, next_states_TA, args, batch_size0, coeffs=None):

    idx = torch.randperm(states_TA.shape[0])
    train_num, test_num = int(0.9 * len(idx)), len(idx) - int(0.9 * len(idx))
    idx_train, idx_test = idx[:train_num], idx[train_num:]
    print("dataset size:", train_num, test_num)
    
    coeff_lst_train, coeff_lst_test = [coeffs[idx_train].to(device).double()] if coeffs is not None else [], [coeffs[idx_test].double()] if coeffs is not None else []
    
    train_loader = RepeatedDataset([states_TA[idx_train].double(), actions_TA[idx_train].double(), terminals_TA[idx_train].double(), next_states_TA[idx_train].double()] + coeff_lst_train, batch_size0) 
    test_loader = RepeatedDataset([states_TA[idx_test].double(), actions_TA[idx_test].double(), terminals_TA[idx_test].double(), next_states_TA[idx_test].double()] + coeff_lst_test, min(len(idx_test), batch_size0))
        
    return train_loader, test_loader  

def get_dataset(dataset, cut_len=1e100, fill=False):
    original_len = len(dataset)
    # print(np.array([dataset[i]["action"] for i in range(min(cut_len, len(dataset)))]))
    print("len:", original_len, dataset[0]["state"].shape, dataset[0]["next_state"].shape)
    states = torch.from_numpy(np.array([dataset[i]["state"] for i in range(min(cut_len, len(dataset)))])).double()
    actions = torch.from_numpy(np.array([dataset[i]["action"] for i in range(min(cut_len, len(dataset)))])).double()
    next_states = torch.from_numpy(np.array([dataset[i]["next_state"] for i in range(min(cut_len, len(dataset)))])).double()
    terminals = torch.from_numpy(np.array([dataset[i]["terminal"] for i in range(min(cut_len, len(dataset)))])).double()
    steps = torch.from_numpy(np.array([dataset[i]["step"] for i in range(len(dataset))])).double()
    
    if "reward" in dataset[0]: rewards = torch.from_numpy(np.array([dataset[i]["reward"] for i in range(len(dataset))])).double()
    else: rewards = []
    print("terminal sum:", terminals.sum())

    if fill:  
        states, actions, next_states = states.repeat((original_len - 1) // cut_len + 1, 1), actions.repeat((original_len - 1) // cut_len + 1, 1), next_states.repeat((original_len - 1) // cut_len + 1, 1)
        terminals, steps = terminals.repeat((original_len - 1) // cut_len + 1), steps.repeat((original_len - 1) // cut_len + 1)
    return states, actions, next_states, terminals, steps, rewards

def concatlist(lst): # return a concatenation of list
    return sum(lst, [])

def list2gen(lst):
    return (_ for _ in lst)

NAME = "halfcheetah"
 
def truncate_dataset(path, TA_name, TS_name, num_step):
    TA = torch.load(path+"/"+TA_name+".pt")
    TS = torch.load(path+"/"+TS_name+".pt")
    steps_TA = torch.from_numpy(np.array([TA[i]["step"] for i in range(len(TA))])).double()
    steps_TS = torch.from_numpy(np.array([TS[i]["step"] for i in range(len(TS))])).double()
    # print("shape:", steps_TA.shape, steps_TS.shape)
    print(steps_TA[:100], steps_TS[:100])
    idx_TA, idx_TS = torch.nonzero(steps_TA < num_step).view(-1), torch.nonzero(steps_TS < num_step).view(-1)
    print("idx_TA:", idx_TA[:100], "len-idx-TA", len(idx_TA))
    print("idx_TS:", idx_TS[:100], "len-idx-TS", len(idx_TS))
    new_TA, new_TS = [], []
    for i in tqdm(range(len(idx_TA))): new_TA.append(TA[idx_TA[i]])
    for i in tqdm(range(len(idx_TS))): new_TS.append(TS[idx_TS[i]])
    torch.save(new_TA, "data/"+NAME+"/TA-truncate"+str(num_step)+".pt")
    torch.save(new_TS, "data/"+NAME+"/TS-truncate"+str(num_step)+".pt")
if __name__ == "__main__":
    """
    import d4rl
    import gym
    env = gym.make("hopper-random-v2") # hopper-random's ending state is correct; halfcheetah has no ending state;
    dataset = env.get_dataset()
    print(np.nonzero(dataset["terminals"]))
    nz = np.nonzero(dataset["terminals"])[0]
    #for i in range(0, nz.shape[0]):
    #    print(nz[i], end = " ")
    """
    TA_dataset = torch.load("data/hopper/TA-read-again-unnormalized.pt")
    
    states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA = get_dataset(TA_dataset)
    states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA = add_terminals(states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA, 0, torch.cat([torch.zeros_like(torch.from_numpy(TA_dataset[0]["state"])).double(), torch.ones(1).double()]).view(1, -1))
    
    initials = torch.nonzero(steps_TA == 0).view(-1)
    ends = torch.nonzero(terminals_TA == 1).view(-1)
    NS = next_states_TA[ends]
    for i in range(116633, 116638):
        print("i:", i, "step:", steps_TA[i], "state:", states_TA[i], "next_states_TA:", next_states_TA[i])
    """
    for i in range(10):
        s, s2 =((NS[i].view(1, -1).repeat(states_TA.shape[0], 1) - states_TA) ** 2).sum(dim=1).min(), ((NS[i].view(1, -1).repeat(states_TA.shape[0], 1) - states_TA) ** 2).sum(dim=1).argmin()
        print(i, "min dist:", s, "argmin:", s2, "current:", ends[i], "argmin step:", steps_TA[s2], "step:", steps_TA[ends[i]]) 
    """
    #print(initials)
    #print(ends)
    #for i in range(len(nz)):
    #    print(ends[-i] - nz[-i], end=" ")
    """
    for i in range(len(nz)):
        print(ends[i+200] - nz[i], end=" ")
    """
    #truncate_dataset("data/"+NAME, "TA-read-again-unnormalized", "TS-read-again-unnormalized", 150)