import d4rl
import gym
import numpy as np
from dataset import get_dataset
import torch
from tqdm import tqdm
import h5py
from envs.kitchen_envs import *
env_names = ['halfcheetah', 'hopper', 'walker2d', 'ant']
np.random.seed(888)

def get_data_pre(env, num_traj=1e100, dataset=None, is_TS=False, missing_num=0): # SMODICE format
        
        initial_obs_, obs_, next_obs_, action_, reward_, done_, step_ = [], [], [], [], [], [], []
        
        if dataset is None:
            dataset = env.get_dataset() 
        N = dataset['rewards'].shape[0]

        use_timeouts = ('timeouts' in dataset)
        traj_count = 0
        episode_step = 0
        for i in range(N-1):
            # only use this condition when num_traj < 2000
            if traj_count == num_traj:
                break
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i+1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])
            # if not expert_data: print(dataset['terminals'][i])
            is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
            if is_final_timestep and not is_TS:
                # Skip this transition and don't apply terminals on the last step of an episode
                traj_count += 1
                episode_step = 0
                continue
            if missing_num == 0 or i % missing_num != missing_num // 2:
                # if traj_count > 0 or not expert_data:     
                obs_.append(obs)
                next_obs_.append(new_obs)
                action_.append(action)
                reward_.append(reward)
                done_.append(done_bool) 
                step_.append(episode_step)
            episode_step += 1

            if done_bool or is_final_timestep:
                traj_count += 1
                episode_step = 0
        
        dataset = {
            'observations': np.array(obs_, dtype=np.float32),
            'actions': np.array(action_, dtype=np.float32),
            'next_observations': np.array(next_obs_, dtype=np.float32),
            'rewards': np.array(reward_, dtype=np.float32),
            'terminals': np.array(done_, dtype=np.float32),
            'steps': np.array(step_, dtype=np.float32)
        }
        
        return dataset


def get_data(env, lim=1e100, missing_num=0, is_TS=False, dataset=None):
    data = []
    tot_traj = 0
    
    if dataset is None: dataset = get_data_pre(env, lim, missing_num=missing_num, is_TS=is_TS)

    use_timeouts = ('timeouts' in dataset)
    
    FLAG = int(dataset is None or 'next_observations' in dataset) 
    
    for i in range(dataset['actions'].shape[0] - (1 - FLAG)):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['next_observations'][i].astype(np.float32) if FLAG else dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        episode_step = dataset['steps'][i]
        if not is_TS: data.append({"state": obs, "action": action, "next_state": new_obs, "terminal": done_bool, "step": episode_step, 'reward': reward})
        else: data.append({"state": obs, "action": action, "next_state": new_obs, "terminal": done_bool, "step": episode_step}) 
        is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
        if is_final_timestep:
            tot_traj += 1
            if tot_traj >= lim: break
    return data


##################################################################### NORMAL #######################################################

env_names = ['hopper', 'halfcheetah', 'walker2d', 'ant'] 

for env_name in env_names: 
    
    
    data = get_data(gym.make(env_name + "-expert-v2"), 200 if env_name != 'walker2d' else 100) + get_data(gym.make(env_name + "-random-v2"))
    
    print(env_name)
        
    print("totlen:", len(data))
    
    torch.save(data, "data/"+env_name+"/TA-read-again-unnormalized.pt")
    
    data = get_data(gym.make(env_name + "-expert-v2"), 40) + get_data(gym.make(env_name + "-random-v2"))
    torch.save(data, "data/"+env_name+"/TA-read-again-unnormalizedexpert40.pt")
        
    data = get_data(gym.make(env_name + "-expert-v2"), 1, is_TS=True)

    torch.save(data, "data/"+env_name+"/TS-read-again-unnormalized.pt")