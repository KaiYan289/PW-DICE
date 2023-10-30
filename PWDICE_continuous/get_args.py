import argparse
import subprocess
def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--data_index", help="data_index", type=int, default=0)
    # parser.add_argument("--TS_type", type=str, default="full") # "full" or "goal"
    # parser.add_argument("--transition", type=str, default="estimated") # "ideal" or "estimated"
    parser.add_argument("--clip_var", type=int, default=0)
    parser.add_argument("--scale_dist_with_occupancy", type=float, default=0)
    parser.add_argument("--scale_dist_with_occupancy_product", type=float, default=0)
    parser.add_argument("--scale_dist_with_context", type=float, default=0) 
    parser.add_argument("--distance", type=str, default="euclidean")
    parser.add_argument("--BC_optimize", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--idm_lr", type=float, default=0.0003) # inverse dynamic model learning rate
    parser.add_argument("--wbc_lr", type=float, default=0.001) # weighted behavior cloning learning rate
    parser.add_argument("--normalizing_factor", type=float, default=1)
    parser.add_argument("--joint", type=int, default=1)
    parser.add_argument("--N_idm", type=int, default=200)
    parser.add_argument("--skip_suffix_TA", type=str, default="")
    parser.add_argument("--skip_suffix_TS", type=str, default="")
    parser.add_argument("--N", type=int,default=100)
    parser.add_argument("--lipschitz", type=float, default=10)
    parser.add_argument("--initial", type=str, default="TS")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--wbc_weight_decay", type=float, default=0)
    parser.add_argument("--save_const", type=int, default=400)
    parser.add_argument("--epsilon_1", type=float, default=0.02)
    parser.add_argument("--train_with_terminal", type=int, default=0)
    parser.add_argument("--interlock_E", type=int, default=0)
    parser.add_argument("--BC_with_terminal", type=int, default=1)
    parser.add_argument("--use_s1", type=int, default=1)
    parser.add_argument("--use_s2", type=int, default=1)
    parser.add_argument("--_use_policy_entropy_constraint",type=int,default=0)
    parser.add_argument("--load_smodice_reward", type=int,default=0)
    parser.add_argument("--BC_stage_per_weight_epoch", type=int, default=1)
    parser.add_argument("--BC_epoch_per_stage", type=int, default=1)
    parser.add_argument("--BC_stage_per_eval", type=int, default=1)
    parser.add_argument("--d_e", type=str, default="exp")
    parser.add_argument("--ratio_start_BC_training", type=int, default=100)
    parser.add_argument("--epsilon_2", type=float, default=0.02)
    parser.add_argument("--BCdebug_N", type=int, default=0)
    parser.add_argument("--uniform", type=int, default=0)
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--use_bn", type=str, default="no") # batch normalization
    parser.add_argument("--decay", type=float, default=0.99) # LRdecay for learning rate scheduler
    parser.add_argument("--normalize_obs", type=int, default=1)
    parser.add_argument("--scheduler_n", type=int, default=-1) # <0 is not using learning rate scheduler
    parser.add_argument("--GT1", type=int, default=0)
    parser.add_argument("--BCdebug", type=int, default=0)
    parser.add_argument("--absorb", type=int, default=1)
    parser.add_argument("--eval_deter", type=int, default=1)
    parser.add_argument("--absorb_distance", type=int, default=0) # deprecated
    parser.add_argument("--load_net", type=int, default=0)
    parser.add_argument("--max_random_traj", type=int, default=9999999)
    parser.add_argument("--max_expert_traj", type=int, default=9999999)
    parser.add_argument("--dist_scale", type=float, default=1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--wbc_scheduler", type=int, default=0)
    parser.add_argument("--cheat", type=int, default=0)
    parser.add_argument("--lambda2_reg", type=float, default=0)
    parser.add_argument("--clip_wbc", type=float, default=0)
    parser.add_argument("--batch_size0", type=int, default=512) # inverse dynamic model
    parser.add_argument("--batch_size1", type=int, default=512)  # uniform state sampling
    parser.add_argument("--batch_size2", type=int, default=512) # rho_I sampling
    parser.add_argument("--batch_size3", type=int, default=32) # use with IDM
    parser.add_argument("--batch_size4", type=int, default=512) # initial state sampling
    parser.add_argument("--batch_size5", type=int, default=512) # rho_E sampling
    parser.add_argument("--env_name", type=str, default='hopper')
    parser.add_argument("--smooth", type=str, default="no")
    parser.add_argument("--smooth_coeff", type=float, default=0)
    parser.add_argument("--use_s3", type=int, default=0)
    parser.add_argument("--EMA", type=float, default=0)
    parser.add_argument("--auto", type=int, default=1)
    parser.add_argument("--mixed_ratio", type=float, default=0)
    args = parser.parse_args()
    return args
    
def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()
    
def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()