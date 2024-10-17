# PW-DICE
Codebase for ICML 2024 paper "Offline Imitation from Observation via Primal Wasserstein State Occupancy Matching". The code for tabular environments is in PWDICE_tabular_wrapup, while the code for continuous environments is in PWDICE_continuous.



## Tabular Environments

The .py files with suffix "arbitrary" are the scripts used in our experiments; run "ours_newKL_solver_arbitrary" for our results.

## Continuous Environments

run datagen.py to generate data and put them in "data" directory under PWDICE_continuous; then run "ours.py" for our results with KL regularizer and "ours_chi.py" for our results with chi-square regularizer.

Example usage: 

**Normal:**

python ours.py --N 250000 --env_name walker2d --lr 0.0003 --distance learned_pd --absorb 0 --dist_scale 5 --log_interval 1000 --BC_stage_per_eval 5 --ratio_start_BC_training 10000 --BC_optimize 1 --initial TA --normalizing_factor 1 --epsilon_1 0.5 --epsilon_2 0.5 --scale_dist_with_occupancy 1 --normalize_obs 1 --scale_dist_with_context 0 --batch_size1 1024 --batch_size2 1024 --batch_size4 1024 --batch_size5 1024 --save_const 100 --mixed_ratio 0.01 --seed 10004

where epsilon_1, epsilon_2 controls the the regularizer coefficient, and learned_pd is our learned distance metric (can be substituted as "euclidean" and "cosine").

**Mismatch dynamics:**

python ours.py --N 250000 --env_name halfcheetah --lr 0.00003 --distance learned_pd --absorb 0 --dist_scale 5 --log_interval 200 --BC_stage_per_eval 5 --ratio_start_BC_training 10000 --BC_optimize 1 --initial TA --normalizing_factor 1 --epsilon_1 0.1 --epsilon_2 0.5 --scale_dist_with_occupancy 1 --weight_decay 0.00001 --normalize_obs 1 --lambda2_reg 0.00001 --scale_dist_with_context 0 --skip_suffix_TS mismatch --batch_size1 1024 --batch_size2 1024 --batch_size4 1024 --batch_size5 1024 --save_const 100 --auto 1

python ours.py --N 250000 --env_name ant --lr 0.00003 --distance learned_pd --absorb 0 --dist_scale 0.0001 --log_interval 200 --BC_stage_per_eval 5 --ratio_start_BC_training 10000 --BC_optimize 1 --initial TA --normalizing_factor 1 --epsilon_1 0.01 --epsilon_2 1 --scale_dist_with_occupancy 1 --weight_decay 0.00001 --normalize_obs 1 --lambda2_reg 0.00001 --scale_dist_with_context 0 --skip_suffix_TS mismatch --batch_size1 1024 --batch_size2 1024 --batch_size4 1024 --batch_size5 1024 --save_const 100 --auto 1 --seed 18

**Distorted state representation:**

python ours.py --N 250000 --env_name hopper --lr 0.0003 --distance learned_pd --absorb 0 --dist_scale 1 --log_interval 1000 --BC_stage_per_eval 5 --ratio_start_BC_training 10000 --BC_optimize 1 --initial TA --normalizing_factor 1 --epsilon_1 0.1 --epsilon_2 0.5 --scale_dist_with_occupancy 1 --normalize_obs 1 --scale_dist_with_context 0 --batch_size1 1024 --batch_size2 1024 --batch_size4 1024 --batch_size5 1024 --save_const 100 --mixed_ratio 0.01 --seed 10006 --skip_suffix_TS distorted1 --skip_suffix_TA distorted1
