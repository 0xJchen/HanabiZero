#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 main.py --env Hanabi-Small --case hanabi --opr train --seed 1 --num_gpus 4 --num_cpus 96 --force \
  --cpu_actor 5 --gpu_actor 20 \
  --p_mcts_num 16\
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.999 \
  --amp_type 'torch_amp' \
  --info 'global-state-full' \
  --actors 8 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 100 \
  --decay_rate 1\
  --decay_step 200000 \
  --lr 0.1 \
  --stack 4 \
  --mdp_type 'global'
