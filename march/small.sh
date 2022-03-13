sh clean.sh
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 ../main.py --env Hanabi-Small --case hanabi --opr train --seed 1 --num_gpus 4 --num_cpus 96 --force \
  --cpu_actor 14 --gpu_actor 26 \
  --p_mcts_num 20\
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.999 \
  --amp_type 'torch_amp' \
  --info 'global-state-small' \
  --actors 16 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 500 \
  --decay_rate 1\
  --decay_step 200000 \
  --lr 0.1 \
  --stack 4
