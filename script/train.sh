#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 ../main.py --env Hanabi-Full --case hanabi --opr train --seed 10 --num_gpus 4 --num_cpus 110 --force\
  --cpu_actor 6 --gpu_actor 16 \
  --p_mcts_num 24\
  --extra 2022 \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.999 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'decay_n_op' \
  --actors 40 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 100 \
  --lr 0.1 \
  --decay_rate 0.1 \
  --decay_step 200000 \
  --stack 1 \
  --optim 'sgd'



