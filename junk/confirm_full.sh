#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
#58
#46/9
python3 ../main.py --env Hanabi-Full --case hanabi --opr train --seed 10 --num_gpus 1 --num_cpus 10 --force --batch_actor 1\
  --p_mcts_num 8\
  --extra reproduce_best \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.999 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'full_share_rescale_legel_4_stack_td_5_sim_80' \
  --actors 1 \
  --simulations 80 \
  --batch_size 2 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 100 \
  --lr 0.1 \
  --decay_rate 0.5 \
  --decay_step 200000 \
  --stack 1

  #--debug_batch \

#revisivt#@wjc current disabled from 0.99
