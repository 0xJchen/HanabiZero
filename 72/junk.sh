#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
#58
python3 ../main.py --env Hanabi-Small --case hanabi --opr train --seed 1 --num_gpus 4 --num_cpus 110 --force \
  --cpu_actor 35 --gpu_actor 25 \
  --p_mcts_num 8\
  --extra junk \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'junk' \
  --actors 16 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 6 \
  --debug_interval 100 \
  --debug_batch \
  --lr 0.1 \
  --decay_rate 1\
  --stack 1
 # --load_model \
#  --model_path 'model_confirm_small_200k.p'


#revisivt#@wjc current disabled from 0.99
