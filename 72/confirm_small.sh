#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
#58
python3 ../main.py --env Hanabi-Small --case hanabi --opr test --seed 1 --num_gpus 2 --num_cpus 20 --force --batch_actor 1\
  --p_mcts_num 8\
  --extra 3p \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'none' \
  --info 'small_full_confirm_reanal_3player' \
  --actors 2 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 100 \
  --debug_batch \
  --lr 0.01 \
  --stack 4 \
  --load_model \
  --model_path 'small_best.p'


#revisivt#@wjc current disabled from 0.99
