#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
python3 ../main.py --env Hanabi-Full --case hanabi --opr train --seed 1 --num_gpus 1 --num_cpus 15 --force --batch_actor 10 \
  --p_mcts_num 8\
  --extra junk \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'junk' \
  --actors 2 \
  --simulations 100 \
  --batch_size 128 \
  --load_model \
  --model_path './model/model_50000.p' \
  --debug_batch

#revisivt#@wjc current disabled from 0.99