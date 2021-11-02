#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
python3 main.py --env Hanabi-Small --case hanabi --opr train --seed 1 --num_gpus 2 --num_cpus 40 --force --batch_actor 24\
  --p_mcts_num 16\
  --extra small_partial_obs_best_model_revert_50_partial_real \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'small_partial' \
#  --load_model \
#  --model_path 'results/models/model_tst.p' \


#revisivt#@wjc current disabled from 0.99
