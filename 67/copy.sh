#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
python3 ../main.py --env Hanabi-Full --case hanabi --opr train --seed 1 --num_gpus 4 --num_cpus 45 --force --batch_actor 40\
  --p_mcts_num 8\
  --extra full_2p_test\
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'full_2_player_share_new_model_1024_512_512_no_get_fixed_obj_store_test' \
  --actors 2 \
  --simulations 100 \
  --batch_size 128 \
  --debug_batch \
  --debug_interval 100
#  --load_model \
#  --model_path 'results/models/model_tst.p' \


#revisivt#@wjc current disabled from 0.99
