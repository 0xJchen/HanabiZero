#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=3
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
#58
python3 ../main.py --env Hanabi-Small --case hanabi --opr train --seed 1 --num_gpus 1  --num_cpus 22 --force --batch_actor 4\
  --p_mcts_num 8\
  --extra small_share_2p_28actor_legal_NOreanal_continues_const_2 \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'none' \
  --info 'small_share_2_player_new_model_1024_512_512_no_get_fixed_obj_store_5p_valcoeff' \
  --actors 8 \
  --simulations 50 \
  --batch_size 512 \
  --debug_interval 100 \
  --debug_batch \
  --td_steps 10 \
  --val_coeff 1 \
  --const 2 \
  --lr 0.01
  #--load_model \
  #--model_path 'bad.p' \


#revisivt#@wjc current disabled from 0.99
