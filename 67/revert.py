#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
#58
python3 ../main.py --env Hanabi-Small --case hanabi --opr train --seed 1 --num_gpus 4 --num_cpus 70 --force --batch_actor 46\
  --p_mcts_num 8\
  --extra full_2p_28actor_legal_NOreanal_continues_from_bad \
  --use_priority \
  --use_max_priority \
  --revisit_policy_search_rate 0.99 \
  --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'small_full_confirm' \
  --actors 8 \
  --simulations 50 \
  --batch_size 256 \
  --val_coeff 0.25 \
  --td_step 5 \
  --debug_interval 100 \
  --debug_batch \