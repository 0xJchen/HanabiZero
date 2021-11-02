set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3,5,6

# ./results/atari/Pong-v4/revisit_rate_0/val_coeff_1/no_target/with_prio/no_max/init_zero/not_norm/actor_48_reply_16/mcts_50/seed_1/skip=4_stack=4_v5/model/model_80000.p
python main.py --env Pong-v4 --case atari --opr make_dataset --test_episodes 0 --seed 0 --num_gpus 4 --num_cpus 40 --force --multi_mcts --use_priority \
  --replay_number 16 \
  --load_model \
  --info 'make_dataset_3model_0.1data'