set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0
# /home/yeweirui/data/3model_0.1data
# /mnt/data1/yeweirui/muzero/make_dataset_3model_0.1data

# /home/yeweirui/code/Muzero/results/atari/Pong-v4/debug/prior/reward_block=True/cosineLr_0.2_10wsteps/spr_consist_0.5/seed_999/super_debug_spr_V4/model/model_20000.p
# /root/Muzero/results/atari/
python main.py --env Pong-v4 --case atari --opr show --seed 666 --num_gpus 1 --num_cpus 40 --force --multi_mcts --replay_number 16 \
  --use_priority \
  --data_path '/mnt/data1/yeweirui/muzero/make_dataset_3model_0.1data' \
  --model_path '' \
  --info 'show'