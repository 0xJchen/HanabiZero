set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0
# /home/yeweirui/data/3model_0.1data
# /mnt/data1/yeweirui/muzero/make_dataset_3model_0.1data
python main.py --env Breakout-v4 --case atari --opr super --seed 999 --num_gpus 1 --num_cpus 40 --force --multi_mcts --replay_number 16 \
  --use_priority \
  --data_path '/mnt/data1/yeweirui/breakout/1model_250k_data' \
  --info '1model_250k'