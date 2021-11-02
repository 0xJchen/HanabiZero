set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2

python main.py --env Pong-v4 --case atari --opr debug --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 1 \
  --use_priority \
  --priority_top 1 \
  --revisit_policy_search_rate 0.8 \
  --info 'online_reanalyze_V0' # > time.txt