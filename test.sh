set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1

#python main.py --env Pong-v4 --case atari --opr train --seed 0 --num_gpus 8 --num_cpus 80 --force --batch_actor 36 \
python3 main.py --env hanabi --case hanabi --opr test --seed 0 --num_gpus 2 --num_cpus 60 --force --render \
  --test_episodes 24 \
  --load_model \
  --amp_type 'none' \
  --model_path './best_model/model/model_200000.p' \
  --info 'test1'
