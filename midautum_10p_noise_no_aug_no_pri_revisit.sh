#set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=3,4,5,6

#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 3 --num_cpus 60 --force --batch_actor 28 \
#python main.py --env KangarooNoFrameskip-v4 --case atari --opr train --seed 0 --num_gpus 4 --num_cpus 48 --force --batch_actor 42 \
python3 main.py --env hanabi --case hanabi --opr train --seed 99 --num_gpus 2 --num_cpus 72 --force --batch_actor 7 \
  --p_mcts_num 16\
  --extra oldreward_40move_0consist_10p_noise \
  --revisit_policy_search_rate 0.99 \
    --amp_type 'torch_amp' \
  --reanalyze_part 'paper' \
  --info 'hanabi_small_noise_aux_no_consist'
#  --load_model \
#  --model_path 'results/models/model_tst.p' \


#revisivt#@wjc current disabled from 0.99
