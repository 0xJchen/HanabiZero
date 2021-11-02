set -ex
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python main.py --env CartPole-v1 --case classic_control --opr train --seed 0 --num_gpus 8 --num_cpus 50 --force --multi_mcts --use_target_model --revisit_policy_search_rate 0.8 --value_loss_coeff 0.25
kernprof -l python main.py --env Pong-v4 --case atari --opr train --seed 1 --num_gpus 8 --force --use_target_model --multi_mcts --revisit_policy_search_rate 0.8 --value_loss_coeff 0.25
#python main.py --env BinDrop-v0 --case bin_drop --opr train --seed 0 --num_gpus 5 --force
# python -m line_profiler main.py.lprof