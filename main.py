import argparse
import logging.config
import os

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train, super, make_dataset,test_mcts
from core.train_blocks import supervised_train
from core.utils import init_logger, make_results_dir, set_seed
if __name__ == '__main__':
    # Lets gather arguments

    parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--extra',default='none',type=str,help='extra information in log directory')
    parser.add_argument('--case', required=True, choices=['atari', 'mujoco', 'classic_control', 'box2d', 'bin_drop','hanabi'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test', 'make_dataset', 'super', 'debug','mcts'])
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--batch_actor', type=int, default=1, help='batch actor')
    parser.add_argument('--replay_number', type=int, default=1, help='Add batch buffer')
    parser.add_argument('--p_mcts_num', type=int, default=64, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int, default=8, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=80, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.8,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--amp_type', required=True, choices=['torch_amp', 'nvidia_apex', 'none'],
                        help='choose automated mixed precision type')
    parser.add_argument('--use_priority', action='store_true', default=False,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=False,
                        help='max priority')
    parser.add_argument('--use_latest_model', action='store_true', default=False,
                        help='Use target model for bootstrap value estimation (default: %(default)s)')
    parser.add_argument('--write_back', action='store_true', default=False,
                        help='write back')
    parser.add_argument('--target_moving_average', action='store_true', default=False,
                        help='Use moving average target model or not')
    # parser.add_argument('--priority_top', type=float, default=np.inf, help='max priority value')
    parser.add_argument('--test_episodes', type=int, default=32,
                        help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--reanalyze_part', type=str, default='none', help='reanalyze part',
                        choices=['none', 'value', 'policy', 'paper', 'all'])
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none', help='debug string')
    parser.add_argument('--opt_level', type=str, default='O1', help='opt level in amp')
    parser.add_argument('--data_path', type=str, default=None, help='dataset path')
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
    parser.add_argument('--make_dataset_model_path', type=list, default=[
                        '/root/models/v5/model_60000.p',
                        '/root/models/v5/model_80000.p',
                        '/root/models/v5/model_100000.p'],
                        help='load model path for making dataset')
    parser.add_argument('--test_begin', type=int, default=1, help='begin model idx')
    parser.add_argument('--test_end', type=int, default=5, help='end model idx')
    parser.add_argument('--use_critic', action='store_true', default=False,
                        help='use global critic')

    parser.add_argument('--val_coeff', type=float, default='0.25', help='val coeff')
    parser.add_argument('--lr', type=float, default='0.1', help='learning rate')
    parser.add_argument('--td_steps', type=int, default=5, help='td step')
    parser.add_argument('--const', type=float, default='0.', help='consistent loss')
    parser.add_argument('--actors', type=int, default=2, help='actors')
    parser.add_argument('--simulations', type=int, default=200, help='simulation')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--debug_batch', action='store_true', default=False, help='show empty batch steps')
    # Process arguments
    parser.add_argument('--debug_interval', type=int, default=500, help='show batch time interval')
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'

    if args.opr == 'train':
        ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus,
              object_store_memory=300*1024*1024*1024,dashboard_port=8267, dashboard_host='0.0.0.0')
                #   object_store_memory=200*1024*1024*1024, dashboard_port=9999,dashboard_host='0.0.0.0'  )
		#object_store_memory=150 * 1024 * 1024 * 1024)
    else:
        ray.init()

    # seeding random iterators
    set_seed(args.seed)

    # import corresponding configuration , neural networks and envs
    if args.case == 'atari':
        from config.atari import muzero_config, AtariConfig
    elif args.case == 'mujoco':
        from config.mujoco import muzero_config
    elif args.case == 'bin_drop':
        from config.bin_drop import muzero_config
    elif args.case == 'box2d':
        from config.classic_control import muzero_config  # just using same config as classic_control for now
    elif args.case == 'classic_control':
        from config.classic_control import muzero_config
    elif args.case == 'hanabi':
        from config.hanabi_control import HanabiControlConfig,HanabiControlConfigFull
    else:
        raise Exception('Invalid --case option')
    if args.env=='Hanabi-Small':
        muzero_config=HanabiControlConfig(args)
    elif args.env=='Hanabi-Full':
        muzero_config=HanabiControlConfigFull(args)
    else:
        print("wrong env name: ",args.env)
        assert False
    # set config as per arguments
    exp_path = muzero_config.set_config(args)
    exp_path, log_base_path = make_results_dir(exp_path, args)

    # set-up logger
    init_logger(log_base_path)

    try:
        if args.opr == 'train':
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            print("path: ",args.load_model)
            print("exist",args.model_path,os.path.exists(args.model_path))
            if args.load_model:
                assert(os.path.exists(args.model_path))
                model_path = args.model_path
                print("============>successfully loaded model, path=",model_path, flush=True)
            else:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
                model_path = None
            #assert False

            model, weights = train(muzero_config, summary_writer, model_path)
            model.set_weights(weights)
            total_steps = muzero_config.training_steps + muzero_config.last_steps
            test_score, test_path = test(muzero_config, model.to('cuda'), total_steps, muzero_config.test_episodes,
                                         'cuda', render=False, save_video=False, final_test=True)
            mean_score = sum(test_score) / len(test_score)
            logging.getLogger('test').info('Test Score: {}'.format(test_score))
            logging.getLogger('test').info('Test Mean Score: {}'.format(mean_score))
            logging.getLogger('test').info('Saving video in path: {}'.format(test_path))
        elif args.opr == 'mcts':
            print("start datawork testing!")
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            if args.load_model and os.path.exists(args.model_path):
                model_path = args.model_path
            else:
                model_path = None
            model, weights = test_mcts(muzero_config, summary_writer, model_path)
        elif args.opr == 'test':
            assert args.load_model
            if args.model_path is None:
                model_path = muzero_config.model_path
            else:
                print("loading my model!")
                model_path = args.model_path
            parent_model_path=model_path
            for idx in range(args.test_begin,args.test_end+1):
                model_path=parent_model_path+"/model_"+str(int(idx*10000))+".p"
                assert os.path.exists(model_path), 'model not found at {}'.format(model_path)

                model = muzero_config.get_uniform_network().to('cuda')
                # for name, param in model.named_parameters():
                #     print(name,param.shape)
                new_model=torch.load(model_path)
                # print("====>",type(new_model))
                # for k,v in new_model.items():
                #     print(k,v.shape)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
                test_score, test_path = test(muzero_config, model, 0, args.test_episodes, device='cuda', render=False,save_video=False, final_test=True)
                mean_score = sum(test_score) / len(test_score)
                print("model: {}, mean: {}".format(idx*10000, mean_score),flush=True)
                # logging.getLogger('test').info('Test Score: {}'.format(test_score))
                # logging.getLogger('test').info('Test Mean Score: {}'.format(mean_score))
                # logging.getLogger('test').info('Saving video in path: {}'.format(test_path))
        elif args.opr == 'make_dataset':
            assert args.load_model
            if args.make_dataset_model_path is None:
                model_path = muzero_config.model_path
                assert os.path.exists(model_path), 'model not found at {}'.format(model_path)
            else:
                for pt in args.make_dataset_model_path:
                    assert os.path.exists(pt), 'model not found at {}'.format(pt)
                muzero_config.model_path = args.make_dataset_model_path
                muzero_config.model_num = len(args.make_dataset_model_path)

                model_path = muzero_config.model_path[0]

            model = muzero_config.get_uniform_network().to('cuda')

            print('load model from {}'.format(model_path))
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

            if args.test_episodes > 0:
                print('Testing {} epochs...'.format(args.test_episodes))
                test_score = test(muzero_config, model, args.test_episodes, device='cuda', render=args.render,
                                  save_video=True)
                logging.getLogger('test').info('Test Score: {}'.format(test_score))
            make_dataset(muzero_config, model)
        elif args.opr == 'super':
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            assert os.path.exists(args.data_path), 'data path not found at {}'.format(args.data_path)
            super(muzero_config, args.data_path, summary_writer)
        elif args.opr == 'debug':
            from core.debug import debug
            debug(muzero_config)
        else:
            raise Exception('Please select a valid operation(--opr) to be performed')
        ray.shutdown()
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
