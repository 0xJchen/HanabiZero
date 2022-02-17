import os
import torch
import numpy as np
import datetime
from .game import Game


class DiscreteSupport(object):
    def __init__(self, min: int, max: int, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1, delta)
        self.size = len(self.range)
        self.delta = delta


class BaseMuZeroConfig(object):

    def __init__(self,
                 training_steps: int,
                 last_steps: int,
                 test_interval: int,
                 test_episodes: int,
                 checkpoint_interval: int,
                 target_model_interval: int,
                 save_ckpt_interval: int,
                 log_interval: int,
                 vis_interval: int,
                 max_moves: int,
                 test_max_moves: int,
                 history_length: int,
                 discount: float,
                 dirichlet_alpha: float,
                 value_delta_max: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_warm_up: float,
                 lr_type: str,
                 lr_init: float,
                 lr_decay_rate: float,
                 lr_decay_steps: float,
                 start_window_size: int,
                 window_size: int = int(1e6),
                 transition_num: float = 25,
                 consist_type='spr',
                 gray_scale: bool = False,
                 episode_life: bool = False,
                 change_temperature: bool = False,
                 init_zero: bool = False,
                 state_norm: bool = False,
                 clip_reward: bool = False,
                 random_start: bool = False,
                 use_epsilon_greedy: bool = False,
                 cvt_string: bool = False,
                 image_based: bool = False,
                 frame_skip: int = 1,
                 stacked_observations: int = 16,
                 uniform_ratio: float = 0,
                 priority_reward_ratio: float = 0,
                 tail_ratio: float = 0,
                 reward_loss_coeff: float = 1,
                 value_loss_coeff: float = 1,
                 policy_loss_coeff: float = 1,
                 consistency_coeff: float = 0,#@wjc
                 reg_loss_coeff: float = 1e-4,
                 debug_batch: bool = False,
                 debug_interval: int = 100,
                 value_support: DiscreteSupport = None,
                 reward_support: DiscreteSupport = None,
                 rmsprop: int = 0
                 ):
        #@wjc
        self.rmsprop=rmsprop
        # Self-Play
        self.debug_batch = debug_batch
        self.debug_interval = debug_interval
        self.action_space_size = None
        self.num_actors = num_actors
        self.gray_scale = gray_scale
        self.episode_life = episode_life
        self.change_temperature = change_temperature
        self.init_zero = init_zero
        self.state_norm = state_norm
        self.clip_reward = clip_reward
        self.random_start = random_start
        self.use_epsilon_greedy = use_epsilon_greedy
        self.cvt_string = cvt_string
        self.image_based = image_based

        self.max_moves = max_moves
        self.test_max_moves = test_max_moves
        self.history_length = history_length
        self.num_simulations = num_simulations
        self.discount = discount
        self.max_grad_norm = 5

        # testing arguments
        self.test_interval = test_interval
        self.test_episodes = test_episodes

        # Root prior exploration noise.
        self.value_delta_max = value_delta_max
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the environment, we can use them to
        # initialize the rescaling. This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.max_value_bound = None
        self.min_value_bound = None

        # Training
        self.training_steps = training_steps
        self.last_steps = last_steps
        self.checkpoint_interval = checkpoint_interval
        self.target_model_interval = target_model_interval
        self.save_ckpt_interval = save_ckpt_interval
        self.log_interval = log_interval
        self.vis_interval = vis_interval
        self.start_window_size = start_window_size
        self.window_size = window_size
        self.transition_num = transition_num
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.frame_skip = frame_skip
        self.stacked_observations = stacked_observations
        self.uniform_ratio = uniform_ratio
        self.priority_reward_ratio = priority_reward_ratio
        self.tail_ratio = tail_ratio
        self.reward_loss_coeff = reward_loss_coeff
        self.value_loss_coeff = value_loss_coeff
        self.policy_loss_coeff = policy_loss_coeff
        self.consistency_coeff = consistency_coeff
        self.reg_loss_coeff = reg_loss_coeff
        self.device = 'cuda'
        self.exp_path = None  # experiment path
        self.debug = False
        self.model_path = None
        self.seed = None
        self.transforms = None
        self.value_support = value_support
        self.reward_support = reward_support

        # optimization control
        self.weight_decay = 1e-4
        self.momentum = 0.9
        assert lr_warm_up <= 0.1 and lr_warm_up >= 0.
        self.lr_warm_up = lr_warm_up
        self.lr_type = lr_type
        self.lr_warm_step = int(self.training_steps * self.lr_warm_up)
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.target_infer_size =64
        self.consist_type = consist_type

        # replay buffer
        self.priority_prob_alpha = 0.6
        self.priority_prob_beta = 0.4
        self.prioritized_replay_eps = 1e-6

        # self play
        self.self_play_moves_ratio = 1

        # env
        self.image_channel = 3

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        raise NotImplementedError

    def set_game(self, env_name):
        raise NotImplementedError

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False) -> Game:
        """ returns a new instance of the game"""
        raise NotImplementedError

    def get_uniform_network(self):
        raise NotImplementedError

    def scalar_loss(self, prediction, target):
        raise NotImplementedError

    def scalar_transform(self, x):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        delta = self.value_support.delta
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x / delta) + 1) - 1 + epsilon * x / delta)
        return output

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support,mode='reward')

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support,mode='value')

    def inverse_scalar_transform(self, logits, scalar_support,mode=None):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        delta = self.value_support.delta
        value_probs = torch.softmax(logits, dim=1)
        value_support = torch.ones(value_probs.shape)
        value_support[:, :] = torch.from_numpy(np.array([x for x in scalar_support.range]))
        value_support = value_support.to(device=value_probs.device)
        #print("in inverse transform: ",value_support.shape,value_probs.shape,flush=True)
        value = (value_support * value_probs).sum(1, keepdim=True) / delta

        epsilon = 0.001
        sign = torch.ones(value.shape).float().to(value.device)
        sign[value < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output * delta

        nan_part = torch.isnan(output)
        # if nan_part.any():
        #     print('===========> in {} transform [ERROR]: NAN in scalar!!!'.format(mode),flush=True) when calling inverse_scalar_transform
        output[nan_part] = 0.
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size: int):
        delta = self.value_support.delta

        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = x - x_low
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min / delta, x_low - min / delta
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    def get_hparams(self):
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def set_config(self, args):
        self.set_game(args.env)
        self.seed = args.seed
        if not args.use_priority:
            self.priority_prob_alpha = 0
        self.amp_type = args.amp_type
        self.use_priority = args.use_priority
        self.use_max_priority = args.use_max_priority if self.use_priority else False
        self.use_latest_model = args.use_latest_model
        self.debug = args.debug
        self.device = args.device
        self.batch_actor = args.batch_actor
        self.replay_number = args.replay_number
        self.opt_level = args.opt_level
        # self.priority_top = args.priority_top
        self.write_back = args.write_back
        self.p_mcts_num = args.p_mcts_num
        self.cpu_actor=args.cpu_actor
        self.gpu_actor=args.gpu_actor
        # self.use_root_value = args.use_root_value
        self.reanalyze_part = args.reanalyze_part
        self.target_moving_average = args.target_moving_average
        self.extra=args.extra
        # augmentation
        if self.consistency_coeff > 0 and self.image_based:
            self.use_augmentation = True
            self.augmentation = args.augmentation
        else:
            self.use_augmentation = False

        if args.revisit_policy_search_rate is not None:
            self.revisit_policy_search_rate = args.revisit_policy_search_rate

        prior_tag = 'no_prior'
        if args.use_priority:
            # prior_tag = 'prior_max={}'.format(self.priority_top)
            prior_tag = 'not_max'
            if args.use_max_priority:
                prior_tag = 'max_prior'
            prior_tag += '-alpha={}-beta={}-reward={}-tail={}-uniform={}'.format(self.priority_prob_alpha, self.priority_prob_beta, self.priority_reward_ratio, self.tail_ratio, self.uniform_ratio)

        aug_tag = 'no_aug'
        if self.use_augmentation:
            assert len(self.augmentation) > 0
            aug_tag = '+'.join(self.augmentation)

        consist_tag = '{}_consist_{}'.format(self.consist_type, self.consistency_coeff)

        game_tag = 'max_moves={}-start={}k-skip={}-stack={}'.format(self.max_moves, self.start_window_size // 1000, self.frame_skip, self.stacked_observations)
        if self.clip_reward:
            game_tag = 'clip_reward-' + game_tag
        if self.gray_scale:
            game_tag = 'gray-' + game_tag
        if self.episode_life:
            game_tag = 'one_life-' + game_tag

        replay_tag = 'buffer_size={}m-slices={}'.format(self.transition_num, self.history_length)
        self_play_tag = 'actor={}-mcts_num={}-ckpt_update={}'.format(self.num_actors, self.p_mcts_num, self.checkpoint_interval)
        if self.use_epsilon_greedy:
            self_play_tag = 'greedy-' + self_play_tag
        if self.random_start:
            self_play_tag = 'random-' + self_play_tag
        train_tag = '{}Lr={}-steps=({}+{})k-batch={}-target_update={}-discount={:<0.4f}-dirichlet={}(delta={})'.format(self.lr_type, self.lr_init, self.training_steps // 1000, self.last_steps // 1000, self.batch_size, self.target_model_interval, self.discount, self.root_dirichlet_alpha, self.value_delta_max)
        if self.state_norm:
            train_tag = 'state_norm-' + train_tag
        if self.write_back:
            train_tag = 'write_back-' + train_tag
        if self.change_temperature:
            train_tag = 'temperature-' + train_tag

        tradeoff_tag = 'reward_coeff={}-val_coeff={}-policy_coeff={}'.format(self.reward_loss_coeff, self.value_loss_coeff, self.policy_loss_coeff)
        reanalyze_tag = 'reanalyze_part={}-revisit={}'.format(self.reanalyze_part, self.revisit_policy_search_rate)
        seed_tag = '{}-seed={}-node={}-extra={}'.format(datetime.date.today(), self.seed, self.num_simulations,self.extra)

        self.exp_path = os.path.join(args.result_dir, args.case, args.env, prior_tag, aug_tag, consist_tag, game_tag,
                                     replay_tag, self_play_tag, train_tag, tradeoff_tag, reanalyze_tag, args.info,
                                     seed_tag)

        self.model_path = os.path.join(self.exp_path, 'model.p')
        self.model_dir = os.path.join(self.exp_path, 'model')
        return self.exp_path
