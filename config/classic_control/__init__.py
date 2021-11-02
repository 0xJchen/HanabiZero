import gym
import torch
from baselines.common.atari_wrappers import WarpFrame, EpisodicLifeEnv
from core.config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import ClassicControlWrapper
from .model import MuZeroNet
from core.utils import make_atari


class ClassicControlConfig(BaseMuZeroConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            training_steps=100000,
            last_steps=0,
            test_interval=10000,
            log_interval=1000,
            vis_interval=1000,
            test_episodes=32,
            checkpoint_interval=1000,
            target_model_interval=1000,
            save_ckpt_interval=10000,
            max_moves=12000,
            test_max_moves=12000,
            history_length=12001,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.006,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=2,
            # network initialization/ & normalization
            gray_scale=False,
            change_temperature=False,
            episode_life=True,
            init_zero=True,
            state_norm=False,
            clip_reward=True,
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_type='step',
            lr_init=0.1,
            lr_decay_rate=0.1,
            lr_decay_steps=500000,
            # replay window
            start_window_size=1,#mannualy changed to 1 for debugging
            window_size=125000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=4,
            stacked_observations=4,
            # spr
            consist_type='spr',
            consistency_coeff=2,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            # value reward support
            value_support=DiscreteSupport(-300, 300, delta=1),
            reward_support=DiscreteSupport(-300, 300, delta=1))

        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_window_size = self.start_window_size * 1000 // self.frame_skip
        self.start_window_size = max(1, self.start_window_size)
        self.image_channel = 1

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * self.training_steps:
                return 1.0
            elif trained_steps < 0.75 * self.training_steps:
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game()
        
        self.obs_shape = game.reset().shape[0] * self.stacked_observations
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        print("obs shape,",self.obs_shape,self.action_space_size, self.new_game().reset().shape)
        return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform, state_norm=self.state_norm)
        # return MuZeroNet(self.obs_shape, self.action_space_size, 2 * self.reward_support + 1, 2 * self.value_support + 1,
        #                  self.inverse_value_transform, self.inverse_reward_transform)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip
            else:
                max_moves = self.test_max_moves
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
        else:
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.max_moves)

        if self.episode_life and not test:
            env = EpisodicLifeEnv(env)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return ClassicControlWrapper(env, discount=self.discount)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        pass


muzero_config = ClassicControlConfig()
