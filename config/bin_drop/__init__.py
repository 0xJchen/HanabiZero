import gym
import torch
import robosuite as suite
from core.config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import AtariWrapper
from .model import MuZeroNet


class BinDropConfig(BaseMuZeroConfig):
    def __init__(self):
        super(BinDropConfig, self).__init__(
            training_steps=50000,
            test_interval=200,
            test_episodes=10,
            checkpoint_interval=50,
            max_moves=6,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=5,
            batch_size=128,
            td_steps=5,
            num_actors=3,
            lr_init=0.02,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            window_size=3000,
            value_loss_coeff=1,
            value_support=DiscreteSupport(-20, 20),
            reward_support=DiscreteSupport(-5, 5))

        self.stacked_observations = 3   # Number of previous observations and previous actions to add to the current observation
        self.blocks = 4  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 64  # Number of channels in reward head
        self.reduced_channels_value = 64  # Number of channels in value head
        self.reduced_channels_policy = 64  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.dynamic_depth = False

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game()
        self.obs_shape = game.reset().shape
        self.action_space_size = game.action_space_size

    # def change_max_moves(self, step=10, max=2000):
    #     if self.max_moves + step <= max:
    #         self.max_moves += step
    #     else:
    #         self.max_moves = max
    #
    # def change_simulations(self, step=1, max=200):
    #     if self.num_simulations + step <= max:
    #         self.num_simulations += step
    #     else:
    #         self.num_simulations = max

    def get_uniform_network(self):
        return MuZeroNet(
            self.obs_shape,
            self.stacked_observations,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None):
        # env = gym.make(self.env_name)
        import ipdb
        ipdb.set_trace()
        env = suite.make(
            'BinPackPlace',
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            control_freq=1,
            render_drop_freq=20,
            camera_height=64,
            camera_width=64,
            video_height=64,
            video_width=64,

            random_take=True
        )
        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return AtariWrapper(env, discount=self.discount, k=self.stacked_observations)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


muzero_config = BinDropConfig()
