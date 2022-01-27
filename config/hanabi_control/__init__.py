import gym
import torch
from baselines.common.atari_wrappers import WarpFrame, EpisodicLifeEnv
from core.config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import HanabiControlWrapper
from .model import MuZeroNet,MuZeroNetFull
from core.utils import make_atari
from envs import HanabiEnv
import numpy as np
class HanabiControlConfig(BaseMuZeroConfig):
    def __init__(self,args):
        super(HanabiControlConfig, self).__init__(
            training_steps=200000,
            last_steps=0,
            test_interval=1000,#changed to 1000
            log_interval=1000,
            vis_interval=1000,
            test_episodes=40,
            checkpoint_interval=1000,
            target_model_interval=100,#changed from 1k @wjc
            save_ckpt_interval=10000,
            max_moves=60,#@wjc
            test_max_moves=60,#@wjc
            history_length=12001,#changed to 80 from 12001
            discount=0.999,#from 0.997
            dirichlet_alpha=0.3,#maybe for small, it should be smaller?
            value_delta_max=0.006,
            num_simulations=args.simulations,
            batch_size=args.batch_size,
            td_steps=args.td_steps,
            num_actors=args.actors,
            # network initialization/ & normalization
            gray_scale=False,
            change_temperature=False,
            episode_life=True,
            init_zero=True,
            state_norm=False,
            clip_reward=False,#no need
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.001,
            lr_type='step',
            lr_init=args.lr,
            lr_decay_rate=args.decay_rate,
            lr_decay_steps=20000,
            # replay window
            start_window_size=40,#@wjc mannualy changed from 40 to 1 for debugging
            window_size=125000,#useless
            transition_num=1,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=args.stack,#changed to 0
            # spr
            consist_type='spr',
            consistency_coeff=args.const,#@wjc
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=args.val_coeff,
            policy_loss_coeff=1,
            # value reward support
            value_support=DiscreteSupport(-25, 25, delta=1),
            reward_support=DiscreteSupport(-25, 25, delta=1))#@wjc what?

        self.discount **= self.frame_skip
        # self.max_moves //= self.frame_skip#@wjc max move should just be 80
        # self.test_max_moves //= self.frame_skip
        self.const=args.const
        self.start_window_size = self.start_window_size * 100 // self.frame_skip#whta fuck?
        self.start_window_size = max(1, self.start_window_size)
        self.image_channel = 1
        self.game_name=None
    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        assert self.change_temperature==False
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
        print("==>setting env = ",env_name)
        self.env_name = env_name
        game = self.new_game()

        reset_obs,_=game.reset()
        reset_obs=np.asarray(reset_obs)
        # print("in set_game",reset_obs.shape)
        self.obs_shape = reset_obs.shape[0] * self.stacked_observations
        self.action_space_size = game.action_space_size
        print("action space =",self.action_space_size,flush=True)
    def get_uniform_network(self):
        # print("obs shape,",self.obs_shape,self.action_space_size, len(self.new_game().reset()[0]))

        if self.env_name=='Hanabi-Small':
            if self.const>0:
                return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform, state_norm=self.state_norm, proj=True)
            else:
                return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform, state_norm=self.state_norm)
        #elif self.env_name='Hanabi-Full':
         #   return MuZeroNetFull(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
          #               self.inverse_value_transform, self.inverse_reward_transform, state_norm=self.state_norm)
        else:
            print("wrong env name in init network")
            assert False
            # return MuZeroNet(self.obs_shape, self.action_space_size, 2 * self.reward_support + 1, 2 * self.value_support + 1,
        #                  self.inverse_value_transform, self.inverse_reward_transform)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        # if test:
        #     if final_test:
        #         max_moves = 108000 // self.frame_skip
        #     else:
        #         max_moves = self.test_max_moves
        #     env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
        # else:
        #     env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.max_moves)

        # if self.episode_life and not test:
        #     env = EpisodicLifeEnv(env)
        #print("====>seed=",seed)
        if seed is not None:
            arg={"hanabi_name":self.env_name,"seed":seed}
        else:
            arg={"hanabi_name":self.env_name,"seed":None}
        env=HanabiEnv(arg)
        #print("===================>",env.num_moves(),flush=True)
        # if seed is not None:
        #     env.seed(seed)

        # if save_video:
        #     from gym.wrappers import Monitor
        #     env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return HanabiControlWrapper(env, discount=self.discount)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        pass

class HanabiControlConfigFull(BaseMuZeroConfig):
    def __init__(self,args):
        super(HanabiControlConfigFull, self).__init__(
            training_steps=2000000,
            last_steps=100,
            test_interval=4000,#changed to 1000
            log_interval=1000,
            vis_interval=1000,
            test_episodes=80,
            checkpoint_interval=2000,
            target_model_interval=200,
            save_ckpt_interval=10000,
            max_moves=160,#@wjc
            test_max_moves=160,#@wjc
            history_length=12001,#changed to 80 from 12001
            discount=0.999,#changed from 0.997
            dirichlet_alpha=0.3,
            value_delta_max=0.006,
            num_simulations=args.simulations,
            batch_size=args.batch_size,
            td_steps=args.td_steps,
            num_actors=args.actors,
            # network initialization/ & normalization
            gray_scale=False,
            change_temperature=False,
            episode_life=True,
            init_zero=True,
            state_norm=False,
            clip_reward=False,#no need
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.0001,#changed from 0.002 @jc, this is fraction of warm up steps
            lr_type='step',
            lr_init=args.lr,
            lr_decay_rate=args.decay_rate,
            lr_decay_steps=args.decay_step,#changed from 0.5m
            # replay window
            start_window_size=40,#@wjc mannualy changed to 10(final model) for debugging
            window_size=125000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=args.stack,
            # spr
            consist_type='spr',
            consistency_coeff=0,#@wjc
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=args.val_coeff,
            policy_loss_coeff=1,
            debug_batch=args.debug_batch,
            debug_interval=args.debug_interval,
        # value reward support
            value_support=DiscreteSupport(-100, 100, delta=1),
            reward_support=DiscreteSupport(-100, 100, delta=1))#@wjc what?

        self.discount **= self.frame_skip
        # self.max_moves //= self.frame_skip#@wjc max move should just be 80
        # self.test_max_moves //= self.frame_skip

        self.start_window_size = self.start_window_size * 100 // self.frame_skip#whta fuck?
        self.start_window_size = max(1, self.start_window_size)
        self.image_channel = 1
        self.game_name=None
    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.25 * self.training_steps:
                return 1.0
            elif trained_steps < 0.75 * self.training_steps:
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        print("==>setting env = ",env_name)
        self.env_name = env_name
        game = self.new_game()

        reset_obs,_=game.reset()
        reset_obs=np.asarray(reset_obs)
        # print("in set_game",reset_obs.shape)
        self.obs_shape = reset_obs.shape[0] * self.stacked_observations
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        # print("obs shape,",self.obs_shape,self.action_space_size, len(self.new_game().reset()[0]))

        if self.env_name=='Hanabi-Small':
            assert False
            return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform, state_norm=self.state_norm)
        elif self.env_name=='Hanabi-Full':
            return MuZeroNetFull(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform, state_norm=self.state_norm)
        else:
            print("wrong env name in init network")
            assert False
            # return MuZeroNet(self.obs_shape, self.action_space_size, 2 * self.reward_support + 1, 2 * self.value_support + 1,
        #                  self.inverse_value_transform, self.inverse_reward_transform)

        print("action space =",self.action_space_size,flush=True)
    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        # if test:
        #     if final_test:
        #         max_moves = 108000 // self.frame_skip
        #     else:
        #         max_moves = self.test_max_moves
        #     env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
        # else:
        #     env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.max_moves)

        # if self.episode_life and not test:
        #     env = EpisodicLifeEnv(env)
        #print("======>seed=",seed)
        # if seed is not None:
        arg={"hanabi_name":self.env_name,"seed":seed}
        # else:
        #     arg={"hanabi_name":self.env_name,"seed":None}
        #print(arg.keys(),flush=True)
        #print("=======================>starting new simulation")
        env=HanabiEnv(arg)

        # if seed is not None:
        #     env.seed(seed)

        # if save_video:
        #     from gym.wrappers import Monitor
        #     env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return HanabiControlWrapper(env, discount=self.discount)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        pass


#muzero_config = HanabiControlConfig()
