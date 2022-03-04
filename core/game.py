from typing import List
from .utils import retype_observation, prepare_observation, prepare_observation_lst, str_to_arr
from core.mcts import MCTS
from core.ctree.cytree import Node, Roots
from core.model import concat_output, concat_output_value
from torch.cuda.amp import autocast as autocast
import numpy as np
import torch
import os
import ray
import copy


class Player(object):
    def __init__(self, id=1):
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id


class Game:
    def __init__(self, env, action_space_size: int, discount: float, config=None):
        self.env = env
        self.action_space_size = action_space_size
        self.discount = discount
        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class GameHistory:
    """
        Store only usefull information of a self-play game.
        """

    def __init__(self, action_space, max_length=200, config=None):
        self.action_space = action_space
        self.max_length = max_length
        self.config = config

        self.stacked_observations = config.stacked_observations
        self.discount = config.discount
        self.action_space_size = config.action_space_size

        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []

        #@wjc
        self.legal_actions = []
        self.ks=['visits', 'root', 'actions', 'obs', 'reward', 'tar_v', 'tar_r', 'tar_p']

    def init(self, init_observations, init_legal_action):
        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []
        self.target_values = []
        self.target_rewards = []
        self.target_policies = []

        #@wjc
        self.legal_actions = []

        assert len(init_observations) == self.stacked_observations
        for observation in init_observations:
            self.obs_history.append(copy.deepcopy(observation))

        #@wjc
        self.legal_actions.append(init_legal_action)

    def pad_over(self, next_block_observations, next_block_rewards, next_block_root_values, next_block_child_visits, next_legal_a):
        assert len(next_block_observations) <= self.config.num_unroll_steps
        assert len(next_block_child_visits) <= self.config.num_unroll_steps
        assert len(next_block_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_block_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # notice: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_block_observations:
            self.obs_history.append(copy.deepcopy(observation))
        #newly added
        for la in next_legal_a:
            self.legal_actions.append(copy.deepcopy(la))

        for reward in next_block_rewards:
            self.rewards.append(reward)

        for value in next_block_root_values:
            self.root_values.append(value)

        for child_visits in next_block_child_visits:
            self.child_visits.append(child_visits)

    def is_full(self):
        return self.__len__() >= self.max_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def load_file(self, gdict):
        # assert os.path.exists(path)
        self.target_values = gdict['tar_v']
        self.target_rewards = gdict['tar_r']
        self.target_policies = gdict['tar_p']
        self.child_visits = gdict['vis']
        self.root_values = gdict['root']

        self.actions = gdict['a']
        self.obs_history = ray.put(gdict['o'])
        self.rewards = gdict['r']
        self.legal_actions=gdict['la']

    def save_file(self):

        return {'vis':np.array(self.child_visits), 'root':np.array(self.root_values), 'a':np.array(self.actions),\
        'o':np.array(ray.get(self.obs_history)), 'r':np.array(self.rewards), 'tar_v':np.array(self.target_values),\
        'tar_r':np.array(self.target_rewards), 'tar_p':np.array(self.target_policies), 'la':np.array(self.legal_actions) }
        

    def append(self, action, obs, reward, legal_action):
        self.actions.append(action)
        self.obs_history.append(obs)
        self.rewards.append(reward)

        self.legal_actions.append(legal_action)
    def obs_object(self):
        return self.obs_history

    def obs(self, i, extra_len=0, padding=False):
        frames = ray.get(self.obs_history)[i:i + self.stacked_observations + extra_len]#@wjc
        if padding:
            pad_len = self.stacked_observations + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def zero_obs(self):
        if self.config.image_based:
            return [np.zeros((96, 96, self.config.image_channel), dtype=np.uint8) for _ in range(self.stacked_observations)]
        else:
            return [np.zeros(self.config.obs_shape // self.stacked_observations) for _ in range(self.stacked_observations)]

    def step_obs(self):
        index = len(self.rewards)
        frames = self.obs_history[index:index + self.stacked_observations]
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def get_targets(self, i):
        return self.target_values[i], self.target_rewards[i], self.target_policies[i]

    def game_over(self):
        self.rewards = np.array(self.rewards)
        self.obs_history = ray.put(np.array(self.obs_history))
        self.actions = np.array(self.actions)
        self.child_visits = np.array(self.child_visits)
        self.root_values = np.array(self.root_values)

        #@wjc
        self.legal_actions=np.array(self.legal_actions)

    def store_search_stats(self, visit_counts, root_value, idx: int = None, set_flag=False):
        if set_flag:
            self.child_visits.setflags(write=1)
            self.root_values.setflags(write=1)

        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visits.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_values.append(root_value)
        else:
            self.child_visits[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_values[idx] = root_value

        if set_flag:
            self.child_visits.setflags(write=0)
            self.root_values.setflags(write=0)

    def action_history(self, idx=None) -> list:
        if idx is None:
            return self.actions
        else:
            return self.actions[:idx]


    def __len__(self):
        return len(self.actions)
