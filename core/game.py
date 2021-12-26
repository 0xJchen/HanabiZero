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
        #print("len of init",len(init_observations),flush=True)
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

    def load_file(self, path):
        assert os.path.exists(path)

        self.child_visits = np.load(os.path.join(path, 'visits.npy'))
        self.root_values = np.load(os.path.join(path, 'root.npy'))

        self.actions = np.load(os.path.join(path, 'actions.npy')).tolist()
        self.obs_history = np.load(os.path.join(path, 'obs.npy'))
        self.rewards = np.load(os.path.join(path, 'reward.npy'))

        # last_observations = [self.obs_history[-1] for i in range(self.config.num_unroll_steps)]
        # self.obs_history = np.concatenate((self.obs_history, last_observations))

        self.target_values = np.load(os.path.join(path, 'target_values.npy'))
        self.target_rewards = np.load(os.path.join(path, 'target_rewards.npy'))
        self.target_policies = np.load(os.path.join(path, 'target_policies.npy'))

    def save_file(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        np.save(os.path.join(path, 'visits.npy'), np.array(self.child_visits))
        np.save(os.path.join(path, 'root.npy'), np.array(self.root_values))

        np.save(os.path.join(path, 'actions.npy'), np.array(self.actions))
        np.save(os.path.join(path, 'obs.npy'), np.array(self.obs_history))
        np.save(os.path.join(path, 'reward.npy'), np.array(self.rewards))

        np.save(os.path.join(path, 'target_values.npy'), np.array(self.target_values))
        np.save(os.path.join(path, 'target_rewards.npy'), np.array(self.target_rewards))
        np.save(os.path.join(path, 'target_policies.npy'), np.array(self.target_policies))

    def append(self, action, obs, reward, legal_action):
        self.actions.append(action)
        self.obs_history.append(obs)
        self.rewards.append(reward)

        #@wjc
        #print("***before",self.legal_actions,flush=True)
        self.legal_actions.append(legal_action)
        #print("***after",self.legal_actions,flush=True)
    def obs_object(self):
        return self.obs_history

    def obs(self, i, extra_len=0, padding=False):
        frames = ray.get(self.obs_history)[i:i + self.stacked_observations + extra_len]#@wjc
        #frames = self.obs_history[i:i + self.stacked_observations + extra_len]
        if padding:
            pad_len = self.stacked_observations + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames
    # def make_batch_legal_action(self,i):
    #     legal_actions=ray.get(self.legal_actions[])

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
        #self.obs_history = np.array(self.obs_history)#@wjc
        self.actions = np.array(self.actions)
        self.child_visits = np.array(self.child_visits)
        self.root_values = np.array(self.root_values)

        #@wjc
        self.legal_actions=np.array(self.legal_actions,dtype=object)
        #try:
        #print("reward={},obs={},action={},legal={},child_visit={},root_val={}".format(self.rewards.shape,ray.get(self.obs_history).shape,self.actions.shape,self.legal_actions.shape,np.array(self.child_visits).shape,np.array(self.root_values).shape),flush=True)
        #except:
        #    print("gg")
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

    # def to_play(self) -> Player:
    #     return Player()

    def __len__(self):
        return len(self.actions)
def prepare_multi_target_test(games, state_index_lst, config, model):
    batch_values, batch_rewards, batch_policies = [], [], []

    zero_obs = games[0].zero_obs()
    device = next(model.parameters()).device
    obs_lst = []
    value_mask = []

    root_val_lst=[]

    with torch.no_grad():
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            #print("obs length:",len(ray.get(game.obs_history)),flush=True)
            #print("root length:",len(game.root_values),game.root_values,flush=True)
            game_obs = game.obs(state_index + config.td_steps, config.num_unroll_steps)
            #game_value=game.root_values[state_index + config.td_steps, config.num_unroll_steps]
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # obs = game.obs(bootstrap_index)
                    beg_index = bootstrap_index - (state_index + config.td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                    # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
             #       print("root value: ")
                    try:
                        root_val_lst.append(game.root_values[bootstrap_index])
                    except:
                        print("bootstrap idx:",bootstrap_index,flush=True)
                else:
                    value_mask.append(0)
                    obs = zero_obs
                    root_val_lst.append(0)

                obs_lst.append(obs)

        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst, image_based=config.image_based)
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = obs_lst[beg_index:end_index]

            if config.image_based:
                m_obs = torch.from_numpy(m_obs).to(device).float() / 255.0#no divice 255
            else:
                m_obs = torch.from_numpy(m_obs).to(device).float().reshape(m_obs.shape[0], -1)
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)
            network_output.append(m_output)

        value_lst = concat_output_value(network_output)
        #print("original value_lst:",value_lst.shape,flush=True)
        value_lst=np.array(root_val_lst)
        #print("value_lst shape:", value_lst.shape,flush=True)

        #value_lst = value_lst.reshape(-1) * config.discount ** config.td_steps
        value_lst = value_lst * config.discount ** config.td_steps

        # get last value
        value_lst = value_lst * np.array(value_mask)
        value_lst = value_lst.tolist()

        value_index = 0
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            target_values = []
            target_rewards = []

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps
                for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    target_rewards.append(game.rewards[current_index])
                else:
                    target_values.append(0)
                    target_rewards.append(0)
                value_index += 1

            batch_values.append(target_values)
            batch_rewards.append(target_rewards)

    # for policy
    policy_mask = []  # 0 -> out of traj, 1 -> old policy
    for game, state_index in zip(games, state_index_lst):
        traj_len = len(game)
        target_policies = []

        for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
            if current_index < traj_len:
                target_policies.append(game.child_visits[current_index])
                policy_mask.append(1)
            else:
                target_policies.append([0 for _ in range(config.action_space_size)])
                policy_mask.append(0)

        batch_policies.append(target_policies)

    return batch_values, batch_rewards, batch_policies

def prepare_multi_target_none(games, state_index_lst, config, model):
    gg=games[0]
    #print("child_visit={},root val={},a={},la={},obs={},r={}".format(len(gg.child_visits),len(gg.root_values),len(gg.actions),len(gg.legal_actions),len(ray.get(gg.obs_history)),len(gg.rewards)),flush=True)
    batch_values, batch_rewards, batch_policies = [], [], []

    zero_obs = games[0].zero_obs()
    device = next(model.parameters()).device
    obs_lst = []
    value_mask = []

    root_val_lst=[]

    with torch.no_grad():
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            #import sys
            #np.set_printoptions(threshold=sys.maxsize)
            #print("state idx= {}/{}".format(state_index,len(state_index_lst)),flush=True)
           # print("obs length:",len(ray.get(game.obs_history)),flush=True)
          #  print("root length:",len(game.root_values),flush=True)
            # game_obs = game.obs(state_index + config.td_steps, config.num_unroll_steps)
            #game_value=game.root_values[state_index + config.td_steps, config.num_unroll_steps]
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps
         #       print("current idx:", current_index,flush=True)
                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # obs = game.obs(bootstrap_index)
                    beg_index = bootstrap_index - (state_index + config.td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = zero_obs
                    # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
             #       print("root value: ")
                    #try:
                    root_val_lst.append(game.root_values[current_index])
                    #except:
                    #    print("bootstrap idx:",bootstrap_index,flush=True)
                else:
                    value_mask.append(0)
                    # obs = zero_obs
                    root_val_lst.append(0)

                #obs_lst.append(obs)
        #print("===========>finish preparing value, ",len(root_val_lst),flush=True)
        network_output = []
        #print("original value_lst:",value_lst.shape,flush=True)
        value_lst=np.array(root_val_lst)
        #value_lst = value_lst.reshape(-1) * config.discount ** config.td_steps
        value_lst = value_lst * config.discount ** config.td_steps

        # get last value
        value_lst = value_lst * np.array(value_mask)
        value_lst = value_lst.tolist()

        value_index = 0
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            target_values = []
            target_rewards = []

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps
                for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    target_rewards.append(game.rewards[current_index])
                else:
                    target_values.append(0)
                    target_rewards.append(0)
                value_index += 1

            batch_values.append(target_values)
            batch_rewards.append(target_rewards)

    # for policy
    policy_mask = []  # 0 -> out of traj, 1 -> old policy
    for game, state_index in zip(games, state_index_lst):
        traj_len = len(game)
        target_policies = []

        for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
            if current_index < traj_len:
                target_policies.append(game.child_visits[current_index])
                policy_mask.append(1)
            else:
                target_policies.append([0 for _ in range(config.action_space_size)])
                policy_mask.append(0)

        batch_policies.append(target_policies)

    return batch_values, batch_rewards, batch_policies

def prepare_multi_target_only_value(games, state_index_lst, config, model):
    batch_values, batch_rewards, batch_policies = [], [], []

    zero_obs = games[0].zero_obs()
    device = next(model.parameters()).device
    obs_lst = []
    value_mask = []

    with torch.no_grad():
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            game_obs = game.obs(state_index + config.td_steps, config.num_unroll_steps)

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # obs = game.obs(bootstrap_index)
                    beg_index = bootstrap_index - (state_index + config.td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                    # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
                else:
                    value_mask.append(0)
                    obs = zero_obs

                obs_lst.append(obs)

        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst, image_based=config.image_based)
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = obs_lst[beg_index:end_index]

            if config.image_based:
                m_obs = torch.from_numpy(m_obs).to(device).float() / 255.0#no divice 255
            else:
                m_obs = torch.from_numpy(m_obs).to(device).float().reshape(m_obs.shape[0], -1)
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)
            network_output.append(m_output)

        value_lst = concat_output_value(network_output)
        value_lst = value_lst.reshape(-1) * config.discount ** config.td_steps
        # get last value
        value_lst = value_lst * np.array(value_mask)
        value_lst = value_lst.tolist()

        value_index = 0
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            target_values = []
            target_rewards = []

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps
                for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    target_rewards.append(game.rewards[current_index])
                else:
                    target_values.append(0)
                    target_rewards.append(0)
                value_index += 1

            batch_values.append(target_values)
            batch_rewards.append(target_rewards)

    # for policy
    policy_mask = []  # 0 -> out of traj, 1 -> old policy
    for game, state_index in zip(games, state_index_lst):
        traj_len = len(game)
        target_policies = []

        for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
            if current_index < traj_len:
                target_policies.append(game.child_visits[current_index])
                policy_mask.append(1)
            else:
                target_policies.append([0 for _ in range(config.action_space_size)])
                policy_mask.append(0)

        batch_policies.append(target_policies)

    return batch_values, batch_rewards, batch_policies


def prepare_multi_target(replay_buffer, indices, make_time, games, state_index_lst, config, model):
    batch_values, batch_rewards, batch_policies = [], [], []
    zero_obs = games[0].zero_obs()

    device = next(model.parameters()).device
    obs_lst = []
    value_mask = []
    # print("state_index_lst",type(state_index_lst))
    with torch.no_grad():
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            game_obs = game.obs(state_index + config.td_steps, config.num_unroll_steps)

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    # obs = game.obs(bootstrap_index)
                    beg_index = bootstrap_index - (state_index + config.td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                    # assert (np.array(obs) == np.array(game.obs(bootstrap_index))).all()
                else:
                    value_mask.append(0)
                    obs = zero_obs

                obs_lst.append(obs)
        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst, image_based=config.image_based)#unchanged for state-based
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = obs_lst[beg_index:end_index]
            # print("in Game's reanalyze: ",m_obs.shape,flush=True)
            if config.image_based:
                m_obs = torch.from_numpy(m_obs).to(device).float() / 255.0
            else:
                m_obs = torch.from_numpy(m_obs).to(device).float().reshape(m_obs.shape[0], -1)
                # print("in Game's reanalyze reshaped: ",m_obs.shape,flush=True)
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)

            network_output.append(m_output)
        value_lst = concat_output_value(network_output)
        value_lst = value_lst.reshape(-1) * config.discount ** config.td_steps
        # get last value
        value_lst = value_lst * np.array(value_mask)
        value_lst = value_lst.tolist()

        value_index = 0
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            target_values = []
            target_rewards = []

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + config.td_steps
                for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    target_rewards.append(game.rewards[current_index])
                else:
                    target_values.append(0)
                    target_rewards.append(0)
                value_index += 1

            batch_values.append(target_values)
            batch_rewards.append(target_rewards)
        # for policy
        obs_lst = []

        legal_action_lst=[]

        policy_mask = []  # 0 -> out of traj, 1 -> new policy
        #assert False
        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)

            game_obs = game.obs(state_index, config.num_unroll_steps)
            #print("game info, la={},obs={},reward={},action={},c_v={},r_v={}".format(len(game.legal_actions),len(game.obs_history),len(game.rewards),len(game.actions),len(game.child_visits),len(game.root_values)),flush=True)
            #legal_action_lst.append(game.legal_actions[state_index])
            #print("state idx={},len={}, legal={},game_legal_action={}".format(state_index,len(state_index_lst),np.array(game.legal_actions[state_index]).shape,np.array(game.legal_actions).shape))
            #print("legal:",game.legal_actions[state_index])
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                if current_index < traj_len:
                    policy_mask.append(1)
                    # obs = game.obs(current_index)
                    beg_index = current_index - state_index
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                    # assert (np.array(obs) == np.array(game.obs(current_index))).all()
                    legal_action_lst.append(game.legal_actions[current_index])
                else:
                    policy_mask.append(0)
                    obs = zero_obs
                    legal_action_lst.append([0 for _ in range(config.action_space_size)])
                obs_lst.append(obs)

                #print("state_idx={},obs={},obs_lst={}".format(state_index,len(obs),len(obs_lst)),flush=True)
        #print("====>all legal action",np.array(legal_action_lst).shape,flush=True)
        batch_num = len(obs_lst)
        obs_lst = prepare_observation_lst(obs_lst, image_based=config.image_based)
        legal_action_lst=np.array(legal_action_lst)
        m_batch = config.target_infer_size
        slices = batch_num // m_batch
        if m_batch * slices < batch_num:
            slices += 1
        network_output = []

        #legal_a=[]
#slice=6,batch_num=378,m_batch=64
        #print("slice={},batch_num={},m_batch={}".format(slices,batch_num,m_batch),flush=True)
        for i in range(slices):
            beg_index = m_batch * i
            end_index = m_batch * (i + 1)
            m_obs = obs_lst[beg_index:end_index]
            if config.image_based:
                m_obs = torch.from_numpy(m_obs).to(device).float() / 255.0
            else:
                m_obs = torch.from_numpy(m_obs).to(device).float().reshape(m_obs.shape[0], -1)
            if config.amp_type == 'torch_amp':
                with autocast():
                    m_output = model.initial_inference(m_obs)
            else:
                m_output = model.initial_inference(m_obs)
            network_output.append(m_output)

        #    legal_a.append(legal_action_lst[beg_index:end_index])

         #   print("slice shape",i, np.array(legal_action_lst[beg_index:end_index]).shape,flush=True)

        #print("finish slice",flush=True)
        _, reward_pool, policy_logits_pool, hidden_state_roots = concat_output(network_output)
        reward_pool = reward_pool.squeeze().tolist()
        #print("policy logits pool shape,",len(policy_logits_pool),policy_logits_pool[0].shape,flush=True)
        #nan_part=np.isnan(policy_logits_pool)
        #if nan_part.any():
        #    print("batchworker, prepare,policy logits are nan!",flush=True)
        policy_logits_pool = policy_logits_pool.tolist()

        roots = Roots(len(obs_lst), config.action_space_size, config.num_simulations)
        #mask noise
        noises = [(np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(np.float32) * np.array(legal_action_lst[idx]) ).tolist() for idx in range(len(obs_lst))]
        #mock_legal_actions=[np.ones(20).astype(int) for _ in range(len(policy_logits_pool))]
        mock_legal_actions=[np.array(i).astype(int) for i in legal_action_lst]
        roots.prepare(config.root_exploration_fraction, noises, reward_pool, policy_logits_pool,mock_legal_actions)

        MCTS(config).run_multi(roots, model, hidden_state_roots)

        roots_distributions = roots.get_distributions()
        roots_values = roots.get_values()
        policy_index = 0
        game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times = [], [], [], [], []
        for game, state_index, game_idx, mt in zip(games, state_index_lst, indices, make_time):
            target_policies = []

            current_index_lst, distributions_lst, value_lst, make_time_lst = [], [], [], []
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                distributions, value = roots_distributions[policy_index], roots_values[policy_index]

                if policy_mask[policy_index] == 0:
                    target_policies.append([0 for _ in range(config.action_space_size)])
                else:
                    # game.store_search_stats(distributions, value, current_index)
                    sum_visits = sum(distributions)
                    policy = [visit_count / sum_visits for visit_count in distributions]
                    target_policies.append(policy)

                    current_index_lst.append(current_index)
                    distributions_lst.append(distributions)
                    value_lst.append(value)
                    make_time_lst.append(mt)

                policy_index += 1

            batch_policies.append(target_policies)
            if config.write_back:
                game_idx_lst.append(game_idx)
                current_index_lsts.append(current_index_lst)
                distributions_lsts.append(distributions_lst)
                value_lsts.append(value_lst)
                make_times.append(make_time_lst)
        if config.write_back:
            replay_buffer.update_games.remote(game_idx_lst, current_index_lsts, distributions_lsts, value_lsts, make_times)

    return batch_values, batch_rewards, batch_policies
