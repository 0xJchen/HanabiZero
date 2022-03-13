import logging
import psutil
import os
import ray
import math
import torch
import random
from random import randint
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from ray.util.queue import Queue
from ray.util.multiprocessing import Pool

import core.ctree.cytree as cytree
from .mcts import MCTS, get_node_distribution
from .replay_buffer import ReplayBuffer
from .test import test
from .utils import select_action, profile, prepare_observation_lst, LinearSchedule
from .vis import show_tree
from .game import GameHistory, prepare_multi_target, prepare_multi_target_only_value, prepare_multi_target_none
from .model import concat_output, concat_output_value
import time
import numpy as np
try:
    from apex import amp
except:
    pass

gpu_num=0.06


@ray.remote
class BatchWorker_CPU(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, mcts_storage, config):
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.batch_storage = batch_storage
        self.mcts_storage = mcts_storage
        self.config = config

        self.last_model_index = -1
        self.batch_max_num = 20
        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps, initial_p=config.priority_prob_beta, final_p=1.0)
        self.mcts_buffer_max = 20

    def _prepare_reward_value_context(self, indices, games, state_index_lst, total_transitions):
        zero_obs = games[0].zero_obs()

        config = self.config
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []
        td_steps = config.td_steps
        # td_steps_lst = []
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # # off-policy correction: shorter horizon of td steps
            # delta_td = (total_transitions - idx) // config.auto_td_steps
            
            # td_steps = np.clip(td_steps, 1, 5).astype(np.int)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
            rewards_lst.append(game.rewards)
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                # td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)

        value_obs_lst=prepare_observation_lst(value_obs_lst)
        value_obs_lst = ray.put(value_obs_lst)
        reward_value_context = [value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens]
        return reward_value_context


    def _prepare_policy_non_re_context(self, indices, games, state_index_lst):

        child_visits = []
        traj_lens = []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            child_visits.append(game.child_visits)

        policy_non_re_context = [state_index_lst, child_visits, traj_lens]
        return policy_non_re_context

    def _prepare_policy_re_context(self, indices, games, state_index_lst):
        zero_obs = games[0].zero_obs()
        config = self.config

        with torch.no_grad():
            # for policy
            policy_obs_lst = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            legal_action_lst = []
            rewards, child_visits, traj_lens = [], [], []
            for game, state_index in zip(games, state_index_lst):
                traj_len = len(game)
                traj_lens.append(traj_len)
                rewards.append(game.rewards)
                child_visits.append(game.child_visits)
                # prepare the corresponding observations
                game_obs = game.obs(state_index, config.num_unroll_steps)
                for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + config.stacked_observations
                        obs = game_obs[beg_index:end_index]
                        legal_action_lst.append(game.legal_actions[current_index])
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                        legal_action_lst.append([0 for _ in range(config.action_space_size)])
                    policy_obs_lst.append(obs)

        policy_obs_lst = prepare_observation_lst(policy_obs_lst)
        legal_action_lst=np.array(legal_action_lst)

        policy_obs_lst = ray.put(policy_obs_lst)
        policy_re_context = [policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens, np.array(legal_action_lst)]
        return policy_re_context



    def make_batch(self, batch_context, ratio, weights=None):

        # obtain the batch context from replay buffer
        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            game_pos = game_pos_lst[i]
            _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]
            _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]

            obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))
          
            action_lst.append(_actions)
            mask_lst.append(_mask)

        re_num = int(batch_size * ratio)
        # formalize the input observations
        obs_lst = prepare_observation_lst(obs_lst,image_based=False)
        # formalize the inputs of a batch
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = ray.get(self.replay_buffer.get_total_len.remote())

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(indices_lst, game_lst, game_pos_lst, total_transitions)

        # 0:re_num -> reanalyzed policy, re_num:end -> non reanalyzed policy
        # reanalyzed policy
        if re_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_re_context(indices_lst[:re_num], game_lst[:re_num], game_pos_lst[:re_num])
        else:
            policy_re_context = None

        # non reanalyzed policy
        if re_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_re_context(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:])
        else:
            policy_non_re_context = None

        countext = reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, weights
        self.mcts_storage.push(countext)

    def run(self):
        # start making mcts contexts to feed the GPU batch maker
        start = False
        while True:
            if not start:
                # print("CPU worker waiting",flush=True)
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(2)
                continue
            # TODO: use latest weights for policy reanalyze
            ray_data_lst = [self.storage.get_counter.remote(), self.storage.get_target_weights.remote()]
            trained_steps, target_weights = ray.get(ray_data_lst)

            beta = self.beta_schedule.value(trained_steps)
            batch_context = ray.get(self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta))

            if trained_steps >= self.config.training_steps + self.config.last_steps:
                print("batchworker finished working",flush=True)
                break

            new_model_index = trained_steps // self.config.target_model_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
            else:
                target_weights = None

            if self.mcts_storage.get_len() < self.mcts_buffer_max:
                #should be zero as no batch is pushed
                try:
                    self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights)

                # self.batch_storage.push(batch)
                except:
                    print('=====================>Data is deleted...')
                    time.sleep(0.1)
                #     #assert False


@ray.remote(num_gpus=gpu_num)
class BatchWorker_GPU(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, mcts_storage, config):

        self.replay_buffer = replay_buffer
        self.config = config
        self.worker_id = worker_id

        self.model = config.get_uniform_network()
        self.model.to(config.device)
        self.model.eval()

        self.mcts_storage = mcts_storage
        self.storage = storage
        self.batch_storage = batch_storage

        self.last_model_index = 0

    def _prepare_reward_value(self, reward_value_context):
        
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens = reward_value_context
        value_obs_lst = ray.get(value_obs_lst)
        device = self.config.device
        batch_size = len(value_obs_lst)
        batch_values, batch_rewards = [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_lst(value_obs_lst)
            # split a full batch into slices of target_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.target_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                # m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float()
                m_obs=m_obs.reshape(m_obs.shape[0],-1)
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            value_lst = concat_output_value(network_output)
            # print("[after inference], value_lst={},v_mask={},shape={},{}".format(value_lst,value_mask,np.array(value_lst).shape,np.array(value_mask).shape))
            # get last state value
            value_lst = value_lst.reshape(-1) * self.config.discount ** self.config.td_steps
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            # print("[after td], value_lst={},v_mask={},reward_lst={} shape={},{},{} ".format(value_lst,value_mask,rewards_lst,np.array(value_lst).shape,np.array(value_mask).shape,np.array(rewards_lst).shape))
            value_index = 0
            for traj_len_non_re, reward_lst, state_index in zip(traj_lens, rewards_lst, state_index_lst):
                # traj_len = len(game)
                target_values = []
                target_rewards = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    bootstrap_index = current_index + self.config.td_steps
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        # try:
                            # print("value_lst={},reward_lst={},disc={},i={},idx={}".format(value_lst, reward,self.config.discount,i,value_index))
                        value_lst[value_index] += reward * self.config.discount ** i
                        # except:
                            
                            # assert False
                    if current_index < traj_len_non_re:
                        target_values.append(value_lst[value_index])
                        target_rewards.append(reward_lst[current_index])
                    else:
                        target_values.append(0)
                        target_rewards.append(0)
                    value_index += 1

                batch_values.append(target_values)
                batch_rewards.append(target_rewards)
        batch_values=np.array(batch_values)
        batch_rewards=np.array(batch_rewards)
        return batch_values,batch_rewards


    def _prepare_policy_re(self, policy_re_context):
        """prepare policy targets from the reanalyzed context of policies
        """
        batch_policies_re = []
        if policy_re_context is None:
            return batch_policies_re

        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens, legal_action_lst = policy_re_context
        policy_obs_lst = ray.get(policy_obs_lst)
        batch_size = len(policy_obs_lst)
        device = self.config.device

        with torch.no_grad():
            policy_obs_lst = prepare_observation_lst(policy_obs_lst)
            # split a full batch into slices of target_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.target_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                # m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float()
                m_obs=m_obs.reshape(m_obs.shape[0],-1)
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots = concat_output(network_output)
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()

            roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
            noises = [(np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32)*legal_action_lst[idx]).tolist() for idx in range(batch_size)]
            mock_legal_actions=[i.astype(int) for i in legal_action_lst]
            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool,mock_legal_actions)
            # do MCTS for a new policy with the recent target model
            MCTS(self.config).run_multi(roots, self.model, hidden_state_roots)

            roots_distributions = roots.get_distributions()
            policy_index = 0
            for state_index, game_idx in zip(state_index_lst, indices):
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                    else:
                        # game.store_search_stats(distributions, value, current_index)
                        sum_visits = sum(distributions)
                        policy = [visit_count / sum_visits for visit_count in distributions]
                        target_policies.append(policy)

                    policy_index += 1

                batch_policies_re.append(target_policies)

        batch_policies_re = np.asarray(batch_policies_re)
        return batch_policies_re


    def _prepare_policy_non_re(self, policy_non_re_context):

        batch_policies_non_re = []
        if policy_non_re_context is None:
            return batch_policies_non_re

        state_index_lst, child_visits, traj_lens = policy_non_re_context
        with torch.no_grad():
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, state_index_lst):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, state_index_lst):
                # traj_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    if current_index < traj_len:
                        target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                    else:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                        policy_mask.append(0)

                batch_policies_non_re.append(target_policies)
        batch_policies_non_re = np.asarray(batch_policies_non_re)
        return batch_policies_non_re


    def _prepare_target_gpu(self):
        input_countext = self.mcts_storage.pop()
        if input_countext is None:
            # print("GPU worker waiting context", flush=True)
            time.sleep(1)
        else:
            reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, target_weights = input_countext
            if target_weights is not None:
                self.model.load_state_dict(target_weights)
                self.model.to(self.config.device)
                self.model.eval()

            # target reward, value
            batch_values, batch_rewards = self._prepare_reward_value(reward_value_context)
            # target policy
            batch_policies_re = self._prepare_policy_re(policy_re_context)
            batch_policies_non_re = self._prepare_policy_non_re(policy_non_re_context)
            batch_policies = np.concatenate([batch_policies_re, batch_policies_non_re])

            targets_batch = [batch_rewards, batch_values, batch_policies]
            # a batch contains the inputs and the targets; inputs is prepared in CPU workers
            self.batch_storage.push([inputs_batch, targets_batch])

    def run(self):
        start = False
        while True:
            # waiting for start signal
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                # print("GPU worker waiting")
                time.sleep(2)
                # time.sleep(5)
                continue
            # trained_steps = ray.get(self.storage.get_counter.remote())
            # if trained_steps >= self.config.training_steps + self.config.last_steps:
            #     time.sleep(30)
            #     break

            self._prepare_target_gpu()
