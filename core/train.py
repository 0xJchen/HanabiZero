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
import time
import numpy as np
try:
    from apex import amp
except:
    pass
###
train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_test')

gpu_num=0.13#0.13 for full

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def _log(config, step_count, log_data, model, replay_buffer, lr, shared_storage, summary_writer, vis_result):
    loss_data, td_data, priority_data = log_data
    total_loss, weighted_loss, loss, reg_loss, policy_loss, reward_loss, value_loss, consistency_loss = loss_data
    if vis_result:
        new_priority, target_reward, target_value, trans_target_reward, trans_target_value, target_reward_phi, target_value_phi, \
        pred_reward, pred_value, target_policies, predicted_policies, state_lst, other_loss, other_log, other_dist = td_data
        batch_weights, batch_indices = priority_data

    replay_episodes_collected, replay_buffer_size, priorities, total_num, worker_logs = ray.get([
        replay_buffer.episodes_collected.remote(), replay_buffer.size.remote(),
        replay_buffer.get_priorities.remote(), replay_buffer.get_total_len.remote(),
        shared_storage.get_worker_logs.remote()])

    worker_ori_reward, worker_reward, worker_reward_max, worker_eps_len, worker_eps_len_max, mean_test_score, max_test_score, temperature, visit_entropy, priority_self_play, distributions = worker_logs

    _msg = '#{:<10} Total Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Loss: {:<8.3f} Consistency Loss: {:<8.3f} ] ' \
           'Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Transition Number: {:<8.3f}k ' \
           'Batch Size: {:<10d} Lr: {:<8.3f}'
    _msg = _msg.format(step_count, total_loss, weighted_loss, policy_loss, value_loss, reward_loss, consistency_loss,
                       replay_episodes_collected, replay_buffer_size, total_num / 1000, config.batch_size, lr)
    train_logger.info(_msg)

    if mean_test_score is not None:
        test_msg = '#{:<10} Test Mean Score: {:<10}(Max: {:<10})'.format(step_count, mean_test_score, max_test_score)
        test_logger.info(test_msg)

    if summary_writer is not None:
        if config.debug:
            for name, W in model.named_parameters():
                summary_writer.add_histogram('after_grad_clip' + '/' + name + '_grad', W.grad.data.cpu().numpy(),
                                             step_count)
                summary_writer.add_histogram('network_weights' + '/' + name, W.data.cpu().numpy(), step_count)
            pass
        tag = 'Train'
        if vis_result:
            summary_writer.add_histogram('{}_replay_data/replay_buffer_priorities'.format(tag),
                                         priorities,
                                         step_count)
            summary_writer.add_histogram('{}_replay_data/batch_weight'.format(tag), batch_weights, step_count)
            summary_writer.add_histogram('{}_replay_data/batch_indices'.format(tag), batch_indices, step_count)
            # TODO: print out the reward to check the distribution (few 0 out) mean std
            target_reward = target_reward.flatten()
            pred_reward = pred_reward.flatten()
            target_value = target_value.flatten()
            pred_value = pred_value.flatten()
            new_priority = new_priority.flatten()

            summary_writer.add_scalar('{}_statistics/new_priority_mean'.format(tag), new_priority.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/new_priority_std'.format(tag), new_priority.std(), step_count)

            summary_writer.add_scalar('{}_statistics/target_reward_mean'.format(tag), target_reward.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/target_reward_std'.format(tag), target_reward.std(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_reward_mean'.format(tag), pred_reward.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_reward_std'.format(tag), pred_reward.std(), step_count)

            summary_writer.add_scalar('{}_statistics/target_value_mean'.format(tag), target_value.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/target_value_std'.format(tag), target_value.std(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_value_mean'.format(tag), pred_value.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_value_std'.format(tag), pred_value.std(), step_count)

            summary_writer.add_histogram('{}_data_dist/new_priority'.format(tag), new_priority, step_count)
            summary_writer.add_histogram('{}_data_dist/target_reward'.format(tag), target_reward - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/target_value'.format(tag), target_value - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/transformed_target_reward'.format(tag), trans_target_reward,
                                         step_count)
            summary_writer.add_histogram('{}_data_dist/transformed_target_value'.format(tag), trans_target_value,
                                         step_count)
            summary_writer.add_histogram('{}_data_dist/pred_reward'.format(tag), pred_reward - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/pred_value'.format(tag), pred_value - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/pred_policies'.format(tag), predicted_policies.flatten(),
                                         step_count)
            summary_writer.add_histogram('{}_data_dist/target_policies'.format(tag), target_policies.flatten(),
                                         step_count)

            summary_writer.add_histogram('{}_data_dist/hidden_state'.format(tag), state_lst.flatten(), step_count)

            for key, val in other_loss.items():
                if val >= 0:
                    summary_writer.add_scalar('{}_metric/'.format(tag) + key, val, step_count)

            for key, val in other_log.items():
                summary_writer.add_scalar('{}_weight/'.format(tag) + key, val, step_count)

            for key, val in other_dist.items():
                summary_writer.add_histogram('{}_dist/'.format(tag) + key, val, step_count)

        summary_writer.add_scalar('{}/total_loss'.format(tag), total_loss, step_count)
        summary_writer.add_scalar('{}/loss'.format(tag), loss, step_count)
        summary_writer.add_scalar('{}/weighted_loss'.format(tag), weighted_loss, step_count)
        summary_writer.add_scalar('{}/reg_loss'.format(tag), reg_loss, step_count)
        summary_writer.add_scalar('{}/policy_loss'.format(tag), policy_loss, step_count)
        summary_writer.add_scalar('{}/value_loss'.format(tag), value_loss, step_count)
        summary_writer.add_scalar('{}/reward_loss'.format(tag), reward_loss, step_count)
        summary_writer.add_scalar('{}/consistency_loss'.format(tag), consistency_loss, step_count)
        summary_writer.add_scalar('{}/episodes_collected'.format(tag), replay_episodes_collected,
                                  step_count)
        summary_writer.add_scalar('{}/replay_buffer_len'.format(tag), replay_buffer_size, step_count)
        summary_writer.add_scalar('{}/total_node_num'.format(tag), total_num, step_count)
        summary_writer.add_scalar('{}/lr'.format(tag), lr, step_count)

        if worker_reward is not None:
            summary_writer.add_scalar('workers/ori_reward', worker_ori_reward, step_count)
            summary_writer.add_scalar('workers/clip_reward', worker_reward, step_count)
            summary_writer.add_scalar('workers/clip_reward_max', worker_reward_max, step_count)
            summary_writer.add_scalar('workers/eps_len', worker_eps_len, step_count)
            summary_writer.add_scalar('workers/eps_len_max', worker_eps_len_max, step_count)
            summary_writer.add_scalar('workers/temperature', temperature, step_count)
            summary_writer.add_scalar('workers/visit_entropy', visit_entropy, step_count)
            summary_writer.add_scalar('workers/priority_self_play', priority_self_play, step_count)
            for key, val in distributions.items():
                if len(val) == 0:
                    continue

                val = np.array(val).flatten()
                summary_writer.add_histogram('workers/{}'.format(key), val, step_count)

        if mean_test_score is not None:
            summary_writer.add_scalar('train/test_score', mean_test_score, step_count)
            summary_writer.add_scalar('train/test_max_score', max_test_score, step_count)


@ray.remote
class SharedStorage(object):
    def __init__(self, model, target_model, latest_model):
        self.step_counter = 0
        self.model = model
        self.target_model = target_model
        self.latest_model = latest_model
        self.ori_reward_log = []
        self.reward_log = []
        self.reward_max_log = []
        self.test_log = []
        self.eps_lengths = []
        self.eps_lengths_max = []
        self.temperature_log = []
        self.visit_entropies_log = []
        self.priority_self_play_log = []
        self.distributions_log = {
            'depth': [],
            'visit': []
        }
        self.start = False

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def get_target_weights(self):
        return self.target_model.get_weights()

    def set_target_weights(self, weights):
        return self.target_model.set_weights(weights)

    def get_latest_weights(self):
        return self.latest_model.get_weights()

    def set_latest_weights(self, weights):
        return self.latest_model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_len_max, eps_ori_reward, eps_reward, eps_reward_max, temperature, visit_entropy, priority_self_play, distributions):
        self.eps_lengths.append(eps_len)
        self.eps_lengths_max.append(eps_len_max)
        self.ori_reward_log.append(eps_ori_reward)
        self.reward_log.append(eps_reward)
        self.reward_max_log.append(eps_reward_max)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)
        self.priority_self_play_log.append(priority_self_play)

        for key, val in distributions.items():
            self.distributions_log[key] += val

    def add_test_log(self, score):
        self.test_log.append(score)

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            ori_reward = sum(self.ori_reward_log) / len(self.ori_reward_log)
            reward = sum(self.reward_log) / len(self.reward_log)
            reward_max = sum(self.reward_max_log) / len(self.reward_max_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            eps_lengths_max = sum(self.eps_lengths_max) / len(self.eps_lengths_max)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)
            priority_self_play = sum(self.priority_self_play_log) / len(self.priority_self_play_log)
            distributions = self.distributions_log

            self.ori_reward_log = []
            self.reward_log = []
            self.reward_max_log = []
            self.eps_lengths = []
            self.eps_lengths_max = []
            self.temperature_log = []
            self.visit_entropies_log = []
            self.priority_self_play_log = []
            self.distributions_log = {
                'depth': [],
                'visit': []
            }

        else:
            ori_reward = None
            reward = None
            reward_max = None
            eps_lengths = None
            eps_lengths_max = None
            temperature = None
            visit_entropy = None
            priority_self_play = None
            distributions = None

        if len(self.test_log) > 0:
            self.test_log = np.array(self.test_log).flatten()
            mean_test_score = sum(self.test_log) / len(self.test_log)
            max_test_score = max(self.test_log)
            self.test_log = []
        else:
            mean_test_score = None
            max_test_score = None

        return ori_reward, reward, reward_max, eps_lengths, eps_lengths_max, mean_test_score, max_test_score, temperature, visit_entropy, priority_self_play, distributions


@ray.remote(num_gpus=gpu_num)
class DataWorker(object):
    def __init__(self, rank, config, shared_storage, replay_buffer):
        self.rank = rank
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.trajectory_pool = []
        self.pool_size = 1
        self.device = 'cuda'
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1

    def put(self, data):
        self.trajectory_pool.append(data)

    def put_last_trajectory(self, i, last_game_histories, last_game_priorities, game_histories):
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_legal_a= game_histories[i].legal_actions[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst, pad_legal_a)
        last_game_histories[i].game_over()

        self.put((last_game_histories[i], last_game_priorities[i]))
        self.free()

        # reset last block
        last_game_histories[i] = None
        last_game_priorities[i] = None

    def len_pool(self):
        return len(self.trajectory_pool)

    def free(self):
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool, self.gap_step)
            del self.trajectory_pool[:]

    def get_priorities(self, i, pred_values_lst, search_values_lst):

        if self.config.use_priority and not self.config.use_max_priority:
            # traj_len = len(pred_values_lst[i])
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            priorities = None

        return priorities

    def run_multi(self):
        # number of parallel mcts
        env_nums = self.config.p_mcts_num
        if self.config.amp_type == 'nvidia_apex':
            model = amp.initialize(self.config.get_uniform_network().cuda())
        else:
            model = self.config.get_uniform_network()
        model.to(self.device)
        model.eval()

        start_training = False
        envs = [self.config.new_game(self.config.seed + self.rank * i) for i in range(env_nums)]

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # 100k benchmark
        total_transitions = 0
        #max_transitions = 500 * 1000 // self.config.num_actors
        max_transitions = 500 * 5000 // self.config.num_actors
        with torch.no_grad():
            while True:
                trained_steps = ray.get(self.shared_storage.get_counter.remote())
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    break

                # init_obses = [env.reset() for env in envs]
                #@wjc
                init_obses=[]
                init_legal_action=[]
                for env in envs:
                    # print(env.reset())
                    o,a=env.reset()

                    init_obses.append(o)
                    #print("rest game: init actions=",a,flush=True)
                    init_legal_action.append(a)

                #assert False

                dones = np.array([False for _ in range(env_nums)])
                game_histories = [GameHistory(envs[_].env.action_space, max_length=self.config.history_length,
                                              config=self.config) for _ in range(env_nums)]
                last_game_histories = [None for _ in range(env_nums)]
                last_game_priorities = [None for _ in range(env_nums)]

                # stack observation windows in boundary: s198, s199, s200, current s1 -> for not init trajectory
                stack_obs_windows = [[] for _ in range(env_nums)]

                for i in range(env_nums):
                    stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
                    game_histories[i].init(stack_obs_windows[i],init_legal_action[i])
                    #print("after init: ",game_histories[i].legal_actions,flush=True)
                # this the root value of MCTS
                search_values_lst = [[] for _ in range(env_nums)]
                # predicted value of target network
                pred_values_lst = [[] for _ in range(env_nums)]
                # observed n-step return
                # obs_value_lst = [[] for _ in range(env_nums)]

                # end_tags = [False for _ in range(env_nums)]

                eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums)
                step_counter = 0

                _temperature = np.array(
                    [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps) for env in
                     envs])

                self_play_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []
                depth_distribution = []
                visit_count_distribution = []

                while not dones.all() and (step_counter <= self.config.max_moves * self.config.self_play_moves_ratio):
                    # if self_play_episodes %2 ==0:
                        # print("parallel env={0},in dataworker, played {1} trajectory already".format(env_nums,self_play_episodes),flush=True)
                    if not start_training:
                        start_training = ray.get(self.shared_storage.get_start_signal.remote())

                    # get model
                    trained_steps = ray.get(self.shared_storage.get_counter.remote())
                    if trained_steps >= self.config.training_steps + self.config.last_steps:
                        print("training finished",flush=True)
                        # training is finished
                        return
                  #  if start_training and (total_transitions / max_transitions) > (trained_steps / self.config.training_steps):
                        # self-play is faster or finished
                        #print("===========> self play is faster,total_tran={0},{1},trained_step={2},{3};{4} > {5}".format(total_transition,max_transition,trained_steps,self.config.training_steps,total_transitions / max_transitions,trained_steps / self.config.training_steps))
                        #print("=====>self play faster!",flush=True)
                   #     a=total_transitions / max_transitions
                    #    b=trained_steps / self.config.training_steps
                    #    print("self-play={0},train={1}".format(a,b),flush=True)
                     #   time.sleep(0.3)
                      #  continue
                   # print("start training",flush=True)

                    new_model_index = trained_steps // self.config.checkpoint_interval
                    if new_model_index > self.last_model_index:
                        self.last_model_index = new_model_index
                        # update model
                        weights = ray.get(self.shared_storage.get_weights.remote())
                        model.set_weights(weights)
                        model.to(self.device)
                        model.eval()

                        # log
                        if env_nums > 1:
                            if len(self_play_visit_entropy) > 0:
                                visit_entropies = np.array(self_play_visit_entropy).mean()
                                visit_entropies /= max_visit_entropy
                            else:
                                visit_entropies = 0.

                            if self_play_episodes > 0:
                                log_self_play_moves = self_play_moves / self_play_episodes
                                log_self_play_rewards = self_play_rewards / self_play_episodes
                                log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                            else:
                                log_self_play_moves = 0
                                log_self_play_rewards = 0
                                log_self_play_ori_rewards = 0

                            # depth_distribution = np.array(depth_distribution)
                            # visit_count_distribution = np.array(visit_count_distribution)
                            self.shared_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                            log_self_play_ori_rewards, log_self_play_rewards,
                                                                            self_play_rewards_max, _temperature.mean(),
                                                                            visit_entropies, 0,
                                                                            {'depth': depth_distribution,
                                                                             'visit': visit_count_distribution})
                            self_play_rewards_max = - np.inf

                    step_counter += 1
                    ## reset env if finished
                    for i in range(env_nums):
                        if dones[i]:

                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # store current block trajectory
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], priorities))
                            self.free()

                            envs[i].close()
                            #@wjc
                            init_obs, init_legal_actions = envs[i].reset()
                            game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            last_game_histories[i] = None
                            last_game_priorities[i] = None
                            stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                            #@wjc
                            stack_legal_actions[i] = init_legal_actions
                            game_histories[i].init(stack_obs_windows[i],stack_legal_actions[i])

                            self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_rewards += eps_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1

                            pred_values_lst[i] = []
                            search_values_lst[i] = []
                            # end_tags[i] = False
                            eps_steps_lst[i] = 0
                            eps_reward_lst[i] = 0
                            eps_ori_reward_lst[i] = 0
                            visit_entropies_lst[i] = 0

                    stack_obs = [game_history.step_obs() for game_history in game_histories]
                    if self.config.image_based:
                        stack_obs = prepare_observation_lst(stack_obs)
                        # stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                        stack_obs = torch.from_numpy(stack_obs).to(self.device).float()
                        assert False
                    else:
                        stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device).reshape(env_nums, -1)

                    #@wjc
                    stack_legal_actions=[game_history.legal_actions[-1] for game_history in game_histories]
                    #test legal action
                    #for tt in range(len(game_histories)):
                        #print("======================================")
                        #print(game_histories[tt].legal_actions,flush=True)
                    if self.config.amp_type == 'torch_amp':
                        with autocast():
                            network_output = model.initial_inference(stack_obs.float())
                    else:
                        network_output = model.initial_inference(stack_obs.float())
                    hidden_state_roots = network_output.hidden_state
                    reward_pool = network_output.reward
                    policy_logits_pool = network_output.policy_logits.tolist()

                    roots = cytree.Roots(env_nums, self.config.action_space_size, self.config.num_simulations)
                    noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(env_nums)]
                    roots.prepare(self.config.root_exploration_fraction, noises, reward_pool, policy_logits_pool,stack_legal_actions)

                    MCTS(self.config).run_multi(roots, model, hidden_state_roots)

                    roots_distributions = roots.get_distributions()
                    roots_values = roots.get_values()
                    for i in range(env_nums):
                        if start_training:
                            distributions, value, temperature, env = roots_distributions[i], roots_values[i], _temperature[i], envs[i]

                            deterministic = False
                            # do greedy action
                            if self.config.use_epsilon_greedy:
                                if random.random() < min(trained_steps / self.config.training_steps, 0.1):
                                    deterministic = True
                        else:
                            value, temperature, env = roots_values[i], _temperature[i], envs[i]
                            distributions = np.ones(self.config.action_space_size)
                            deterministic = False
                        # print("before select action: ",len(stack_legal_actions))
                        action, visit_entropy = select_action(distributions, temperature=temperature, deterministic=deterministic, legal_actions=stack_legal_actions[i])
                        obs, ori_reward, done, info, legal_action = env.step(action)
                        #if not np.any(legal_action):
                        #    print("step:",done,legal_action,flush=True)
                        #    print("after step, legal a type",type(legal_action))
                        if self.config.clip_reward:
                            clip_reward = np.sign(ori_reward)
                        else:
                            clip_reward = ori_reward

                        game_histories[i].store_search_stats(distributions, value)
                        game_histories[i].append(action, obs, clip_reward, legal_action)

                        eps_reward_lst[i] += clip_reward
                        eps_ori_reward_lst[i] += ori_reward
                        dones[i] = done
                        visit_entropies_lst[i] += visit_entropy

                        eps_steps_lst[i] += 1
                        if start_training:
                            total_transitions += 1

                        if self.config.use_priority and not self.config.use_max_priority and start_training:
                            pred_values_lst[i].append(network_output.value[i].item())
                            search_values_lst[i].append(roots_values[i])

                        # fresh stack windows
                        # print("check type: ",type(stack_obs_windows[0]),len(stack_obs_windows[0]),type(stack_legal_actions[0]),len(stack_legal_actions[0]))
                        # print("after check: ",len(obs))
                        # for obss in stack_obs_windows:
                        #     print(len(obss))

                        del stack_obs_windows[i][0]
                        stack_obs_windows[i].append(obs)
                        #@wjc
                        #now stack_legal_actions' input is numpy array
                        # del stack_legal_actions[i][0]
                        #stack_legal_actions[i].append(legal_action) #seems to be useless

                        # if game history is full
                        if game_histories[i].is_full():
                            assert False
                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # calculate priority
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                            # save block trajectory
                            last_game_histories[i] = game_histories[i]
                            last_game_priorities[i] = priorities

                            # new block trajectory
                            game_histories[i] = GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            game_histories[i].init(stack_obs_windows[i],stack_legal_actions[i])
                #print("##########one iter all parallel env done",flush=True)
                #try:
                #    print("#########played {0} trajectory already".format(self_play_episodes),flush=True)
                #except:
                #    print("########gg",flush=True)
                for i in range(env_nums):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        # pad over last block trajectory
                        if last_game_histories[i] is not None:
                            self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                        # store current block trajectory
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], priorities))
                        self.free()

                        self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                        self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                        self_play_rewards += eps_reward_lst[i]
                        self_play_ori_rewards += eps_ori_reward_lst[i]
                        self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                        self_play_moves += eps_steps_lst[i]
                        self_play_episodes += 1
                    else:
                        # not save this data
                        total_transitions -= len(game_histories[i])


                visit_entropies = np.array(self_play_visit_entropy).mean()
                visit_entropies /= max_visit_entropy

                if self_play_episodes > 0:
                    log_self_play_moves = self_play_moves / self_play_episodes
                    log_self_play_rewards = self_play_rewards / self_play_episodes
                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                else:
                    log_self_play_moves = 0
                    log_self_play_rewards = 0
                    log_self_play_ori_rewards = 0

                # depth_distribution = np.array(depth_distribution)
                # visit_count_distribution = np.array(visit_count_distribution)
                self.shared_storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                log_self_play_ori_rewards, log_self_play_rewards,
                                                                self_play_rewards_max, _temperature.mean(),
                                                                visit_entropies, 0,
                                                                {'depth': depth_distribution,
                                                                 'visit': visit_count_distribution})


def update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result=False):
    #total_transitions = ray.get(replay_buffer.get_total_len.remote())
    total_transitions=0
    obs_batch_ori, action_batch, mask_batch, target_reward, target_value, target_policy, indices, weights_lst, make_time = batch
    # print("original obs batch: ",torch.from_numpy(obs_batch_ori).shape,flush=True)
    #n=["obs","a","mask","re","val","p","idx","weight_idx","make_time"]
    #for idx,item in enumerate(batch):
    #    print("in update weights: "+str(n[idx]),item.shape,flush=True)

    # [:, 0: config.stacked_observations * 3,:,:]
    if config.image_based:
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float() / 255.0
        obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
        obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]
    else:
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float()
        obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :]
        obs_target_batch = obs_batch_ori[:, config.image_channel:, :]

    if config.use_augmentation:
        # TODO: use different augmentation in target observations respectively
        obs_batch = config.transform(obs_batch)
        obs_target_batch = config.transform(obs_target_batch)

    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_reward = torch.from_numpy(target_reward).to(config.device).float()
    target_value = torch.from_numpy(target_value).to(config.device).float()
    target_policy = torch.from_numpy(target_policy).to(config.device).float()
    weights = torch.from_numpy(weights_lst).to(config.device).float()

    batch_size = obs_batch.size(0)
    assert batch_size == config.batch_size == target_reward.size(0)
    metric_loss = torch.nn.L1Loss()

    # transform targets to categorical representation
    # Reference:  Appendix F
    other_log = {}
    other_dist = {}

    other_loss = {
        'l1': -1,
        'l1_1': -1,
        'l1_-1': -1,
        'l1_0': -1,
    }
    for i in range(config.num_unroll_steps):
        key = 'unroll_' + str(i + 1) + '_l1'
        other_loss[key] = -1
        other_loss[key + '_1'] = -1
        other_loss[key + '_-1'] = -1
        other_loss[key + '_0'] = -1
    ratio_reward_1 = (target_reward == 1).sum().item() / (batch_size * 5)
    ratio_reward_0 = (target_reward == 0).sum().item() / (batch_size * 5)
    ratio_reward_ne1 = (target_reward == -1).sum().item() / (batch_size * 5)
    other_loss['target_reward_1'] = ratio_reward_1
    other_loss['target_reward_0'] = ratio_reward_0
    other_loss['target_reward_-1'] = ratio_reward_ne1

    transformed_target_reward = config.scalar_transform(target_reward)
    target_reward_phi = config.reward_phi(transformed_target_reward)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)
    #print("target: ",target_value.shape,transformed_target_value.shape,target_value_phi.shape,flush=True)

    with autocast():
        value, _, policy_logits, hidden_state = model.initial_inference(obs_batch.reshape(batch_size, -1))
    scaled_value = config.inverse_value_transform(value)
    #print("inference shape: ",value.shape,policy_logits.shape,hidden_state.shape,flush=True)

    #print("inference shape: value={}, policy_logits={}, hidden_state={}, scaled_value={}".format(value.shape, policy_logits.shape,hidden_state.shape,scaled_value.shape),flush=True)
    if vis_result:
        state_lst = hidden_state.detach().cpu().numpy()

    predicted_rewards = []
    # Note: Following line is just for logging.
    if vis_result:
        predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(policy_logits, dim=1).detach().cpu()

    # Reference: Appendix G
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps
    reward_priority = []

    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    reward_loss = torch.zeros(batch_size, device=config.device)
    consistency_loss = torch.zeros(batch_size, device=config.device)

    target_reward_cpu = target_reward.detach().cpu()
    final_indices = indices > total_transitions * 0.95
    gradient_scale = 1 / config.num_unroll_steps
    with autocast():
        for step_i in range(config.num_unroll_steps):
            value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)

            if config.consistency_coeff > 0:
                #will not run here
                #assert False
                _, _, _, presentation_state = model.initial_inference(obs_target_batch[:, beg_index:end_index, :].reshape(batch_size, -1))
                if config.consist_type is 'contrastive':
                    temp_loss = model.contrastive_loss(hidden_state, presentation_state) * mask_batch[:, step_i]
                else:
                    dynamic_proj = model.project(hidden_state, with_grad=True)
                    observation_proj = model.project(presentation_state, with_grad=False)
                    temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
            #print("==================>",reward_loss.shape,reward.shape, target_reward_phi[:, step_i].shape,flush=True)
            reward_loss += config.scalar_reward_loss(reward, target_reward_phi[:, step_i])
            hidden_state.register_hook(lambda grad: grad * 0.5)

            scaled_rewards = config.inverse_reward_transform(reward.detach())

            l1_prior = torch.nn.L1Loss(reduction='none')(scaled_rewards.squeeze(-1), target_reward[:, step_i])
            reward_priority.append(l1_prior.detach().cpu().numpy())
            if vis_result:
                scaled_rewards_cpu = scaled_rewards.detach().cpu()

                predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                # scaled_rewards = config.inverse_reward_transform(reward)
                predicted_rewards.append(scaled_rewards_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                reward_indices_0 = (target_reward_cpu[:, step_i].unsqueeze(-1) == 0)
                reward_indices_n1 = (target_reward_cpu[:, step_i].unsqueeze(-1) == -1)
                reward_indices_1 = (target_reward_cpu[:, step_i].unsqueeze(-1) == 1)

                target_reward_base = target_reward_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_rewards_cpu, target_reward_base)
                if reward_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(scaled_rewards_cpu[reward_indices_1], target_reward_base[reward_indices_1])
                if reward_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(scaled_rewards_cpu[reward_indices_n1], target_reward_base[reward_indices_n1])
                if reward_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(scaled_rewards_cpu[reward_indices_0], target_reward_base[reward_indices_0])

                if final_indices.any():
                    # last 5% data
                    key = '5%_' + key

                    target_reward_cpu_test = target_reward_cpu[final_indices]
                    scaled_rewards_cpu_test = scaled_rewards_cpu[final_indices]
                    target_reward_base_test = target_reward_base[final_indices]

                    reward_indices_0 = (target_reward_cpu_test[:, step_i].unsqueeze(-1) == 0)
                    reward_indices_n1 = (target_reward_cpu_test[:, step_i].unsqueeze(-1) == -1)
                    reward_indices_1 = (target_reward_cpu_test[:, step_i].unsqueeze(-1) == 1)

                    if reward_indices_1.any():
                        other_loss[key + '_1'] = metric_loss(scaled_rewards_cpu_test[reward_indices_1], target_reward_base_test[reward_indices_1])
                    if reward_indices_n1.any():
                        other_loss[key + '_-1'] = metric_loss(scaled_rewards_cpu_test[reward_indices_n1], target_reward_base_test[reward_indices_n1])
                    if reward_indices_0.any():
                        other_loss[key + '_0'] = metric_loss(scaled_rewards_cpu_test[reward_indices_0], target_reward_base_test[reward_indices_0])

    # ----------------------------------------------------------------------------------
    # optimize
    loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss + config.value_loss_coeff * value_loss + config.reward_loss_coeff * reward_loss)

    #loss = ( config.policy_loss_coeff * policy_loss +
     #       config.value_loss_coeff * value_loss + config.reward_loss_coeff * reward_loss)
    #print("==========>loss",weights.shape,loss.shape,flush=True)
    weighted_loss = (weights * loss).mean()

    # L2 reg
    parameters = model.parameters()
    if config.amp_type == 'torch_amp':
        with autocast():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
    else:
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
    optimizer.zero_grad()

    if config.amp_type == 'nvidia_apex':
        # TODO: use torch.cuda.amp
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    elif config.amp_type == 'none':
        total_loss.backward()
    elif config.amp_type == 'torch_amp':
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if config.amp_type == 'torch_amp':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # ----------------------------------------------------------------------------------
    # update priority
    reward_priority = np.mean(reward_priority, 0)
    new_priority = (1 - config.priority_reward_ratio) * value_priority + config.priority_reward_ratio * reward_priority
    replay_buffer.update_priorities.remote(indices, new_priority, make_time)

    # packing data for logging
    loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                 reward_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean())
    if vis_result:
        reward_w_dist, representation_mean, dynamic_mean, reward_mean = model.get_params_mean()
        other_dist['reward_weights_dist'] = reward_w_dist
        other_log['representation_weight'] = representation_mean
        other_log['dynamic_weight'] = dynamic_mean
        other_log['reward_weight'] = reward_mean

        # reward l1 loss
        reward_indices_0 = (target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
        reward_indices_n1 = (target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
        reward_indices_1 = (target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

        target_reward_base = target_reward_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

        predicted_rewards = torch.stack(predicted_rewards).transpose(1, 0).squeeze(-1)
        if final_indices.any():
            predicted_rewards_test = predicted_rewards[final_indices].reshape(-1).unsqueeze(-1)
        predicted_rewards = predicted_rewards.reshape(-1).unsqueeze(-1)
        other_loss['l1'] = metric_loss(predicted_rewards, target_reward_base)
        if reward_indices_1.any():
            other_loss['l1_1'] = metric_loss(predicted_rewards[reward_indices_1], target_reward_base[reward_indices_1])
        if reward_indices_n1.any():
            other_loss['l1_-1'] = metric_loss(predicted_rewards[reward_indices_n1], target_reward_base[reward_indices_n1])
        if reward_indices_0.any():
            other_loss['l1_0'] = metric_loss(predicted_rewards[reward_indices_0], target_reward_base[reward_indices_0])

        if final_indices.any():
            # last 5% data
            target_reward_cpu_test = target_reward_cpu[final_indices]
            target_reward_base_test = target_reward_cpu[final_indices, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

            reward_indices_0 = (target_reward_cpu_test[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
            reward_indices_n1 = (target_reward_cpu_test[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
            reward_indices_1 = (target_reward_cpu_test[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

            other_loss['5%_l1'] = metric_loss(predicted_rewards_test, target_reward_base_test)
            if reward_indices_1.any():
                other_loss['5%_l1_1'] = metric_loss(predicted_rewards_test[reward_indices_1], target_reward_base_test[reward_indices_1])
            if reward_indices_n1.any():
                other_loss['5%_l1_-1'] = metric_loss(predicted_rewards_test[reward_indices_n1], target_reward_base_test[reward_indices_n1])
            if reward_indices_0.any():
                other_loss['5%_l1_0'] = metric_loss(predicted_rewards_test[reward_indices_0], target_reward_base_test[reward_indices_0])

        td_data = (new_priority, target_reward.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                   transformed_target_reward.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                   target_reward_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                   predicted_rewards.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                   target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst,
                   other_loss, other_log, other_dist)
        priority_data = (weights, indices)
    else:
        td_data, priority_data = _, _

    return loss_data, td_data, priority_data, scaler


def consist_loss_func(f1, f2):
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count, scheduler):
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if config.lr_type is 'cosine':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
        else:
            lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    return lr


def add_batch(batch, m_batch):
    # obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices_lst, weights_lst, make_time
    for i, m_bt in enumerate(m_batch):
        batch[i].append(m_bt)


class BatchStorage(object):
    def __init__(self, threshold=15, size=20):#8,16
        self.threshold = threshold
        self.batch_queue = Queue(maxsize=size)

    def push(self, batch):
        if self.batch_queue.qsize() <= self.threshold:
            self.batch_queue.put(batch)
        else:
            pass
            #print("full",flush=True)

    def pop(self):
        if self.batch_queue.qsize() > 0:
            return self.batch_queue.get()
        else:
            return None

    def get_len(self):
        return self.batch_queue.qsize()
    def is_full(self):
        if self.get_len()>=self.threshold:
            print("full",flush=True)
            return True
        else:
            return False
        #print("full!",flush=True)
        #return self.get_len()>=self.threshold

@ray.remote(num_gpus=gpu_num,num_cpus=1)
class BatchWorker(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, config):
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.batch_storage = batch_storage
        self.config = config

        if config.amp_type == 'nvidia_apex':
            self.target_model = amp.initialize(config.get_uniform_network().to(self.config.device))
        else:
            self.target_model = config.get_uniform_network()
            self.target_model.to('cuda')
        self.target_model.eval()

        self.last_model_index = -1
        self.batch_max_num = 40#from 20
        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps, initial_p=config.priority_prob_beta, final_p=1.0)

    def run(self):
        start = False
        # print("batch worker initialize",flush=True)
        while True:
            # wait for starting
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(1)
                # print("batch worker waiting replay_buffer.get_total_len.remote()) >= config.start_window_size",flush=True)

                continue
            # print("batch worker starting run",flush=True)
            # TODO: use latest weights for policy reanalyze
            ray_data_lst = [self.storage.get_counter.remote(), self.storage.get_target_weights.remote()]
            trained_steps, target_weights = ray.get(ray_data_lst)

            beta = self.beta_schedule.value(trained_steps)
            batch_context = ray.get(self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta))

            #@wjc
            game_lst, game_pos_lst, _, _, _=batch_context
            #temporarily save the obs list


            # print("batch worker finished prepare batch context",flush=True)
            # break
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                # print("batchworker finished working",flush=True)
                break

            new_model_index = trained_steps // self.config.target_model_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
            else:
                target_weights = None

            if self.batch_storage.get_len() < self.batch_max_num:
                #should be zero as no batch is pushed
                # print("batch storage size={0}/20 ".format(self.batch_storage.get_len()),flush=True)
                try:
                    batch = self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights, batch_num=2)
                    # print("batch worker finish makeing batch, start to push",flush=True)
                    #if self.batch_storage.is_full():
                     #   print("{} is sleeping, buffer={}".format(self.worker_id,self.batch_storage.get_len()),flush=True)
                     #   time.sleep(15)
                    self.batch_storage.push(batch)
                    #time.sleep(randint(1,4))
                except:
                    print('=====================>Data is deleted...')
                    #assert False

    def split_context(self, batch_context, split_num):#game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_context_lst = []

        context_num = len(batch_context)
        batch_size = len(batch_context[0])
        split_size = batch_size // split_num

        assert split_size * split_num == batch_size

        for i in range(split_num):
            beg_index = split_size * i
            end_index = split_size * (i + 1)

            _context = []
            for j in range(context_num):
                _context.append(batch_context[j][beg_index:end_index])

            batch_context_lst.append(_context)

        return batch_context_lst

    def concat_batch(self, batch_lst):
        # print("4",flush=True)
        batch = [[] for _ in range(8 + 1)]

        for i in range(len(batch_lst)):
            for j in range(len(batch_lst[0])):
                if i == 0:
                    batch[j] = batch_lst[i][j]
                else:
                    batch[j] = np.concatenate((batch[j], batch_lst[i][j]), axis=0)
        # print("5",flush=True)
        return batch

    def make_batch(self, batch_context, ratio, weights=None, batch_num=2):
        # print("1",flush=True)
        batch_context_lst = self.split_context(batch_context, batch_num)#game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        # print("2",flush=True)
        batch_lst = []
        for i in range(len(batch_context_lst)):
            # print("start making batch = ",i,flush=True)
            # print("batch_context_lst size=",len(batch_context_lst),flush=True)

            batch_lst.append(self._make_batch(batch_context_lst[i], ratio, weights))
            # print("finish making batch = ",i,len(batch_context_lst),flush=True)
            # print("2."+str(i),flush=True)
        # print("3",flush=True)
        return self.concat_batch(batch_lst)

    def _make_batch(self, batch_context, ratio, weights=None):
        if weights is not None:
            self.target_model.set_weights(weights)
            self.target_model.to('cuda')
            self.target_model.eval()
            # print('weight is not none! change weights!',flush=True)
        # print("m1",flush=True)s
        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        # print("in batchwoker's make_batch,(64/128?)batchsize=",batch_size)
        obs_lst, action_lst, mask_lst = [], [], []
        # print("m2",flush=True)
        for i in range(batch_size):
            # print("in _make_batch, ",i,flush=True)
            game = game_lst[i]
            # print("00",flush=True)
            game_pos = game_pos_lst[i]
            # print("11",flush=True)
            _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
            # print("22",flush=True)
            _mask = [1. for i in range(len(_actions))]
            # print("33",flush=True)
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]
            # print("44",flush=True)
            _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]
            # print("55",flush=True)
            #try:
                # game_lst[i]
                # print("game_lst-i ",game_lst[i],flush=True)
                # print()
            obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))#<===== problem
            #except:
            #    assert False

            # print("66",flush=True)
            action_lst.append(_actions)
            mask_lst.append(_mask)
        # print("m3",flush=True)
        re_num = int(batch_size * ratio)#ratio is config.revisit_policy_search_rate
        obs_lst = prepare_observation_lst(obs_lst, image_based=self.config.image_based)
        batch = [obs_lst, action_lst, mask_lst, [], [], [], indices_lst, weights_lst, make_time_lst]
        # print("m4",flush=True)
        if self.config.reanalyze_part == 'paper':
            # value + policy (reanalyze part)
            # print("start reanalyze",flush=True)
            re_value, re_reward, re_policy = prepare_multi_target(self.replay_buffer, indices_lst[:re_num],
                                                                  make_time_lst[:re_num],
                                                                  game_lst[:re_num], game_pos_lst[:re_num],
                                                                  self.config, self.target_model)
            batch[3].append(re_reward)
            batch[4].append(re_value)
            batch[5].append(re_policy)
            # print("ffinish multi target",flush=True)
            # only value
            if re_num < batch_size:
                re_value, re_reward, re_policy = prepare_multi_target_only_value(game_lst[re_num:], game_pos_lst[re_num:],
                                                                                 self.config, self.target_model)
                batch[3].append(re_reward)
                batch[4].append(re_value)
                batch[5].append(re_policy)
        elif self.config.reanalyze_part == 'none':
            #print("gg",flush=True)
            re_value, re_reward, re_policy = prepare_multi_target_none(game_lst[:], game_pos_lst[:],
                                                                                 self.config, self.target_model)
            batch[3].append(re_reward)
            batch[4].append(re_value)
            batch[5].append(re_policy)
        else:
            assert self.config.reanalyze_part == 'all'
            re_value, re_reward, re_policy = prepare_multi_target(self.replay_buffer, indices_lst, make_time_lst,
                                                                  game_lst, game_pos_lst, self.config,
                                                                  self.target_model)
            batch[3].append(re_reward)
            batch[4].append(re_value)
            batch[5].append(re_policy)
        # print("finish reanalyze,len(batch)=",len(batch),flush=True)
        for i in range(len(batch)):
            if i in range(3, 6):
                batch[i] = np.concatenate(batch[i])
            else:
                batch[i] = np.asarray(batch[i])

        # weights_lst = batch[-2]
        # weight_max = np.percentile(weights_lst, 90)
        # weights_lst = (weights_lst / weight_max).clip(0., 1.0)
        # batch[-2] = weights_lst
        # print("start returning batch", flush=True)
       # for i in batch:
       #     print("batch shape",i.shape,flush=True,end="")
        return batch


def _train(model, target_model, latest_model, config, shared_storage, replay_buffer, batch_storage, summary_writer):

    # ----------------------------------------------------------------------------------
    model = model.to(config.device)
    target_model = target_model.to(config.device)

    model.train()
    target_model.eval()
    latest_model.eval()

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)

    if config.amp_type == 'nvidia_apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level)
    scaler = GradScaler()
    # ----------------------------------------------------------------------------------

    if config.lr_type is 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.training_steps + config.last_steps - config.lr_warm_step)
    else:
        scheduler = None

    if config.use_augmentation:
        config.set_transforms()

    # wait for all replay buffer to be non-empty
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_window_size):
        print("waiting in _train,buffer size ={0} /{1}".format(ray.get(replay_buffer.get_total_len.remote()),config.start_window_size),flush=True)
        time.sleep(1)
        pass
    print('in _train, Begin training...')
    shared_storage.set_start_signal.remote()

    step_count = 0
    batch_count = 0
    make_time = 0.
    lr = 0.

    recent_weights = model.get_weights()
    time_100k=time.time()
    _interval=config.debug_interval
    _debug_batch=config.debug_batch
    while step_count < config.training_steps + config.last_steps:

        # @profile
        # def f(batch_count, step_count, lr, make_time):
        if step_count % 500 == 0:#@wjc changed to 100 for debugging
            replay_buffer.remove_to_fit.remote()
        # while True:
        #@wjc
        #if step_count%100==0:
            #print("in _train,step={2}: step_count={0}, batch storage size={1} ".format(step_count, batch_storage.get_len(),step_count),flush=True)

        #    print("in _train,step={}:  ".format(step_count),flush=True)

        batch = batch_storage.pop()
        # before_btch=time.time()
        if batch is None:
            time.sleep(0.1)#0.3->2
            #print("_train(): waiting batch storage,step/current batch storage_Size=",step_count,batch_storage.get_len(),config.debug_batch,flush=True)
            if _debug_batch:
                #print("_train(): waiting batch storage,step=",step_count,flush=True)
                print("LEARNER WAITING!",flush=True)
            continue
        # print("making one batch takes: ", time.time()-before_btch,flush=True)
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count, scheduler)

        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())

        # if config.target_moving_average:
        #     tau = 1 / config.target_model_interval
        #     soft_update(target_model, model, tau=tau)
        #     target_model.eval()
        #     shared_storage.set_target_weights.remote(target_model.get_weights())
        # else:
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights)
            recent_weights = model.get_weights()

        if config.use_latest_model:
            soft_update(latest_model, model.detach(), tau=1)
            shared_storage.set_latest_model.remote(latest_model.get_weights())

        if step_count % config.vis_interval == 0:
            vis_result = True
        else:
            vis_result = False

        if config.amp_type == 'torch_amp':
            if step_count >= 1:
                scaler = scaler_prev
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, True)
            scaler_prev = log_data[3]
        else:
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result)

        if step_count % config.log_interval == 0:
            _log(config, step_count, log_data[0:3], model, replay_buffer, lr, shared_storage, summary_writer, vis_result)

        step_count += 1
     #   if config.debug_batch:
    #        _interval=config.debug_interval

        if step_count%_interval==0:
        #if step_count%1==0:
           # print("===========>100 lr step, cost [{}] s, buffer = {}".format(time.time()-time_100k,ray.get(replay_buffer.get_total_len.remote())),flush=True)
            _time=time.time()-time_100k
            print("===========>{} lr step, cost [{}] s; <==>[{}] s/100lr".format(_interval,_time,_time/(_interval/500)),flush=True)
            time_100k=time.time()
        #    if step_count % 1000 ==0:
        #        print("buffer transitions=",ray.get(replay_buffer.get_total_len.remote()),flush=True)
        # if(step_count%50==0):
        #     _test(config, shared_storage)
       # if step_count % 100000==0:
       #    replay_buffer.save_files.remote()
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(model.get_weights())
    return model.get_weights()


@ray.remote(num_gpus=gpu_num)#@wjc, changed to 0.25
def _test(config, shared_storage):
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
    episodes = 0
    while True:
        counter = ray.get(shared_storage.get_counter.remote())
        if counter >= config.training_steps + config.last_steps:
            break
        if counter >= config.test_interval * episodes:
            episodes += 1
            test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            test_model.eval()

            test_score, _ = test(config, test_model, counter, config.test_episodes, 'cuda', False, save_video=False)
            mean_score = sum(test_score) / len(test_score)
            if mean_score >= best_test_score:
                best_test_score = mean_score
                torch.save(test_model.state_dict(), config.model_path)

            shared_storage.add_test_log.remote(test_score)
        #print("============================>Sleeping Test!")
        time.sleep(120)
        #print("============================>Waking up Test!")


def train(config, summary_writer=None, model_path=None):
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    latest_model = config.get_uniform_network()
    #assert model_path is not None
    if model_path:
        print('>>>>>>>>>>>>>>>>resume model from path: ', model_path,flush=True)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)
        latest_model.load_state_dict(weights)

    storage = SharedStorage.remote(model, target_model, latest_model)

    batch_storage = BatchStorage(30, 50)

    replay_buffer = ReplayBuffer.remote(replay_buffer_id=0, config=config)

    #batch_actor=5
    batch_workers = [BatchWorker.remote(idx, replay_buffer, storage, batch_storage, config)
                     for idx in range(config.batch_actor)]#by wjc

    # self-play
    #num_actors=2
    workers = [DataWorker.remote(rank, config, storage, replay_buffer) for rank in range(config.num_actors)] #changed to 1 actor
    workers = [worker.run_multi.remote() for worker in workers]
    # ray.get(replay_buffer.random_init_trajectory.remote(200))

    # batch maker
    workers += [batch_worker.run.remote() for batch_worker in batch_workers]
    # time.sleep(5)
    # storage.set_start_signal.remote()
    # for batch_worker in batch_workers:
        # print("launch batch")
        # batch_worker.run()
    # while True:
        # print("batch storage size ",(batch_storage.get_len()),flush=True)
        # time.sleep(2)
    # test
    workers += [_test.remote(config, storage)]
    # train
    final_weights = _train(model, target_model, latest_model, config, storage, replay_buffer, batch_storage, summary_writer)
    # wait all
    ray.wait(workers)

    return model, final_weights

def test_mcts(config, summary_writer=None, model_path=None):
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    latest_model = config.get_uniform_network()
    if model_path:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)
        latest_model.load_state_dict(weights)

    storage = SharedStorage.remote(model, target_model, latest_model)

    batch_storage = BatchStorage(8, 16)

    replay_buffer = ReplayBuffer.remote(replay_buffer_id=0, config=config)

    batch_workers = [BatchWorker.remote(idx, replay_buffer, storage, batch_storage, config)
                     for idx in range(config.batch_actor)]

    dw=DataWorker(0, config, storage, replay_buffer).run_multi()
    # # self-play
    # workers = [DataWorker.remote(rank, config, storage, replay_buffer) for rank in range(0, config.num_actors)]
    # workers = [worker.run_multi.remote() for worker in workers]
    # # ray.get(replay_buffer.random_init_trajectory.remote(200))

    # # batch maker
    # workers += [batch_worker.run.remote() for batch_worker in batch_workers]
    # # test
    # # workers += [_test.remote(config, storage)]
    # # train
    # final_weights = _train(model, target_model, latest_model, config, storage, replay_buffer, batch_storage, summary_writer)
    # # wait all
    # ray.wait(workers)

    return model, final_weights

def super(config, data_path, summary_writer=None):
    storage = SharedStorage.remote(config.get_uniform_network())
    assert (config.batch_size // config.replay_number * config.replay_number) == config.batch_size
    replay_buffer_lst = [
        ReplayBuffer.remote(num_replays=config.replay_number, config=config, replay_buffer_id=i, make_dataset=True) for
        i in range(config.replay_number)]

    ray.get([replay_buffer.load_files.remote(data_path) for replay_buffer in replay_buffer_lst])

    _train(config, storage, replay_buffer_lst, summary_writer, None)

    return config.get_uniform_network().set_weights(ray.get(storage.get_weights.remote()))


def make_dataset(config, model):
    storage = SharedStorage.remote(config.get_uniform_network())
    storage.set_weights.remote(model.get_weights())
    replay_buffer_lst = [
        ReplayBuffer.remote(num_replays=config.replay_number, config=config, make_dataset=True, replay_buffer_id=i) for
        i in range(config.replay_number)]

    workers = [DataWorker.remote(rank, config, storage, replay_buffer_lst[rank % config.replay_number]) for rank in
               range(0, config.num_actors)]
    workers = [worker.run_multi.remote() for worker in workers]

    ray.wait(workers)

    data_workers = [replay_buffer.save_files.remote() for replay_buffer in replay_buffer_lst]
    ray.get(data_workers)
