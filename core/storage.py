
import ray
from ray.util.queue import Queue
import numpy as np

class BatchStorage(object):
    def __init__(self, threshold=15, size=20,name=''):#8,16
        self.threshold = threshold
        self.batch_queue = Queue(maxsize=size)
        self.name=name

    def push(self, batch):
        if self.batch_queue.qsize() <= self.threshold:
            self.batch_queue.put(batch)
        else:
            pass
            #print(self.name+"full",flush=True)

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

