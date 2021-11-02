import math
import random
import numpy as np
import torch
# from random import sample
# from graphviz import Digraph

# colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']
# colors = ['skyblue']


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # if value >= self.maximum:
            #     return self.maximum
            # elif value <= self.minimum:
            #     return self.minimum
            # We normalize only when we have set the maximum and minimum values.
            # results better
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.
        self.children = {}
        self.hidden_state = None
        self.reward = 0.
        self.tag = str(random.randint(0, 1000000))

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, network_output, idx=None):
        if idx is not None:
            self.hidden_state = np.expand_dims(network_output.hidden_state[idx], 0)
            self.reward = network_output.reward[idx]
            policy = {a: math.exp(network_output.policy_logits[idx][a]) for a in actions}
        else:
            self.hidden_state = network_output.hidden_state
            self.reward = network_output.reward[0][0]
            # softmax over policy logits
            policy = {a: math.exp(network_output.policy_logits[0][a]) for a in actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)

    def get_distribution(self):
        distribution = []
        for a, child in self.children.items():
            distribution.append(child.visit_count)
        return distribution

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def run_multi(self, roots, action_histories, model):
        num = len(roots)
        min_max_stats_lst = [MinMaxStats() for i in range(num)]

        for _ in range(self.config.num_simulations):
            search_paths = []
            hidden_states = []
            last_actions = []
            nodes = []
            histories = []

            for i in range(num):
                history = action_histories[i].copy()
                node = roots[i]
                min_max_stats = min_max_stats_lst[i]
                search_path = [node]
                search_paths.append(search_path)

                while node.expanded():
                    action, node = self.select_child(node, min_max_stats)
                    history.append(action)
                    search_path.append(node)

                parent = search_path[-2]
                hidden_states.append(parent.hidden_state.squeeze(0))
                last_actions.append([history[-1]])
                nodes.append(node)
                histories.append(history)

            hidden_states = torch.from_numpy(np.array(hidden_states)).to('cuda')

            last_actions = torch.from_numpy(np.array(last_actions)).to('cuda')

            network_output = model.recurrent_inference(hidden_states, last_actions)

            for i in range(num):
                node = nodes[i]
                history = histories[i]
                search_path = search_paths[i]
                min_max_stats = min_max_stats_lst[i]

                node.expand(np.arange(self.config.action_space_size), network_output, i)

                self.backpropagate(search_path, network_output.value[i].item(), min_max_stats)

    def select_child(self, node, min_max_stats):
        _, action, child = max((self.ucb_score(node, child, min_max_stats), action, child)
                               for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent, child, min_max_stats) -> float:
        pb_c = math.log(
            (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.discount * value

    def search_path_to_str(self, path):
        info = '**********\t Search Path \t**********\n'
        for node in path:
            info += str(node) + '\n'

        info += '**********\t Search Path Over \t**********\n'
        return info