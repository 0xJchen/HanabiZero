import numpy as np
import torch
import core.ctree.cytree as tree
from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def run_multi(self, roots, model, hidden_state_roots):

        with torch.no_grad():
            model.eval()

            num = roots.num
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            hidden_state_pool = [hidden_state_roots]
            hidden_state_index_x = 0
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)

            total_simulation=self.config.num_simulations
            for index_simulation in range(self.config.num_simulations):
                if (index_simulation == total_simulation-1):
                    continue
                hidden_states = []
                results = tree.ResultsWrapper(num)
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                hidden_states = np.asarray(hidden_states)
                hidden_states = torch.from_numpy(hidden_states)
                hidden_states = hidden_states.to('cuda')
                last_actions = torch.from_numpy(np.asarray(last_actions)).to('cuda').unsqueeze(1).long()
                
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states, last_actions)
                else:
                    network_output = model.recurrent_inference(hidden_states, last_actions)

                hidden_state_nodes = network_output.hidden_state
                reward_pool = network_output.reward.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits
                nan_part = np.isnan(policy_logits_pool)
                policy_logits_pool[nan_part] = 0.0
                policy_logits_pool = policy_logits_pool.tolist()

                hidden_state_pool.append(hidden_state_nodes)
                hidden_state_index_x += 1

                tree.multi_back_propagate(hidden_state_index_x, discount,
                                          reward_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results)



def get_node_distribution(root):
    depth_lst = []
    visit_count_lst = []

    # bfs
    node_stack = [root]
    while len(node_stack) > 0:
        node = node_stack.pop()

        if node.is_leaf():
            depth_lst.append(node.depth)
            visit_count_lst.append(node.visit_count)

        for action, child in node.children.items():
            if child.visit_count > 0:
                node_stack.append(child)
    return depth_lst, visit_count_lst
