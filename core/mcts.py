import math
import random
import numpy as np
import torch
import core.ctree.cytree as tree
from core.vis import DrawNode, DrawTree
from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def run_multi(self, roots, model, hidden_state_roots):

        with torch.no_grad():
            model.eval()

            num = roots.num
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            #print("batchworker, hidden state root: ", np.array(hidden_state_roots).shape,flush=True)
            hidden_state_pool = [hidden_state_roots]
            hidden_state_index_x = 0
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)

            # DrawNode.clear()
            # d_root = DrawNode(0)
            # draw_tree = DrawTree(d_root)
            total_simulation=self.config.num_simulations
            for index_simulation in range(self.config.num_simulations):
                if (index_simulation == total_simulation-1):
                    continue
                hidden_states = []
                results = tree.ResultsWrapper(num)
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    #print("{},{},pool={}".format(ix,iy,np.array(hidden_state_pool).shape),flush=True)
                    hidden_states.append(hidden_state_pool[ix][iy])
                hidden_states = np.asarray(hidden_states)
                #hidden_nan=np.isnan(hidden_states)
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
                #print(type(policy_logits_pool),flush=True)
                nan_part = np.isnan(policy_logits_pool)
                #if nan_part.any():
                    #there's problem is disable here, when no-reanalyze
                    #print("simulation={},node_simluation_cnt={},node_parallel_cnt={}".format(hidden_state_index_x,hidden_state_index_x_lst,hidden_state_index_y_lst),flush=True)
                    #print("simulation={}".format(hidden_state_index_x),flush=True)
                    #print("hidden_state_pool shape=",np.array(hidden_state_pool).shape,flush=True)
                    #print('=========>mcts,simulation={},[ERROR]: NAN in policy scalar!!!'.format(hidden_state_index_x), flush=True)
                    #if hidden_nan.any():
                    #    print("=============>hidden nan",flush=True)
                    #breakpoint()
                    #print('=========>mcts,[ERROR]: NAN in scalar!!!, last_action={}'.format(last_actions),flush=True)
                #    pass
                policy_logits_pool[nan_part] = 0.0
                #print("policy logits pool shape",policy_logits_pool[0].shape,flush=True)
                policy_logits_pool = policy_logits_pool.tolist()
                # print('predicted reward: {} and value: {}'.format(reward_pool, value_pool))

                hidden_state_pool.append(hidden_state_nodes)
                hidden_state_index_x += 1

                tree.multi_back_propagate(hidden_state_index_x, discount,
                                          reward_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results)
                # print(roots.get_distributions())
                # print(roots.get_values())
            #     trajs = roots.get_trajectories()
            #     print(trajs[0])
            #     d_root.add_traj(trajs[0])
            #     draw_tree.build()
            #
            # import ipdb
            # ipdb.set_trace()
            # draw_tree.make_video()


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

    # print(depth_lst)
    # assert (np.array(depth_lst) > 0).all()
    return depth_lst, visit_count_lst
