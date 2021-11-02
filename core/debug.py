import torch
import numpy as np
import random
from tqdm import tqdm

from config.atari import muzero_config
import core.ctree.cytree as cytree
from core.mcts import MCTS as c_MCTS
from core.py_mcts import MCTS as py_MCTS
from core.py_mcts import Node as py_Node


class Config:
    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.997
        self.num_simulations = 50
        self.action_space = 6
        self.root_num = 32
        self.batch_size = 128


def set_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > (2**32 - 1):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def test(model, config, seed=0):
    root_num = config.root_num
    # ================================================================================================================
    # prepare data
    set_seed(seed)

    batch_size = config.batch_size
    obs_batch = torch.from_numpy(np.random.random((batch_size, 3 * 4, 96, 96))).cuda().float()

    network_output = model.initial_inference(obs_batch)
    reward_pool = network_output.reward
    policy_logits_pool = network_output.policy_logits.tolist()
    hidden_state_roots = network_output.hidden_state

    histories_len, action_histories = [], []
    for i in range(root_num):
        h_len = random.randint(0, 200)
        histories_len.append(h_len)

        history = []
        for _ in range(h_len):
            action = random.randint(0, config.action_space - 1)
            history.append(action)
        action_histories.append(history)
    # ================================================================================================================

    # ================================================================================================================
    # for py mcts
    set_seed(seed)
    py_roots = [py_Node(0) for _ in range(root_num)]
    for i in range(root_num):
        root = py_roots[i]
        root.expand(np.arange(config.action_space), network_output, i)

    py_MCTS(config).run_multi(py_roots, action_histories, model)
    py_roots_distributions, py_roots_values = [], []
    for root in py_roots:
        py_roots_distributions.append(root.get_distribution())
        py_roots_values.append(root.value())
    # ================================================================================================================

    # ================================================================================================================
    # for c mcts
    set_seed(seed)
    c_roots = cytree.Roots(root_num, config.action_space, config.num_simulations)
    c_roots.prepare_no_noise(reward_pool, policy_logits_pool)

    c_MCTS(config).run_multi(c_roots, histories_len, action_histories, model, hidden_state_roots)

    c_roots_distributions = c_roots.get_distributions()
    c_roots_values = c_roots.get_values()
    # ================================================================================================================

    assert (np.array(py_roots_distributions) == np.array(c_roots_distributions)).all()
    assert (np.array(py_roots_values) == np.array(c_roots_values)).all()


def debug_c_MCTS(muzero_config):
    config = Config()
    model = muzero_config.get_uniform_network().to('cuda')
    model.eval()

    test_times = 1000
    with torch.no_grad():
        for seed in tqdm(range(test_times)):
            test(model, config, seed=seed)


def debug(muzero_config):
    debug_c_MCTS(muzero_config)
