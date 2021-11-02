import torch
import numpy as np
from .replay_buffer import ReplayBuffer


def supervised_train(config, data_path=None):
    replay_buffer = ReplayBuffer(config=config)
    replay_buffer.load_files(data_path)

    replay_buffer.sample_batch_efficient()