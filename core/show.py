import ray
import torch

import numpy as np

from tqdm import tqdm
from core.train import add_batch, concat_batch
from core.replay_buffer import ReplayBuffer


@ray.remote
def make_batch(replay_buffer_lst, beg_index, random=True):
    if random:
        result_ids = [replay_buffer.sample_batch_testing.remote(idx) for (idx, replay_buffer) in enumerate(replay_buffer_lst)]
    else:
        result_ids = [replay_buffer.sample_batch_in_order.remote(idx, beg_index) for (idx, replay_buffer) in enumerate(replay_buffer_lst)]

    batch = [[] for _ in range(9 + 1)]
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        m_batch, replay_id = ray.get(done_id[0])
        assert m_batch is not None

        add_batch(batch, m_batch)
        batch[-1].append([replay_id])

    concat_batch(batch)
    return batch


def log():
    pass


@torch.no_grad()
def test_batch(model, batch, config):
    obs_batch_ori, action_batch, mask_batch, target_reward, target_value, target_policy, _, weights, make_time, replay_id = batch

    obs_batch = torch.from_numpy(obs_batch_ori).to(config.device).float()
    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    # mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_reward = torch.from_numpy(target_reward).to(config.device).float()
    # target_value = torch.from_numpy(target_value).to(config.device).float()
    # target_policy = torch.from_numpy(target_policy).to(config.device).float()
    # weights = torch.from_numpy(weights).to(config.device).float()

    # batch_size = obs_batch.size(0)

    # transformed_target_reward = config.scalar_transform(target_reward)
    # target_reward_phi = config.reward_phi(transformed_target_reward)

    # transformed_target_value = config.scalar_transform(target_value)
    # target_value_phi = config.value_phi(transformed_target_value)

    value, _, policy_logits, hidden_state = model.initial_inference(obs_batch[:, 0:config.stacked_observations * 3, :, :])
    reward_l1_error = None

    for step_i in range(1):
        value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])

        beg_index = 3 * (step_i + 1)
        end_index = 3 * (step_i + 1 + config.stacked_observations)
        _, _, _, presentation_state = model.initial_inference(obs_batch[:, beg_index:end_index, :, :])

        scaled_rewards = config.inverse_reward_transform(reward.detach())

        reward_l1_error = torch.nn.L1Loss(reduction='none')(scaled_rewards, target_reward[:, step_i].unsqueeze(1))

    return reward_l1_error.detach(), obs_batch_ori


def show(config, data_path, summary_writer):
    replay_buffer_lst = [ReplayBuffer.remote(num_replays=config.replay_number, config=config, replay_buffer_id=i, make_dataset=True) for i in range(config.replay_number)]

    ray.get([replay_buffer.load_files.remote(data_path) for replay_buffer in replay_buffer_lst])

    total_lens = ray.get(replay_buffer_lst[0].get_total_len.remote())
    replay_batch = config.batch_size // config.replay_number

    model = config.get_uniform_network().to(config.device)
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cuda')))
    model.eval()

    # not drop last
    # batch_num = np.ceil(total_lens / replay_batch).astype(np.int)
    batch_num = 20
    reward_error_array = None
    obs_array = []


    # test all batch
    for beg_index in tqdm(range(0, batch_num)):

        result_ids = [make_batch.remote(replay_buffer_lst, beg_index, random=True) for _ in range(2)]  # config.batch_buffer
        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            batch = ray.get(done_id[0])
            assert batch is not None

            reward_l1_error, obs_batch = test_batch(model, batch, config)

            if reward_error_array is None:
                reward_error_array = reward_l1_error
            else:
                reward_error_array = torch.cat((reward_error_array, reward_l1_error))
            obs_array.append(obs_batch)

    # top 100 images
    topK = 10

    reward_error_array = reward_error_array.squeeze(-1)

    top_errors, top_indices = torch.topk(reward_error_array, topK)

    # log data
    print('Logging images...')
    rank = 0
    for (error, indice) in zip(top_errors, top_indices):
        # replay_id = replay_id_array[indice]

        one_idx = indice // config.batch_size
        two_idx = indice % config.batch_size

        stack_observations = obs_array[one_idx][two_idx]

        concat_observations = None

        for step in range(9):
            beg_index = 3 * step
            end_index = 3 * (step + 1)

            observation = stack_observations[beg_index:end_index, :, :]
            if concat_observations is None:
                concat_observations = observation
            else:
                concat_observations = np.concatenate((concat_observations, observation), 1)

        concat_observations = torch.from_numpy(concat_observations)
        tag = 'top {} of {} samples, error={:.4f}'.format(rank, config.batch_size * batch_num, error)
        summary_writer.add_image(tag, concat_observations)

        rank += 1
