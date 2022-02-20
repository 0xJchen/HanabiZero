import os
from unittest import mock

import torch
import numpy as np

from .mcts import MCTS
import core.ctree.cytree as cytree
from .utils import select_action, prepare_observation_lst, set_seed
from .game import GameHistory
import multiprocessing
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from core.py_mcts import MCTS as py_MCTS
from core.py_mcts import Node as py_Node


def test(config, model, counter, test_episodes, device, render, save_video=False, final_test=False):
    #print("start testing!!!",flush=True)
    test_episodes=100
    model.to(device)
    model.eval()


    # torch.set_printoptions(profile='full')
    # mock_next_state=torch.load('/home/game/revert/lsy/lst_best_confirm_copy/HanabiZero/72/next_state_41')
    # mock_next_state=mock_next_state.to(device)
    # # for parallel_idx in range(mock_next_state.shape[0]):
    #     # if torch.isnan(mock_next_state[parallel_idx]).any():
    # parallel_idx=41
    # mk=mock_next_state[parallel_idx].float().unsqueeze(0)

    # # mock_reward=model._dynamics_reward(mk)
    # l1=model._dynamics_reward[0](mk)
    # bn=model._dynamics_reward[1](l1)
    # relu=model._dynamics_reward[2](bn)
    # mock_reward=model._dynamics_reward[3](relu)
    # if torch.isnan(mock_reward).any():
    #     print(parallel_idx)
    #     print("next state=",torch.isnan(mk).any(),flush=True)
    #     print("mock reward=",torch.isnan(mock_reward).any(),flush=True)
    #     print('state',mk)
    #     print('reward',mock_reward)
    #     print('l1',l1)
    #     print('bn',bn)
    #     print('relu',relu)
    #     print('l2',mock_reward)



    # assert False

    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))


    with torch.no_grad():
        envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                              video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]

        # init_obses = [env.reset() for env in envs]
        init_obses=[]
        init_legal_actions=[]
        for env in envs:
            o,l_a=env.reset()
            init_obses.append(o)
            init_legal_actions.append(l_a)

        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(envs[_].env.action_space, max_length=config.max_moves, config=config) for
            _ in
            range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)],init_legal_actions[i])

        step = 0
        ep_step=[0 for I in range(test_episodes)]
        ep_ori_rewards = [0 for _ in range(test_episodes)]
        ep_clip_rewards = [0 for _ in range(test_episodes)]
        ep_final_rewards = [0 for _ in range(test_episodes)]
        while not dones.all():
            if render:
                for i in range(test_episodes):
                    envs[i].render()

            stack_obs = [game_history.step_obs() for game_history in game_histories]

            #@wjc
            stack_legal_actions=[game_history.legal_actions[-1] for game_history in game_histories]

            if config.image_based:
                stack_obs = prepare_observation_lst(stack_obs)
                # stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
                stack_obs = torch.from_numpy(stack_obs).to(device).float()
            else:
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device).reshape(test_episodes, -1)

            network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state
            reward_pool = network_output.reward
            policy_logits_pool = network_output.policy_logits.tolist()

            roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations )
            roots.prepare_no_noise(reward_pool, policy_logits_pool,stack_legal_actions)

            MCTS(config).run_multi(roots, model, hidden_state_roots)

            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            # print('===>[test]---step {}---'.format(step),flush=True)
            for i in range(test_episodes):
                if dones[i]:
                    #print("===>I finished testing!",flush=True)
                    continue

                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]

                action, _ = select_action(distributions, temperature=1, deterministic=True,legal_actions=stack_legal_actions[i])

                obs, ori_reward, done, info, legal_a = env.step(int(action))
                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward,legal_a)

                dones[i] = done
                if dones[i]:
                    #print("done in one environment, final reward=",ori_reward)
                    ep_final_rewards[i]=info.item()['score']
                if not dones[i]:
                    ep_ori_rewards[i] += ori_reward
                    ep_clip_rewards[i] += clip_reward
                ep_step[i]+=1
            step += 1

    #    for i in range(test_episodes):
    #        print('===========>Test episode {}:  reward={}, step={}, final_reward={}'.format(i, ep_ori_rewards[i],ep_step[i],ep_final_rewards[i]),flush=True)

    #    print(ep_final_rewards,ep_ori_rewards)
        env.close()
    return ep_final_rewards, save_path
