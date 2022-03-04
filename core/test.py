import os
import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst
from .device import test_gpu as gpu_num

@ray.remote(num_gpus=gpu_num)
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
        time.sleep(180)



def test(config, model, counter, test_episodes, device, render, save_video=False, final_test=False):
    test_episodes=1000
    model.to(device)
    model.eval()
    s_t=time.time()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))


    with torch.no_grad():
        envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                              video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]
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
            for i in range(test_episodes):
                if dones[i]:
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
                    ep_final_rewards[i]=info.item()['score']
                if not dones[i]:
                    ep_ori_rewards[i] += ori_reward
                    ep_clip_rewards[i] += clip_reward
                ep_step[i]+=1
            step += 1

        env.close()
    print("finish in {}s".format(time.time()-s_t),flush=True)
    return ep_final_rewards, save_path

