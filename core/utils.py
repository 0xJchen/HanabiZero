import logging
import os
import random
import shutil
import numpy as np
import cv2
import gym
from baselines.common.atari_wrappers import NoopResetEnv, TimeLimit
from scipy.stats import entropy
import torch
import torch.nn as nn


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# class MaxAndSkipEnv(gym.Wrapper):
#     def __init__(self, env, skip=4, use_max=True):
#         """Return only every `skip`-th frame"""
#         gym.Wrapper.__init__(self, env)
#         # most recent raw observations (for max pooling across time steps)
#         self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
#         self._skip       = skip
#         self._use_max    = use_max
#         self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

#     def step(self, action):
#         """Repeat action, sum reward, and max over last observations."""
#         total_reward = 0.0
#         done = None
#         for i in range(self._skip):
#             obs, reward, done, info = self.env.step(action)
#             if i == self._skip - 2: self._obs_buffer[0] = obs
#             if i == self._skip - 1: self._obs_buffer[1] = obs
#             total_reward += reward
#             if done:
#                 break
#         # Note that the observation on the done=True frame
#         # doesn't matter
#         if self._use_max:
#             self.max_frame = self._obs_buffer.max(axis=0)
#         else:
#             self.max_frame = self._obs_buffer[-1]

#         return self.max_frame, total_reward, done, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

#     def render(self, mode='human', **kwargs):
#         img = self.max_frame
#         img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
#         if mode == 'rgb_array':
#             return img
#         elif mode == 'human':
#             from gym.envs.classic_control import rendering
#             if self.viewer is None:
#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(img)
#             return self.viewer.isopen


def make_atari(env_id, skip=4, image_based=True, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip, use_max=image_based)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

#@wjc
def select_action(visit_counts, temperature=1, deterministic=True,legal_actions=None):
    assert (legal_actions is not None)
    # print("in select action: ",np.asarray(legal_actions).shape,np.asarray(visit_counts).shape)
    # print(legal_actions,visit_counts)
    for action_idx,_ in enumerate(legal_actions):
        if (legal_actions[action_idx]==0) and visit_counts[action_idx]>=1:
            # print("illegal action detected during decision with %d  visit" % visit_counts[action_idx])
            visit_counts[action_idx]=0

    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy


def time_calculate(func):
    import time
    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print('|func: [%r] took: %2.4f seconds|' % (func.__name__, end_time - start_time))
        return result
    return time_wrapper


def profile(func):
    from line_profiler import LineProfiler
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()

        return result
    return wrapper


def retype_observation(observation):
    # if observation.dtype == 'uint8':
    observation = np.asarray(observation, dtype=np.float32) / 255.0
    return observation


def prepare_observation(observation):
    observation = np.asarray(observation, dtype=np.float32) / 255.0
    observation = np.moveaxis(observation, -1, 0)

    return observation


def prepare_observation_lst(observation_lst, image_based=False):
    if image_based:
        # B, S, W, H, C
        observation_lst = np.array(observation_lst, dtype=np.uint8)
        observation_lst = np.moveaxis(observation_lst, -1, 2)

        shape = observation_lst.shape
        observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))
    else:
        observation_lst = np.array(observation_lst)

    return observation_lst


def arr_to_str(arr):
    img_str = cv2.imencode('.jpg', arr)[1].tostring()

    return img_str


def str_to_arr(s, gray_scale=False):
    nparr = np.fromstring(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise
