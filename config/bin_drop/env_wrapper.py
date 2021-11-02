import cv2
from collections import deque
import numpy as np
from core.game import Game, Action


class BinDropWrapper(Game):
    def __init__(self, env, k: int, discount: float):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount)
        self.k = k
        self.frames = deque([], maxlen=k)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_AREA)
        observation = np.array(observation, dtype=np.float32) / 255.0
        observation = np.moveaxis(observation, -1, 0)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(observation)

        return self.obs(len(self.rewards)), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_AREA)
        observation = np.array(observation, dtype=np.float32) / 255.0
        observation = np.moveaxis(observation, -1, 0)


        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.k):
            self.obs_history.append(observation)

        return self.obs(0)

    def obs(self, i):
        frames = self.obs_history[i:i + self.k]
        return np.concatenate(frames)
        # return np.array(frames)

    def close(self):
        self.env.close()
