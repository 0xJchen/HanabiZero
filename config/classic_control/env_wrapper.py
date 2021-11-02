import numpy as np

from core.game import Game


class ClassicControlWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=False):
        """

                :param env: instance of gym environment
                :param k: no. of observations to stack
                """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = state.reshape(-1) / 255.0

        return state, reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        state = state.reshape(-1) / 255.0

        return state

    def close(self):
        self.env.close()