import numpy as np

from core.game import Game


class HanabiControlWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=False, local_state=False):
        """
                :param env: instance of gym environment
                :param k: no. of observations to stack
        """
        super().__init__(env, env.num_moves(), discount)
        self.cvt_string = cvt_string
        self.local = local_state

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        global_state, state, reward, done, info, legalActions = self.env.step(
            action)
        if self.local:
            return np.array(state), np.array(reward), np.array(done), np.array(info), np.array(legalActions)
        else:
            return np.array(global_state), np.array(reward), np.array(done), np.array(info), np.array(legalActions)

    def reset(self, **kwargs):
        global_state, state, legal_actions = self.env.reset()

        if self.local:
            return np.array(state), np.array(legal_actions)
        else:
            return np.array(global_state), np.array(legal_actions)

    def close(self):
        pass
