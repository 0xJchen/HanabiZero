import numpy as np

from core.game import Game


class HanabiControlWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=False,mdp='global'):
        """
                :param env: instance of gym environment
                :param k: no. of observations to stack
                """
        super().__init__(env, env.num_moves(), discount)
        self.cvt_string = cvt_string
        self.mdp=mdp
    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        global_state, state, reward, done, info , legalActions= self.env.step(action)
        if self.mdp=='global':
            return np.array(global_state), np.array(reward), np.array(done), np.array(info), np.array(legalActions)
        if self.mdp=='local':
            return np.array(state), np.array(reward), np.array(done), np.array(info), np.array(legalActions)

    def reset(self, **kwargs):
        global_state, state,legal_actions = self.env.reset()
        # state = state.reshape(-1) / 255.0
        if self.mdp=='global':
            return np.array(global_state),np.array(legal_actions)
        if self.mdp=='local':
            return np.array(state),np.array(legal_actions)

    def close(self):
        pass
