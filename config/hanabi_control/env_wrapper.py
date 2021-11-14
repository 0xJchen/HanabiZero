import numpy as np

from core.game import Game


class HanabiControlWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=False):
        """

                :param env: instance of gym environment
                :param k: no. of observations to stack
                """
        super().__init__(env, env.num_moves(), discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        state, reward, done, info , legalActions= self.env.step(action)#here legal action is np array
        # print("in wrapper, type=",type(legalActions))
        #print("***********************************",legalActions,flush=True)
        # state = state.reshape(-1) / 255.0#mannualy disabled @wjc
        #if done==True and reward <0:
            #reward=(-reward)
            #print("reward",reward)
        return np.array(state), np.array(reward), np.array(done), np.array(info), np.array(legalActions)

    def reset(self, **kwargs):
        state,legal_actions = self.env.reset()
        # state = state.reshape(-1) / 255.0

        return np.array(state),np.array(legal_actions)

    def close(self):
        # self.env.close()
        pass
    #no close in hanabi
