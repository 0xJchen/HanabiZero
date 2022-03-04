import typing
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


# from core.game import Action
class Action(object):
    pass

class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

def concat_output_value(output_lst):
    # for numpy
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst

def concat_output(output_lst):
    # for numpy
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.reward)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst


class BaseMuZeroNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform):
        super(BaseMuZeroNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

        if not self.training:
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            actor_logit = actor_logit.detach().cpu().numpy()

        num = obs.size(0)
        return NetworkOutput(value, [0. for _ in range(num)], actor_logit, state)


    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)

        if not self.training:
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            reward = self.inverse_reward_transform(reward).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            actor_logit = actor_logit.detach().cpu().numpy()

        return NetworkOutput(value, reward, actor_logit, state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)