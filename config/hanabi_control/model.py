import torch
import torch.nn as nn

from core.model import BaseMuZeroNet

class BasicBlock(nn.Module):
    def __init__(self, in_dim):
        super(BasicBlock, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)

        self.fc2 = nn.Linear(in_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        _x = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x += _x

        x = nn.functional.relu(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownSampleBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.downsample=nn.Sequential(nn.Linear(in_dim,out_dim),nn.BatchNorm1d(out_dim))

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)

        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        _x = self.downsample(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)

        x += _x
        x = nn.functional.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, repeat=1):
        super(ResNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.block1=DownSampleBlock(in_dim, out_dim)
        self.block2=nn.Sequential(*[BasicBlock(out_dim) for _ in range(repeat)])
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        return x

class ResMLP(nn.Module):
    def __init__(self, in_dim):
        super(ResMLP, self).__init__()
        self.in_dim = in_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)

        self.fc2 = nn.Linear(in_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        _x = x

        x = self.fc1(x)
        x = self.bn1(x)

        x += _x
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        return x

class NewResMLP(nn.Module):
    def __init__(self, in_dim):
        super(NewResMLP, self).__init__()
        self.in_dim = in_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)

        self.fc2 = nn.Linear(in_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        _x = x

        x = self.fc1(x)
        x = self.bn1(x)


        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x += _x

        x = nn.functional.relu(x)
        return x



class DynamicNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DynamicNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, state_dim)
        self.bn1 = nn.BatchNorm1d(state_dim)

        self.fc2 = nn.Linear(state_dim, state_dim)
        self.bn2 = nn.BatchNorm1d(state_dim)

        self.fc3 = nn.Linear(state_dim, state_dim)
        self.bn3 = nn.BatchNorm1d(state_dim)

    def forward(self, state_action):
        state = state_action[:, :self.state_dim]
        x = self.fc1(state_action)
        x = self.bn1(x)

        x += state
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        next_state = nn.functional.relu(x)
        return next_state

class NewDynamicNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NewDynamicNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, state_dim)
        self.bn1 = nn.BatchNorm1d(state_dim)

        self.fc2 = nn.Linear(state_dim, state_dim)
        self.bn2 = nn.BatchNorm1d(state_dim)

        self.fc3 = nn.Linear(state_dim, state_dim)
        self.bn3 = nn.BatchNorm1d(state_dim)

    def forward(self, state_action):
        state = state_action[:, :self.state_dim]
        x = self.fc1(state_action)
        x = self.bn1(x)


        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)

        x += state
        next_state = nn.functional.relu(x)
        return next_state

class MuZeroNet(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size,
                 inverse_value_transform, inverse_reward_transform, state_norm=False, proj=False):
        super(MuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.state_norm = state_norm
        self.action_space_n = action_space_n
        self.feature_size = 512#chengded from 512
        self.hidden_size = 128 #changed from 128
       # print("=================>init muzero net, repr input_size=%d, action_Size=%d",input_size,action_space_n,flush=True)
        #self._representation = nn.Sequential(nn.Linear(input_size, self.feature_size//2),
        #                                     nn.BatchNorm1d(self.feature_size//2),
        #                                     nn.ReLU(),
        #                                     nn.Linear(self.feature_size // 2, self.feature_size // 2),
        #                                     nn.BatchNorm1d(self.feature_size // 2),
        #                                     nn.ReLU(),
        #                                     nn.Linear(self.feature_size //2 , self.feature_size),
        #                                     nn.BatchNorm1d(self.feature_size),
        #                                     nn.ReLU()
        #                                     )
        self._representation = nn.Sequential(nn.Linear(input_size,self.feature_size),nn.BatchNorm1d(self.feature_size),nn.ReLU(),ResMLP(self.feature_size))
        self._dynamics_state = DynamicNet(self.feature_size, action_space_n)
        # self._dynamics_state = ResNet(self.feature_size + action_space_n, self.feature_size)
        self._dynamics_reward = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size),
                                              nn.BatchNorm1d(self.hidden_size),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size, reward_support_size))
        self._prediction_actor = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size),
                                               nn.BatchNorm1d(self.hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_size, action_space_n))
        self._prediction_value = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size),
                                               nn.BatchNorm1d(self.hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_size, value_support_size))

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)
        self._prediction_actor[-1].weight.data.fill_(0)
        self._prediction_actor[-1].bias.data.fill_(0)

        self.proj_hid = 512
        self.proj_out = 512
        self.pred_hid = 128
        self.pred_out = 512

        if proj:
            assert False
            self.projection = nn.Sequential(
                 nn.Linear(self.feature_size, self.proj_hid),
                 nn.BatchNorm1d(self.proj_hid),
                nn.ReLU(),
                  nn.Linear(self.proj_hid, self.proj_hid),
                 nn.BatchNorm1d(self.proj_hid),
              nn.ReLU(),
              nn.Linear(self.proj_hid, self.proj_out),
              nn.BatchNorm1d(self.proj_out)
          )
        #@wjc simplufy useless module
        #    self.projection = nn.Sequential(
        #     nn.Linear(self.feature_size, self.proj_out),
        #    )

            self.projection_head = nn.Sequential(
                 nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                nn.ReLU(),
                nn.Linear(self.pred_hid, self.pred_out),
            )
        #@wjc

        #    self.projection_head = nn.Sequential(
        #        nn.Linear(self.proj_out, self.pred_out),
        #   )

    def project(self, hidden_state, with_grad=True):
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        # print("representation net: ",obs_history.shape)
        if not self.state_norm:
            return self._representation(obs_history)
        else:
            state = self._representation(obs_history)
            min_state = state.view(-1, state.shape[1]).min(1, keepdim=True)[0]
            max_state = state.view(-1, state.shape[1]).max(1, keepdim=True)[0]
            scale_state = max_state - min_state
            scale_state[scale_state < 1e-5] += 1e-5
            state_normalize = (state - min_state) / scale_state
            return state_normalize

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(next_state)

        if not self.state_norm:
            return next_state, reward
        else:
        # Scale encoded state between [0, 1] (See paper appendix Training)
            min_next_state = next_state.view(-1, next_state.shape[1]).min(1, keepdim=True)[0]
            max_next_state = next_state.view(-1,next_state.shape[1]).max(1, keepdim=True)[0]
            scale_next_state = max_next_state - min_next_state
            scale_next_state[scale_next_state < 1e-5] += 1e-5
            next_state_normalized = (next_state - min_next_state) / scale_next_state
            return next_state_normalized, reward

    def get_params_mean(self):
        return 0, 0, 0, 0

    def num_params(self):
        def cnt(md):
            return sum(p.numel() for p in md.parameters())
        print("params",cnt(self))
        print("repr={},dyn={},policy={}".format(cnt(self._representation),cnt(self._dynamics_state),cnt(self._prediction_actor)))


class MuZeroNetFull(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size,
                 inverse_value_transform, inverse_reward_transform, state_norm=False):
        super(MuZeroNetFull, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.state_norm = state_norm
        self.action_space_n = action_space_n
        self.feature_size = 512
        # self.init_size= 1024
        # self.actor_hidden = 128
        self.hidden_size = 256
        # self.hidden_sec = 128
        #print("=================>init muzero net, repr input_size=%d, action_Size=%d",input_size,action_space_n,flush=True)
       # self._representation = nn.Sequential(nn.Linear(input_size, self.init_size),
       #                                      nn.BatchNorm1d(self.init_size),
       #                                      nn.ReLU(),
       #                                      nn.Linear(self.init_size, self.feature_size),
       #                                      nn.BatchNorm1d(self.feature_size),
       #                                      nn.ReLU(),
       #                                      nn.Linear(self.feature_size, self.feature_size),
       #                                      nn.BatchNorm1d(self.feature_size),
       #                                      nn.ReLU()
       #                                      )
        #self._representation = nn.Sequential(nn.Linear(input_size,self.init_size),nn.BatchNorm1d(self.init_size),nn.ReLU(),\
        #                                     nn.Linear(self.init_size,self.feature_size),nn.BatchNorm1d(self.feature_size),nn.ReLU(),\
        #                                     ResMLP(self.feature_size))
        self._representation = nn.Sequential(nn.Linear(input_size,self.feature_size),nn.BatchNorm1d(self.feature_size),nn.ReLU(),NewResMLP(self.feature_size))
        self._dynamics_state = NewDynamicNet(self.feature_size, action_space_n)
        self._dynamics_reward = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size),
                                              nn.BatchNorm1d(self.hidden_size),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size, self.hidden_size),
                                              nn.BatchNorm1d(self.hidden_size),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size, reward_support_size))
        self._prediction_actor = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size),
                                               nn.BatchNorm1d(self.hidden_size),
                                               nn.ReLU(),
                                               NewResMLP(self.hidden_size),
                                            #    nn.Linear(self.hidden_size,self.hidden_sec),
                                            #    nn.BatchNorm1d(self.hidden_sec),
                                            #    nn.ReLU(),
                                            #    nn.Linear(self.hidden_sec,self.hidden_sec//2), #512->256->128->64->20
                                            #    nn.BatchNorm1d(self.hidden_sec//2),
                                            #    nn.ReLU(),
                                               nn.Linear(self.hidden_size,self.action_space_n)
                                               )
        self._prediction_value = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size),
                                               nn.BatchNorm1d(self.hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_size, self.hidden_size),
                                               nn.BatchNorm1d(self.hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_size, value_support_size))

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)
        self._prediction_actor[-1].weight.data.fill_(0)
        self._prediction_actor[-1].bias.data.fill_(0)

        self.proj_hid = 512
        self.proj_out = 512
        self.pred_hid = 128
        self.pred_out = 512

        # self.projection = nn.Sequential(
        #     nn.Linear(self.feature_size, self.proj_hid),
        #     nn.BatchNorm1d(self.proj_hid),
        #     nn.ReLU(),
        #     nn.Linear(self.proj_hid, self.proj_hid),
        #     nn.BatchNorm1d(self.proj_hid),
        #     nn.ReLU(),
        #     nn.Linear(self.proj_hid, self.proj_out),
        #     nn.BatchNorm1d(self.proj_out)
        # )
        #@wjc simplufy useless module

        #self.projection = nn.Sequential(
         #   nn.Linear(self.feature_size, self.proj_out),
        #)

        # self.projection_head = nn.Sequential(
        #     nn.Linear(self.proj_out, self.pred_hid),
        #     nn.BatchNorm1d(self.pred_hid),
        #     nn.ReLU(),
        #     nn.Linear(self.pred_hid, self.pred_out),
        # )
        #@wjc

        #self.projection_head = nn.Sequential(
         #   nn.Linear(self.proj_out, self.pred_out),
        #)

    def project(self, hidden_state, with_grad=True):
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        #print("representation net: ",obs_history.shape,flush=True)
        if not self.state_norm:
            return self._representation(obs_history)
        else:
            state = self._representation(obs_history)
            min_state = state.view(-1, state.shape[1]).min(1, keepdim=True)[0]
            max_state = state.view(-1, state.shape[1]).max(1, keepdim=True)[0]
            scale_state = max_state - min_state
            scale_state[scale_state < 1e-5] += 1e-5
            state_normalize = (state - min_state) / scale_state
            return state_normalize

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)#x is fine here.
        next_state = self._dynamics_state(x)#next_state.max() might contain inf, which leads to inf when
        reward = self._dynamics_reward(next_state)

        # for nan_idx in range(reward.shape[0]):
        #     r_nan=torch.isnan(reward[nan_idx]).any()
        #     x_nan=torch.isnan(x[nan_idx]).any()
        #     n_nan=torch.isnan(next_state[nan_idx]).any()
        #     if r_nan:
        #         print("batch idx=",nan_idx,flush=True)
        #         print("in dynamcis,x={},next={},reward={}".format(x_nan,n_nan,r_nan),flush=True)
        #         print("===>x",x_nan,x[nan_idx].max(),x[nan_idx].min(),flush=True)
        #         print("===>next_state",n_nan,next_state[nan_idx].max(),flush=True)
        #         # print("===>reward",reward[nan_idx],flush=True)
        #         torch.save(next_state[nan_idx],'next_state_'+str(nan_idx))

        if not self.state_norm:
            return next_state, reward
        else:
            assert False
        # Scale encoded state between [0, 1] (See paper appendix Training)
            min_next_state = next_state.view(-1, next_state.shape[1]).min(1, keepdim=True)[0]
            max_next_state = next_state.view(-1,next_state.shape[1]).max(1, keepdim=True)[0]
            scale_next_state = max_next_state - min_next_state
            scale_next_state[scale_next_state < 1e-5] += 1e-5
            next_state_normalized = (next_state - min_next_state) / scale_next_state
            return next_state_normalized, reward

    def get_params_mean(self):
        return 0, 0, 0, 0



class MuZeroNet_v2(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size,
                 inverse_value_transform, inverse_reward_transform):
        super(MuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.state_dim = 64
        self.hidden_dim = 128

        # 1 -> 2 layer
        self._representation = nn.Sequential(nn.Linear(input_size, self.hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(self.hidden_dim, self.state_dim),
                                             nn.Tanh())

        # 2 -> 3 layer
        self._dynamics_state = nn.Sequential(nn.Linear(self.state_dim + action_space_n, self.hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                             nn.LeakyReLU(),
                                             nn.Linear(self.hidden_dim // 2, self.state_dim),
                                             nn.Tanh())

        # 2 -> 3 layer
        self._dynamics_reward = nn.Sequential(nn.Linear(self.state_dim + action_space_n, self.hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                              nn.LeakyReLU(),
                                              nn.Linear(self.hidden_dim // 2, reward_support_size))

        # 2 -> 3 layer
        self._prediction_actor = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                               nn.LeakyReLU(),
                                               nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                               nn.LeakyReLU(),
                                               nn.Linear(self.hidden_dim // 2, action_space_n))

        # 2 -> 3 layer
        self._prediction_value = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                               nn.LeakyReLU(),
                                               nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                               nn.LeakyReLU(),
                                               nn.Linear(self.hidden_dim // 2, value_support_size))
        self.action_space_n = action_space_n

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        obs_history = obs_history.float()

        return self._representation(obs_history)
        # state = self._representation(obs_history)
        # min_state = state.view(-1, state.shape[1]).min(1, keepdim=True)[0]
        # max_state = state.view(-1, state.shape[1]).max(1, keepdim=True)[0]
        # scale_state = max_state - min_state
        # scale_state[scale_state < 1e-5] += 1e-5
        # state_normalize = (state - min_state) / scale_state
        # return state_normalize

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward

        # Scale encoded state between [0, 1] (See paper appendix Training)
        # min_next_state = next_state.view(-1, next_state.shape[1]).min(1, keepdim=True)[0]
        # max_next_state = next_state.view(-1,next_state.shape[1]).max(1, keepdim=True)[0]
        # scale_next_state = max_next_state - min_next_state
        # scale_next_state[scale_next_state < 1e-5] += 1e-5
        # next_state_normalized = (next_state - min_next_state) / scale_next_state
        # return next_state_normalized, reward
