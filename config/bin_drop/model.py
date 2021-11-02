import torch
import torch.nn as nn

from core.model import BaseMuZeroNet


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=None):
        super().__init__()
        size_list = [input_size] + layer_sizes
        layers = []
        if 1 < len(size_list):
            for i in range(len(size_list) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(size_list[i], size_list[i + 1]),
                        torch.nn.LeakyReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(size_list[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        # self.resblocks1 = torch.nn.ModuleList(
        #     [ResidualBlock(out_channels // 2) for _ in range(2)]
        # )
        self.resblocks1_1 = ResidualBlock(out_channels // 2)
        self.resblocks1_2 = ResidualBlock(out_channels // 2)

        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        # self.resblocks2 = torch.nn.ModuleList(
        #     [ResidualBlock(out_channels) for _ in range(3)]
        # )
        self.resblocks2_1 = ResidualBlock(out_channels)
        self.resblocks2_2 = ResidualBlock(out_channels)
        self.resblocks2_3 = ResidualBlock(out_channels)

        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.resblocks3 = torch.nn.ModuleList(
        #     [ResidualBlock(out_channels) for _ in range(3)]
        # )
        self.resblocks3_1 = ResidualBlock(out_channels)
        self.resblocks3_2 = ResidualBlock(out_channels)
        self.resblocks3_3 = ResidualBlock(out_channels)

        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)

        # for block in self.resblocks1:
        #     out = block(out)
        out = self.resblocks1_1(out)
        out = self.resblocks1_2(out)

        out = self.conv2(out)

        # for block in self.resblocks2:
        #     out = block(out)
        out = self.resblocks2_1(out)
        out = self.resblocks2_2(out)
        out = self.resblocks2_3(out)

        out = self.pooling1(out)

        # for block in self.resblocks3:
        #     out = block(out)
        out = self.resblocks3_1(out)
        out = self.resblocks3_2(out)
        out = self.resblocks3_3(out)

        out = self.pooling2(out)
        return out


class RepresentationNetwork(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            downsample,
    ):
        super().__init__()
        self.use_downsample = downsample
        if self.use_downsample:
            self.downsample = DownSample(
                observation_shape[0],
                num_channels,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.use_downsample:
            out = self.downsample(x)
        else:
            out = self.conv(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)

        for block in self.resblocks:
            out = block(out)
        return out


class DynamicsNetwork(torch.nn.Module):
    def __init__(
            self,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            fc_reward_layers,
            full_support_size,
            block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.block_output_size_reward = block_output_size_reward
        self.fc = FullyConnectedNetwork(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.nn.functional.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        out = self.conv1x1_reward(out)
        out = out.view(-1, self.block_output_size_reward)
        reward = self.fc(out)
        return state, reward
        return out


class PredictionNetwork(torch.nn.Module):
    def __init__(
            self,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = FullyConnectedNetwork(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = FullyConnectedNetwork(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        value = self.conv1x1_value(out)
        policy = self.conv1x1_policy(out)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroNet(BaseMuZeroNet):
    def __init__(
            self,
            observation_shape,
            stacked_observations,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            reduced_channels_value,
            reduced_channels_policy,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            reward_support_size,
            value_support_size,
            downsample,
            inverse_value_transform,
            inverse_reward_transform):
        super(MuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        # self.hx_size = 32

        self.action_space_size = action_space_size
        if downsample:
            block_output_size_reward = (reduced_channels_reward * (observation_shape[1] // 16) * (observation_shape[2] // 16))
            block_output_size_value = (reduced_channels_value * (observation_shape[1] // 16) * (observation_shape[2] // 16))
            block_output_size_policy = (reduced_channels_policy * (observation_shape[1] // 16) * (observation_shape[2] // 16))
        else:
            block_output_size_reward = (reduced_channels_reward * observation_shape[1] * observation_shape[2])
            block_output_size_value = (reduced_channels_value * observation_shape[1] * observation_shape[2])
            block_output_size_policy = (reduced_channels_policy * observation_shape[1] * observation_shape[2])

        self._representation_network = RepresentationNetwork(
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            downsample,
        )

        self._dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels + 1,
            reduced_channels_reward,
            fc_reward_layers,
            reward_support_size,
            block_output_size_reward,
        )

        self._prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            block_output_size_value,
            block_output_size_policy,
        )

    def prediction(self, state):
        actor_logit, value = self._prediction_network(state)
        return actor_logit, value

    def representation(self, obs_history):
        encoded_state = self._representation_network(obs_history)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
                                           encoded_state - min_encoded_state
                                   ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
                .to(action.device)
                .float()
        )
        action_one_hot = (
                action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self._dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
                                                next_encoded_state - min_next_encoded_state
                                        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward
