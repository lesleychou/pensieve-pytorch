# This is from Runzhe_MORL/multimario/model.py

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

from torch.distributions.categorical import Categorical


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class NaiveMoCnnActorNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(NaiveMoCnnActorNetwork, self).__init__()
        linear = nn.Linear
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, output_size),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = self.feature(state)
        x = torch.cat((x, preference), dim=1)
        policy = self.actor(x)
        return policy


class NaiveMoCnnCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, reward_size):
        super(NaiveMoCnnCriticNetwork, self).__init__()
        linear = nn.Linear
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )

        self.critic = nn.Sequential(
            linear(512+reward_size, 128),
            nn.LeakyReLU(),
            linear(128, 1),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, preference):
        x = self.feature(state)
        x = torch.cat((x, preference), dim=1)
        value = self.critic(x)
        return value
