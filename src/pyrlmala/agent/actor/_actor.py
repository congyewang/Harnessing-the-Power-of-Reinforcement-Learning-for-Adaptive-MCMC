import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv


class PolicyNetwork(nn.Module):
    def __init__(self, envs: SyncVectorEnv):
        super().__init__()
        self.input_layer = nn.Linear(
            np.array(envs.single_observation_space.shape).prod() >> 1, 8
        )
        self.hidden_layer = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, 1)

    def phi(self, x: torch.Tensor):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)

        return x

    def forward(self, observation: torch.Tensor):
        current_sample, proposed_sample = torch.tensor_split(observation, 2, dim=1)

        current_phi = self.phi(current_sample)
        proposed_phi = self.phi(proposed_sample)

        action = torch.concatenate([current_phi, proposed_phi])

        return action
