import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv


class QNetwork(nn.Module):
    def __init__(self, envs: SyncVectorEnv):
        super().__init__()
        self.input_layer = nn.Linear(
            np.array(envs.single_observation_space.shape).prod()
            + np.prod(envs.single_action_space.shape),
            8,
        )
        self.hidden_layer = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        x = torch.cat([observation, action], 1)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
