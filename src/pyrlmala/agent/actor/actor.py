import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from jaxtyping import Float

from ...config import PolicyNetworkConfigParser
from ..agent_network import AgentNetworkBase


class PolicyNetwork(AgentNetworkBase):
    def __init__(self, envs: SyncVectorEnv, config: PolicyNetworkConfigParser) -> None:
        super().__init__(envs=envs, config=config)

    def _get_input_size(self) -> int:
        return np.array(self.envs.single_observation_space.shape).prod() >> 1

    def forward(
        self, observation: Float[torch.Tensor, "current_sample proposed_sample"]
    ) -> Float[torch.Tensor, "current_step_size proposed_step_size"]:
        current_sample, proposed_sample = torch.tensor_split(observation, 2, dim=1)

        current_phi = self.network(current_sample)
        proposed_phi = self.network(proposed_sample)

        action = torch.concatenate([current_phi, proposed_phi], dim=1)

        return action
