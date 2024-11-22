from typing import Dict, List, Union

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from jaxtyping import Float

from ..agent_network import AgentNetworkBase


class PolicyNetwork(AgentNetworkBase):
    def __init__(
        self, envs: SyncVectorEnv, config: Dict[str, Union[List[int], str]]
    ) -> None:
        super().__init__(envs=envs, config=config)

    def _get_input_size(self) -> int:
        return int(np.array(self.envs.single_observation_space.shape).prod())

    def forward(
        self, observation: Float[torch.Tensor, "current_sample, proposed_sample"]
    ) -> Float[torch.Tensor, "current_step_size, proposed_step_size"]:
        current_sample, proposed_sample = torch.split(observation, 2)

        current_phi = self.network(current_sample)
        proposed_phi = self.network(proposed_sample)

        action = torch.concatenate([current_phi, proposed_phi])

        return action
