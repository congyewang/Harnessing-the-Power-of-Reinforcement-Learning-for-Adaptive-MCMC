import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from jaxtyping import Float

from ...config import QNetworkConfigParser
from ..agent_network import AgentNetworkBase


class QNetwork(AgentNetworkBase):
    def __init__(self, envs: SyncVectorEnv, config: QNetworkConfigParser):
        super().__init__(envs=envs, config=config)

    def _get_input_size(self) -> int:
        return int(
            np.array(self.envs.single_observation_space.shape).prod()
            + np.array(self.envs.single_action_space.shape).prod()
        )

    def forward(
        self,
        observation: Float[torch.Tensor, "current_sample proposed_sample"],
        action: Float[torch.Tensor, "current_step_size proposed_step_size"],
    ) -> Float[torch.Tensor, "q_value"]:
        x = torch.cat([observation, action], 1)
        return self.network(x)
