import copy
from typing import Any, Dict, Tuple, TypeAliasType

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from ...datastructures import DynamicTopK
from .actor import PolicyNetwork

ActorWeights = TypeAliasType("ActorWeights", Dict[str, Any])


class EnsemblePolicyNetwork(nn.Module):
    def __init__(
        self,
        actor: PolicyNetwork,
        dynamic_top_k_weight: DynamicTopK[
            Tuple[np.float64, Dict[str, ActorWeights | int]]
        ],
    ) -> None:
        """
        Initialize the EnsemblePolicyNetwork.

        Attributes:
            actor (PolicyNetwork): The actor network.
            dynamic_top_k_weight (DynamicTopK[Tuple[np.float64, Dict[str, ActorWeights | int]]]): The top-k weights.
        """
        super(EnsemblePolicyNetwork, self).__init__()

        self.models = nn.ModuleList()
        self._load_from_topk(actor, dynamic_top_k_weight)

    def _load_from_topk(
        self,
        actor: PolicyNetwork,
        dynamic_top_k_weight: DynamicTopK[
            Tuple[np.float64, Dict[str, ActorWeights | int]]
        ],
    ) -> None:
        """
        Load models from the top-k policy.

        Args:
            actor (PolicyNetwork): The actor network.
            dynamic_top_k_weight (DynamicTopK[Tuple[np.float64, Dict[str, ActorWeights | int]]]): The top-k weights.
        """
        for _, store_dict in dynamic_top_k_weight.topk():
            model = copy.deepcopy(actor)
            model.load_state_dict(store_dict["actor"])
            model.eval()
            self.models.append(model)

    def forward(self, x: Float[torch.Tensor, "state"]) -> Float[torch.Tensor, "action"]:
        """
        Forward pass of the ensemble policy.
        Args:
            x (Float[torch.Tensor, "state"]): The input state.

        Returns:
            Float[torch.Tensor, "action"]: The mean action from the ensemble.
        """
        with torch.no_grad():
            actions = torch.stack([model(x) for model in self.models])

            return actions.mean(dim=0)
