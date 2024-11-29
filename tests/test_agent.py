import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from gymnasium.vector import SyncVectorEnv
from pytest_mock import MockerFixture

from pyrlmala.agent import PolicyNetwork, QNetwork
from pyrlmala.config.config_parser import (
    PolicyNetworkConfigParser,
    QNetworkConfigParser,
)

SAMPLE_DIM_LIST = [1, 2, 3, 5, 33, 66]


def create_mock_env(
    mocker: MockerFixture, sample_dim: int, include_action_space: bool = False
) -> SyncVectorEnv:
    mock_envs = mocker.Mock(spec=SyncVectorEnv)
    mock_envs.single_observation_space = Box(
        -np.inf, np.inf, (sample_dim << 1,), np.float64
    )
    if include_action_space:
        mock_envs.single_action_space = Box(-np.inf, np.inf, (2,), np.float64)
    return mock_envs


class TestPolicyNetwork:
    """
    Test that PolicyNetwork can be initialized with a mocked SyncVectorEnv
    and a mocked PolicyNetworkConfigParser. Also verifies that the output
    dimensions match the expected shape when passing valid input tensors.
    """

    @pytest.mark.parametrize("sample_dim", SAMPLE_DIM_LIST)
    def test_initialization_with_valid_env_and_config(
        self, mocker: MockerFixture, sample_dim: int
    ) -> None:
        """
        Initializes PolicyNetwork with valid SyncVectorEnv and PolicyNetworkConfigParser.
        """
        mock_envs = create_mock_env(mocker, sample_dim)
        mock_config = mocker.Mock(spec=PolicyNetworkConfigParser)
        policy_network = PolicyNetwork(envs=mock_envs, config=mock_config)

        assert policy_network(torch.randn(1, sample_dim << 1)).shape == (
            1,
            2,
        ), "Output shape mismatch."
        assert (
            policy_network.envs == mock_envs
        ), "Environment instance should be retained."
        assert isinstance(
            policy_network, PolicyNetwork
        ), "PolicyNetwork instance creation failed."


class TestQNetwork:
    """
    Test that QNetwork can be initialized with a mocked SyncVectorEnv
    and a mocked QNetworkConfigParser. Also verifies that the output
    dimensions match the expected shape when passing valid input tensors.
    """

    @pytest.mark.parametrize("sample_dim", SAMPLE_DIM_LIST)
    def test_initialization_with_valid_env_and_config(
        self, mocker: MockerFixture, sample_dim: int
    ) -> None:
        """
        Initializes QNetwork with valid SyncVectorEnv and QNetworkConfigParser.
        """
        mock_envs = create_mock_env(mocker, sample_dim, include_action_space=True)
        mock_config = mocker.Mock(spec=QNetworkConfigParser)
        q_network = QNetwork(envs=mock_envs, config=mock_config)

        assert q_network(torch.randn(1, sample_dim << 1), torch.randn(1, 2)).shape == (
            1,
            1,
        ), "Output shape mismatch."
        assert q_network.envs == mock_envs, "Environment instance should be retained."
        assert isinstance(q_network, QNetwork), "QNetwork instance creation failed."
