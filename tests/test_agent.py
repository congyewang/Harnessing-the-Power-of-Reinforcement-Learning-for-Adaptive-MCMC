import torch
from gymnasium.vector import SyncVectorEnv

from pyrlmala.agent.actor.actor import PolicyNetwork
from pyrlmala.config.config_parser import PolicyNetworkConfigParser


class TestPolicyNetwork:

    # Initializes PolicyNetwork with valid SyncVectorEnv and PolicyNetworkConfigParser
    def test_initialization_with_valid_env_and_config(self, mocker):
        mock_env = mocker.Mock(spec=SyncVectorEnv)
        mock_config = mocker.Mock(spec=PolicyNetworkConfigParser)
        policy_network = PolicyNetwork(envs=mock_env, config=mock_config)
        assert policy_network.envs == mock_env
        assert isinstance(policy_network, PolicyNetwork)

    # Handles empty or zero-dimensional observation space gracefully
    def test_handle_empty_observation_space(self, mocker):
        mock_env = mocker.Mock(spec=SyncVectorEnv)
        mock_env.single_observation_space.shape = ()
        mock_config = mocker.Mock(spec=PolicyNetworkConfigParser)
        policy_network = PolicyNetwork(envs=mock_env, config=mock_config)
        observation = torch.empty((1, 0))
        result = policy_network.forward(observation)
        assert result.shape == (1, 2)


class TestQNetwork:

    # Initializes QNetwork with valid SyncVectorEnv and QNetworkConfigParser
    def test_initialization_with_valid_inputs(self, mocker):
        mock_envs = mocker.Mock(spec=SyncVectorEnv)
        mock_config = mocker.Mock(spec=QNetworkConfigParser)
        q_network = QNetwork(envs=mock_envs, config=mock_config)
        assert q_network.envs == mock_envs
        assert isinstance(q_network.network, torch.nn.Sequential)

    # Handles empty or zero-dimensional observation and action spaces gracefully
    def test_empty_observation_action_spaces(self, mocker):
        mock_envs = mocker.Mock(spec=SyncVectorEnv)
        mock_envs.single_observation_space.shape = ()
        mock_envs.single_action_space.shape = ()
        mock_config = mocker.Mock(spec=QNetworkConfigParser)
        q_network = QNetwork(envs=mock_envs, config=mock_config)
        input_size = q_network._get_input_size()
        assert input_size == 0
