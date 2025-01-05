import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from pyrlmala.learning.buffers import NStepReplayBuffer
from pyrlmala.learning.learning import LearningDDPG, LearningTD3


class TestLearningDDPG:

    # Initialize LearningDDPG with valid parameters and verify correct setup
    def test_initialize_learning_ddpg_with_valid_parameters(self, mocker):
        # Mocking necessary components
        env = mocker.Mock(spec=gym.spaces.Box)
        predicted_env = mocker.Mock(spec=gym.spaces.Box)
        actor = mocker.Mock(spec=torch.nn.Module)
        target_actor = mocker.Mock(spec=torch.nn.Module)
        critic = mocker.Mock(spec=torch.nn.Module)
        target_critic = mocker.Mock(spec=torch.nn.Module)
        actor_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        critic_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        replay_buffer = mocker.Mock(spec=ReplayBuffer)

        # Initialize LearningDDPG
        ddpg = LearningDDPG(
            env=env,
            predicted_env=predicted_env,
            actor=actor,
            target_actor=target_actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            replay_buffer=replay_buffer,
        )

        # Assertions to verify correct setup
        assert ddpg.env == env
        assert ddpg.predicted_env == predicted_env
        assert ddpg.actor == actor
        assert ddpg.target_actor == target_actor
        assert ddpg.critic == critic
        assert ddpg.target_critic == target_critic

    # Initialize LearningDDPG with invalid environment types and expect ValueError
    def test_initialize_learning_ddpg_with_invalid_environment(self, mocker):
        # Mocking necessary components with invalid environment type
        env = mocker.Mock()
        predicted_env = mocker.Mock()
        actor = mocker.Mock(spec=torch.nn.Module)
        target_actor = mocker.Mock(spec=torch.nn.Module)
        critic = mocker.Mock(spec=torch.nn.Module)
        target_critic = mocker.Mock(spec=torch.nn.Module)
        actor_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        critic_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        replay_buffer = mocker.Mock(spec=ReplayBuffer)

        # Expect ValueError due to invalid environment type
        with pytest.raises(
            ValueError, match="only continuous observation space is supported"
        ):
            LearningDDPG(
                env=env,
                predicted_env=predicted_env,
                actor=actor,
                target_actor=target_actor,
                critic=critic,
                target_critic=target_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                replay_buffer=replay_buffer,
            )


class TestLearningTD3:

    # Initialize LearningTD3 with valid parameters and ensure no exceptions are raised
    def test_initialize_with_valid_parameters(self, mocker):
        mock_env = mocker.Mock()
        mock_env.single_observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        mock_env.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        mock_env.reset.return_value = (np.array([0.0, 0.0, 0.0]), {})
        mock_predicted_env = mocker.Mock()
        mock_predicted_env.single_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,)
        )
        mock_predicted_env.reset.return_value = (np.array([0.0, 0.0, 0.0]), {})

        actor = mocker.Mock(spec=torch.nn.Module)
        target_actor = mocker.Mock(spec=torch.nn.Module)
        critic = mocker.Mock(spec=torch.nn.Module)
        target_critic = mocker.Mock(spec=torch.nn.Module)
        critic2 = mocker.Mock(spec=torch.nn.Module)
        target_critic2 = mocker.Mock(spec=torch.nn.Module)
        actor_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        critic_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        replay_buffer = mocker.Mock(spec=ReplayBuffer)

        try:
            td3 = LearningTD3(
                env=mock_env,
                predicted_env=mock_predicted_env,
                actor=actor,
                target_actor=target_actor,
                critic=critic,
                target_critic=target_critic,
                critic2=critic2,
                target_critic2=target_critic2,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                replay_buffer=replay_buffer,
            )
        except Exception as e:
            pytest.fail(f"Initialization failed with exception: {e}")

    # Initialize LearningTD3 with an invalid environment type and expect a ValueError
    def test_initialize_with_invalid_environment(self, mocker):
        mock_env = mocker.Mock()
        mock_env.single_observation_space = gym.spaces.Discrete(
            3
        )  # Invalid type for this context
        mock_predicted_env = mocker.Mock()
        mock_predicted_env.single_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,)
        )

        actor = mocker.Mock(spec=torch.nn.Module)
        target_actor = mocker.Mock(spec=torch.nn.Module)
        critic = mocker.Mock(spec=torch.nn.Module)
        target_critic = mocker.Mock(spec=torch.nn.Module)
        critic2 = mocker.Mock(spec=torch.nn.Module)
        target_critic2 = mocker.Mock(spec=torch.nn.Module)
        actor_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        critic_optimizer = mocker.Mock(spec=torch.optim.Optimizer)
        replay_buffer = mocker.Mock(spec=ReplayBuffer)

        with pytest.raises(
            ValueError, match="only continuous observation space is supported"
        ):
            LearningTD3(
                env=mock_env,
                predicted_env=mock_predicted_env,
                actor=actor,
                target_actor=target_actor,
                critic=critic,
                target_critic=target_critic,
                critic2=critic2,
                target_critic2=target_critic2,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                replay_buffer=replay_buffer,
            )


class TestNStepReplayBuffer:
    @pytest.fixture
    def buffer(self):
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
        return NStepReplayBuffer(
            buffer_size=100,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            n_envs=1,
            n_step=3,
            gamma=0.99,
        )

    def test_add_and_store_transition(self, buffer):
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        next_obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        action = np.array([0.1, -0.1], dtype=np.float64)
        reward = 1.0
        done = False

        # Add transitions
        for _ in range(3):
            buffer.add(obs, next_obs, action, reward, done)

        # The trajectory buffer should have one stored transition
        assert len(buffer.traj_buffer) == 1

        # Check the values inside the main buffer after transition
        assert np.allclose(buffer.observations[0, 0], obs)
        assert np.allclose(buffer.actions[0, 0], action)
        assert np.allclose(buffer.next_observations[0, 0], next_obs)
        assert buffer.rewards[0, 0] == pytest.approx(
            1.99, rel=1e-3
        )  # 1 + 0.99 + 0.99**2

    def test_buffer_overflow(self, buffer):
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        next_obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        action = np.array([0.1, -0.1], dtype=np.float64)
        reward = 1.0

        for i in range(110):
            done = i % 10 == 0  # Add some done flags to break the trajectories
            buffer.add(obs, next_obs, action, reward, done)

        # Check that the buffer size is respected
        assert buffer.full
        assert buffer.pos == 10

    def test_sample(self, buffer):
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        next_obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        action = np.array([0.1, -0.1], dtype=np.float64)
        reward = 1.0

        for _ in range(10):
            buffer.add(obs, next_obs, action, reward, False)

        sample = buffer.sample(batch_size=5)
        assert isinstance(sample, ReplayBufferSamples)

        # Check sample shapes
        assert sample.observations.shape == (5, 4)
        assert sample.actions.shape == (5, 2)
        assert sample.next_observations.shape == (5, 4)
        assert sample.rewards.shape == (5, 1)
        assert sample.dones.shape == (5, 1)

    def test_transition_on_done(self, buffer):
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        next_obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        action = np.array([0.1, -0.1], dtype=np.float64)
        reward = 1.0

        buffer.add(obs, next_obs, action, reward, done=True)

        # Since the done is True, the transition should be stored immediately
        assert len(buffer.traj_buffer) == 0
        assert np.allclose(buffer.observations[0, 0], obs)
        assert buffer.rewards[0, 0] == reward
