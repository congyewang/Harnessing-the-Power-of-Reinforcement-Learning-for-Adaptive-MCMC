from collections import deque
from typing import Any, Deque, Tuple

import numpy as np
import torch
from gymnasium import spaces
from numpy import typing as npt
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class NStepReplayBuffer(BaseBuffer):
    """
    N-step TD replay buffer used in off-policy algorithms like DDPG/TD3.

    Args:
        buffer_size (int): Max number of element in the buffer
        observation_space (spaces.Space[Any]): Observation space
        action_space (spaces.Space[Any]): Action space
        device (torch.device | str): PyTorch device
        n_envs (int): Number of parallel environments
        n_step (int): Number of steps to look ahead for TD
        gamma (float): Discount factor
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        device: torch.device | str = "auto",
        n_envs: int = 1,
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        """
        N-step TD replay buffer used in off-policy algorithms like DDPG/TD3.

        Args:
            buffer_size (int): Max number of element in the buffer (n-step transitions)
            observation_space (spaces.Space[Any]): Observation space
            action_space (spaces.Space[Any]): Action space
            device (torch.device | str): PyTorch device
            n_envs (int): Number of parallel environments
            n_step (int): Number of steps to look ahead for TD
            gamma (float): Discount factor
        """
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.n_step = n_step
        self.gamma = gamma

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float64
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool)
        self.traj_buffer: Deque[
            Tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                np.float64,
                bool,
            ]
        ] = deque([])

    def add(
        self,
        obs: npt.NDArray[np.float64],
        next_obs: npt.NDArray[np.float64],
        action: npt.NDArray[np.float64],
        reward: np.float64,
        done: bool,
    ) -> None:
        """
        Add a new transition to the buffer.

        Args:
            obs (npt.NDArray[np.float64]): observation
            next_obs (npt.NDArray[np.float64]): next observation
            action (npt.NDArray[np.float64]): action
            reward (np.float64): reward
            done (bool): done
        """
        transition = (obs, next_obs, action, reward, done)
        self.traj_buffer.append(transition)

        if len(self.traj_buffer) >= self.n_step or done:
            self._store_transition()

    def _store_transition(self) -> None:
        """
        Store n-step transitions in the replay buffer. The n-step transition is calculated as the sum of discounted rewards over the n-step trajectory.
        """
        total_reward = 0
        for idx, (_, next_obs, _, reward, done) in enumerate(self.traj_buffer):
            total_reward += (self.gamma**idx) * reward
            if done:
                next_obs = next_obs
                break
        self._insert_transition(
            self.traj_buffer[0][0], next_obs, self.traj_buffer[0][2], total_reward, done
        )
        self.traj_buffer.popleft()

    def _insert_transition(self, obs, next_obs, action, reward, done) -> None:
        """
        Insert a transition in the replay buffer.

        Args:
            obs (npt.NDArray[np.float64]): observation
            next_obs (npt.NDArray[np.float64]): next observation
            action (npt.NDArray[np.float64]): action
            reward (np.float64): reward
            done (bool): done
        """
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample a batch of transitions.

        Args:
            batch_size (int): Number of element to sample in the batch from the replay buffer.

        Returns:
            ReplayBufferSamples: A batch of transitions sampled from the replay buffer with n-step rewards and next observations calculated.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: npt.NDArray[np.float64]) -> ReplayBufferSamples:
        """
        Get samples from the replay buffer.

        Args:
            batch_inds (npt.NDArray[np.float64]): Indices of the samples to get from the replay buffer.

        Returns:
            ReplayBufferSamples: A batch of transitions sampled from the replay buffer with n-step rewards and next observations calculated.
        """
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            self.next_observations[batch_inds, env_indices, :],
            self.dones[batch_inds, env_indices].reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
