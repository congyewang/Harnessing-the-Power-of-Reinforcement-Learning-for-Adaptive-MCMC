import time
from abc import ABC, abstractmethod
from typing import List

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from jaxtyping import Float
from stable_baselines3.common.buffers import ReplayBuffer
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm.auto import trange

from ..envs import MCMCEnvBase
from ..utils import Toolbox


class LearningInterface(ABC):
    def __init__(
        self,
        env: gym.spaces.Box,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        random_seed: int = 42,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
    ) -> None:
        if not isinstance(env.single_observation_space, gym.spaces.Box):
            raise ValueError("only continuous observation space is supported")
        self.env = env
        self.random_seed = random_seed

        self.obs, self.infos = env.reset(seed=random_seed)

        _single_envs: List[MCMCEnvBase] = env.envs
        if hasattr(_single_envs[0].unwrapped, "sample_dim"):
            self.sample_dim: int = _single_envs[0].unwrapped.sample_dim
        else:
            self.sample_dim: int = np.prod(env.single_observation_space.shape) >> 1
        if hasattr(_single_envs[0].unwrapped, "initial_step_size"):
            self.initial_step_size: npt.NDArray[np.float64] = _single_envs[
                0
            ].unwrapped.initial_step_size
        else:
            self.initial_step_size: npt.NDArray[np.float64] = np.array([1.0])
        if hasattr(_single_envs[0].unwrapped, "total_timesteps"):
            self.total_timesteps: int = _single_envs[0].unwrapped.total_timesteps
        else:
            self.total_timesteps: int = 500_000

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.replay_buffer = replay_buffer

        self.learning_starts = learning_starts
        self.batch_size = batch_size

        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.tau = tau

        self.device = device
        self.verbose = verbose

        self.predicted_env: gym.spaces.Box | None = None
        self.predicted_timesteps: int | None = None

        self.critic_loss: List[float] = []
        self.actor_loss: List[float] = []

        self.predicted_observation: List[npt.NDArray[np.float64]] = []
        self.predicted_action: List[npt.NDArray[np.float64]] = []
        self.predicted_reward: List[npt.NDArray[np.float64]] = []

    def soft_clipping(
        self, g: Float[torch.Tensor, "sample_dim"], t: float = 1.0, p: int = 2
    ) -> Float[torch.Tensor, "sample_dim"]:
        """Soft clipping function for gradient clipping.

        Args:
            g (torch.Tensor): Gradient.
            t (float, optional): Threshold. Defaults to 1.0.
            p (int, optional): Norm. Defaults to 2.

        Returns:
            torch.Tensor: Clipped gradient.
        """
        norm = torch.norm(g, p=p)

        return t / (t + norm) * g

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError("train method is not implemented")

    def predict(
        self,
        predicted_env: MCMCEnvBase,
    ) -> None:
        if not isinstance(predicted_env.single_action_space, gym.spaces.Box):
            raise ValueError("only continuous action space is supported")

        self.predicted_env = predicted_env
        _single_predicted_envs: List[MCMCEnvBase] = predicted_env.envs
        if hasattr(_single_predicted_envs[0].unwrapped, "total_timesteps"):
            self.predicted_timesteps: int = _single_predicted_envs[
                0
            ].unwrapped.total_timesteps
        else:
            self.predicted_timesteps: int = 10_000

        # Reset the environment
        predicted_obs, _ = predicted_env.reset(seed=self.random_seed)

        # Store predicted obs, action, and reward
        predicted_observation: List[npt.NDArray[np.float64]] = []
        predicted_action: List[npt.NDArray[np.float64]] = []
        predicted_reward: List[npt.NDArray[np.float64]] = []

        for _ in trange(self.predicted_timesteps, disable=not self.verbose):
            with torch.no_grad():
                predicted_actions = self.actor(
                    torch.from_numpy(predicted_obs).to(self.device)
                )

            predicted_obs, predicted_rewards, _, _, _ = predicted_env.step(
                predicted_actions.detach().cpu().numpy()
            )

            predicted_observation.append(predicted_obs)
            predicted_action.append(predicted_actions.view(-1).detach().cpu().numpy())
            predicted_reward.append(predicted_rewards)

        self.predicted_observation = np.array(predicted_observation).reshape(
            -1, np.prod(predicted_env.single_observation_space.shape)
        )
        self.predicted_action = np.array(predicted_action)
        self.predicted_reward = np.array(predicted_reward).flatten()

    @abstractmethod
    def save(self, folder_path: str) -> None:
        raise NotImplementedError("save method is not implemented")


class LearningDDPG(LearningInterface):
    def __init__(
        self,
        env: gym.spaces.Box,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        random_seed: int = 42,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            actor=actor,
            target_actor=target_actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            replay_buffer=replay_buffer,
            learning_starts=learning_starts,
            batch_size=batch_size,
            exploration_noise=exploration_noise,
            gamma=gamma,
            policy_frequency=policy_frequency,
            tau=tau,
            random_seed=random_seed,
            device=device,
            verbose=verbose,
        )

    def train(self, gradient_clipping: bool = False) -> None:
        """
        Training Session for DDPG.
        """
        if gradient_clipping:
            for p_critic in self.critic.parameters():
                p_critic.register_hook(self.soft_clipping)

            for p_actor in self.actor.parameters():
                p_actor.register_hook(self.soft_clipping)

        for global_step in trange(self.total_timesteps, disable=not self.verbose):
            if global_step < self.learning_starts:
                actions = np.concatenate(
                    [self.initial_step_size, self.initial_step_size], axis=0
                ).reshape(1, -1)
            else:
                with torch.no_grad():
                    actions = self.actor(torch.from_numpy(self.obs).to(self.device))
                    actions += torch.normal(
                        0, torch.ones_like(actions) * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low,
                            self.env.single_action_space.high,
                        )
                    )
            next_obs, rewards, terminations, _, self.infos = self.env.step(actions)

            real_next_obs = next_obs.copy()
            self.replay_buffer.add(
                self.obs, real_next_obs, actions, rewards, terminations, self.infos
            )

            self.obs = next_obs

            if global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions = self.target_actor(data.next_observations)
                    critic_next_target = self.target_critic(
                        data.next_observations, next_state_actions
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (critic_next_target).view(-1)
                critic_a_values = self.critic(data.observations, data.actions).view(-1)
                critic_loss = F.mse_loss(critic_a_values, next_q_value)

                # optimize the model
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic(
                        data.observations, self.actor(data.observations)
                    ).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.target_actor.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic.parameters(), self.target_critic.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.critic_loss.append(critic_loss.item())
                    self.actor_loss.append(actor_loss.item())

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/ddpg.{time.time()}.pth"
        Toolbox.create_folder(model_path)
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            model_path,
        )


class LearningTD3(LearningInterface):
    def __init__(
        self,
        env: gym.spaces.Box,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        critic2: torch.nn.Module,
        target_critic2: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        random_seed: int = 42,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            actor=actor,
            target_actor=target_actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            replay_buffer=replay_buffer,
            learning_starts=learning_starts,
            batch_size=batch_size,
            exploration_noise=exploration_noise,
            gamma=gamma,
            policy_frequency=policy_frequency,
            tau=tau,
            random_seed=random_seed,
            device=device,
            verbose=verbose,
        )

        self.critic2 = critic2
        self.target_critic2 = target_critic2

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def train(self, gradient_clipping: bool = False) -> None:
        """
        Training Session for TD3.
        """
        if gradient_clipping:
            for p_critic in self.critic.parameters():
                p_critic.register_hook(self.soft_clipping)

            for p_critic2 in self.critic2.parameters():
                p_critic2.register_hook(self.soft_clipping)

            for p_actor in self.actor.parameters():
                p_actor.register_hook(self.soft_clipping)

        for global_step in trange(self.total_timesteps, disable=not self.verbose):
            if global_step < self.learning_starts:
                actions = np.concatenate(
                    [self.initial_step_size, self.initial_step_size], axis=0
                ).reshape(1, -1)
            else:
                with torch.no_grad():
                    actions = self.actor(torch.from_numpy(self.obs).to(self.device))
                    actions += torch.normal(
                        0, torch.ones_like(actions) * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low,
                            self.env.single_action_space.high,
                        )
                    )

            next_obs, rewards, terminations, _, self.infos = self.env.step(actions)

            real_next_obs = next_obs.copy()
            self.replay_buffer.add(
                self.obs, real_next_obs, actions, rewards, terminations, self.infos
            )

            self.obs = next_obs

            if global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.batch_size)
                with torch.no_grad():
                    clipped_noise = (
                        torch.randn_like(data.actions, device=self.device)
                        * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_state_actions = (
                        self.target_actor(data.next_observations) + clipped_noise
                    )

                    critic_next_target = self.target_critic(
                        data.next_observations, next_state_actions
                    )
                    critic2_next_target = self.target_critic2(
                        data.next_observations, next_state_actions
                    )
                    min_critic_next_target = torch.min(
                        critic_next_target, critic2_next_target
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (min_critic_next_target).view(-1)

                critic_a_values = self.critic(data.observations, data.actions).view(-1)
                critic2_a_values = self.critic2(data.observations, data.actions).view(
                    -1
                )
                critic_loss = F.mse_loss(critic_a_values, next_q_value)
                critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
                critic_loss = critic_loss + critic2_loss

                # optimize the model
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic(
                        data.observations, self.actor(data.observations)
                    ).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.target_actor.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic.parameters(), self.target_critic.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic2.parameters(), self.target_critic2.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.critic_loss.append(critic_loss.item())
                    self.actor_loss.append(actor_loss.item())

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/td3.{time.time()}.pth"
        Toolbox.create_folder(model_path)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            model_path,
        )
