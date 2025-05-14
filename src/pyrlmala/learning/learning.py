import copy
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from jaxtyping import Float
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..agent import EnsemblePolicyNetwork
from ..datastructures import DynamicTopK
from ..utils import Toolbox
from .events import EventManager, TrainEvents
from .logging import DummyWriter

T = TypeVar("T")


class LearningInterface(ABC):
    """
    Learning Interface.

    Attributes:
        env (SyncVectorEnv): Environment.
        predicted_env (SyncVectorEnv): Predicted environment.
        random_seed (int): Random seed.
        sample_dim (int): Sample dimension.
        initial_step_size (npt.NDArray[np.float64]): Initial step size.
        total_timesteps (int): Total timesteps.
        actor (torch.nn.Module): Actor.
        target_actor (torch.nn.Module): Target actor.
        critic (torch.nn.Module): Critic.
        target_critic (torch.nn.Module): Target critic.
        actor_optimizer (torch.optim.Optimizer): Actor optimizer.
        critic_optimizer (torch.optim.Optimizer): Critic optimizer.
        replay_buffer (ReplayBuffer): Replay buffer.
        actor_gradient_clipping (bool): Actor gradient clipping.
        actor_gradient_threshold (Optional[float]): Actor gradient threshold.
        actor_gradient_norm (Optional[int]): Actor gradient norm.
        critic_gradient_clipping (bool): Critic gradient clipping.
        critic_gradient_threshold (Optional[float]): Critic gradient threshold.
        critic_gradient_norm (Optional[int]): Critic gradient norm.
        learning_starts (int): Learning starts.
        batch_size (int): Batch size.
        exploration_noise (float): Exploration noise.
        gamma (float): Gamma.
        policy_frequency (int): Policy frequency.
        tau (float): Tau.
        device (torch.device): Device.
        verbose (bool): Verbose.
        predicted_timesteps (int | None): Predicted timesteps.
        critic_values (List[float]): Critic values.
        critic_loss (List[float]): Critic loss.
        actor_loss (List[float]): Actor loss.
        predicted_observation (List[npt.NDArray[np.float64]]): Predicted observation.
        predicted_action (List[npt.NDArray[np.float64]]): Predicted action.
        predicted_reward (List[npt.NDArray[np.float64]]): Predicted reward.
        best_policy (Optional[Dict[str, Any]]): Best policy.
        writer (SummaryWriter): Summary writer.
        event_manager (EventManager): Event manager.
    """

    def __init__(
        self,
        env: SyncVectorEnv,
        predicted_env: SyncVectorEnv,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        actor_gradient_clipping: bool = False,
        actor_gradient_threshold: Optional[float] = 1.0,
        actor_gradient_norm: Optional[int] = 2,
        actor_scheduler: Optional[LRScheduler] = None,
        critic_gradient_clipping: bool = False,
        critic_gradient_threshold: Optional[float] = 1.0,
        critic_gradient_norm: Optional[int] = 2,
        critic_scheduler: Optional[LRScheduler] = None,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        random_seed: int = 42,
        num_of_top_policies: int = 5,
        reward_centering: bool = True,
        r_bar: float = 0.0,
        rbar_alpha: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        track: bool = False,
        verbose: bool = True,
        run_name: str = "rlmcmc",
    ) -> None:
        """
        Initialize the Learning Interface.

        Args:
            env (SyncVectorEnv): Environment.
            predicted_env (SyncVectorEnv): Predicted environment.
            actor (torch.nn.Module): Actor.
            target_actor (torch.nn.Module): Target actor.
            critic (torch.nn.Module): Critic.
            target_critic (torch.nn.Module): Target critic.
            actor_optimizer (torch.optim.Optimizer): Actor optimizer.
            critic_optimizer (torch.optim.Optimizer): Critic optimizer.
            replay_buffer (ReplayBuffer): Replay buffer.
            actor_gradient_clipping (bool, optional): Actor gradient clipping. Defaults to False.
            actor_gradient_threshold (Optional[float], optional): Actor gradient threshold. Defaults to 1.0.
            actor_gradient_norm (Optional[int], optional): Actor gradient norm. Defaults to 2.
            critic_gradient_clipping (bool, optional): Critic gradient clipping. Defaults to False.
            critic_gradient_threshold (Optional[float], optional): Critic gradient threshold. Defaults to 1.0.
            critic_gradient_norm (Optional[int], optional): Critic gradient norm. Defaults to 2.
            learning_starts (int, optional): Learning starts. Defaults to 32.
            batch_size (int, optional): Batch size. Defaults to 32.
            exploration_noise (float, optional): Exploration noise. Defaults to 0.1.
            gamma (float, optional): Gamma. Defaults to 0.99.
            policy_frequency (int, optional): Policy frequency. Defaults to 2.
            tau (float, optional): Tau. Defaults to 0.005.
            random_seed (int, optional): Random seed. Defaults to 42.
            num_of_top_policies (int, optional): Number of top policies to keep. Defaults to 5.
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            track (bool, optional): Track. Defaults to False.
            verbose (bool, optional): Verbose. Defaults to True.
            run_name (str, optional): Run name. Defaults to "rlmcmc".

        Raises:
            ValueError: If the observation space is not continuous
        """
        if not isinstance(env.single_observation_space, gym.spaces.Box):
            raise ValueError("only continuous observation space is supported")
        else:
            self.env = env
        if not isinstance(predicted_env.single_observation_space, gym.spaces.Box):
            raise ValueError("only continuous observation space is supported")
        else:
            self.predicted_env = predicted_env
        self.random_seed = random_seed

        self.obs, _ = env.reset(seed=random_seed)

        _single_envs: List[RecordEpisodeStatistics] = env.envs
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

        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.replay_buffer = replay_buffer

        self.actor_gradient_clipping = actor_gradient_clipping
        self.actor_gradient_threshold = actor_gradient_threshold
        self.actor_gradient_norm = actor_gradient_norm

        self.critic_gradient_clipping = critic_gradient_clipping
        self.critic_gradient_threshold = critic_gradient_threshold
        self.critic_gradient_norm = critic_gradient_norm

        self.learning_starts = learning_starts
        self.batch_size = batch_size

        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.tau = tau

        self.device = device
        self.verbose = verbose

        # Losses
        self.critic_values: List[float] = []
        self.critic_loss: List[float] = []
        self.actor_loss: List[float] = []

        # Predicted
        self.predicted_timesteps: int | None = None

        # Best Policy
        self.best_episodic_return = -np.inf
        self.best_policy_step: Optional[int] = None
        self.best_policy: Optional[Dict[str, Any]] = None

        # Last Policy
        self.last_policy: Optional[Dict[str, Any]] = None

        # Top-K Policy
        if num_of_top_policies < 1:
            raise ValueError("Number of policies must be greater than 0")
        self.num_of_top_policies = num_of_top_policies
        self.topk_policy: DynamicTopK[
            Tuple[np.float64, Dict[str, Dict[str, Any] | int]]
        ] = DynamicTopK(num_of_top_policies, key=lambda x: x[0])

        # Tensorboard
        if track:
            self.writer = SummaryWriter(f"runs/{run_name}")
        else:
            self.writer = DummyWriter()

        # Event Manager Callback
        self.event_manager = EventManager()

        # Reward Centering
        self.reward_centering = reward_centering
        self.r_bar = r_bar
        self.rbar_alpha = rbar_alpha

    def soft_clipping(
        self, g: Float[torch.Tensor, "gradient"], t: float = 1.0, p: int = 2
    ) -> Float[torch.Tensor, "gradient"]:
        """
        Soft clipping function for gradient clipping.

        Args:
            g (torch.Tensor): Gradient.
            t (float, optional): Threshold. Defaults to 1.0.
            p (int, optional): Norm. Defaults to 2.

        Returns:
            torch.Tensor: Clipped gradient.
        """
        norm = torch.norm(g, p=p)

        return t / (t + norm) * g

    def currying_gradient_clipping(
        self,
        gradient_threshold: float = 1.0,
        gradient_norm: int = 2,
    ) -> Callable[[Float[torch.Tensor, "gradient"]], Float[torch.Tensor, "gradient"]]:
        """
        Currying function for gradient clipping. It returns a function that clips the gradient.

        Raises:
            ValueError: Threshold must be non-negative.
            ValueError: Norm must be positive integer.

        Returns:
            Callable[[Float[torch.Tensor, "gradient"]], Float[torch.Tensor, "gradient"]]: Clipped gradient.
        """
        if gradient_threshold < 0.0:
            raise ValueError("Threshold must be non-negative")
        if isinstance(gradient_norm, int) and gradient_norm < 1:
            raise ValueError("Norm must be positive integer")

        return partial(self.soft_clipping, t=gradient_threshold, p=gradient_norm)

    def actor_gradient_clipping_function(self) -> None:
        """
        Actor gradient clipping.
        """
        if self.actor_gradient_clipping:
            for p_actor in self.actor.parameters():
                p_actor.register_hook(
                    self.currying_gradient_clipping(
                        self.actor_gradient_threshold, self.actor_gradient_norm
                    )
                )

    def critic_gradient_clipping_function(self) -> None:
        """
        Critic gradient clipping.
        """
        if self.critic_gradient_clipping:
            for p_critic in self.critic.parameters():
                p_critic.register_hook(
                    self.currying_gradient_clipping(
                        self.critic_gradient_threshold, self.critic_gradient_norm
                    )
                )

    @abstractmethod
    def trainning_loop(self) -> None:
        """
        Training process. Must be implemented in the subclass.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("train_process method is not implemented")

    @property
    def current_step(self) -> int:
        """
        Get the current step.

        Returns:
            int: Current step.
        """
        return self.env.get_attr("current_step")[0]

    @property
    def predicted_step(self) -> int:
        """
        Get the predicted step.

        Returns:
            int: Predicted step.
        """
        return self.predicted_env.get_attr("current_step")[0]

    def train(self) -> None:
        """
        Training method. Must be implemented in the subclass.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        # Set the actor and critic to training mode
        self.actor.train()
        self.critic.train()

        # Trigger before step event
        self.event_manager.trigger(TrainEvents.BEFORE_TRAIN)

        # Gradient clipping
        if self.actor_gradient_clipping:
            self.actor_gradient_clipping_function()

        if self.critic_gradient_clipping:
            self.critic_gradient_clipping_function()

        progress_bar = tqdm(
            total=self.total_timesteps, disable=not self.verbose, desc="Training"
        )

        # Training loop
        while self.current_step < self.total_timesteps:
            self.trainning_loop()

            # Trigger within train event
            self.event_manager.trigger(TrainEvents.WITHIN_TRAIN)

            progress_bar.n = self.current_step
            progress_bar.refresh()

        self.last_policy = self.actor.state_dict()

        # Trigger after step event
        self.event_manager.trigger(TrainEvents.AFTER_TRAIN)

    def predict(
        self,
        load_policy: str = "ensemble",
    ) -> None:
        """
        Predict the observation, action, and reward.

        Args:
            load_policy (str, optional): Load policy. Defaults to "ensemble".
                - "ensemble": Ensemble policy.
                - "best": Best policy.
                - "last": Last policy.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        self.predicted_timesteps = self.predicted_env.get_attr("total_timesteps")[0]

        # Reset the environment
        predicted_obs, _ = self.predicted_env.reset(seed=self.random_seed)
        predicted_actor = copy.deepcopy(self.actor)

        match load_policy:
            case "ensemble":
                predicted_actor = EnsemblePolicyNetwork(
                    predicted_actor, self.topk_policy
                )
            case "best":
                predicted_actor.load_state_dict(self.best_policy)
            case "last":
                predicted_actor.load_state_dict(self.last_policy)
            case _:
                raise ValueError(
                    "Invalid load_policy. Must be 'ensemble', 'swa', 'best', or 'last'."
                )

        # Set the actor to evaluation mode
        predicted_actor.eval()

        progress_bar = tqdm(
            total=self.predicted_timesteps, disable=not self.verbose, desc="Prediction"
        )

        while self.predicted_step < self.predicted_timesteps:
            with torch.no_grad():
                predicted_actions = predicted_actor(
                    torch.from_numpy(predicted_obs).to(self.device)
                )

            predicted_obs, _, _, _, _ = self.predicted_env.step(
                predicted_actions.detach().cpu().numpy()
            )

            progress_bar.n = self.predicted_step
            progress_bar.refresh()

    @property
    def predicted_observation(self) -> npt.NDArray[np.float64]:
        """
        Get the predicted observation.

        Returns:
            npt.NDArray[np.float64]: Predicted observation.
        """
        return self.predicted_env.get_attr("store_accepted_sample")[0]

    @property
    def predicted_action(self):
        """
        Get the predicted action.

        Returns:
            npt.NDArray[np.float64]: Predicted action.
        """
        return self.predicted_env.get_attr("store_action")[0]

    @property
    def predicted_reward(self):
        """
        Get the predicted reward.

        Returns:
            npt.NDArray[np.float64]: Predicted reward.
        """
        return self.predicted_env.get_attr("store_reward")[0]

    def switch_actor_weights(
        self,
        load_policy: str = "swa",
    ) -> None:
        """
        Switch the actor weights.
        """
        match load_policy:
            case "best":
                self.actor.load_state_dict(self.best_policy)
            case "last":
                self.actor.load_state_dict(self.last_policy)
            case _:
                raise ValueError(
                    "Invalid load_policy. Must be 'swa', 'best', or 'last'."
                )


class LearningDDPG(LearningInterface):
    """
    DDPG Learning Interface.

    Attributes:
        env (SyncVectorEnv): Environment.
        predicted_env (SyncVectorEnv): Predicted environment.
        random_seed (int): Random seed.
        sample_dim (int): Sample dimension.
        initial_step_size (npt.NDArray[np.float64]): Initial step size.
        total_timesteps (int): Total timesteps.
        actor (torch.nn.Module): Actor.
        target_actor (torch.nn.Module): Target actor.
        critic (torch.nn.Module): Critic.
        target_critic (torch.nn.Module): Target critic.
        actor_optimizer (torch.optim.Optimizer): Actor optimizer.
        critic_optimizer (torch.optim.Optimizer): Critic optimizer.
        replay_buffer (ReplayBuffer): Replay buffer.
        actor_gradient_clipping (bool): Actor gradient clipping.
        actor_gradient_threshold (Optional[float]): Actor gradient threshold.
        actor_gradient_norm (Optional[int]): Actor gradient norm.
        critic_gradient_clipping (bool): Critic gradient clipping.
        critic_gradient_threshold (Optional[float]): Critic gradient threshold.
        critic_gradient_norm (Optional[int]): Critic gradient norm.
        learning_starts (int): Learning starts.
        batch_size (int): Batch size.
        exploration_noise (float): Exploration noise.
        gamma (float): Gamma.
        policy_frequency (int): Policy frequency.
        tau (float): Tau.
        num_of_top_policies (int): Number of top policies to keep.
        device (torch.device): Device.
        verbose (bool): Verbose.
        critic_values (List[float]): Critic values.
        critic_loss (List[float]): Critic loss.
        actor_loss (List[float]): Actor loss.
        best_policy (Optional[Dict[str, Any]]): Best policy.
        writer (SummaryWriter): Summary writer.
        event_manager (EventManager): Event manager.
        track (bool): Track training progress.
    """

    def __init__(
        self,
        env: SyncVectorEnv,
        predicted_env: SyncVectorEnv,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        actor_scheduler: Optional[LRScheduler] = None,
        actor_gradient_clipping: bool = False,
        actor_gradient_threshold: Optional[float] = 1.0,
        actor_gradient_norm: Optional[int] = 2,
        critic_scheduler: Optional[LRScheduler] = None,
        critic_gradient_clipping: bool = False,
        critic_gradient_threshold: Optional[float] = 1.0,
        critic_gradient_norm: Optional[int] = 2,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        random_seed: int = 42,
        num_of_top_policies: int = 5,
        reward_centering: bool = True,
        r_bar: float = 0.0,
        rbar_alpha: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        track: bool = False,
        verbose: bool = True,
        run_name: str = "rlmcmc",
    ) -> None:
        """
        Initialize the DDPG Learning Interface.

        Args:
            env (SyncVectorEnv): Environment.
            predicted_env (SyncVectorEnv): Predicted environment.
            actor (torch.nn.Module): Actor.
            target_actor (torch.nn.Module): Target actor.
            critic (torch.nn.Module): Critic.
            target_critic (torch.nn.Module): Target critic.
            actor_optimizer (torch.optim.Optimizer): Actor optimizer.
            critic_optimizer (torch.optim.Optimizer): Critic optimizer.
            replay_buffer (ReplayBuffer): Replay buffer.
            actor_gradient_clipping (bool, optional): Actor gradient clipping. Defaults to False.
            actor_gradient_threshold (Optional[float], optional): Actor gradient threshold. Defaults to 1.0.
            actor_gradient_norm (Optional[int], optional): Actor gradient norm. Defaults to 2.
            critic_gradient_clipping (bool, optional): Critic gradient clipping. Defaults to False.
            critic_gradient_threshold (Optional[float], optional): Critic gradient threshold. Defaults to 1.0.
            critic_gradient_norm (Optional[int], optional): Critic gradient norm. Defaults to 2.
            learning_starts (int, optional): Learning starts. Defaults to 32.
            batch_size (int, optional): Batch size. Defaults to 32.
            exploration_noise (float, optional): Exploration noise. Defaults to 0.1.
            gamma (float, optional): Gamma. Defaults to 0.99.
            policy_frequency (int, optional): Policy frequency. Defaults to 2.
            tau (float, optional): Tau. Defaults to 0.005.
            random_seed (int, optional): Random seed. Defaults to 42.
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            verbose (bool, optional): Verbose. Defaults to True.
            run_name (str, optional): Run name. Defaults to "rlmcmc".

        Raises:
            ValueError: If the observation space is not continuous.
        """
        super().__init__(
            env=env,
            predicted_env=predicted_env,
            actor=actor,
            target_actor=target_actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            replay_buffer=replay_buffer,
            actor_scheduler=actor_scheduler,
            actor_gradient_clipping=actor_gradient_clipping,
            actor_gradient_threshold=actor_gradient_threshold,
            actor_gradient_norm=actor_gradient_norm,
            critic_scheduler=critic_scheduler,
            critic_gradient_clipping=critic_gradient_clipping,
            critic_gradient_threshold=critic_gradient_threshold,
            critic_gradient_norm=critic_gradient_norm,
            learning_starts=learning_starts,
            batch_size=batch_size,
            exploration_noise=exploration_noise,
            gamma=gamma,
            policy_frequency=policy_frequency,
            tau=tau,
            random_seed=random_seed,
            num_of_top_policies=num_of_top_policies,
            reward_centering=reward_centering,
            r_bar=r_bar,
            rbar_alpha=rbar_alpha,
            device=device,
            track=track,
            verbose=verbose,
            run_name=run_name,
        )

    def trainning_loop(self) -> None:
        """
        Training Session for DDPG.
        """

        if self.current_step < self.learning_starts:
            initial_step_size_unconstrained = Toolbox.inverse_softplus(
                self.initial_step_size
            )
            actions = np.concatenate(
                [initial_step_size_unconstrained, initial_step_size_unconstrained],
                axis=0,
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
        next_obs, rewards, terminations, _, infos = self.env.step(actions)

        self.writer.add_scalars(
            "training/trace",
            {
                f"x{idx}": val.item()
                for idx, val in enumerate(self.obs[0, 0 : self.sample_dim])
            },
            self.current_step,
        )
        self.writer.add_scalar(
            "training/step_size",
            Toolbox.softplus(actions[0, 0]).item(),
            self.current_step,
        )

        if self.current_step == self.learning_starts:
            max_steps_per_episode = self.env.get_attr("max_steps_per_episode")[0]
            episodic_return = (
                self.env.get_attr("store_reward")[0][0 : self.learning_starts].mean()
                * max_steps_per_episode
            )
            self.topk_policy.add(
                (
                    episodic_return,
                    {
                        "actor": self.actor.state_dict(),
                        "step": self.current_step,
                    },
                )
            )

        if "episode" in infos:
            episodic_return = float(infos["episode"]["r"][0])

            self.topk_policy.add(
                (
                    episodic_return,
                    {
                        "actor": self.actor.state_dict(),
                        "step": self.current_step,
                    },
                )
            )

            if self.current_step > self.total_timesteps >> 1:
                if episodic_return > self.best_episodic_return:
                    self.best_episodic_return = episodic_return
                    self.best_policy = self.actor.state_dict()
                    self.best_policy_step = self.current_step

            self.writer.add_scalar(
                "charts/episodic_return", episodic_return, self.current_step
            )
            self.writer.add_scalar(
                "charts/episodic_length", infos["episode"]["l"], self.current_step
            )

        real_next_obs = next_obs.copy()
        self.replay_buffer.add(
            self.obs, real_next_obs, actions, rewards, terminations, infos
        )

        self.obs = next_obs

        if self.current_step == self.learning_starts:
            if self.reward_centering:
                self.r_bar = np.mean(
                    self.env.get_attr("store_reward")[0][0 : self.current_step]
                )
        elif self.current_step > self.learning_starts:
            data = self.replay_buffer.sample(self.batch_size)
            with torch.no_grad():
                next_state_actions = self.target_actor(data.next_observations)
                critic_next_target = self.target_critic(
                    data.next_observations, next_state_actions
                )
                if self.reward_centering:
                    rewards_centered = data.rewards.flatten() - self.r_bar
                    next_q_value = rewards_centered + (
                        1 - data.dones.flatten()
                    ) * self.gamma * critic_next_target.view(-1)
                else:
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (critic_next_target).view(-1)

            critic_a_values = self.critic(data.observations, data.actions).view(-1)
            critic_loss = F.mse_loss(critic_a_values, next_q_value)

            # optimize the model
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.current_step % self.policy_frequency == 0:
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

            if self.reward_centering:
                batch_mean_r = data.rewards.flatten().mean().item()
                delta = batch_mean_r - self.r_bar
                self.r_bar += self.rbar_alpha * delta

            if (
                self.current_step % 100 == 0
                and self.current_step > self.policy_frequency
            ):
                self.writer.add_scalar(
                    "losses/critic_values",
                    critic_a_values.mean().item(),
                    self.current_step,
                )
                self.writer.add_scalar(
                    "losses/critic_loss", critic_loss.item(), self.current_step
                )
                self.writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), self.current_step
                )

                self.critic_values.append(critic_a_values.mean().item())
                self.critic_loss.append(critic_loss.item())
                self.actor_loss.append(actor_loss.item())
        else:
            pass

        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

    def save(self, folder_path: str) -> None:
        """
        Save the model.

        Args:
            folder_path (str): Folder path.
        """
        model_path = f"{folder_path}/ddpg.{time.time()}.pth"
        Toolbox.create_folder(model_path)
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            model_path,
        )


class LearningTD3(LearningInterface):
    """
    TD3 Learning Interface.

    Attributes:
        env (SyncVectorEnv): Environment.
        predicted_env (SyncVectorEnv): Predicted environment.
        random_seed (int): Random seed.
        sample_dim (int): Sample dimension.
        initial_step_size (npt.NDArray[np.float64]): Initial step size.
        total_timesteps (int): Total timesteps.
        actor (torch.nn.Module): Actor.
        target_actor (torch.nn.Module): Target actor.
        critic (torch.nn.Module): Critic.
        target_critic (torch.nn.Module): Target critic.
        actor_optimizer (torch.optim.Optimizer): Actor optimizer.
        critic_optimizer (torch.optim.Optimizer): Critic optimizer.
        replay_buffer (ReplayBuffer): Replay buffer.
        actor_gradient_clipping (bool): Actor gradient clipping.
        actor_gradient_threshold (Optional[float]): Actor gradient threshold.
        actor_gradient_norm (Optional[int]): Actor gradient norm.
        critic_gradient_clipping (bool): Critic gradient clipping.
        critic_gradient_threshold (Optional[float]): Critic gradient threshold.
        critic_gradient_norm (Optional[int]): Critic gradient norm.
        learning_starts (int): Learning starts.
        batch_size (int): Batch size.
        exploration_noise (float): Exploration noise.
        gamma (float): Gamma.
        policy_frequency (int): Policy frequency.
        tau (float): Tau.
        device (torch.device): Device.
        verbose (bool): Verbose.
        critic_values (List[float]): Critic values.
        critic_loss (List[float]): Critic loss.
        actor_loss (List[float]): Actor loss.
        policy_noise (float): Policy noise.
        noise_clip (float): Noise clip.
        critic2 (torch.nn.Module): Critic 2.
        target_critic2 (torch.nn.Module): Target critic 2.
        critic2_values (List[float]): Critic 2 values.
        critic2_loss (List[float]): Critic 2 loss.
        best_policy (Optional[Dict[str, Any]]): Best policy.
        writer (SummaryWriter): Summary writer.
        event_manager (EventManager): Event manager.
        track (bool): Track training progress.
    """

    def __init__(
        self,
        env: SyncVectorEnv,
        predicted_env: SyncVectorEnv,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        critic2: torch.nn.Module,
        target_critic2: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        actor_scheduler: Optional[LRScheduler] = None,
        actor_gradient_clipping: bool = False,
        actor_gradient_threshold: Optional[float] = 1.0,
        actor_gradient_norm: Optional[int] = 2,
        critic_scheduler: Optional[LRScheduler] = None,
        critic_gradient_clipping: bool = False,
        critic_gradient_threshold: Optional[float] = 1.0,
        critic_gradient_norm: Optional[int] = 2,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        random_seed: int = 42,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        num_of_top_policies: int = 5,
        reward_centering: bool = True,
        r_bar: float = 0.0,
        rbar_alpha: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        track: bool = False,
        verbose: bool = True,
        run_name: str = "rlmcmc",
    ) -> None:
        """
        Initialize the TD3 Learning Interface.

        Args:
            env (SyncVectorEnv): Environment.
            predicted_env (SyncVectorEnv): Predicted environment.
            actor (torch.nn.Module): Actor.
            target_actor (torch.nn.Module): Target actor.
            critic (torch.nn.Module): Critic.
            target_critic (torch.nn.Module): Target critic.
            critic2 (torch.nn.Module): Critic 2.
            target_critic2 (torch.nn.Module): Target critic 2.
            actor_optimizer (torch.optim.Optimizer): Actor optimizer.
            critic_optimizer (torch.optim.Optimizer): Critic optimizer.
            replay_buffer (ReplayBuffer): Replay buffer.
            actor_gradient_clipping (bool, optional): Actor gradient clipping. Defaults to False.
            actor_gradient_threshold (Optional[float], optional): Actor gradient threshold. Defaults to 1.0.
            actor_gradient_norm (Optional[int], optional): Actor gradient norm. Defaults to 2.
            critic_gradient_clipping (bool, optional): Critic gradient clipping. Defaults to False.
            critic_gradient_threshold (Optional[float], optional): Critic gradient threshold. Defaults to 1.0.
            critic_gradient_norm (Optional[int], optional): Critic gradient norm. Defaults to 2.
            learning_starts (int, optional): Learning starts. Defaults to 32.
            batch_size (int, optional): Batch size. Defaults to 32.
            exploration_noise (float, optional): Exploration noise. Defaults to 0.1.
            gamma (float, optional): Gamma. Defaults to 0.99.
            policy_frequency (int, optional): Policy frequency. Defaults to 2.
            tau (float, optional): Tau. Defaults to 0.005.
            random_seed (int, optional): Random seed. Defaults to 42.
            policy_noise (float, optional): Policy noise. Defaults to 0.2.
            noise_clip (float, optional): Noise clip. Defaults to 0.5.
            num_of_top_policies (int, optional): Number of top policies to keep. Defaults to 5.
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            track (bool, optional): Track. Defaults to False.
            verbose (bool, optional): Verbose. Defaults to True.
            run_name (str, optional): Run name. Defaults to "rlmcmc".

        Raises:
            ValueError: If the observation space is not continuous.
        """
        super().__init__(
            env=env,
            predicted_env=predicted_env,
            actor=actor,
            target_actor=target_actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            replay_buffer=replay_buffer,
            actor_scheduler=actor_scheduler,
            actor_gradient_clipping=actor_gradient_clipping,
            actor_gradient_threshold=actor_gradient_threshold,
            actor_gradient_norm=actor_gradient_norm,
            critic_scheduler=critic_scheduler,
            critic_gradient_clipping=critic_gradient_clipping,
            critic_gradient_threshold=critic_gradient_threshold,
            critic_gradient_norm=critic_gradient_norm,
            learning_starts=learning_starts,
            batch_size=batch_size,
            exploration_noise=exploration_noise,
            gamma=gamma,
            policy_frequency=policy_frequency,
            tau=tau,
            random_seed=random_seed,
            num_of_top_policies=num_of_top_policies,
            reward_centering=reward_centering,
            r_bar=r_bar,
            rbar_alpha=rbar_alpha,
            device=device,
            track=track,
            verbose=verbose,
            run_name=run_name,
        )

        self.critic2 = critic2
        self.target_critic2 = target_critic2

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.critic2_values: List[float] = []
        self.critic1_loss: List[float] = []
        self.critic2_loss: List[float] = []

    @property
    def critic1_values(self) -> List[float]:
        """
        Get the critic1 values.

        Returns:
            List[float]: Critic1 values.
        """
        return self.critic_values

    def critic_gradient_clipping(self) -> None:
        if self.critic_gradient_clipping:
            for p_critic in self.critic.parameters():
                p_critic.register_hook(
                    self.currying_gradient_clipping(
                        self.critic_gradient_threshold, self.critic_gradient_norm
                    )
                )

            for p_critic2 in self.critic2.parameters():
                p_critic2.register_hook(
                    self.currying_gradient_clipping(
                        self.critic_gradient_threshold, self.critic_gradient_norm
                    )
                )

    def trainning_loop(self) -> None:
        """
        Training Session for TD3.
        """

        if self.current_step < self.learning_starts:
            initial_step_size_unconstrained = Toolbox.inverse_softplus(
                self.initial_step_size
            )
            actions = np.concatenate(
                [initial_step_size_unconstrained, initial_step_size_unconstrained],
                axis=0,
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

        next_obs, rewards, terminations, _, infos = self.env.step(actions)

        if self.current_step == self.learning_starts:
            max_steps_per_episode = self.env.get_attr("max_steps_per_episode")[0]
            episodic_return = (
                self.env.get_attr("store_reward")[0][0 : self.learning_starts].mean()
                * max_steps_per_episode
            )
            self.topk_policy.add(
                (
                    episodic_return,
                    {
                        "actor": self.actor.state_dict(),
                        "step": self.current_step,
                    },
                )
            )

        if "episode" in infos:
            episodic_return = infos["episode"]["r"][0]

            self.topk_policy.add(
                (
                    episodic_return,
                    {
                        "actor": self.actor.state_dict(),
                        "step": self.current_step,
                    },
                )
            )

            if self.current_step > self.total_timesteps >> 1:
                if episodic_return > self.best_episodic_return:
                    self.best_episodic_return = episodic_return
                    self.best_policy = self.actor.state_dict()
                    self.best_policy_step = self.current_step

            self.writer.add_scalar(
                "charts/episodic_return", episodic_return, self.current_step
            )
            self.writer.add_scalar(
                "charts/episodic_length", infos["episode"]["l"], self.current_step
            )

        real_next_obs = next_obs.copy()
        self.replay_buffer.add(
            self.obs, real_next_obs, actions, rewards, terminations, infos
        )

        self.obs = next_obs

        if self.current_step == self.learning_starts:
            if self.reward_centering:
                self.r_bar = np.mean(
                    self.env.get_attr("store_reward")[0][0 : self.current_step]
                )
        elif self.current_step > self.learning_starts:
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

                if self.reward_centering:
                    rewards_centered = data.rewards.flatten() - self.r_bar
                    next_q_value = rewards_centered + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (min_critic_next_target).view(-1)
                else:
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (min_critic_next_target).view(-1)

            critic1_a_values = self.critic(data.observations, data.actions).view(-1)
            critic2_a_values = self.critic2(data.observations, data.actions).view(-1)
            critic1_loss = F.mse_loss(critic1_a_values, next_q_value)
            critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
            critic_loss = critic1_loss + critic2_loss

            # optimize the model
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.current_step % self.policy_frequency == 0:
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

            if self.reward_centering:
                batch_mean_r = data.rewards.flatten().mean().item()
                delta = batch_mean_r - self.r_bar
                self.r_bar += self.rbar_alpha * delta

            if self.current_step % 100 == 0:
                self.writer.add_scalar(
                    "losses/critic1_values",
                    critic1_a_values.mean().item(),
                    self.current_step,
                )
                self.writer.add_scalar(
                    "losses/critic2_values",
                    critic2_a_values.mean().item(),
                    self.current_step,
                )
                self.writer.add_scalar(
                    "losses/critic1_loss", critic1_loss.item(), self.current_step
                )
                self.writer.add_scalar(
                    "losses/critic2_loss", critic2_loss.item(), self.current_step
                )
                self.writer.add_scalar(
                    "losses/critic_loss", critic_loss.item() / 2.0, self.current_step
                )
                self.writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), self.current_step
                )

                self.critic_values.append(critic1_a_values.mean().item())
                self.critic2_values.append(critic2_a_values.mean().item())
                self.critic1_loss.append(critic_loss.item())
                self.critic2_loss.append(critic2_loss.item())
                self.critic_loss.append(critic_loss.item() / 2.0)
                self.actor_loss.append(actor_loss.item())
        else:
            pass

        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

    def save(self, folder_path: str) -> None:
        """
        Save the model.

        Args:
            folder_path (str): Folder path.
        """
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
