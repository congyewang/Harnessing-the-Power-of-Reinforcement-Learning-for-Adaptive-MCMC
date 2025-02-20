import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    SupportsFloat,
    Tuple,
)

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


class MCMCEnvBase(gym.Env[npt.NDArray[np.float64], npt.NDArray[np.float64]], ABC):
    """
    MCMC Environment Base Class

    Attributes:
        log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the log target probability density function without numerical stabilization.
        grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the gradient of the log target probability density function without numerical stabilization.
        initial_sample (npt.NDArray[np.float64]): Initial Sample.
        initial_covariance (npt.NDArray[np.float64]): Initial Covariance.
        initial_step_size (npt.NDArray[np.float64]): Initial Step Size.
        total_timesteps (int): The number of the total time steps in the whole episode.
        log_mode (bool): The controller if reward function returns the logarithmic form.
        sample_dim (int): Sample Dimension.
        steps (int): Iteration Time.
        observation_space (gym.spaces.Box): Observation Specification.
        action_space (gym.spaces.Box): Action Specification.
        state (npt.NDArray[np.float64]): State.
        covariance (npt.NDArray[np.float64]): Covariance. Defaults to initial_covariance.
    """

    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        grad_log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        initial_step_size: npt.NDArray[np.float64] = np.array([1.0]),
        total_timesteps: int = 500_000,
        max_steps_per_episode: int = 500,
        log_mode: bool = True,
    ) -> None:
        """
        Initialize the Environment.

        Args:
            log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
                Function to compute the log target probability density function without numerical stabilization.
            grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
                Function to compute the gradient of the log target probability density function without numerical stabilization.
            initial_sample (npt.NDArray[np.float64]): Initial Sample.
            initial_covariance (npt.NDArray[np.float64], optional): Initial Covariance. Defaults to Identity Matrix.
            initial_step_size (npt.NDArray[np.float64], optional): Initial Step Size. Defaults to 1.0.
            total_timesteps (int, optional): The number of the total time steps in the whole episode. Defaults to 500_000.
            max_steps_per_episode (int, optional): Maximum Steps per Episode. Defaults to 500.
            log_mode (bool, optional): The controller if reward function returns the logarithmic form. Defaults to True.
        """
        super().__init__()

        self.sample_dim: int = int(np.prod(initial_sample.shape))  # Sample Dimension
        self.current_step: int = 0  # Iteration Time
        self.total_timesteps = total_timesteps  # Total Time Steps
        self.max_steps_per_episode = max_steps_per_episode  # Maximum Steps per Episode

        # Log Target Probability Density Functions without Numerical Stabilization
        self.log_target_pdf_unsafe = log_target_pdf_unsafe
        self.grad_log_target_pdf_unsafe = grad_log_target_pdf_unsafe

        self.initial_sample = initial_sample
        if initial_covariance is None:
            initial_covariance = np.eye(self.sample_dim)
        self.initial_covariance: npt.NDArray[np.float64] = initial_covariance
        self.covariance: npt.NDArray[np.float64] = initial_covariance
        self.initial_step_size: npt.NDArray[np.float64] = initial_step_size
        self.log_mode = log_mode

        # Observation specification
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.left_shift(self.sample_dim, 1),),
            dtype=np.float64,
        )
        # Action specification
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float64,
        )

        # Announce State
        self.state: npt.NDArray[np.float64]

        # Store
        self.store_observation: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim << 1)
        )
        self.store_action: npt.NDArray[np.float64] = np.empty((total_timesteps, 2))
        self.store_log_acceptance_rate: npt.NDArray[np.float64] = np.empty(
            total_timesteps
        )
        self.store_accepted_status: npt.NDArray[np.bool] = np.full(
            total_timesteps, False, dtype=bool
        )
        self.store_reward: npt.NDArray[np.float64] = np.empty(total_timesteps)

        self.store_current_sample: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
        )
        self.store_proposed_sample: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
        )

        self.store_current_mean: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
        )
        self.store_proposed_mean: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
        )

        self.store_current_covariance: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim, self.sample_dim)
        )
        self.store_proposed_covariance: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim, self.sample_dim)
        )

        self.store_log_target_proposed: npt.NDArray[np.float64] = np.empty(
            total_timesteps
        )
        self.store_log_target_current: npt.NDArray[np.float64] = np.empty(
            total_timesteps
        )
        self.store_log_proposal_proposed: npt.NDArray[np.float64] = np.empty(
            total_timesteps
        )
        self.store_log_proposal_current: npt.NDArray[np.float64] = np.empty(
            total_timesteps
        )

        self.store_accepted_mean: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
        )
        self.store_accepted_sample: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
        )
        self.store_accepted_covariance: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim, self.sample_dim)
        )

        # Track the items in reward
        self.reward_items: DefaultDict[str, List[np.float64]] = defaultdict(list)

    def softplus(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Softplus Function for Numerical Stability.

        Args:
            x (npt.NDArray[np.float64]): Input array.

        Returns:
            npt.NDArray[np.float64]: Softplus of the input with numerical stabilization.
        """
        return np.logaddexp(x, 0)

    def inverse_softplus(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Inverse softplus Function for Numerical Stability. y = log(exp(x) - 1).

        Args:
            x (npt.NDArray[np.float64]): Input array.

        Returns:
            npt.NDArray[np.float64]: Inverse softplus of the input with numerical stabilization.
        """
        return x + np.log1p(-np.exp(-x))

    def expected_entropy_reward(
        self,
        log_target_current: npt.NDArray[np.float64],
        log_target_proposed: npt.NDArray[np.float64],
        log_proposal_current: npt.NDArray[np.float64],
        log_alpha: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Expected Entropy Reward

        Args:
            log_target_current (npt.NDArray[np.float64]): Log target density at current sample.
            log_target_proposed (npt.NDArray[np.float64]): Log target density at proposed sample.
            log_proposal_current (npt.NDArray[np.float64]): Log proposal density at current sample.
            log_alpha (npt.NDArray[np.float64]): Log acceptance rate.

        Returns:
            npt.NDArray[np.float64]: Expected Entropy Reward.
        """
        log_one_minus_alpha = np.log1p(-np.exp(log_alpha))
        if np.isinf(log_one_minus_alpha) or np.isnan(log_one_minus_alpha):
            log_one_minus_alpha = -np.finfo(np.float64).max

        transient_item = np.exp(log_alpha) * (log_target_proposed - log_target_current)
        entropy_item = (
            -np.exp(log_one_minus_alpha) * log_one_minus_alpha
            - np.exp(log_alpha) * log_alpha
        )
        expected_square_jump_distance_item = -np.exp(log_alpha) * log_proposal_current

        self.reward_items["transient"].append(transient_item.item())
        self.reward_items["entropy"].append(entropy_item.item())
        self.reward_items["expected_square_jump_distance"].append(
            expected_square_jump_distance_item.item()
        )

        if self.log_mode:
            res = transient_item + entropy_item + expected_square_jump_distance_item
        else:
            res = np.exp(
                transient_item + entropy_item + expected_square_jump_distance_item
            )

        return res

    def expected_square_jump_distance(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        log_alpha: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Expected Square Jump Distance Reward Function.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample.
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample.
            log_alpha (npt.NDArray[np.float64]): Log Acceptance Rate.

        Returns:
            npt.NDArray[np.float64]: Expected Square Jump Distance Reward.
        """
        if self.log_mode:
            reward = (
                2 * np.log(np.linalg.norm(current_sample - proposed_sample, 2))
                + log_alpha
            )
        else:
            reward = np.linalg.norm(current_sample - proposed_sample, 2) ** 2 * np.exp(
                log_alpha
            )

        return reward

    def log_target_pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Log Target Probability Density Function.

        Args:
            x (npt.NDArray[np.float64]): Sample.

        Returns:
            npt.NDArray[np.float64]: Log Target Probability Density at x.
        """
        res = self.log_target_pdf_unsafe(x)

        if np.isinf(res):
            res = -np.finfo(np.float64).max
            warnings.warn(f"log_target_pdf is inf or -inf, where x: {x}.")
        return res

    def grad_log_target_pdf(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Gradient of Log Target Probability Density Function.

        Args:
            x (npt.NDArray[np.float64]): Sample.

        Returns:
            npt.NDArray[np.float64]: Gradient of Log Target Probability Density at x.
        """
        res = self.grad_log_target_pdf_unsafe(x)

        if np.isinf(res).any():
            res = np.where(np.isinf(res), 0, res)
            warnings.warn(f"grad_log_target_pdf is inf or -inf, where x: {x}.")
        return res

    def log_target_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Log Target Process.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample.
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                Log Target Current, Log Target Proposed, Log Target Ratio.
        """
        log_target_current = self.log_target_pdf(current_sample)
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_ratio = log_target_proposed - log_target_current

        return log_target_current, log_target_proposed, log_target_ratio

    @abstractmethod
    def log_proposal_pdf(
        self, x: npt.NDArray[np.float64], *args, **kwargs
    ) -> npt.NDArray[np.float64]:
        """
        Log Proposal Probability Density Function.

        Args:
            x (npt.NDArray[np.float64]): Sample.

        Raises:
            NotImplementedError: log_proposal_pdf is not implemented.

        Returns:
            npt.NDArray[np.float64]: Log Proposal Probability Density.
        """
        raise NotImplementedError("log_proposal_pdf is not implemented.")

    @abstractmethod
    def log_proposal_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        *args,
        **kwargs,
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Log Proposal Process.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample
            proposed_sample (np.float64, npt.NDArray[np.float64]): Proposed Sample

        Raises:
            NotImplementedError: log_proposal_process is not implemented.

        Returns:
            Tuple[np.float64, np.float64, np.float64]:
                Log Proposal Current, Log Proposal Proposed, Log Proposal Ratio
        """
        raise NotImplementedError("log_proposal_process is not implemented.")

    def store_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        current_mean: npt.NDArray[np.float64],
        proposed_mean: npt.NDArray[np.float64],
        current_covariance: npt.NDArray[np.float64],
        proposed_covariance: npt.NDArray[np.float64],
        log_target_current: npt.NDArray[np.float64],
        log_target_proposed: npt.NDArray[np.float64],
        log_proposal_current: npt.NDArray[np.float64],
        log_proposal_proposed: npt.NDArray[np.float64],
        accepted_status: bool,
        log_alpha: npt.NDArray[np.float64],
        accepted_sample: npt.NDArray[np.float64],
        accepted_mean: npt.NDArray[np.float64],
        accepted_covariance: npt.NDArray[np.float64],
    ) -> None:
        """
        Store Process. Store the information of the current step.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample
            current_mean (npt.NDArray[np.float64]): Current Mean
            proposed_mean (npt.NDArray[np.float64]): Proposed Mean
            current_covariance (npt.NDArray[np.float64]): Current Covariance
            proposed_covariance (npt.NDArray[np.float64]): Proposed Covariance
            log_target_current (np.float64): Log Target Probability Density at Current Sample
            log_target_proposed (np.float64): Log Target Probability Density at Proposed Sample
            log_proposal_current (np.float64): Log Proposal Probability Density at Current Sample
            log_proposal_proposed (np.float64): Log Proposal Probability Density at Proposed Sample
            accepted_status (bool): Whether the proposed sample is accepted
            log_alpha (np.float64): Log Acceptance Rate
            accepted_sample (npt.NDArray[np.float64]): Accepted Sample
            accepted_mean (npt.NDArray[np.float64]): Accepted Mean
            accepted_covariance (npt.NDArray[np.float64]): Accepted Covariance
        """
        # Store Sample
        self.store_current_sample[self.current_step, :] = current_sample
        self.store_proposed_sample[self.current_step, :] = proposed_sample

        # Store Mean
        self.store_current_mean[self.current_step, :] = current_mean
        self.store_proposed_mean[self.current_step, :] = proposed_mean

        # Store Covariance
        self.store_current_covariance[self.current_step, :, :] = current_covariance
        self.store_proposed_covariance[self.current_step, :, :] = proposed_covariance

        # Store Log Densities
        self.store_log_target_current[self.current_step] = log_target_current
        self.store_log_target_proposed[self.current_step] = log_target_proposed

        self.store_log_proposal_current[self.current_step] = log_proposal_current
        self.store_log_proposal_proposed[self.current_step] = log_proposal_proposed

        # Store Acceptance
        self.store_accepted_status[self.current_step] = accepted_status
        self.store_log_acceptance_rate[self.current_step] = log_alpha

        self.store_accepted_sample[self.current_step, :] = accepted_sample
        self.store_accepted_mean[self.current_step, :] = accepted_mean
        self.store_accepted_covariance[self.current_step, :, :] = accepted_covariance

    @abstractmethod
    def sample_generator(self, *args, **kwargs) -> npt.NDArray[np.float64]:
        """
        Sample generator, which is used to generate the next proposed sample. This function should be implemented.

        Raises:
            NotImplementedError: mcmc_noise is not implemented.

        Returns:
            npt.NDArray[np.float64]: Next Proposed Sample.
        """
        raise NotImplementedError("mcmc_noise is not implemented.")

    @abstractmethod
    def step(
        self, action: npt.NDArray[np.float64]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step Function for Environment. This function should be implemented.

        Args:
            action (npt.NDArray[np.float64]): Step Size Before Softplus Function.

        Raises:
            NotImplementedError: step is not implemented.
        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
                State, Reward, Terminated, Truncated, Info.
        """
        raise NotImplementedError("step is not implemented.")

    def _initialize_state(self) -> None:
        """
        Initialize the state of the environment. This function is used to initialize the state of the environment.
        """
        initial_next_proposed_sample = self.np_random.multivariate_normal(
            mean=self.initial_sample, cov=self.initial_covariance, size=1
        ).flatten()

        self.state = np.concatenate((self.initial_sample, initial_next_proposed_sample))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Reset environment to initial state and return initial observation.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (Dict[str, Any], optional): Defaults to None.

        Returns:
            tuple[Any, Dict[str, Any]]: Initial Observation, Info
        """
        # Call the super class reset to handle seeding and other logic
        super().reset(seed=seed, options=options)

        if self.current_step == 0:
            self._initialize_state()

        return self.state, {}


class BarkerEnv(MCMCEnvBase):
    """
    Barker Environment. This is a simple environment to illustrate how to sample from the Barker proposal.

    Attributes:
        log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the log target probability density function without numerical stabilization.
        grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the gradient of the log target probability density function without numerical stabilization.
        initial_sample (npt.NDArray[np.float64]): Initial Sample.
        initial_covariance (npt.NDArray[np.float64]): Initial Covariance.
        initial_step_size (npt.NDArray[np.float64]): Initial Step Size.
        total_timesteps (int): The number of the total time steps in the whole episode.
        log_mode (bool): The controller if reward function returns the logarithmic form.
        sample_dim (int): Sample Dimension.
        steps (int): Iteration Time.
        observation_space (gym.spaces.Box): Observation Specification.
        action_space (gym.spaces.Box): Action Specification.
        state (npt.NDArray[np.float64]): State.
        covariance (npt.NDArray[np.float64]): Covariance. Defaults to initial_covariance.
    """

    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        grad_log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        initial_step_size: npt.NDArray[np.float64] = np.array([1.0]),
        total_timesteps: int = 500_000,
        max_steps_per_episode: int = 500,
        log_mode: bool = True,
    ) -> None:
        """
        Initialize the Environment.

        Args:
            log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]): _description_
            grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]): _description_
            initial_sample (npt.NDArray[np.float64]): _description_
            initial_covariance (Optional[npt.NDArray[np.float64]], optional): _description_. Defaults to None.
            initial_step_size (npt.NDArray[np.float64], optional): _description_. Defaults to np.array([1.0]).
            total_timesteps (int, optional): _description_. Defaults to 500_000.
            max_steps_per_episode (int, optional): _description_. Defaults to 500.
            log_mode (bool, optional): _description_. Defaults to True.
        """
        super().__init__(
            log_target_pdf_unsafe,
            grad_log_target_pdf_unsafe,
            initial_sample,
            initial_covariance,
            initial_step_size,
            total_timesteps,
            max_steps_per_episode,
            log_mode,
        )

    def _log_mu_sigma(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        step_size: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Log of mu_{sigma}.

        Args:
            x (npt.NDArray[np.float64]): Sample.
            y (npt.NDArray[np.float64]): Sample.
            step_size (np.float64): Step Size.

        Returns:
            npt.NDArray[np.float64]: Log of mu_{sigma}.
        """
        position = (y - x) / step_size

        return -self.sample_dim * np.log(step_size) + multivariate_normal.logpdf(
            position, np.zeros(self.sample_dim), np.eye(self.sample_dim)
        )

    def sample_generator(
        self,
        x: npt.NDArray[np.float64],
        grad_x: npt.NDArray[np.float64],
        step_size: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        A simple function to illustrate how to sample from the Barker proposal (isotropic version, i.e. no preconditioning)

        Args:
            x (npt.NDArray[np.float64]): Sample.
            grad_x (npt.NDArray[np.float64]): Gradient of Log Target Probability Density at x.
            step_size (np.float64): Step Size.

        Returns:
            npt.NDArray[np.float64]: Next Proposed Sample.
        """
        z = self.np_random.normal(0, step_size, self.sample_dim)
        u = self.np_random.uniform(size=self.sample_dim)
        b = 2 * (u < 1 / (1 + np.exp(-grad_x * z))) - 1

        return x + z * b

    def log_proposal_pdf(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        step_size_x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Log Proposal Probability Density Function.

        Args:
            x (npt.NDArray[np.float64]): Sample.
            y (npt.NDArray[np.float64]): Sample.
            step_size_x (npt.NDArray[np.float64]): Step Size at x.

        Returns:
            npt.NDArray[np.float64]: Log Proposal Probability Density.
        """
        return self.sample_dim * np.log(2) + np.sum(
            self._log_mu_sigma(x, y, step_size_x)
            - self.softplus((x - y) * self.grad_log_target_pdf(x))
        )

    def log_proposal_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        current_step_size: npt.NDArray[np.float64],
        proposed_step_size: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Log Proposal Process.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample
            current_step_size (npt.NDArray[np.float64]): Step Size at Current Sample
            proposed_step_size (npt.NDArray[np.float64]): Step Size at Proposed Sample

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                Log Proposal Current, Log Proposal Proposed, Log Proposal Ratio
        """
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, current_step_size
        )
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, proposed_step_size
        )

        log_proposal_ratio = log_proposal_proposed - log_proposal_current

        return log_proposal_current, log_proposal_proposed, log_proposal_ratio

    def accepted_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        current_mean: npt.NDArray[np.float64],
        proposed_mean: npt.NDArray[np.float64],
        current_covariance: npt.NDArray[np.float64],
        proposed_covariance: npt.NDArray[np.float64],
        current_step_size: npt.NDArray[np.float64],
        proposed_step_size: npt.NDArray[np.float64],
    ) -> Tuple[
        bool,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Accepted / Rejected Process. This function is used to determine whether the proposed sample is accepted or rejected.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample.
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample.
            current_mean (npt.NDArray[np.float64]): Current Mean.
            proposed_mean (npt.NDArray[np.float64]): Proposed Mean.
            current_covariance (npt.NDArray[np.float64]): Current Covariance.
            proposed_covariance (npt.NDArray[np.float64]): Proposed Covariance.
            current_step_size (npt.NDArray[np.float64]): Step Size at Current Sample.
            proposed_step_size (npt.NDArray[np.float64]): Step Size at Proposed Sample.

        Returns:
            Tuple[ bool, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64]:
                Accepted Status, Accepted Sample, Accepted Mean, Accepted Covariance, Log Acceptance Rate
        """
        # Calculate Log Target Density
        log_target_current, log_target_proposed, log_target_ratio = (
            self.log_target_process(current_sample, proposed_sample)
        )

        # Calculate Log Proposal Densitys
        log_proposal_current, log_proposal_proposed, log_proposed_ratio = (
            self.log_proposal_process(
                current_sample, proposed_sample, current_step_size, proposed_step_size
            )
        )

        # Calculate Log Acceptance Rate
        log_alpha = np.minimum(0.0, log_target_ratio + log_proposed_ratio)

        # Accept or Reject
        if np.log(self.np_random.random()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_mean = proposed_mean
            accepted_covariance = proposed_covariance
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_mean = current_mean
            accepted_covariance = current_covariance

        # Store
        self.store_process(
            current_sample,
            proposed_sample,
            current_mean,
            proposed_mean,
            current_covariance,
            proposed_covariance,
            log_target_current,
            log_target_proposed,
            log_proposal_current,
            log_proposal_proposed,
            accepted_status,
            log_alpha,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
        )

        return (
            accepted_status,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
            log_alpha,
        )

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]:
        """
        Step Function for Environment. This function is used to update the environment state.

        Args:
            action (npt.NDArray[np.float64]): Step Size Before Softplus Function, which is unconstrained.

        Returns:
            Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]: State, Reward, Terminated, Truncated, Info.
        """
        # Unpack state
        current_sample, proposed_sample = np.split(self.state, 2)

        # Calculate phi
        current_phi, proposed_phi = self.softplus(action)

        # Mean and Coveriance
        [current_mean, proposed_mean] = [current_sample for _ in range(2)]
        [current_covariance, proposed_covariance] = [self.covariance for _ in range(2)]

        # Accept or Reject
        accepted_status, accepted_sample, _, _, log_alpha = self.accepted_process(
            current_sample,
            proposed_sample,
            current_mean,
            proposed_mean,
            current_covariance,
            proposed_covariance,
            current_phi,
            proposed_phi,
        )

        # Update Observation
        accepted_phi = proposed_phi if accepted_status else current_phi

        accepted_grad_log_pdf = self.grad_log_target_pdf(accepted_sample)
        next_proposed_sample = self.sample_generator(
            accepted_sample, accepted_grad_log_pdf, accepted_phi
        )
        observation = np.concatenate((accepted_sample, next_proposed_sample))
        self.state = observation

        # Calculate Reward
        log_target_current, log_target_proposed, _ = self.log_target_process(
            current_sample, proposed_sample
        )

        log_proposal_current, _, _ = self.log_proposal_process(
            current_sample, proposed_sample, current_phi, proposed_phi
        )

        reward = self.expected_entropy_reward(
            log_target_current, log_target_proposed, log_proposal_current, log_alpha
        )

        # Store
        self.store_observation[self.current_step, :] = observation
        self.store_action[self.current_step, :] = action
        self.store_reward[self.current_step] = reward

        # Update Steps
        self.current_step += 1
        truncated: bool = (self.current_step + 1) % self.max_steps_per_episode == 0
        terminated: bool = False
        info: Dict[None, None] = {}

        return self.state, reward.item(), terminated, truncated, info


class BarkerESJDEnv(BarkerEnv):
    """
    Barker Environment with Expected Square Jump Distance Reward.

    Attributes:
        log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the log target probability density function without numerical stabilization.
        grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the gradient of the log target probability density function without numerical stabilization.
        initial_sample (npt.NDArray[np.float64]): Initial Sample.
        initial_covariance (npt.NDArray[np.float64]): Initial Covariance.
        initial_step_size (npt.NDArray[np.float64]): Initial Step Size.
        total_timesteps (int): The number of the total time steps in the whole episode.
        log_mode (bool): The controller if reward function returns the logarithmic form.
        sample_dim (int): Sample Dimension.
        steps (int): Iteration Time.
        observation_space (gym.spaces.Box): Observation Specification.
        action_space (gym.spaces.Box): Action Specification.
        state (npt.NDArray[np.float64]): State.
        covariance (npt.NDArray[np.float64]): Covariance. Defaults to initial_covariance.
    """

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]:
        """
        Step Function for Environment. This function is used to update the environment state.

        Args:
            action (npt.NDArray[np.float64]): Step Size Before Softplus Function, which is unconstrained.

        Returns:
            Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]: State, Reward, Terminated, Truncated, Info.
        """
        # Unpack state
        current_sample, proposed_sample = np.split(self.state, 2)

        # Calculate phi
        current_phi, proposed_phi = self.softplus(action)

        # Mean and Coveriance
        [current_mean, proposed_mean] = [current_sample for _ in range(2)]
        [current_covariance, proposed_covariance] = [self.covariance for _ in range(2)]

        # Accept or Reject
        accepted_status, accepted_sample, _, _, log_alpha = self.accepted_process(
            current_sample,
            proposed_sample,
            current_mean,
            proposed_mean,
            current_covariance,
            proposed_covariance,
            current_phi,
            proposed_phi,
        )

        # Update Observation
        accepted_phi = proposed_phi if accepted_status else current_phi

        accepted_grad_log_pdf = self.grad_log_target_pdf(accepted_sample)
        next_proposed_sample = self.sample_generator(
            accepted_sample, accepted_grad_log_pdf, accepted_phi
        )
        observation = np.concatenate((accepted_sample, next_proposed_sample))
        self.state = observation

        # Calculate Reward
        reward = self.expected_square_jump_distance(
            current_sample, proposed_sample, log_alpha
        )

        # Store
        self.store_observation[self.current_step, :] = observation
        self.store_action[self.current_step, :] = action
        self.store_reward[self.current_step] = reward

        # Update Steps
        self.current_step += 1
        truncated: bool = (self.current_step + 1) % self.max_steps_per_episode == 0
        terminated: bool = False
        info: Dict[None, None] = {}

        return self.state, reward.item(), terminated, truncated, info


class MALAEnv(MCMCEnvBase):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) Environment.

    Attributes:
        log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the log target probability density function without numerical stabilization.
        grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the gradient of the log target probability density function without numerical stabilization.
        initial_sample (npt.NDArray[np.float64]): Initial Sample.
        initial_covariance (npt.NDArray[np.float64]): Initial Covariance.
        initial_step_size (npt.NDArray[np.float64]): Initial Step Size.
        total_timesteps (int): The number of the total time steps in the whole episode.
        log_mode (bool): The controller if reward function returns the logarithmic form.
        sample_dim (int): Sample Dimension.
        steps (int): Iteration Time.
        observation_space (gym.spaces.Box): Observation Specification.
        action_space (gym.spaces.Box): Action Specification.
        state (npt.NDArray[np.float64]): State.
        covariance (npt.NDArray[np.float64]): Covariance. Defaults to initial_covariance.
    """

    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        grad_log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        initial_step_size: npt.NDArray[np.float64] = np.array([1.0]),
        total_timesteps: int = 500_000,
        max_steps_per_episode: int = 500,
        log_mode: bool = True,
    ) -> None:
        """
        Initialize the Environment.

        Args:
            log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]): _description_
            grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]): _description_
            initial_sample (npt.NDArray[np.float64]): _description_
            initial_covariance (Optional[npt.NDArray[np.float64]], optional): _description_. Defaults to None.
            initial_step_size (npt.NDArray[np.float64], optional): _description_. Defaults to np.array([1.0]).
            total_timesteps (int, optional): _description_. Defaults to 500_000.
            max_steps_per_episode (int, optional): _description_. Defaults to 500.
            log_mode (bool, optional): _description_. Defaults to True.
        """
        super().__init__(
            log_target_pdf_unsafe,
            grad_log_target_pdf_unsafe,
            initial_sample,
            initial_covariance,
            initial_step_size,
            total_timesteps,
            max_steps_per_episode,
            log_mode,
        )

    def _compute_mean_and_covariance(
        self,
        sample: npt.NDArray[np.float64],
        step_size: npt.NDArray[np.float64],
        covariance: npt.NDArray[np.float64],
        grad_log_pdf: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute Mean and Covariance. This function is used to compute the mean and covariance.

        Args:
            sample (npt.NDArray[np.float64]): Sample.
            step_size (npt.NDArray[np.float64]): Step Size.
            covariance (npt.NDArray[np.float64]): Covariance.
            grad_log_pdf (npt.NDArray[np.float64]): Gradient of Log Target Probability Density at sample.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Mean, Covariance
        """
        mean = sample + step_size / 2 * covariance @ grad_log_pdf
        covariance = step_size * covariance

        return mean, covariance

    def sample_generator(
        self,
        mean: npt.NDArray[np.float64],
        covariance: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        MALA Proposal. This function is used to generate the next proposed sample.

        Args:
            mean (npt.NDArray[np.float64]): Mean.
            covariance (npt.NDArray[np.float64]): Covariance.

        Returns:
            npt.NDArray[np.float64]: Next Proposed Sample.
        """
        return self.np_random.multivariate_normal(mean, covariance, size=1).flatten()

    def log_proposal_pdf(
        self,
        x: npt.NDArray[np.float64],
        mean: npt.NDArray[np.float64],
        covariance: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Log Proposal Probability Density Function. This function is used to compute the log proposal probability density function.

        Args:
            x (npt.NDArray[np.float64]): Sample.
            mean (npt.NDArray[np.float64]): Mean.
            covariance (npt.NDArray[np.float64]): Covariance.

        Returns:
            npt.NDArray[np.float64]: Log Proposal Probability Density.
        """
        res = multivariate_normal.logpdf(x, mean, covariance)

        if np.isinf(res) or np.isnan(res):
            res = -np.finfo(np.float64).max
            warnings.warn(f"log_proposal_pdf is inf or -inf, where x: {x}.")

        return res

    def log_proposal_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        current_covariance: npt.NDArray[np.float64],
        proposed_covariance: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Log Proposal Process.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample
            current_covariance (npt.NDArray[np.float64]): Current Covariance
            proposed_covariance (npt.NDArray[np.float64]): Proposed Covariance

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                Log Proposal Current, Log Proposal Proposed, Log Proposal Ratio
        """
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, current_covariance
        )
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, proposed_covariance
        )

        log_proposal_ratio = log_proposal_proposed - log_proposal_current

        return log_proposal_current, log_proposal_proposed, log_proposal_ratio

    def accepted_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        current_mean: npt.NDArray[np.float64],
        proposed_mean: npt.NDArray[np.float64],
        current_covariance: npt.NDArray[np.float64],
        proposed_covariance: npt.NDArray[np.float64],
    ) -> Tuple[
        bool,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Accepted / Rejected Process. This function is used to determine whether the proposed sample is accepted or rejected.

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample.
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample.
            current_mean (npt.NDArray[np.float64]): Current Mean.
            proposed_mean (npt.NDArray[np.float64]): Proposed Mean.
            current_covariance (npt.NDArray[np.float64]): Current Covariance.
            proposed_covariance (npt.NDArray[np.float64]): Proposed Covariance.

        Returns:
            Tuple[ bool, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64]:
                Accepted Status, Accepted Sample, Accepted Mean, Accepted Covariance, Log Acceptance Rate
        """
        # Calculate Log Target Density
        log_target_current, log_target_proposed, log_target_ratio = (
            self.log_target_process(current_sample, proposed_sample)
        )

        # Calculate Log Proposal Densitys
        log_proposal_current, log_proposal_proposed, log_proposed_ratio = (
            self.log_proposal_process(
                current_sample, proposed_sample, current_covariance, proposed_covariance
            )
        )

        # Calculate Log Acceptance Rate
        log_alpha = np.minimum(0.0, log_target_ratio + log_proposed_ratio)

        # Accept or Reject
        if np.log(self.np_random.random()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_mean = proposed_mean
            accepted_covariance = proposed_covariance
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_mean = current_mean
            accepted_covariance = current_covariance

        # Store
        self.store_process(
            current_sample,
            proposed_sample,
            current_mean,
            proposed_mean,
            current_covariance,
            proposed_covariance,
            log_target_current,
            log_target_proposed,
            log_proposal_current,
            log_proposal_proposed,
            accepted_status,
            log_alpha,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
        )

        return (
            accepted_status,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
            log_alpha,
        )

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]:
        """
        Step Function for Environment. This function is used to update the environment state.

        Args:
            action (npt.NDArray[np.float64]): Step Size Before Softplus Function, which is unconstrained.

        Returns:
            Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]: State, Reward, Terminated, Truncated, Info.
        """
        # Unpack state
        current_sample, proposed_sample = np.split(self.state, 2)

        # Calculate phi
        current_phi, proposed_phi = self.softplus(action)

        # Mean and Coveriance
        current_grad_log_pdf, proposed_grad_log_pdf = self.grad_log_target_pdf(
            current_sample
        ), self.grad_log_target_pdf(proposed_sample)

        current_mean, current_covariance = self._compute_mean_and_covariance(
            current_sample, current_phi, self.covariance, current_grad_log_pdf
        )
        proposed_mean, proposed_covariance = self._compute_mean_and_covariance(
            proposed_sample, proposed_phi, self.covariance, proposed_grad_log_pdf
        )

        # Accept or Reject
        _, accepted_sample, accepted_mean, accepted_covariance, log_alpha = (
            self.accepted_process(
                current_sample,
                proposed_sample,
                current_mean,
                proposed_mean,
                current_covariance,
                proposed_covariance,
            )
        )

        # Update Observation
        next_proposed_sample = self.sample_generator(accepted_mean, accepted_covariance)
        observation = np.concatenate((accepted_sample, next_proposed_sample))
        self.state = observation

        # Calculate Reward
        log_target_current, log_target_proposed, _ = self.log_target_process(
            current_sample, proposed_sample
        )

        log_proposal_current, _, _ = self.log_proposal_process(
            current_sample, proposed_sample, current_covariance, proposed_covariance
        )

        reward = self.expected_entropy_reward(
            log_target_current, log_target_proposed, log_proposal_current, log_alpha
        )

        # Store
        self.store_observation[self.current_step, :] = observation
        self.store_action[self.current_step, :] = action
        self.store_reward[self.current_step] = reward

        # Update Steps
        self.current_step += 1
        truncated: bool = (self.current_step + 1) % self.max_steps_per_episode == 0
        terminated: bool = False
        info: Dict[None, None] = {}

        return self.state, reward.item(), terminated, truncated, info


class MALAESJDEnv(MALAEnv):
    """
    MALAESJDEnv is an environment for performing MALA (Metropolis-Adjusted Langevin Algorithm) transitions
    with expected square jump distance rewards.

    Attributes:
        log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the log target probability density function without numerical stabilization.
        grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
            Function to compute the gradient of the log target probability density function without numerical stabilization.
        initial_sample (npt.NDArray[np.float64]): Initial Sample.
        initial_covariance (npt.NDArray[np.float64]): Initial Covariance.
        initial_step_size (npt.NDArray[np.float64]): Initial Step Size.
        total_timesteps (int): The number of the total time steps in the whole episode.
        log_mode (bool): The controller if reward function returns the logarithmic form.
        sample_dim (int): Sample Dimension.
        steps (int): Iteration Time.
        observation_space (gym.spaces.Box): Observation Specification.
        action_space (gym.spaces.Box): Action Specification.
        state (npt.NDArray[np.float64]): State.
        covariance (npt.NDArray[np.float64]): Covariance. Defaults to initial_covariance
    """

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]:
        """
        Step Function for Environment. This function is used to update the environment state.

        Args:
            action (npt.NDArray[np.float64]): Step Size Before Softplus Function, which is unconstrained.

        Returns:
            Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]: State, Reward, Terminated, Truncated, Info.
        """
        # Unpack state
        current_sample, proposed_sample = np.split(self.state, 2)

        # Calculate phi
        current_phi, proposed_phi = self.softplus(action)

        # Mean and Coveriance
        current_grad_log_pdf, proposed_grad_log_pdf = self.grad_log_target_pdf(
            current_sample
        ), self.grad_log_target_pdf(proposed_sample)

        current_mean, current_covariance = self._compute_mean_and_covariance(
            current_sample, current_phi, self.covariance, current_grad_log_pdf
        )
        proposed_mean, proposed_covariance = self._compute_mean_and_covariance(
            proposed_sample, proposed_phi, self.covariance, proposed_grad_log_pdf
        )

        # Accept or Reject
        _, accepted_sample, accepted_mean, accepted_covariance, log_alpha = (
            self.accepted_process(
                current_sample,
                proposed_sample,
                current_mean,
                proposed_mean,
                current_covariance,
                proposed_covariance,
            )
        )

        # Update Observation
        next_proposed_sample = self.sample_generator(accepted_mean, accepted_covariance)
        observation = np.concatenate((accepted_sample, next_proposed_sample))
        self.state = observation

        # Calculate Reward
        reward = self.expected_square_jump_distance(
            current_sample, proposed_sample, log_alpha
        )

        # Store
        self.store_observation[self.current_step, :] = observation
        self.store_action[self.current_step, :] = action
        self.store_reward[self.current_step] = reward

        # Update Steps
        self.current_step += 1
        truncated: bool = (self.current_step + 1) % self.max_steps_per_episode == 0
        terminated: bool = False
        info: Dict[None, None] = {}

        return self.state, reward.item(), terminated, truncated, info
