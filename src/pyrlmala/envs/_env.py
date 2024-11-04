import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, SupportsFloat, Tuple, Union, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import scipy
from gymnasium.utils import seeding
from scipy.stats import multivariate_normal


class RLMCMCEnvBase(gym.Env[npt.NDArray[np.float64], npt.NDArray[np.float64]], ABC):
    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        grad_log_target_pdf_unsafe: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Union[npt.NDArray[np.float64], None] = None,
        total_timesteps: int = 500_000,
        log_mode: bool = True,
    ) -> None:
        """Initialize the Environment

        Args:
            log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
                Function to compute the log target probability density function without numerical stabilization.
            grad_log_target_pdf_unsafe (Callable[ [npt.NDArray[np.float64]], npt.NDArray[np.float64] ]):
                Function to compute the gradient of the log target probability density function without numerical stabilization.
            initial_sample (npt.NDArray[np.float64]): Initial Sample.
            initial_covariance (Union[npt.NDArray[np.float64], None], optional): Initial Covariance. Defaults to Identity Matrix.
            total_timesteps (int, optional): The number of the total time steps in the whole episode. Defaults to 500_000.
            log_mode (bool, optional): The controller if reward function returns the logarithmic form. Defaults to True.
        """
        super().__init__()

        self.sample_dim: int = int(np.prod(initial_sample.shape))  # Sample Dimension
        self.steps: int = 1  # Iteration Time

        # Log Target Probability Density Functions without Numerical Stabilization
        self.log_target_pdf_unsafe = log_target_pdf_unsafe
        self.grad_log_target_pdf_unsafe = grad_log_target_pdf_unsafe

        if initial_covariance is None:
            initial_covariance = (2.38 / np.sqrt(self.sample_dim)) * np.eye(
                self.sample_dim
            )
        self.initial_covariance = initial_covariance
        self.covariance = initial_covariance
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

        # Initialize State
        initial_next_proposed_sample = self.np_random.multivariate_normal(
            mean=initial_sample, cov=initial_covariance, size=1
        ).flatten()
        self.state = np.concatenate(
            (initial_sample, initial_next_proposed_sample)
        )  # state at self time, s_{t}

        # Store
        self.store_observation: npt.NDArray[np.float64] = np.empty(
            (total_timesteps, self.sample_dim)
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

    def expected_entropy_reward(
        self,
        log_target_current: npt.NDArray[np.float64],
        log_target_accepted: npt.NDArray[np.float64],
        log_proposal_current: npt.NDArray[np.float64],
        log_alpha: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Expected Entropy Reward

        Args:
            log_target_current (npt.NDArray[np.float64]): Log target density at current sample
            log_target_accepted (npt.NDArray[np.float64]): Log target density at accepted sample
            log_proposal_current (npt.NDArray[np.float64]): Log proposal density at current sample
            log_alpha (npt.NDArray[np.float64]): Log acceptance rate

        Returns:
            npt.NDArray[np.float64]: Expected Entropy Reward
        """
        log_one_minus_alpha = np.log1p(-np.exp(log_alpha))
        if np.isinf(log_one_minus_alpha) or np.isnan(log_one_minus_alpha):
            log_one_minus_alpha = -np.finfo(np.float64).max

        transient_item = log_target_accepted - log_target_current
        entropy_item = (
            -np.exp(log_one_minus_alpha) * log_one_minus_alpha
            - np.exp(log_alpha) * log_alpha
        )
        expected_square_jump_distance_item = -np.exp(log_alpha) * log_proposal_current

        if self.log_mode:
            res = transient_item + entropy_item + expected_square_jump_distance_item
        else:
            res = np.exp(
                transient_item + entropy_item + expected_square_jump_distance_item
            )

        return res

    def log_target_pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Log Target Probability Density Function

        Args:
            x (npt.NDArray[np.float64]): Sample

        Returns:
            npt.NDArray[np.float64]: Log Target Probability Density at x
        """
        res = self.log_target_pdf_unsafe(x)

        if np.isinf(res):
            res = -np.finfo(np.float64).max
            warnings.warn(f"log_target_pdf is inf or -inf, where x: {x}.")
        return res

    def grad_log_target_pdf(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Gradient of Log Target Probability Density Function

        Args:
            x (npt.NDArray[np.float64]): Sample

        Returns:
            npt.NDArray[np.float64]: Gradient of Log Target Probability Density at x
        """
        res = self.grad_log_target_pdf_unsafe(x)

        if np.isinf(res).any():
            res = np.where(np.isinf(res), 0, res)
            warnings.warn(f"grad_log_target_pdf is inf or -inf, where x: {x}.")
        return res

    @abstractmethod
    def log_proposal_pdf(
        self, x: npt.NDArray[np.float64], *args, **kwargs
    ) -> np.float64:
        """Log Proposal Probability Density Function

        Args:
            x (npt.NDArray[np.float64]): Sample

        Raises:
            NotImplementedError: log_proposal_pdf is not implemented.

        Returns:
            np.float64: Log Proposal Probability Density
        """
        raise NotImplementedError("log_proposal_pdf is not implemented.")

    @abstractmethod
    def log_proposal_process(
        self,
        current_sample: Union[np.float64, npt.NDArray[np.float64]],
        proposed_sample: Union[np.float64, npt.NDArray[np.float64]],
        *args,
        **kwargs,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """Log Proposal Process

        Args:
            current_sample (Union[np.float64, npt.NDArray[np.float64]]): Current Sample
            proposed_sample (Union[np.float64, npt.NDArray[np.float64]]): Proposed Sample

        Raises:
            NotImplementedError: log_proposal_process is not implemented.

        Returns:
            Tuple[np.float64, np.float64, np.float64]:
            Log Proposal Current, Log Proposal Proposed, Log Proposal Ratio
        """
        raise NotImplementedError("log_proposal_process is not implemented.")

    def log_target_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Log Target Process

        Args:
            current_sample (npt.NDArray[np.float64]): Current Sample
            proposed_sample (npt.NDArray[np.float64]): Proposed Sample

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            Log Target Current, Log Target Proposed, Log Target Ratio
        """
        log_target_current = self.log_target_pdf(current_sample)
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_ratio = log_target_proposed - log_target_current

        return log_target_current, log_target_proposed, log_target_ratio

    def store_process(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        current_mean: npt.NDArray[np.float64],
        proposed_mean: npt.NDArray[np.float64],
        current_covariance: npt.NDArray[np.float64],
        proposed_covariance: npt.NDArray[np.float64],
        log_target_current: np.float64,
        log_target_proposed: np.float64,
        log_proposal_current: np.float64,
        log_proposal_proposed: np.float64,
        accepted_status: bool,
        log_alpha: np.float64,
        accepted_sample: npt.NDArray[np.float64],
        accepted_mean: npt.NDArray[np.float64],
        accepted_covariance: npt.NDArray[np.float64],
    ) -> None:
        """Store Process

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
        self.store_current_sample[self.step, :] = current_sample
        self.store_proposed_sample[self.step, :] = proposed_sample

        # Store Mean
        self.store_current_mean[self.step, :] = current_mean
        self.store_proposed_mean[self.step, :] = proposed_mean

        # Store Covariance
        self.store_current_covariance[self.step, :, :] = current_covariance
        self.store_proposed_covariance[self.step, :, :] = proposed_covariance

        # Store Log Densities
        self.store_log_target_current[self.step] = log_target_current
        self.store_log_target_proposed[self.step] = log_target_proposed

        self.store_log_proposal_current[self.step] = log_proposal_current
        self.store_log_proposal_proposed[self.step] = log_proposal_proposed

        # Store Acceptance
        self.store_accepted_status[self.step] = accepted_status
        self.store_log_acceptance_rate[self.step] = log_alpha

        self.store_accepted_sample[self.step, :] = accepted_sample
        self.store_accepted_mean[self.step, :] = accepted_mean
        self.store_accepted_covariance[self.step, :, :] = accepted_covariance

    @abstractmethod
    def sample_generator(self, *args, **kwargs) -> npt.NDArray[np.float64]:
        """Sample generator, which is used to generate the next proposed sample.

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
        """Step Function for Environment

        Args:
            action (npt.NDArray[np.float64]): Step Size Before Softplus Function

        Raises:
            NotImplementedError: step is not implemented.
        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
            State, Reward, Terminated, Truncated, Info
        """
        raise NotImplementedError("step is not implemented.")
