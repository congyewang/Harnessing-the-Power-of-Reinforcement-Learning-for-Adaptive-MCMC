import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, SupportsFloat, Tuple, Union, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import scipy
from gymnasium.utils import seeding
from scipy.stats import multivariate_normal

scipy.linalg.cholesky = cast(
    Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], scipy.linalg.cholesky
)


class RLMALAEnvBase(gym.Env[npt.NDArray[np.float64], npt.NDArray[np.float64]], ABC):
    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [Union[float, np.float64, npt.NDArray[np.float64]]],
            Union[float, np.float64],
        ],
        grad_log_target_pdf_unsafe: Callable[
            [Union[float, np.float64, npt.NDArray[np.float64]]],
            Union[float, np.float64],
        ],
        initial_sample: Union[np.float64, npt.NDArray[np.float64]],
        initial_covariance: Union[np.float64, npt.NDArray[np.float64], None] = None,
    ) -> None:
        super().__init__()

        self.sample_dim: int = int(np.prod(initial_sample.shape))  # sample dimension
        self.steps: int = 1  # iteration time
        # log target probability density function without numerical stabilization
        self.log_target_pdf_unsafe = log_target_pdf_unsafe
        self.grad_log_target_pdf_unsafe = grad_log_target_pdf_unsafe

        if initial_covariance is None:
            initial_covariance = (2.38 / np.sqrt(self.sample_dim)) * np.eye(
                self.sample_dim
            )
        self.initial_covariance = initial_covariance

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
        )  # state at this time, s_{t}

        # Store
        self.store_observation: List[npt.NDArray[np.float64]] = []
        self.store_action: List[npt.NDArray[np.float64]] = []
        self.store_log_acceptance_rate: List[Union[np.float64, float]] = []
        self.store_accepted_status: List[bool] = []
        self.store_reward: List[Union[np.float64, float]] = []

        self.store_current_sample: List[npt.NDArray[np.float64]] = []
        self.store_proposed_sample: List[npt.NDArray[np.float64]] = []

        self.store_current_mean: List[npt.NDArray[np.float64]] = []
        self.store_proposed_mean: List[npt.NDArray[np.float64]] = []

        self.store_current_covariance: List[npt.NDArray[np.float64]] = []
        self.store_proposed_covariance: List[npt.NDArray[np.float64]] = []

        self.store_log_target_proposed: List[Union[np.float64, float]] = []
        self.store_log_target_current: List[Union[np.float64, float]] = []
        self.store_log_proposal_proposed: List[Union[np.float64, float]] = []
        self.store_log_proposal_current: List[Union[np.float64, float]] = []

        self.store_accepted_mean: List[npt.NDArray[np.float64]] = []
        self.store_accepted_sample: List[npt.NDArray[np.float64]] = []
        self.store_accepted_covariance: List[npt.NDArray[np.float64]] = []

    def log_target_pdf(self, x: Union[float, np.float64, npt.NDArray[np.float64]]):
        res = self.log_target_pdf_unsafe(x)

        # Numerical stability
        if np.any(np.isinf(res)) or np.any(np.isnan(res)):
            res = -np.finfo(np.float64).max
            warnings.warn(f"log_target_pdf has inf or -inf.", UserWarning)

        return res

    def grad_log_target_pdf(self, x: Union[float, np.float64, npt.NDArray[np.float64]]):
        res = self.grad_log_target_pdf_unsafe(x)

        # Numerical stability
        if np.any(np.isinf(res)) or np.any(np.isnan(res)):
            nan_or_inf_indices = np.isnan(res) | np.isinf(res)
            res = np.where(nan_or_inf_indices, -np.finfo(np.float64).max, res)
            warnings.warn(f"grad_log_target_pdf has inf or -inf.", UserWarning)

        return res

    def log_proposal_pdf(
        self,
        x: npt.NDArray[np.float64],
        mean: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64],
    ) -> np.float64:
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)

    def mcmc_noise(
        self, mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.np_random.multivariate_normal(mean=mean, cov=cov, size=1).flatten()

    def reward_function(
        self,
        current_sample: npt.NDArray[np.float64],
        proposed_sample: npt.NDArray[np.float64],
        log_alpha: Union[np.float64, float],
        log_mode: bool = True,
    ) -> np.float64:
        if log_mode:
            res = (
                2 * np.log(np.linalg.norm(current_sample - proposed_sample, 2))
                + log_alpha
            )
        else:
            res = np.power(
                np.linalg.norm(current_sample - proposed_sample, 2), 2
            ) * np.exp(log_alpha)

        return res.item()

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
        np.float64,
    ]:
        # Calculate Log Target Density
        log_target_current = self.log_target_pdf(current_sample)
        log_target_proposed = self.log_target_pdf(proposed_sample)

        # Calculate Log Proposal Densitys
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_mean, proposed_covariance
        )
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_mean, current_covariance
        )

        # Calculate Log Acceptance Rate
        log_alpha = np.minimum(
            0.0,
            log_target_proposed
            - log_target_current
            + log_proposal_current
            - log_proposal_proposed,
        )

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
        # Store Sample
        self.store_current_sample.append(current_sample)
        self.store_proposed_sample.append(proposed_sample)

        # Store Mean
        self.store_current_mean.append(current_mean)
        self.store_proposed_mean.append(proposed_mean)

        # Store Covariance
        self.store_current_covariance.append(current_covariance)
        self.store_proposed_covariance.append(proposed_covariance)

        # Store Log Densities
        self.store_log_target_current.append(log_target_current)
        self.store_log_target_proposed.append(log_target_proposed)

        self.store_log_proposal_current.append(log_proposal_current)
        self.store_log_proposal_proposed.append(log_proposal_proposed)

        # Store Acceptance
        self.store_accepted_status.append(accepted_status)
        self.store_log_acceptance_rate.append(log_alpha)

        self.store_accepted_sample.append(accepted_sample)
        self.store_accepted_mean.append(accepted_mean)
        self.store_accepted_covariance.append(accepted_covariance)

        return (
            accepted_status,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
            log_alpha,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset environment to initial state and return initial observation.

        Args:
            seed (int | None, optional): Random seed. Defaults to None.
            options (dict[str, Any] | None, optional): Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]
        """
        # Gym Recommandation
        super().reset(seed=seed, options=options)

        # Set Random Seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return self.state, {}

    @abstractmethod
    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError("step is not implemented.")


class RLMALAEnv(RLMALAEnvBase):
    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [Union[float, np.float64, npt.NDArray[np.float64]]],
            Union[float, np.float64],
        ],
        grad_log_target_pdf_unsafe: Callable[
            [Union[float, np.float64, npt.NDArray[np.float64]]],
            Union[float, np.float64],
        ],
        initial_sample: Union[np.float64, npt.NDArray[np.float64]],
        initial_covariance: Union[np.float64, npt.NDArray[np.float64], None] = None,
    ) -> None:
        super().__init__(
            log_target_pdf_unsafe,
            grad_log_target_pdf_unsafe,
            initial_sample,
            initial_covariance,
        )

    def softplus(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.log(1 + np.exp(x))

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]:
        # Unpack state
        current_sample, proposed_sample = np.split(self.state, 2)

        # Unpack action
        current_psi, proposed_psi = np.split(action, 2)

        # Calculate phi
        current_phi = self.softplus(current_psi)
        proposed_phi = self.softplus(proposed_psi)

        # Mean and Coveriance
        print(current_sample)
        current_grad_log_pdf = self.grad_log_target_pdf(current_sample)
        proposed_grad_log_pdf = self.grad_log_target_pdf(proposed_sample)

        current_mean = (
            current_sample
            + current_phi * self.initial_covariance @ current_grad_log_pdf
        )
        proposed_mean = (
            proposed_sample
            + proposed_phi * self.initial_covariance @ proposed_grad_log_pdf
        )

        current_covariance = 2 * current_phi**2 * self.initial_covariance
        proposed_covariance = 2 * proposed_phi**2 * self.initial_covariance

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
        next_proposed_sample = self.mcmc_noise(accepted_mean, accepted_covariance)
        observation = np.concatenate((accepted_sample, next_proposed_sample))
        self.state = observation

        # Store
        self.store_observation.append(observation)
        self.store_action.append(action)

        # Calculate Reward
        reward = self.reward_function(current_sample, proposed_sample, log_alpha)
        self.store_reward.append(reward)

        # Update Steps
        self.steps += 1
        terminated: bool = False
        truncated: bool = False
        info: Dict[None, None] = {}

        return self.state, reward, terminated, truncated, info
