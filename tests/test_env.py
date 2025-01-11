from functools import partial
from itertools import product

import numpy as np
import pytest
from scipy.stats import multivariate_normal, wishart

from pyrlmala.envs.env import BarkerEnv, MALAEnv

SAMPLE_DIM_LIST = [1, 3, 5, 33, 66]
RANDOM_SEED_LIST = [0, 1, 2, 42, 1234]


class TestBarkerEnv:
    def create_env(self, sample_dim: int) -> None:
        """
        Create a basic BarkerEnv instance for testing
        """
        self.env = BarkerEnv(
            log_target_pdf_unsafe=partial(
                multivariate_normal.logpdf,
                mean=np.zeros(sample_dim),
                cov=np.eye(sample_dim),
            ),
            grad_log_target_pdf_unsafe=lambda x: -x,
            initial_sample=np.random.normal(size=sample_dim),
            initial_covariance=wishart.rvs(
                df=(sample_dim + 1), scale=np.eye(sample_dim)
            ).reshape(sample_dim, sample_dim),
            initial_step_size=np.random.uniform(0.1, 5.0),
            total_timesteps=1_000,
            log_mode=True,
        )

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_sample_generator_happy_path(
        self, sample_dim: int, random_seed: int
    ) -> None:
        self.create_env(sample_dim)
        self.env.reset(seed=random_seed)

        x = np.random.normal(size=(sample_dim,))
        grad_x = np.random.normal(size=(sample_dim,))
        step_size = np.random.uniform(0.1, 5.0, size=(sample_dim,))

        sample = self.env.sample_generator(x, grad_x, step_size)

        assert sample.shape == x.shape

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_accepted_process_edge_case(
        self, sample_dim: int, random_seed: int
    ) -> None:
        self.create_env(sample_dim)
        self.env.reset(seed=random_seed)

        current_sample = np.random.normal(size=(sample_dim,))
        proposed_sample = np.random.normal(size=(sample_dim,))
        current_mean = np.random.normal(size=(sample_dim,))
        proposed_mean = np.random.normal(size=(sample_dim,))
        current_covariance = wishart.rvs(
            df=(sample_dim + 1), scale=np.eye(sample_dim)
        ).reshape(sample_dim, sample_dim)
        proposed_covariance = wishart.rvs(
            df=(sample_dim + 1), scale=np.eye(sample_dim)
        ).reshape(sample_dim, sample_dim)
        current_step_size = np.random.uniform(0.1, 5.0, size=(sample_dim,))
        proposed_step_size = np.random.uniform(0.1, 5.0, size=(sample_dim,))

        (
            accepted_status,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
            log_alpha,
        ) = self.env.accepted_process(
            current_sample,
            proposed_sample,
            current_mean,
            proposed_mean,
            current_covariance,
            proposed_covariance,
            current_step_size,
            proposed_step_size,
        )

        assert isinstance(accepted_status, bool)
        assert accepted_sample.shape == current_sample.shape
        assert accepted_mean.shape == current_mean.shape
        assert accepted_covariance.shape == current_covariance.shape
        assert isinstance(log_alpha, np.float64)

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_step_happy_path(self, sample_dim: int, random_seed: int) -> None:
        self.create_env(sample_dim)
        self.env.reset(seed=random_seed)

        action = np.random.normal(size=2)
        state, reward, terminated, truncated, info = self.env.step(action)

        assert state.shape == (sample_dim << 1,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_initialization_consistancy(
        self, sample_dim: int, random_seed: int
    ) -> None:
        self.create_env(sample_dim)

        self.env.current_step = 0
        state1, _ = self.env.reset(seed=random_seed)

        self.env.current_step = 0
        state2, _ = self.env.reset(seed=random_seed)

        assert np.array_equal(state1, state2)


class TestMALAEnv:
    def create_env(self, sample_dim: int) -> None:
        """
        Create a basic BarkerEnv instance for testing
        """
        self.env = MALAEnv(
            log_target_pdf_unsafe=partial(
                multivariate_normal.logpdf,
                mean=np.zeros(sample_dim),
                cov=np.eye(sample_dim),
            ),
            grad_log_target_pdf_unsafe=lambda x: -x,
            initial_sample=np.random.normal(size=sample_dim),
            initial_covariance=wishart.rvs(
                df=(sample_dim + 1), scale=np.eye(sample_dim)
            ).reshape(sample_dim, sample_dim),
            initial_step_size=np.random.uniform(0.1, 5.0),
            total_timesteps=1_000,
            log_mode=True,
        )

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_sample_generator_happy_path(
        self, sample_dim: int, random_seed: int
    ) -> None:
        self.create_env(sample_dim)
        self.env.reset(seed=random_seed)

        x = np.random.normal(size=(sample_dim,))
        mean = np.random.normal(size=(sample_dim,))
        covariance = wishart.rvs(df=(sample_dim + 1), scale=np.eye(sample_dim)).reshape(
            sample_dim, sample_dim
        )

        sample = self.env.sample_generator(mean, covariance)

        assert sample.shape == x.shape

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_accepted_process_edge_case(
        self, sample_dim: int, random_seed: int
    ) -> None:
        self.create_env(sample_dim)
        self.env.reset(seed=random_seed)

        current_sample = np.random.normal(size=(sample_dim,))
        proposed_sample = np.random.normal(size=(sample_dim,))
        current_mean = np.random.normal(size=(sample_dim,))
        proposed_mean = np.random.normal(size=(sample_dim,))
        current_covariance = wishart.rvs(
            df=(sample_dim + 1), scale=np.eye(sample_dim)
        ).reshape(sample_dim, sample_dim)
        proposed_covariance = wishart.rvs(
            df=(sample_dim + 1), scale=np.eye(sample_dim)
        ).reshape(sample_dim, sample_dim)

        (
            accepted_status,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
            log_alpha,
        ) = self.env.accepted_process(
            current_sample,
            proposed_sample,
            current_mean,
            proposed_mean,
            current_covariance,
            proposed_covariance,
        )

        assert isinstance(accepted_status, bool)
        assert accepted_sample.shape == current_sample.shape
        assert accepted_mean.shape == current_mean.shape
        assert accepted_covariance.shape == current_covariance.shape
        assert isinstance(log_alpha, np.float64), f"value:{log_alpha}, type: {type(log_alpha)}"

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_step_happy_path(self, sample_dim: int, random_seed: int) -> None:
        self.create_env(sample_dim)
        self.env.reset(seed=random_seed)

        action = np.random.normal(size=2)
        state, reward, terminated, truncated, info = self.env.step(action)

        assert state.shape == (sample_dim << 1,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.parametrize(
        "sample_dim, random_seed", product(SAMPLE_DIM_LIST, RANDOM_SEED_LIST)
    )
    def test_initialization_consistancy(
        self, sample_dim: int, random_seed: int
    ) -> None:
        self.create_env(sample_dim)

        self.env.current_step = 0
        state1, _ = self.env.reset(seed=random_seed)

        self.env.current_step = 0
        state2, _ = self.env.reset(seed=random_seed)

        assert np.array_equal(state1, state2)
