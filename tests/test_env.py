import numpy as np

from pyrlmala.envs.env import BarkerEnv


class TestBarkerEnv:
    def setup_method(self):
        # Setup a basic BarkerEnv instance for testing
        self.env = BarkerEnv(
            log_target_pdf_unsafe=lambda x: -0.5 * np.sum(x**2, axis=-1),
            grad_log_target_pdf_unsafe=lambda x: -x,
            initial_sample=np.array([0.0]),
            initial_covariance=np.array([[1.0]]),
            initial_step_size=np.array([0.1]),
            total_timesteps=1000,
            log_mode=False,
        )
        self.env.sample_dim = 1
        self.env.np_random = np.random.default_rng(seed=42)

    def test_sample_generator_happy_path(self):
        x = np.array([0.0])
        grad_x = np.array([0.0])
        step_size = np.array([0.1])
        sample = self.env.sample_generator(x, grad_x, step_size)
        assert sample.shape == x.shape

    def test_accepted_process_edge_case(self):
        current_sample = np.array([0.0])
        proposed_sample = np.array([0.0])
        current_mean = np.array([0.0])
        proposed_mean = np.array([0.0])
        current_covariance = np.array([[1.0]])
        proposed_covariance = np.array([[1.0]])
        current_step_size = np.array([0.1])
        proposed_step_size = np.array([0.1])

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

    def test_step_happy_path(self):
        action = np.array([0.0, 0.0])
        state, reward, terminated, truncated, info = self.env.step(action)

        assert state.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
