import numpy as np
from scipy.stats import multivariate_normal

from src.pyrlmala.learning import LearningFactory


# Environment
sample_dim = 2

log_target_pdf = lambda x: multivariate_normal.logpdf(
    x, mean=np.zeros(sample_dim), cov=np.eye(sample_dim)
)
grad_log_target_pdf = lambda x: -x

initial_sample = np.zeros(sample_dim)
learning_instance = LearningFactory.create_learning_instance(
    algorithm="td3",
    log_target_pdf=log_target_pdf,
    grad_log_target_pdf=grad_log_target_pdf,
    initial_sample=initial_sample,
    hyperparameter_config_path="src/pyrlmala/config/default/td3.toml",
    actor_config_path="src/pyrlmala/config/default/actor.toml",
    critic_config_path="src/pyrlmala/config/default/critic.toml",
)
learning_instance.train()
