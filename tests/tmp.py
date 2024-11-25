import numpy as np
from scipy.stats import multivariate_normal

from pyrlmala.pyrlmala.learning import LearningFactory

# Environment
sample_dim = 2

log_target_pdf = lambda x: multivariate_normal.logpdf(
    x, mean=np.zeros(sample_dim), cov=np.eye(sample_dim)
)
grad_log_target_pdf = lambda x: -x

initial_sample = np.zeros(sample_dim)
algorithm = "td3"

# Learning
learning_instance = LearningFactory.create_learning_instance(
    algorithm=algorithm,
    log_target_pdf=log_target_pdf,
    grad_log_target_pdf=grad_log_target_pdf,
    initial_sample=initial_sample,
)

learning_instance.train()
learning_instance.predict()
