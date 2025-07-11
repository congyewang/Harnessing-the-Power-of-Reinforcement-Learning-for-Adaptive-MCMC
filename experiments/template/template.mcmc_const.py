import random

import numpy as np
import torch

from pyrlmala.envs import {{ env_name }}
from pyrlmala.utils import CalculateMMD, Toolbox
from pyrlmala.utils.target import PosteriorDatabaseTargetPDF


RANDOM_SEED = {{ random_seed }}
STEP_SIZE = {{ step_size }}

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set Model and Posterior Database Path
model_name = "{{ model_name }}"
posteriordb_path = "../../../posteriordb/posterior_database"

# Generate Target PDF and Gradient of Target PDF
posteriordb_generator = PosteriorDatabaseTargetPDF(
    model_name=model_name, posteriordb_path=posteriordb_path
)

# Set MCMC Environment
sample_dim = 2
initial_sample = 0.1 * np.ones(sample_dim)
initial_step_size = np.array([STEP_SIZE])

mcmc = {{ env_name }}(
    log_target_pdf_unsafe=posteriordb_generator.log_target_pdf,
    grad_log_target_pdf_unsafe=posteriordb_generator.grad_log_target_pdf,
    initial_sample=initial_sample,
    initial_covariance=None,
    initial_step_size=Toolbox.inverse_softplus(initial_step_size),
    total_timesteps=500_000,
    max_steps_per_episode=500,
    log_mode=True,
)

# Run MCMC
_, _ = mcmc.reset(seed=RANDOM_SEED)

for i in range(mcmc.total_timesteps):
    _, _, _, _, _ = mcmc.step(
        np.repeat(Toolbox.inverse_softplus(initial_step_size), 2)
    )

# Calculate MMD
gs = Toolbox.gold_standard(model_name, posteriordb_path)
accepted_sample = mcmc.store_accepted_sample[-len(gs):]
mmd = CalculateMMD.calculate(gs, accepted_sample)

# Print MMD
res = f"""step_size, MMD
{STEP_SIZE}, {mmd}
"""
with open("mmd.csv", "w") as f:
    f.write(res)
