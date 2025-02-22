import random

import numpy as np
import torch

from pyrlmala.envs import BarkerEnv
from pyrlmala.learning.preparation import PosteriorDBFunctionsGenerator
from pyrlmala.utils import CalculateMMD, Toolbox

RANDOM_SEED = 42
STEP_SIZE = 16.5

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set Model and Posterior Database Path
model_name = ""
posteriordb_path = "../../posteriordb/posterior_database"

# Generate Target PDF and Gradient of Target PDF
posteriordb_generator = PosteriorDBFunctionsGenerator(
    model_name=model_name, posteriordb_path=posteriordb_path, posterior_data=None
)
log_target_pdf = posteriordb_generator.make_log_pdf()
grad_log_target_pdf = posteriordb_generator.make_grad_log_pdf()

# Set MCMC Environment
sample_dim = 2
initial_sample = 0.1 * np.ones(sample_dim)
initial_step_size = np.array([STEP_SIZE])

mcmc = BarkerEnv(
    log_target_pdf_unsafe=log_target_pdf,
    grad_log_target_pdf_unsafe=grad_log_target_pdf,
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