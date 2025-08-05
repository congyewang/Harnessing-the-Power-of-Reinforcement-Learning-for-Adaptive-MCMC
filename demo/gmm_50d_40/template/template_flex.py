import os

import jax
import numpy as np
from jax import numpy as jnp
from jaxtarget.gmm40 import GMM40

from pyrlmala.learning import LearningFactory
from pyrlmala.utils import Toolbox
import warnings

warnings.filterwarnings("ignore")
output_dir = "results/"
model_name = "gmm_50d_40"


gmm = GMM40(dim=50)

log_prob_jit = jax.jit(gmm.log_prob)
grad_log_target_pdf_jit = jax.jit(jax.jacrev(log_prob_jit))
hess_log_target_pdf_jit = jax.jit(jax.hessian(log_prob_jit))


def log_target_pdf(x):
    x_jnp = jnp.array(x)
    res_jnp = log_prob_jit(x_jnp)
    res = np.array(res_jnp)
    return res


def grad_log_target_pdf(x):
    x_jnp = jnp.array(x)
    res_jnp = grad_log_target_pdf_jit(x_jnp)
    res = np.array(res_jnp)
    return res


def hess_log_target_pdf(x):
    x_jnp = jnp.array(x)
    res_jnp = hess_log_target_pdf_jit(x_jnp)
    res = np.array(res_jnp)
    return res


gs = np.array(gmm.sample(jax.random.PRNGKey(0), (5000,)))
sample_dim = 50
initial_sample = gs[0]
initial_step_size = np.array([1.0])
hessian_matrix = hess_log_target_pdf(gs.mean(axis=0))
initial_covariance = -np.linalg.inv(hessian_matrix)
algorithm = "ddpg"
mcmc_env = "{{ mcmc_env }}"

learning_instance = LearningFactory.create_learning_instance(
    algorithm=algorithm,
    log_target_pdf=log_target_pdf,
    grad_log_target_pdf=grad_log_target_pdf,
    initial_sample=initial_sample,
    initial_step_size=initial_step_size,
    hyperparameter_config_path=f"./config/{algorithm}_{mcmc_env}/{algorithm}_{mcmc_env}_seed_{{ random_seed }}.toml",
    actor_config_path="./config/actor.toml",
    critic_config_path="./config/critic.toml",
)
learning_instance.train()


predicted_sample = learning_instance.predicted_observation[:, 0:sample_dim]
mmd = Toolbox.calculate_mmd(predicted_sample, gs, Toolbox.median_trick(gs))

mmd_file = os.path.join(output_dir, f"flex_{mcmc_env}_mmd_{{ random_seed }}.csv")
Toolbox.create_folder(output_dir)
with open(mmd_file, "w") as f:
    f.write("model_name,rl_algorithm,mcmc_env,random_seed,mmd\n")
    f.write(f"{model_name},{algorithm},{mcmc_env},{{ random_seed }},{mmd}\n")
