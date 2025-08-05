import jax
import numpy as np
from jaxtarget.student_t_mixture import StudentTMixtureModel
from jax import numpy as jnp
from mcmclib.metropolis import mala_adapt

from pyrlmala.envs import MALAEnv
from pyrlmala.utils import Toolbox


stmm = StudentTMixtureModel(15, dim=2)

log_prob_jit = jax.jit(stmm.log_prob)
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


gs = np.array(stmm.sample(jax.random.PRNGKey(0), (5000,)))


model_name = "stmm_2d_15"
replicate_num = 10
mmd_file_path = f"./results/baseline_{model_name}.csv"
Toolbox.create_folder(mmd_file_path)


with open(mmd_file_path, "w") as f:
    f.write("model_name,random_seed,mmd\n")


def run_mmd(random_seed: int) -> float:
    hessian_matrix = hess_log_target_pdf(gs.mean(axis=0))
    initial_covariance = -np.linalg.inv(hessian_matrix)
    initial_covariance = Toolbox.nearestPD(initial_covariance)
    # initial_covariance = Toolbox.nearestPD(np.cov(gs, rowvar=False))
    const_mala = mala_adapt(
        fp=log_target_pdf,
        fg=grad_log_target_pdf,
        x0=gs[0],
        h0=0.1,
        c0=initial_covariance,
        alpha=[1.0] * 5,
        epoch=[5_000] * 5,
    )

    step_size = const_mala[-2] ** 2

    baseline_env = MALAEnv(
        log_target_pdf_unsafe=log_target_pdf,
        grad_log_target_pdf_unsafe=grad_log_target_pdf,
        initial_sample=gs[0],
        initial_covariance=initial_covariance,
        initial_step_size=step_size,
        total_timesteps=5_000,
        max_steps_per_episode=500,
        log_mode=True,
    )

    action = Toolbox.softplus(np.tile(step_size, 2))
    baseline_env.reset(seed=random_seed)

    for _ in range(5_000):
        baseline_env.step(action)

    mmd = Toolbox.calculate_mmd(
        gs, baseline_env.store_accepted_sample[-5_000:], Toolbox.median_trick(gs)
    )

    return mmd


for random_seed in range(replicate_num):
    np.random.seed(random_seed)
    mmd = run_mmd(random_seed)
    print(f"Random seed: {random_seed}, MMD: {mmd}")

    with open(mmd_file_path, "a+") as f:
        f.write(f"{model_name},{random_seed},{mmd}\n")


hessian_matrix = hess_log_target_pdf(gs.mean(axis=0))
initial_covariance = -np.linalg.inv(hessian_matrix)
# initial_covariance = Toolbox.nearestPD(np.cov(gs, rowvar=False))
const_mala = mala_adapt(
    fp=log_target_pdf,
    fg=grad_log_target_pdf,
    x0=gs[0],
    h0=0.1,
    c0=initial_covariance,
    alpha=[1.0] * 5,
    epoch=[5_000] * 5,
)
