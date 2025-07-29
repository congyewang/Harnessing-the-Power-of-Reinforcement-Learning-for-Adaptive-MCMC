import numpy as np
from mcmclib.metropolis import mala_adapt

from pyrlmala.envs import MALAEnv
from pyrlmala.utils import Toolbox
from pyrlmala.utils.target import AutoStanTargetPDF
import time

model_name = "{{  model_name }}"
posteriordb_path = "{{ posteriordb_path }}"
replicate_num = 1
mmd_file_path = f"./results/baseline_{model_name}.csv"
Toolbox.create_folder(mmd_file_path)


with open(mmd_file_path, "w") as f:
    f.write("model_name,random_seed,mmd\n")


target = AutoStanTargetPDF(model_name, posteriordb_path)
fp, fg = target.combine_make_log_target_pdf(["pdf", "grad"])
gs = Toolbox.gold_standard(model_name, posteriordb_path)


def get_cpu_time(random_seed: int) -> float:
    hessian_matrix = target.hess_log_target_pdf(gs.mean(axis=0))
    initial_covariance = -np.linalg.inv(hessian_matrix)
    const_mala = mala_adapt(
        fp=fp,
        fg=fg,
        x0=gs[0],
        h0=0.1,
        c0=initial_covariance,
        alpha=[1.0] * 5,
        epoch=[5_000] * 5,
    )

    step_size = const_mala[-2] ** 2

    start_time = time.perf_counter()

    baseline_env = MALAEnv(
        log_target_pdf_unsafe=fp,
        grad_log_target_pdf_unsafe=fg,
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

    end_time = time.perf_counter()

    cpu_time = end_time - start_time

    return cpu_time


for random_seed in range(replicate_num):
    np.random.seed(random_seed)
    cpu_time = get_cpu_time(random_seed)
    print(f"Random seed: {random_seed}, MMD: {cpu_time}")

    with open(mmd_file_path, "a+") as f:
        f.write(f"{model_name},{random_seed},{cpu_time}\n")
