import numpy as np

from pyrlmala.envs import BarkerEnv
from pyrlmala.utils import Toolbox
from pyrlmala.utils.target import AutoStanTargetPDF

model_name = "{{  model_name }}"
posteriordb_path = "{{ posteriordb_path }}"
replicate_num = 10
mmd_file_path = f"./results/baseline_barker_{model_name}.csv"
Toolbox.create_folder(mmd_file_path)


with open(mmd_file_path, "w") as f:
    f.write("model_name,random_seed,mmd\n")


target = AutoStanTargetPDF(model_name, posteriordb_path)
fp, fg = target.combine_make_log_target_pdf(["pdf", "grad"])
gs = Toolbox.gold_standard(model_name, posteriordb_path)


def output_initial_step_size(model_name: str, l: float = 1.65) -> float:
    """
    Output a smooth initial step size based on model dimension using log-linear interpolation.

    Args:
        model_name (str): The name of the model.
        l (float, optional): The scaling factor. Defaults to 1.65.

    Returns:
        float: The initial step size for the model.
    """
    target = AutoStanTargetPDF(model_name, posteriordb_path)
    sigma_inv = -target.hess_log_target_pdf(
        Toolbox.gold_standard(model_name, posteriordb_path).mean(axis=0)
    )
    d = sigma_inv.shape[0]

    lam_max = np.linalg.eigvalsh(sigma_inv).max()
    eps0 = l / np.sqrt(lam_max * d ** (1 / 3))
    initial_step_size = 29.0168 * eps0**3 - 25.6180 * eps0**2 + 3.0239 * eps0 + 1.3476

    return initial_step_size.item()


def run_mmd(random_seed: int) -> float:
    initial_covariance = Toolbox.nearestPD(np.cov(gs, rowvar=False))
    step_size = output_initial_step_size(model_name)

    baseline_barker_env = BarkerEnv(
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
    baseline_barker_env.reset(seed=random_seed)

    for _ in range(5_000):
        baseline_barker_env.step(action)

    mmd = Toolbox.calculate_mmd(
        gs, baseline_barker_env.store_accepted_sample[-5_000:], Toolbox.median_trick(gs)
    )

    return mmd


for random_seed in range(replicate_num):
    np.random.seed(random_seed)
    mmd = run_mmd(random_seed)
    print(f"Random seed: {random_seed}, MMD: {mmd}")

    with open(mmd_file_path, "a+") as f:
        f.write(f"{model_name},{random_seed},{mmd}\n")
