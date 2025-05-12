import numpy as np
from adaptive_mcmc.mala import PrecondESJDMALA

from pyrlmala.envs import MALAEnv
from pyrlmala.utils import Toolbox
from pyrlmala.utils.target import AutoStanTargetPDF

model_name = "{{  model_name }}"
posteriordb_path = "{{ posteriordb_path }}"
replicate_num = 10
mmd_file_path = f"./results/esjd_baseline_{model_name}.csv"
Toolbox.create_folder(mmd_file_path)


with open(mmd_file_path, "w") as f:
    f.write("model_name,random_seed,mmd\n")


target = AutoStanTargetPDF(model_name, posteriordb_path)
fp, fg = target.combine_make_log_target_pdf(["pdf", "grad"])
gs = Toolbox.gold_standard(model_name, posteriordb_path)


def run_mmd(random_seed: int) -> float:
    initial_covariance = Toolbox.nearestPD(np.cov(gs, rowvar=False))
    esjd_mala = PrecondESJDMALA(
        log_target_pdf=fp,
        grad_target_pdf=fg,
        initial_sample=gs[0],
        initial_covariance=initial_covariance,
        eps0=0.1,
    )
    _, _, eps_hist, _ = esjd_mala.run(n_samples=25_000)

    step_size = eps_hist[-1] ** 2

    esjd_baseline_env = MALAEnv(
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
    esjd_baseline_env.reset(seed=random_seed)

    for _ in range(5_000):
        esjd_baseline_env.step(action)

    mmd = Toolbox.calculate_mmd(
        gs, esjd_baseline_env.store_accepted_sample[-5_000:], Toolbox.median_trick(gs)
    )

    return mmd


for random_seed in range(replicate_num):
    np.random.seed(random_seed)
    mmd = run_mmd(random_seed)
    print(f"Random seed: {random_seed}, MMD: {mmd}")

    with open(mmd_file_path, "a+") as f:
        f.write(f"{model_name},{random_seed},{mmd}\n")
