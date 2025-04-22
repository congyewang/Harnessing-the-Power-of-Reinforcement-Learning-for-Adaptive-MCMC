import numpy as np
from tqdm.auto import trange

from pyrlmala.envs import MALAEnv
from pyrlmala.utils import Toolbox
from pyrlmala.utils.target import AutoStanTargetPDF

model_name = "{{ model_name }}"
posteriordb_path = "../posteriordb/posterior_database"


gs = Toolbox.gold_standard(model_name, posteriordb_path)


sample_dim = gs.shape[1]
initial_sample = gs[0]
initial_covariance = np.cov(gs, rowvar=False)
initial_step_size = np.array([1.0])
total_timesteps = 500_000
algorithm = "ddpg"
env_id = "MALAEnv-v1.0"
random_seed = 0


target = AutoStanTargetPDF(model_name, posteriordb_path)
log_target_pdf, grad_log_target_pdf = target.combine_make_log_target_pdf(
    ["pdf", "grad"]
)


mala_env = Toolbox.make_env(
    env_id=env_id,
    log_target_pdf=log_target_pdf,
    grad_log_target_pdf=grad_log_target_pdf,
    initial_sample=initial_sample,
    initial_covariance=initial_covariance,
    initial_step_size=initial_step_size,
    total_timesteps=total_timesteps,
    max_steps_per_episode=500,
    log_mode=True,
    seed=random_seed,
)()


action = Toolbox.softplus(np.repeat(initial_step_size, 2))
step_size = initial_step_size
_ = mala_env.reset(seed=random_seed)


acc_list = []
step_size_list = []


s = 0
for i in trange(total_timesteps):
    mala_env.step(action)

    current_step = mala_env.get_wrapper_attr("current_step")

    if current_step % 10_000 == 0 and current_step != 0:
        mala_env.set_wrapper_attr("state", np.tile(initial_sample, 2))

        acc = np.mean(
            mala_env.get_wrapper_attr("store_accepted_status")[
                current_step - 10_000 : current_step
            ]
        )

        acc_list.append(acc)
        step_size_list.append(step_size)

        step_size = step_size * np.exp(acc - 0.572)
        action = Toolbox.softplus(np.repeat(step_size, 2))


acc_array = np.array(acc_list)
step_size_array = np.array(step_size_list)
best_idx = np.argmin(np.abs(acc_array - 0.572))
best_acc = acc_array[best_idx]
best_step_size = step_size_array[best_idx]


with open(f"mala_step_size_{model_name}.txt", "w") as f:
    f.write("model_name,best_acc,best_step_size\n")
    f.write(f"{model_name},{best_acc},{best_step_size.item()}\n")
