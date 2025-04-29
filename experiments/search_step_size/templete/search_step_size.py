import numpy as np
from mcmclib.metropolis import mala_adapt

from pyrlmala.utils import Toolbox
from pyrlmala.utils.target import AutoStanTargetPDF

model_name = "{{ model_name }}"
posteriordb_path = "../posteriordb/posterior_database"
output_file_path = f"results/mala_step_size_{model_name}.csv"
repeat_num = 10

gs = Toolbox.gold_standard(model_name, posteriordb_path)


sample_dim = gs.shape[1]
initial_sample = gs[0]
initial_covariance = np.cov(gs, rowvar=False)
initial_step_size = 0.1
step_per_epoch = 5_000
num_epoch = 10


target = AutoStanTargetPDF(model_name, posteriordb_path)
log_target_pdf, grad_log_target_pdf = target.combine_make_log_target_pdf(
    ["pdf", "grad"]
)

Toolbox.create_folder(output_file_path)
with open(output_file_path, "w") as f:
    f.write("model_name,random_seed,best_step_size\n")

for random_seed in range(repeat_num):
    np.random.seed(random_seed)
    const_mala = mala_adapt(
        fp=log_target_pdf,
        fg=grad_log_target_pdf,
        x0=initial_sample,
        h0=initial_step_size,
        c0=initial_covariance,
        alpha=[1.0] * num_epoch,
        epoch=[step_per_epoch] * num_epoch,
    )

    step_size = const_mala[-2].item() ** 2

    with open(output_file_path, "a+") as f:
        f.write(f"{model_name},{random_seed},{step_size}\n")
