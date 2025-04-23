import numpy as np
from mcmclib.metropolis import mala_adapt

from pyrlmala.utils import Toolbox
from pyrlmala.utils.target import AutoStanTargetPDF

model_name = "{{  model_name }}"
posteriordb_path = "{{ posteriordb_path }}"
replicate_num = 10
mmd_file_path = f"./results/baseline_{model_name}.txt"
Toolbox.create_folder(mmd_file_path)


with open(mmd_file_path, "w") as f:
    f.write("model_name,random_seed,mmd\n")


target = AutoStanTargetPDF(model_name, posteriordb_path)
fp, fg = target.combine_make_log_target_pdf(["pdf", "grad"])
gs = Toolbox.gold_standard(model_name, posteriordb_path)


for random_seed in range(replicate_num):
    np.random.seed(random_seed)
    res = mala_adapt(
        fp=fp,
        fg=fg,
        x0=gs[0],
        h0=0.1,
        c0=np.cov(gs, rowvar=False),
        alpha=[0.3] * 10 + [0.0],
        epoch=[5_000] * 10 + [10_000],
    )
    mmd = Toolbox.calculate_mmd(gs, res[0][-1], Toolbox.median_trick(gs))

    with open(mmd_file_path, "a+") as f:
        f.write(f"{model_name},{random_seed},{mmd}\n")
