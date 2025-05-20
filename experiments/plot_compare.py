import glob

import matplotlib.pyplot as plt

from pyrlmala.utils.plot import FlexPipeLine, PlotPipeLine

model_list = [
    "test-laplace_1-test-laplace_1",
    "test-laplace_2-test-laplace_2",
    "test-banana-test-banana",
]
mcmc_env = "mala"

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)

for ax, model_name in zip(axes, model_list):
    pp = PlotPipeLine(log_mode=True, axes=ax)

    const_dir = "./" + model_name.split("-")[1] + "/const"
    for file_path in sorted(glob.glob(f"{const_dir}/*.csv")):
        pp.store_to_dict(file_path)

    flex = FlexPipeLine(
        input_file=f"./{model_name.split('-')[1]}/flex/{model_name}_{mcmc_env}_mmd.txt"
    )

    pp.plot_const(mcmc_env=mcmc_env)
    pp.plot_flex(
        median=flex.median,
        left_quantile=flex.left_quantile,
        right_quantile=flex.right_quantile,
    )

    ax.set_xlabel(r"$\epsilon$")
    if ax is axes[0]:
        ax.set_ylabel("MMD")
    else:
        ax.tick_params(labelleft=False)

plt.savefig("policy_compare.pdf", bbox_inches="tight")
