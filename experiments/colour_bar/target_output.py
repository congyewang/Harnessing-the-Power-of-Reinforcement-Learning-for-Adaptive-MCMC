from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from pyrlmala.utils.target import AutoStanTargetPDF

LATEX_STYLE = True

if LATEX_STYLE:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
    }
)


posteriordb_path = "../posteriordb/posterior_database"


target_laplace_1 = AutoStanTargetPDF("test-laplace_1-test-laplace_1", posteriordb_path)
target_laplace_2 = AutoStanTargetPDF("test-laplace_2-test-laplace_2", posteriordb_path)
target_banana = AutoStanTargetPDF("test-banana-test-banana", posteriordb_path)


def target_plot_2d(
    ax: plt.Axes,
    x_mesh_range: Tuple[float, float, int],
    y_mesh_range: Tuple[float, float, int],
    log_target_pdf: Callable[[np.ndarray], float],
    *,
    norm,
    levels: np.ndarray,
    cmap: str = "turbo",
    show_ticks: bool = False,
    n_xticks: int = 5,
    n_yticks: int = 5,
):
    x0, x1, nx = x_mesh_range
    y0, y1, ny = y_mesh_range
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(x0, x1, nx),
        np.linspace(y0, y1, ny),
    )
    data = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

    vals = np.exp([log_target_pdf(pt) for pt in data])
    Z = vals.reshape(ny, nx)

    cf = ax.contourf(
        mesh_x, mesh_y, Z, levels=levels, norm=norm, cmap=cmap, extend="neither"
    )

    if show_ticks:
        ax.set_xticks(np.linspace(x0, x1, n_xticks))
        ax.set_yticks(np.linspace(y0, y1, n_yticks))
        ax.tick_params(
            axis="both", which="major", labelsize=plt.rcParams["xtick.labelsize"]
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect("auto")
    return cf


boundaries = np.array([0.00, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 3.5])
cmap = plt.cm.turbo
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

funcs = [
    target_laplace_1.log_target_pdf,
    target_laplace_2.log_target_pdf,
    target_banana.log_target_pdf,
]
mesh_ranges = [
    ((-3, 3, 100), (-3, 3, 100)),
    ((-3, 3, 100), (-3, 3, 100)),
    ((-4, 4, 100), (5, 15, 100)),
]

fig, axes = plt.subplots(
    1, 3, figsize=(15, 5), subplot_kw={"aspect": "auto"}, constrained_layout=True
)

for ax, f, (x_rng, y_rng) in zip(axes, funcs, mesh_ranges):
    target_plot_2d(
        ax,
        x_mesh_range=x_rng,
        y_mesh_range=y_rng,
        log_target_pdf=f,
        norm=norm,
        levels=boundaries,
        cmap=cmap,
    )

cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    boundaries=boundaries,
    spacing="proportional",
    ticks=boundaries,
)
cbar.set_label(r"$p(x)$")
cbar.locator = MaxNLocator(nbins=5)
plt.savefig("target_share_plot.pdf", bbox_inches="tight")
