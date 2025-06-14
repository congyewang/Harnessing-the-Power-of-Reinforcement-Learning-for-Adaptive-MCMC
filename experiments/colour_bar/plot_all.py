import glob
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator
from numpy import typing as npt

from pyrlmala.utils.plot import FlexPipeLine, PlotPipeLine
from pyrlmala.utils.target import AutoStanTargetPDF

fig, axes = plt.subplots(
    3, 3, figsize=(15, 15), constrained_layout=True, subplot_kw={"aspect": "auto"}
)

# First
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

for ax, f, (x_rng, y_rng) in zip(axes[0], funcs, mesh_ranges):
    target_plot_2d(
        ax,
        x_mesh_range=x_rng,
        y_mesh_range=y_rng,
        log_target_pdf=f,
        norm=norm,
        levels=boundaries,
        cmap=cmap,
    )

# Create a color bar for the first row, but keep it hidden for now; this is solely for placeholder and alignment purposes.
cbar1 = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes[0],
    boundaries=boundaries,
    spacing="proportional",
    ticks=boundaries,
    shrink=1.0,
    pad=0.02
)
cbar1.set_label(r"$p(x)$")
cbar1.locator = MaxNLocator(nbins=5)


# Second
class nlcmap(object):
    def __init__(self, cmap, levels: np.ndarray):
        self.cmap = cmap
        self.levels = np.asarray(levels, dtype="float64")
        self._x = self.levels
        self.levmax = self.levels.max()
        # Nonlinear "target" scale
        self.transformed_levels = np.linspace(0.0, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)


def plot_heatmaps_nonlinear_shared_colorbar(
    items: List[
        Tuple[
            npt.NDArray[np.float64],
            Tuple[float, float, int],
            Tuple[float, float, int],
            str,
        ]
    ],
    levels: Optional[npt.NDArray[np.float64]] = None,
    base_cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    axes: Optional[List[plt.Axes]] = None,
    cbar_axes: Optional[plt.Axes] = None,
    cbar_label: str = "Value",
    tick_count: Optional[int] = 5,
    save_path: Optional[str] = None,
    shrink: float = 1.0,
    pad: float = 0.02,
) -> None:
    """
    Plot multiple heatmaps in a single row of subplots, using nlcmap for nonlinear colormap, and share a single colorbar with unevenly spaced labels.

    Args:
        items: Each item = (Z, x_range, y_range, title):
            Z: np.ndarray, shape=(nx,ny)
            x_range: (xmin, xmax, nx)
            y_range: (ymin, ymax, ny)
            title: Subplot title
        levels: Segmentation nodes for the nonlinear colormap; if None, automatically use [0%, 10%, ..., 100%] percentiles of all Z
        base_cmap: Name of the underlying linear colormap
        figsize: Figure size; if None, defaults to (5*n, 5)
        axes: List of matplotlib Axes objects to plot on; if None, creates new subplots
        cbar_axes: If you want to manually provide an Axes for the colorbar, pass it here; otherwise, one is automatically generated on the right
        cbar_label: Colorbar label text
        tick_count: Number of ticks on the colorbar; if None, use the length of levels
        save_path: Save path; if None, directly plt.show()
        shrink: Colorbar shrink factor
        pad: Colorbar padding
    """
    n = len(items)

    # Validate that if axes is provided, it has the correct length
    if axes is not None and len(axes) != n:
        raise ValueError(
            f"Number of provided axes ({len(axes)}) must match number of items ({n})"
        )

    # shape validation
    for Z, xr, yr, _ in items:
        if Z.shape != (xr[2], yr[2]):
            raise ValueError(f"Matrix shape {Z.shape} ≠ ranges ({xr[2]},{yr[2]})")

    # 1) calculate levels
    if levels is None:
        all_vals = np.concatenate([Z.ravel() for Z, *_ in items])
        levels = np.percentile(all_vals, np.linspace(0, 100, 11))
    levels = np.unique(levels)

    cmap0 = cm.get_cmap(base_cmap)
    cmap_nl = nlcmap(cmap0, levels)

    # 2) layout - create figure and axes if not provided
    fig = None
    if axes is None:
        if figsize is None:
            figsize = (5 * n, 5)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
        if n == 1:
            axes = [axes]
    else:
        # If axes are provided, get the figure from the first axes
        fig = axes[0].figure

    # 3) Draw each heatmap (first generate RGBA, then imshow)
    for ax, (Z, xr, yr, title) in zip(axes, items):
        # After transposing and remapping, obtain (ny,nx,4)
        Zt = Z.T
        rgba = cmap_nl(Zt)
        ax.imshow(
            rgba,
            extent=[xr[0], xr[1], yr[0], yr[1]],
            origin="lower",
            aspect="auto",
        )
        ax.set_title(title)

        ax.set_xticks([])
        ax.set_yticks([])

    # 4) Use the same parameters for the color bar as in the first row to ensure alignment.
    if cbar_axes is None and axes is not None:
        # Generate a dedicated color bar for the axes in the second row, ensuring it utilizes the identical `shrink` and `pad` parameters as applied elsewhere.
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=cmap0, norm=Normalize(vmin=levels.min(), vmax=levels.max())),
            ax=axes,
            location='right',
            shrink=shrink,
            pad=pad
        )
    elif cbar_axes is not None:
        sm = cm.ScalarMappable(
            cmap=cmap0, norm=Normalize(vmin=levels.min(), vmax=levels.max())
        )
        sm._A = []
        cbar = plt.colorbar(sm, cax=cbar_axes)
    else:
        # If neither `axes` nor `cbar_axes` are provided, create the default colorbar.
        if fig is not None:
            cbar = fig.colorbar(
                cm.ScalarMappable(cmap=cmap0, norm=Normalize(vmin=levels.min(), vmax=levels.max())),
                ax=axes,
                shrink=shrink,
                pad=pad
            )
        else:
            return

    # colorbar Setting
    if 'cbar' in locals():
        # Only equidistant tick_count ticks are selected here.
        if tick_count is not None and tick_count < len(levels):
            # Take tick_count equally spaced indices on [0, N-1]
            idxs = np.linspace(0, len(levels) - 1, tick_count, dtype=int)
        else:
            idxs = np.arange(len(levels))

        # extract the corresponding transformed_levels and text labels
        ticks = cmap_nl.transformed_levels[idxs]
        labels = [f"{levels[i]:.2f}" for i in idxs]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.set_label(cbar_label)

    if save_path is not None:
        # Assuming Toolbox.create_folder exists
        # Toolbox.create_folder(save_path)
        if fig is not None:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig(save_path, bbox_inches="tight")
    else:
        if fig is not None and axes is None:
            plt.show()


policy_laplace_1 = np.load(
    "Data/test-laplace_1-test-laplace_1_ddpg_mala_average_policy.npy"
)
policy_laplace_2 = np.load(
    "Data/test-laplace_2-test-laplace_2_ddpg_mala_average_policy.npy"
)
policy_banana = np.load("Data/test-banana-test-banana_ddpg_mala_average_policy.npy")

laplace_1_xr, laplace_1_yr = ((-3, 3, 100), (-3, 3, 100))
laplace_2_xr, laplace_2_yr = ((-3, 3, 100), (-3, 3, 100))
banana_xr, banana_yr = ((-4, 4, 100), (5, 15, 100))

items = [
    (policy_laplace_1, laplace_1_xr, laplace_1_yr, ""),
    (policy_laplace_2, laplace_2_xr, laplace_2_yr, ""),
    (policy_banana, banana_xr, banana_yr, ""),
]


base_cmap = "rainbow"

all_vals = np.concatenate([Z.ravel() for Z, *_ in items])
vmin, vmax = all_vals.min(), all_vals.max()


levels = np.unique(
    np.concatenate(
        [
            np.linspace(vmin, 0.095, 200),
            np.linspace(0.095, 2.1, 50),
            np.linspace(2.1, vmax, 200),
        ]
    )
)

# When calling the second row of drawing functions, pass the same shrink and pad parameters as the first row.
plot_heatmaps_nonlinear_shared_colorbar(
    items,
    levels=levels,
    base_cmap=base_cmap,
    axes=axes[1],
    figsize=(15, 5),
    cbar_label=r"$\epsilon(x)$",
    shrink=1.0,  # Align with the first color bar.
    pad=0.02     # Align with the first color bar.
)

# Third
model_list = [
    "test-laplace_1-test-laplace_1",
    "test-laplace_2-test-laplace_2",
    "test-banana-test-banana",
]
mcmc_env = "mala"

for ax, model_name in zip(axes[2], model_list):
    pp = PlotPipeLine(log_mode=True, axes=ax)

    const_dir = "../" + model_name.split("-")[1] + "/const"
    for file_path in sorted(glob.glob(f"{const_dir}/*.csv")):
        pp.store_to_dict(file_path)

    flex = FlexPipeLine(
        input_file=f"../{model_name.split('-')[1]}/flex/{model_name}_{mcmc_env}_mmd.txt"
    )

    pp.plot_const(mcmc_env=mcmc_env)
    pp.plot_flex(
        median=flex.median,
        left_quantile=flex.left_quantile,
        right_quantile=flex.right_quantile,
    )

    # Set the y-axis of the third row to a logarithmic scale.
    ax.set_yscale("log")

    ax.set_xlabel(r"$\epsilon$")
    if ax is axes[2][0]:
        ax.set_ylabel("MMD")
    else:
        ax.tick_params(labelleft=False)

plt.savefig("policy_compare.pdf", bbox_inches="tight")