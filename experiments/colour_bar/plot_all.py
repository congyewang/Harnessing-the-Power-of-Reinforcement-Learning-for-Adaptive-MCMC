import glob
from builtins import getattr
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.contour import QuadContourSet
from matplotlib.ticker import MaxNLocator
from numpy import typing as npt

from pyrlmala.utils.plot import FlexPipeLine, PlotPipeLine
from pyrlmala.utils.target import AutoStanTargetPDF


@dataclass
class PlotConfig:
    """
    Config for matplotlib plotting styles and parameters.
    """

    latex_style: bool = True
    font_size: int = 36
    base_cmap: str = "rainbow"
    figure_size: Tuple[float, float] = (15, 15)
    colorbar_shrink: float = 1.0
    colorbar_pad: float = 0.02

    def apply_style(self) -> None:
        """
        Apply the matplotlib style settings based on the configuration.
        """
        if self.latex_style:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                }
            )

        plt.rcParams.update(
            {
                "font.size": self.font_size,
                "axes.titlesize": self.font_size,
                "axes.labelsize": self.font_size,
                "xtick.labelsize": self.font_size,
                "ytick.labelsize": self.font_size,
                "legend.fontsize": self.font_size,
            }
        )


@dataclass
class TargetConfig:
    """
    Configuration for target functions used in the plots.
    """

    name: str
    mesh_range_x: Tuple[float, float, int]
    mesh_range_y: Tuple[float, float, int]
    data_file_pattern: Optional[str] = None


class nlcmap:
    """
    Non-linear color map class.
    """

    def __init__(
        self,
        cmap: str,
        levels: npt.NDArray[np.float64],
    ) -> None:
        """
        Initialize the non-linear color map.

        Args:
            cmap (str): Name of the color map to use.
            levels (npt.NDArray[np.float64]): Levels for the color mapping.
        """
        self.cmap = cmap
        self.levels = np.asarray(levels, dtype="float64")
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(0.0, self.levmax, len(self.levels))

    def __call__(
        self,
        xi: npt.NDArray[np.float64],
        alpha: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """
        Apply the non-linear color map to the input array.

        Args:
            xi (npt.NDArray[np.float64]): Input array to be color-mapped.
            alpha (float, optional): Alpha blending value. Defaults to 1.0.

        Returns:
            npt.NDArray[np.float64]: Color-mapped array.
        """
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)


class ScientificPlotter:
    """
    A class for creating scientific plots.
    """

    def __init__(self, config: PlotConfig, posteriordb_path: str) -> None:
        """
        Initialize the ScientificPlotter with configuration and posteriordb path.

        Args:
            config (PlotConfig): Configuration for plotting styles and parameters.
            posteriordb_path (str): Path to the posterior database.
        """
        self.config = config
        self.config.apply_style()
        self.posteriordb_path = posteriordb_path

        # Prepare color map and normalization for PDF plots
        self.pdf_boundaries = np.array([0.00, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 3.5])
        self.pdf_cmap = getattr(cm, "viridis")
        self.pdf_norm = BoundaryNorm(
            self.pdf_boundaries, ncolors=self.pdf_cmap.N, clip=True
        )

    def create_target_pdf(self, target_name: str) -> AutoStanTargetPDF:
        """
        Create a target PDF object for the specified target name.

        Args:
            target_name (str): Name of the target function.

        Returns:
            AutoStanTargetPDF: An instance of AutoStanTargetPDF for the target.
        """
        return AutoStanTargetPDF(target_name, self.posteriordb_path)

    def plot_target_2d(
        self,
        ax: plt.Axes,
        x_mesh_range: Tuple[float, float, int],
        y_mesh_range: Tuple[float, float, int],
        log_target_pdf: Callable[[np.ndarray], float],
        norm: Optional[BoundaryNorm] = None,
        levels: Optional[npt.NDArray[np.float64]] = None,
        cmap: str = "turbo",
        show_ticks: bool = False,
        n_xticks: int = 5,
        n_yticks: int = 5,
    ) -> QuadContourSet:
        """
        Plot the 2D target function as a contour plot.

        Args:
            ax (plt.Axes): The axes to plot on.
            x_mesh_range (Tuple[float, float, int]): Range and number of points for x-axis mesh.
            y_mesh_range (Tuple[float, float, int]): Range and number of points for y-axis mesh.
            log_target_pdf (Callable[[np.ndarray], float]): Function to compute the log PDF values.
            norm (Optional[BoundaryNorm]): Normalization for the color map.
            levels (Optional[npt.NDArray[np.float64]]): Levels for the contour plot.
            cmap (str): Name of the color map to use.
            show_ticks (bool): Whether to show ticks on the axes.
            n_xticks (int): Number of ticks on the x-axis.
            n_yticks (int): Number of ticks on the y-axis.

        Returns:
            QuadContourSet: The contour set created by the plot.
        """
        if norm is None:
            norm = self.pdf_norm
        if levels is None:
            levels = self.pdf_boundaries

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

        self._setup_ticks(ax, x0, x1, y0, y1, show_ticks, n_xticks, n_yticks)
        ax.set_aspect("auto")
        return cf

    def plot_heatmaps_nonlinear(
        self,
        items: List[
            Tuple[
                npt.NDArray[np.float64],
                Tuple[float, float, int],
                Tuple[float, float, int],
                str,
            ]
        ],
        axes: List[plt.Axes],
        levels: Optional[npt.NDArray[np.float64]] = None,
        cbar_label: str = "Value",
        tick_count: Optional[int] = 5,
    ) -> None:
        """
        Plot the nonlinear heatmaps.

        Args:
            items (List[Tuple[npt.NDArray[np.float64], Tuple[float, float, int], Tuple[float, float, int], str]]):
                List of tuples containing the heatmap data, x and y ranges, and title.
            axes (List[plt.Axes]): List of axes to plot on.
            levels (Optional[npt.NDArray[np.float64]]): Custom levels for the color map.
            cbar_label (str): Label for the color bar.
            tick_count (Optional[int]): Number of ticks on the color bar. If None
        """
        # Calculate levels if not provided
        if levels is None:
            all_vals = np.concatenate([Z.ravel() for Z, *_ in items])
            levels = np.percentile(all_vals, np.linspace(0, 100, 11))
        levels = np.unique(levels)

        cmap0 = plt.get_cmap(self.config.base_cmap)
        cmap_nl = nlcmap(cmap0, levels)

        # Plot each heatmap
        for ax, (Z, xr, yr, title) in zip(axes, items):
            Zt = Z.T
            rgba = cmap_nl(Zt)
            ax.imshow(
                rgba, extent=[xr[0], xr[1], yr[0], yr[1]], origin="lower", aspect="auto"
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        # Create colorbar
        cbar = axes[0].figure.colorbar(
            cm.ScalarMappable(
                cmap=cmap0, norm=Normalize(vmin=levels.min(), vmax=levels.max())
            ),
            ax=axes,
            location="right",
            shrink=self.config.colorbar_shrink,
            pad=self.config.colorbar_pad,
        )

        self._setup_colorbar(cbar, cmap_nl, levels, cbar_label, tick_count)

    def plot_mmd_comparison(
        self,
        axes: List[plt.Axes],
        model_configs: List[TargetConfig],
        mcmc_env: str = "mala",
    ) -> None:
        """
        Plot the MMD comparison.

        Args:
            axes (List[plt.Axes]): List of axes to plot on.
            model_configs (List[TargetConfig]): List of target configurations for MMD comparison.
            mcmc_env (str): MCMC environment to use for loading data.
        """
        for ax, model_config in zip(axes, model_configs):
            pp = PlotPipeLine(log_mode=True, axes=ax)

            # Load constant data
            const_dir = f"../{model_config.name.split('-')[1]}/const"
            for file_path in sorted(glob.glob(f"{const_dir}/*.csv")):
                pp.store_to_dict(file_path)

            # Load flexible data
            flex_file = f"../{model_config.name.split('-')[1]}/flex/{model_config.name}_{mcmc_env}_mmd.txt"
            flex = FlexPipeLine(input_file=flex_file)

            # Plot constant and flexible MMD
            pp.plot_const(mcmc_env=mcmc_env)
            pp.plot_flex(
                median=flex.median,
                left_quantile=flex.left_quantile,
                right_quantile=flex.right_quantile,
            )

            # Set up axes
            ax.set_yscale("log")
            ax.set_xlabel(r"$\epsilon$")

            if ax is axes[0]:
                ax.set_ylabel("MMD")
            else:
                ax.tick_params(labelleft=False)

            ax.set_ylim(0.0002793883107761695, 0.1)

    def _setup_ticks(
        self,
        ax: plt.Axes,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        show_ticks: bool,
        n_xticks: int,
        n_yticks: int,
    ) -> None:
        """
        Set up the ticks for the axes.

        Args:
            ax (plt.Axes): The axes to set up ticks for.
            x0 (float): Minimum x value.
            x1 (float): Maximum x value.
            y0 (float): Minimum y value.
            y1 (float): Maximum y value.
            show_ticks (bool): Whether to show ticks on the axes.
            n_xticks (int): Number of ticks on the x-axis.
            n_yticks (int): Number of ticks on the y-axis.
        """
        if show_ticks:
            ax.set_xticks(np.linspace(x0, x1, n_xticks))
            ax.set_yticks(np.linspace(y0, y1, n_yticks))
            ax.tick_params(
                axis="both", which="major", labelsize=plt.rcParams["xtick.labelsize"]
            )
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    def _setup_colorbar(
        self,
        cbar: Colorbar,
        cmap_nl: nlcmap,
        levels: np.ndarray,
        cbar_label: str,
        tick_count: int,
    ) -> None:
        """
        Set up the colorbar.

        Args:
            cbar (plt.colorbar.Colorbar): The colorbar to set up.
            cmap_nl (nlcmap): Non-linear color map.
            levels (np.ndarray): Levels for the color mapping.
            cbar_label (str): Label for the color bar.
            tick_count (int): Number of ticks on the color bar.
        """
        if tick_count is not None and tick_count < len(levels):
            idxs = np.linspace(0, len(levels) - 1, tick_count, dtype=int)
        else:
            idxs = np.arange(len(levels))

        ticks = cmap_nl.transformed_levels[idxs]
        labels = [f"{levels[i]:.2f}" for i in idxs]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.set_label(cbar_label)


def create_custom_levels(vmin: float, vmax: float) -> npt.NDArray[np.float64]:
    """
    Create custom levels for color mapping.

    Args:
        vmin (float): Minimum value for the levels.
        vmax (float): Maximum value for the levels.

    Returns:
        npt.NDArray[np.float64]: Array of unique levels for color mapping.
    """
    return np.unique(
        np.concatenate(
            [
                np.linspace(vmin, 0.095, 200),
                np.linspace(0.095, 2.1, 50),
                np.linspace(2.1, vmax, 200),
            ]
        )
    )


def main() -> None:
    """
    Main function.
    """
    # Configure plotting styles and parameters
    config = PlotConfig(
        latex_style=True, font_size=36, base_cmap="magma", figure_size=(15, 15)
    )

    posteriordb_path = "../posteriordb/posterior_database"

    # Define target configurations
    target_configs = [
        TargetConfig("test-laplace_1-test-laplace_1", (-3, 3, 200), (-3, 3, 200)),
        TargetConfig("test-laplace_2-test-laplace_2", (-2, 2, 200), (-2, 2, 200)),
        TargetConfig("test-banana-test-banana", (-2.5, 2.5, 200), (7, 17, 200)),
    ]

    # Create the plotter instance
    plotter = ScientificPlotter(config, posteriordb_path)

    # Create figure and subplots
    fig, axes = plt.subplots(
        3,
        3,
        figsize=config.figure_size,
        constrained_layout=True,
        subplot_kw={"aspect": "auto"},
    )

    # Plot the target functions in the first row
    targets = [plotter.create_target_pdf(tc.name) for tc in target_configs]
    funcs = [target.log_target_pdf for target in targets]

    for ax, func, target_config in zip(axes[0], funcs, target_configs):
        plotter.plot_target_2d(
            ax,
            target_config.mesh_range_x,
            target_config.mesh_range_y,
            func,
            norm=plotter.pdf_norm,
            levels=plotter.pdf_boundaries,
            cmap=plotter.pdf_cmap,
        )

    # Create colorbar for the first row
    cbar1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=plotter.pdf_norm, cmap=plotter.pdf_cmap),
        ax=axes[0],
        boundaries=plotter.pdf_boundaries,
        spacing="proportional",
        ticks=plotter.pdf_boundaries,
        shrink=config.colorbar_shrink,
        pad=config.colorbar_pad,
    )
    cbar1.set_label(r"$p(x)$")
    cbar1.locator = MaxNLocator(nbins=5)

    # Plot the policy heatmaps in the second row
    policy_data = []
    for target_config in target_configs:
        policy_file = f"Data/{target_config.name}_ddpg_mala_average_policy.npy"
        policy = np.load(policy_file)
        policy_data.append(
            (policy, target_config.mesh_range_x, target_config.mesh_range_y, "")
        )

    # Create custom levels for the heatmaps
    all_vals = np.concatenate([Z.ravel() for Z, *_ in policy_data])
    vmin, vmax = all_vals.min(), all_vals.max()
    custom_levels = create_custom_levels(vmin, vmax)

    plotter.plot_heatmaps_nonlinear(
        policy_data, axes[1], levels=custom_levels, cbar_label=r"$\epsilon(x)$"
    )

    # Plot the MMD comparison in the third row
    plotter.plot_mmd_comparison(axes[2], target_configs)

    # Save the figure
    plt.savefig("plot_all.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
