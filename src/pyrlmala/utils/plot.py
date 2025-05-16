import copy
import glob
import itertools
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAliasType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cytoolz import pipe, topk
from cytoolz.curried import map
from jaxtyping import Float
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from numpy import typing as npt
from torch import nn
from tqdm.auto import tqdm

from ..agent import PolicyNetwork
from .utils import Toolbox

ActorWeights = TypeAliasType("ActorWeights", Dict[str, Any])
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
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 16,
    }
)


class CleanPipeLine:
    @staticmethod
    def delete_duplicated_lines(input_file: str, output_file: str) -> None:
        pd.read_csv(input_file).drop_duplicates().to_csv(output_file, index=False)

    @staticmethod
    def delete_matrics(input_file: str, output_file: str) -> None:
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        filtered_lines = [
            line for line in lines if not re.match(r"^(Mean:|Median:|SE:)", line)
        ]

        with open(output_file, "w", encoding="utf-8") as fout:
            fout.writelines(filtered_lines)

    @staticmethod
    def sort_by_random_seed(input_file: str, output_file: str) -> None:
        pd.read_csv(input_file).sort_values(by="random_seed").to_csv(
            output_file, index=False
        )

    @classmethod
    def pipline(cls, input_file: str, output_file: str) -> None:
        cls.delete_duplicated_lines(input_file, output_file)
        cls.delete_matrics(output_file, output_file)
        cls.sort_by_random_seed(output_file, output_file)


class FlexPipeLine:
    def __init__(self, input_file: str) -> None:
        self.input_file = input_file
        self.clean_file()
        self.df = pd.read_csv(self.input_file)

    def add_header(self, header: List[str], input_file: str, output_file: str) -> None:
        with open(input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split(",")
        try:
            _ = [float(item) for item in first_line]
            has_header = False
        except ValueError:
            has_header = True

        if has_header:
            pass
        else:
            df = pd.read_csv(input_file, header=None)
            df.columns = header
            df.to_csv(output_file, index=False)

    def clean_file(self) -> None:
        CleanPipeLine.delete_duplicated_lines(self.input_file, self.input_file)
        CleanPipeLine.delete_matrics(self.input_file, self.input_file)
        self.add_header(
            header=["random_seed", "mmd"],
            input_file=self.input_file,
            output_file=self.input_file,
        )

    @property
    def mean(self) -> float:
        return self.df["mmd"].mean().item()

    @property
    def se(self) -> float:
        return float(self.df["mmd"].std(ddof=1) / (self.df["mmd"].count() ** 0.5))

    @property
    def median(self) -> float:
        return self.df["mmd"].median().item()

    @property
    def left_quantile(self) -> float:
        return self.df["mmd"].quantile(0.25).item()

    @property
    def right_quantile(self) -> float:
        return self.df["mmd"].quantile(0.75).item()


class PlotPipeLine:
    def __init__(self, log_mode: bool = True, axes: Optional[Axes] = None) -> None:
        if axes:
            self.ax = axes
        else:
            self.ax = self.make_axes()

        if log_mode:
            self.ax.set_xscale("log")
            self.ax.set_yscale("log")

        self.res: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def store_to_dict(self, file_path: str) -> None:
        df = pd.read_csv(file_path)

        mcmc_env = df["mcmc_env"].unique().item()
        step_size = df["step_size"].unique().item()

        median = df["mmd"].median()
        left_quantile = df["mmd"].quantile(0.25)
        right_quantile = df["mmd"].quantile(0.75)

        self.res[mcmc_env][step_size] = {
            "median": median,
            "left_quantile": left_quantile,
            "right_quantile": right_quantile,
        }

    def make_axes(self) -> Axes:
        _, ax = plt.subplots(figsize=(5, 5))
        return ax

    def plot_const(self, mcmc_env: str = "mala") -> None:
        x_ranges = np.array(sorted([i for i in self.res[mcmc_env].keys()]))
        y_median = np.array([self.res[mcmc_env][float(x)]["median"] for x in x_ranges])
        y_left_quantile = np.array(
            [self.res[mcmc_env][float(x)]["left_quantile"] for x in x_ranges]
        )
        y_right_quantile = np.array(
            [self.res[mcmc_env][float(x)]["right_quantile"] for x in x_ranges]
        )

        self.ax.plot(x_ranges, y_median, label="Constant Policy")
        self.ax.fill_between(x_ranges, y_left_quantile, y_right_quantile, alpha=0.3)

        self.ax.relim()
        self.ax.autoscale_view()

        self.x_ranges = x_ranges

    def plot_flex(
        self,
        median: float | npt.NDArray[np.floating],
        left_quantile: float | npt.NDArray[np.floating],
        right_quantile: float | npt.NDArray[np.floating],
    ) -> None:
        self.ax.axhline(median, color="red", linestyle="--", label="Flexible Policy")
        self.ax.fill_between(
            self.x_ranges, left_quantile, right_quantile, alpha=0.2, color="red"
        )

    def plot_total(
        self,
        mcmc_env: str,
        flex_median: float | npt.NDArray[np.floating],
        flex_left_quantile: float | npt.NDArray[np.floating],
        flex_right_quantile: float | npt.NDArray[np.floating],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        self.plot_const(mcmc_env=mcmc_env)
        self.plot_flex(
            median=flex_median,
            left_quantile=flex_left_quantile,
            right_quantile=flex_right_quantile,
        )

        if title:
            self.ax.set_title(title)

        if save_path:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def execute(
        self,
        mcmc_env: str,
        const_dir: str,
        flex_file_path: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        file_path_list = sorted(glob.glob(f"{const_dir}/*.csv"))

        for file_path in tqdm(file_path_list):
            CleanPipeLine.delete_duplicated_lines(file_path, file_path)
            CleanPipeLine.delete_matrics(file_path, file_path)
            CleanPipeLine.sort_by_random_seed(file_path, file_path)

            self.store_to_dict(file_path)

        flex_pipeline = FlexPipeLine(input_file=flex_file_path)
        self.plot_total(
            mcmc_env=mcmc_env,
            flex_median=flex_pipeline.median,
            flex_left_quantile=flex_pipeline.left_quantile,
            flex_right_quantile=flex_pipeline.right_quantile,
            title=title,
            save_path=save_path,
        )


class PolicyPlot:
    @staticmethod
    def policy_plot_2D_heatmap(
        policy: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        x_range: Float[torch.Tensor, "x"],
        y_range: Float[torch.Tensor, "y"],
        softplus_mode: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        axes: Optional[Axes] = None,
    ) -> None:
        """
        Plot the policy heatmap.

        Args:
            policy (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Policy function.
            x_range (Float[torch.Tensor, "x"], optional): x range. e.g. torch.arange(-5, 5, 0.1).
            y_range (Float[torch.Tensor, "y"], optional): y range. e.g. torch.arange(-5, 5, 0.1).
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
            save_path (Optional[str], optional): Save path. Defaults to None.
            axes (Optional[plt.Axes]): External axes for subplots. If None, creates a new figure.
        """
        if axes is None:
            _, axes = plt.subplots(figsize=(5, 5))

        # Plot heatmap
        heatmap_plot = lambda x: axes.imshow(
            x.T,
            extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )

        if softplus_mode:
            pipe(
                (x_range, y_range),
                lambda ranges: Toolbox.imbalanced_mesh_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                F.softplus,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )
        else:
            pipe(
                (x_range, y_range),
                lambda ranges: Toolbox.imbalanced_mesh_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )

        axes.set_xlabel("x")
        axes.set_ylabel("y")
        if title is not None:
            axes.set_title(title)

        cbar = plt.colorbar(axes.images[0], ax=axes, shrink=0.8)
        cbar.set_label("Step Size")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()


class AverageActor(nn.Module):
    """
    AverageActor.
    """

    def __init__(
        self,
        actor: PolicyNetwork,
        weights: List[ActorWeights],
    ) -> None:
        """
        Initialize the AverageActor.

        Args:
            actor (PolicyNetwork): The actor network.
            weights (List[ActorWeights]): The weights of the given policies.
        """
        super(AverageActor, self).__init__()

        self._check_if_weights_valid(weights)

        self.models = nn.ModuleList()
        self._load_from_weights(
            actor, list(weights)
        )  # Copy the weights to avoid modifying the original list

    def _check_if_weights_valid(
        self,
        weights: List[ActorWeights],
    ) -> None:
        """
        Check if the weights are valid.

        Args:
            weights (List[ActorWeights]): The weights of the given policies.
        """
        if not isinstance(weights, list):
            raise TypeError("weights must be a list of state_dict dictionaries.")

        for weight in weights:
            if not isinstance(weight, dict):
                raise TypeError("weights must be a list of state_dict dictionaries.")

    def _load_from_weights(
        self,
        actor: PolicyNetwork,
        weights: List[ActorWeights],
    ) -> None:
        """
        Load the actor from the weights.

        Args:
            actor (PolicyNetwork): The actor network.
            weights (List[ActorWeights]): The weights of the given policies.
        """
        while weights:
            model = copy.deepcopy(actor)
            model.load_state_dict(weights.pop())
            model.eval()
            self.models.append(model)

    def forward(self, x: Float[torch.Tensor, "state"]) -> Float[torch.Tensor, "action"]:
        """
        Forward pass of the ensemble policy.
        Args:
            x (Float[torch.Tensor, "state"]): The input state.

        Returns:
            Float[torch.Tensor, "action"]: The mean action from the ensemble.
        """
        with torch.no_grad():
            actions = torch.stack([model(x) for model in self.models])

            return actions.mean(dim=0)


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


class AveragePolicy:
    @staticmethod
    def generate_state_mesh(
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> Callable[
        [Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        Float[torch.Tensor, "mesh_2d"],
    ]:
        """
        Generate a mesh grid from ranges. The mesh grid is used to evaluate the policy.

        Args:
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Ranges for the mesh grid.

        Returns:
            Callable[[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], Float[torch.Tensor, "mesh_2d"]]: Mesh grid generator.
        """
        return pipe(
            ranges,
            map(lambda r: torch.linspace(*r)),
            lambda ranges: Toolbox.imbalanced_mesh_2d(*ranges),
            lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
            lambda x: x.double(),
        )

    @classmethod
    def calculate_mean_policy(
        cls,
        actor: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        weights_root_dir: str,
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        last_step_num: int,
        frequency_per_step: int,
    ):
        """
        Calculate the mean policy from the weights.

        Args:
            actor (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Actor function.
            weights_root_dir (str): Path to the weights.
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Range for the state mesh.
            last_step_num (int): Last step number to consider for the plot.
            frequency_per_step (int): Frequency of steps.

        Returns:
            Callable[[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], Float[torch.Tensor, "mesh_2d"]]: Mesh grid generator.

        Raises:
            ValueError: If the weights root directory does not exist.
        """
        if not os.path.exists(weights_root_dir):
            raise ValueError(f"Path {weights_root_dir} does not exist.")

        extract_step_number = lambda x: int(re.search(r"\d+", x).group())
        weights_path = topk(
            last_step_num,
            glob.glob(os.path.join(weights_root_dir, "*.pth")),
            key=extract_step_number,
        )
        weights_path_slices = itertools.islice(
            weights_path, 0, None, frequency_per_step
        )
        weights = [torch.load(i) for i in weights_path_slices]
        average_actor = AverageActor(actor, weights)

        return pipe(ranges, cls.generate_state_mesh, average_actor)

    @classmethod
    def plot_policy(
        cls,
        actor: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        weights_root_dir: str,
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        last_step_num: int,
        frequency_per_step: int,
        softplus_mode: bool = True,
        colorbar_range: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        title: str = "",
        axes: Optional[Axes] = None,
    ) -> None:
        """
        Plot the policy based on the provided actor and weights.

        Args:
            actor (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Actor function.
            weights_root_dir (str): Path to the weights.
            last_step_num (int): Last step number to consider for the plot.
            frequency_per_step (int): Frequency of steps.
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Range for the state mesh.
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
            colorbar_range (Optional[Tuple[float, float]], optional): Colorbar range. Defaults to None.
            save_path (Optional[str], optional): Save path. Defaults to None.
            title (str, optional): Title of the plot. Defaults to "".
            axes (Optional[plt.Axes]): External axes for subplots. If None, creates a new figure.
        """
        if axes is None:
            _, axes = plt.subplots(figsize=(5, 5))

        # Plot heatmap
        heatmap_plot = lambda x: axes.imshow(
            x.T,
            extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=colorbar_range[0] if colorbar_range else None,
            vmax=colorbar_range[1] if colorbar_range else None,
        )

        # Calculate the policy
        if softplus_mode:
            pipe(
                ranges,
                lambda x: cls.calculate_mean_policy(
                    actor,
                    weights_root_dir,
                    x,
                    last_step_num,
                    frequency_per_step,
                ),
                F.softplus,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(ranges[0][2], ranges[1][2]),
                heatmap_plot,
            )
        else:
            pipe(
                ranges,
                lambda x: cls.calculate_mean_policy(
                    actor,
                    weights_root_dir,
                    x,
                    last_step_num,
                    frequency_per_step,
                ),
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(ranges[0][2], ranges[1][2]),
                heatmap_plot,
            )

        if title:
            axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        cbar = plt.colorbar(axes.images[0], ax=axes, shrink=0.8)
        cbar.set_label("Step Size")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @classmethod
    def get_policy_matrix(
        cls,
        actor: Callable[[torch.Tensor], torch.Tensor],
        weights_root_dir: str,
        ranges: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        last_step_num: int,
        frequency_per_step: int,
        softplus_mode: bool = True,
    ) -> npt.NDArray[np.float64]:
        """
        Compute and return the Z matrix of the policy heatmap, shape = (nx, ny).

        Args:
            actor (Callable[[torch.Tensor], torch.Tensor]): Actor function.
            weights_root_dir (str): Path to the weights.
            ranges (Tuple[Tuple[float, float, int], Tuple[float, float, int]]): Range for the state mesh.
            last_step_num (int): Last step number to consider for the plot.
            frequency_per_step (int): Frequency of steps.
            softplus_mode (bool, optional): Softplus mode. Defaults to True.

        Returns:
            npt.NDArray[np.float64]: Z matrix of the policy heatmap.
        """
        # reusing calculate_mean_policy only gets raw tensor
        tensor = cls.calculate_mean_policy(
            actor, weights_root_dir, ranges, last_step_num, frequency_per_step
        )
        if softplus_mode:
            tensor = F.softplus(tensor)
        arr = tensor.detach().numpy()[:, 0]
        # reshape into (nx, ny)
        nx, ny = ranges[0][2], ranges[1][2]

        return arr.reshape(nx, ny)

    @classmethod
    def save_policy_matrix(
        cls,
        actor: Callable[[torch.Tensor], torch.Tensor],
        weights_root_dir: str,
        ranges: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        last_step_num: int,
        frequency_per_step: int,
        save_path: str,
        softplus_mode: bool = True,
        fmt: str = "csv",
    ) -> None:
        """
        Save the policy matrix to local storage. Supports 'csv' or 'npy'.

        Args:
            actor (Callable[[torch.Tensor], torch.Tensor]): Actor function.
            weights_root_dir (str): Path to the weights.
            ranges (Tuple[Tuple[float, float, int], Tuple[float, float, int]]): Range for the state mesh.
            last_step_num (int): Last step number to consider for the plot.
            frequency_per_step (int): Frequency of steps.
            save_path (str): Path to save the policy matrix.
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
            fmt (str, optional): Format of the saved file. Defaults to 'csv'.
        """
        Z = cls.get_policy_matrix(
            actor,
            weights_root_dir,
            ranges,
            last_step_num,
            frequency_per_step,
            softplus_mode,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if fmt.lower() == "npy":
            np.save(save_path, Z)
        else:
            np.savetxt(save_path, Z, delimiter=",")
        print(f"Policy matrix saved to {save_path} (shape={Z.shape})")

    @classmethod
    def plot_policies_shared_colorbar(
        cls,
        configs: List[
            Tuple[Callable[[torch.Tensor], torch.Tensor], str, int, int, str]
        ],
        ranges: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        softplus_mode: bool = True,
        boundaries: Optional[np.ndarray] = None,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = None,
    ) -> None:
        """
        Arrange multiple policy heatmaps into a row of subplots, sharing a single "non-uniform" colorbar.

        Args:
            configs: Each element is (actor, weights_root_dir, last_step_num,
                        frequency_per_step, title).
            ranges: Same as the ranges in plot_policy.
            softplus_mode: Whether to apply softplus before taking the matrix.
            boundaries: If provided, the segmentation boundaries for the non-uniform colorbar; otherwise, automatically generated based on the 10%~90% percentiles.
            cmap: Name of the colormap.
            figsize: Overall canvas size, default is (5*len(configs), 5).
        """
        n = len(configs)
        if figsize is None:
            figsize = (5 * n, 5)
        fig, axes = plt.subplots(1, n, sharey=False, figsize=figsize, squeeze=False)
        axes = axes[0]

        # 1) First calculate all the Z matrices.
        all_Z = []
        for actor, root, last_step, freq, _title in configs:
            Z = cls.get_policy_matrix(
                actor, root, ranges, last_step, freq, softplus_mode
            )
            all_Z.append(Z)

        # 2) Automatically generate non-uniform boundaries (if not provided by the user)
        if boundaries is None:
            concat = np.concatenate([Z.flatten() for Z in all_Z])
            # Take the 0%, 10%, 20%, ..., 100% quantiles as boundaries.
            boundaries = np.percentile(concat, np.linspace(0, 100, 11))

        norm = colors.BoundaryNorm(boundaries, ncolors=256, clip=True)

        # 3) Draw each subplot
        ims = []
        for ax, Z, (_, _, _, _, title) in zip(axes, all_Z, configs):
            im = ax.imshow(
                Z.T,
                extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
                origin="lower",
                cmap=cmap,
                aspect="auto",
                norm=norm,
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ims.append(im)

        # 4) Use a single "non-uniform" colorbar
        cbar = fig.colorbar(
            ims[0],
            ax=axes.tolist(),
            boundaries=boundaries,
            spacing="proportional",
            shrink=0.8,
        )
        cbar.set_label("Step Size")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_matrix(
        Z: npt.NDArray[np.float64],
        ranges: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        axes: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: str = "",
        colorbar: bool = True,
        cbar_label: str = "Step Size",
    ) -> None:
        """
        Plot a heatmap based on the given Z matrix and ranges.

        Args:
            Z (npt.NDArray[np.float64]): Numerical matrix with shape (nx, ny).
            ranges (Tuple[Tuple[float, float, int], Tuple[float, float, int]]): ((x_min, x_max, nx), (y_min, y_max, ny)).
            axes (Optional[plt.Axes]): Matplotlib Axes instance. If None, a new figure will be created.
            cmap (str): Colormap name.
            vmin (Optional[float]), vmax: Colorbar range.
            title (Optional[float]): Subplot title.
            colorbar (bool): Whether to display the colorbar.
            cbar_label (str): Colorbar label text.
        """
        if axes is None:
            _, axes = plt.subplots(figsize=(5, 5))
        im = axes.imshow(
            Z.T,
            extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        if title:
            axes.set_title(title)

        if colorbar:
            cbar = plt.colorbar(im, ax=axes, shrink=0.8)
            cbar.set_label(cbar_label)

    @classmethod
    def plot_matrix_from_file(
        cls,
        path: str,
        ranges: Tuple[Tuple[float, float, int], Tuple[float, float, int]],
        axes: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: str = "",
        colorbar: bool = True,
        cbar_label: str = "Step Size",
    ) -> None:
        """
        Load a matrix from .npy or .csv files and plot a heatmap.

        Args:
            path (str): Local file path ending with .npy or .csv.
            ranges (Tuple[Tuple[float, float, int], Tuple[float, float, int]]): ((x_min, x_max, nx), (y_min, y_max, ny)).
            axes (Optional[plt.Axes], optional): Matplotlib Axes instance. If None, a new figure will be created.
            cmap (str, optional): Colormap name. Defaults to "viridis".
            vmin (Optional[float], optional): Colorbar minimum value. Defaults to None.
            vmax (Optional[float], optional): Colorbar maximum value. Defaults to None.
            title (str, optional): Subplot title. Defaults to "".
            colorbar (bool, optional): Whether to display the colorbar. Defaults to True.
            cbar_label (str, optional): Colorbar label text. Defaults to "Step Size".
        """
        # 1) Load matrix
        ext = path.split(".")[-1].lower()
        if ext == "npy":
            Z = np.load(path)
        elif ext == "csv":
            Z = np.loadtxt(path, delimiter=",")
        else:
            raise ValueError("Unsupported file type: must be .npy or .csv")

        # 2) Check if the shape matches the ranges
        nx, ny = ranges[0][2], ranges[1][2]
        if Z.shape != (nx, ny):
            raise ValueError(f"Matrix shape {Z.shape} doesn't match ranges {nx}×{ny}")

        # 3) Delegated to plot_matrix
        cls.plot_matrix(Z, ranges, axes, cmap, vmin, vmax, title, colorbar, cbar_label)

    @staticmethod
    def plot_matrices_shared_colorbar(
        items: List[
            Tuple[
                np.ndarray,
                Tuple[float, float, int],
                Tuple[float, float, int],
                str,
            ]
        ],
        boundaries: Optional[np.ndarray] = None,
        cmap: str = "viridis",
        figsize: Optional[Tuple[float, float]] = None,
        cbar_label: str = "Step Size",
        shrink: float = 0.8,
    ) -> None:
        """
        Plot a row of heatmaps using the ranges inherent to each matrix, sharing a non-uniform color scale.

        Args:
            items: Each item is (Z, x_range, y_range, title):
                   Z        : np.ndarray, shape=(nx,ny)
                   x_range  : (xmin, xmax, nx)
                   y_range  : (ymin, ymax, ny)
                   title    : Subplot title
            boundaries: Non-uniform color segment boundaries; if None, automatically use the [0%,10%,…,100%] percentiles of all Z
            cmap     : Colormap name
            figsize  : (width, height); if None, then (5*len(items),5)
            cbar_label: Colorbar label
            shrink   : Colorbar shrink ratio
        """
        n = len(items)
        # Verify that the shape of each Z matches its ranges
        for Z, xr, yr, _ in items:
            nx, ny = xr[2], yr[2]
            if Z.shape != (nx, ny):
                raise ValueError(f"Matrix shape {Z.shape} != ranges ({nx},{ny})")

        # 1) Calculate boundary
        if boundaries is None:
            all_vals = np.concatenate([Z.ravel() for Z, *_ in items])
            boundaries = np.percentile(all_vals, np.linspace(0, 100, 11))

        norm = colors.BoundaryNorm(boundaries, ncolors=256, clip=True)

        # 2) Create a subgraph
        if figsize is None:
            figsize = (5 * n, 5)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
        if n == 1:
            axes = [axes]

        # 3) Draw one by one
        first_im = None
        for ax, (Z, xr, yr, title) in zip(axes, items):
            im = ax.imshow(
                Z.T,
                extent=[xr[0], xr[1], yr[0], yr[1]],
                origin="lower",
                cmap=cmap,
                aspect="auto",
                norm=norm,
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            first_im = first_im or im

        # 4) Uniform colorbar
        cbar = fig.colorbar(
            first_im,
            ax=axes,
            boundaries=boundaries,
            spacing="proportional",
            shrink=shrink,
        )
        cbar.set_label(cbar_label)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmaps_nonlinear_shared_colorbar(
        items: List[
            Tuple[
                np.ndarray,
                Tuple[float, float, int],
                Tuple[float, float, int],
                str,
            ]
        ],
        levels: Optional[np.ndarray] = None,
        base_cmap: str = "viridis",
        figsize: Optional[Tuple[float, float]] = None,
        cbar_axes: Optional[plt.Axes] = None,
        cbar_label: str = "Value",
        tick_count: Optional[int] = 5,
        save_path: Optional[str] = None,
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
            cbar_axes: If you want to manually provide an Axes for the colorbar, pass it here; otherwise, one is automatically generated on the right
            cbar_label: Colorbar label text
            tick_count: Number of ticks on the colorbar; if None, use the length of levels
            save_path: Save path; if None, directly plt.show()
        """
        n = len(items)
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

        # 2) layout
        if figsize is None:
            figsize = (5 * n, 5)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
        if n == 1:
            axes = [axes]

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

        # 4) colorbar
        if cbar_axes is None:
            cbar_axes = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        sm = cm.ScalarMappable(
            cmap=cmap0, norm=Normalize(vmin=levels.min(), vmax=levels.max())
        )
        sm._A = []

        cbar = plt.colorbar(sm, cax=cbar_axes)
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

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()


class GeneralPlot:
    """
    A class for general plotting functions.
    """

    @staticmethod
    def imbalanced_mesh_2d(
        x_range: Float[torch.Tensor, "x_range"], y_range: Float[torch.Tensor, "y_range"]
    ) -> Float[torch.Tensor, "mesh_2d"]:
        """
        Construct a 2D meshgrid from x and y ranges.

        Args:
            x_range (Float[torch.Tensor, "x_range"]): x range.
            y_range (Float[torch.Tensor, "y_range"]): y range.

        Returns:
            Float[torch.Tensor, "mesh_2d"]: 2D meshgrid.
        """
        x_repeat = x_range.repeat_interleave(len(y_range))
        y_tile = y_range.repeat(len(x_range))

        return torch.stack([x_repeat, y_tile], dim=1)

    @staticmethod
    def plot_agent(
        indicate: npt.NDArray[np.float64],
        steps_per_episode: int = 100,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the agent's performance.

        Args:
            indicate (npt.NDArray[np.float64]): Indicate array.
            steps_per_episode (int, optional): Steps per episode. Defaults to 100.
            title (str, optional): Title of the plot. Defaults to "".
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        time_points = np.arange(
            steps_per_episode,
            steps_per_episode * (len(indicate) + 1),
            steps_per_episode,
        )

        plt.plot(time_points, indicate)
        if title:
            plt.title(title)

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @classmethod
    def policy_plot_2D_heatmap(
        cls,
        policy: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        x_range: Float[torch.Tensor, "x"],
        y_range: Float[torch.Tensor, "y"],
        softplus_mode: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        axes: Optional[Axes] = None,
    ) -> None:
        """
        Plot the policy heatmap.

        Args:
            policy (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Policy function.
            x_range (Float[torch.Tensor, "x"], optional): x range. e.g. torch.arange(-5, 5, 0.1).
            y_range (Float[torch.Tensor, "y"], optional): y range. e.g. torch.arange(-5, 5, 0.1).
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
            save_path (Optional[str], optional): Save path. Defaults to None.
            axes (Optional[plt.Axes]): External axes for subplots. If None, creates a new figure.
        """
        if axes is None:
            _, axes = plt.subplots()

        # Plot heatmap
        heatmap_plot = lambda x: axes.imshow(
            x.T,
            extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )

        if softplus_mode:
            pipe(
                (x_range, y_range),
                lambda ranges: cls.imbalanced_mesh_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                F.softplus,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )
        else:
            pipe(
                (x_range, y_range),
                lambda ranges: cls.imbalanced_mesh_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )

        if title:
            axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        cbar = plt.colorbar(axes.images[0], ax=axes, shrink=0.8)
        cbar.set_label("Step Size")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def reward_plot(
        reward: npt.NDArray[np.float64],
        step_per_episode: int = 500,
        window_size: int = 5,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the average reward and moving average. Save the plot if save_path is provided.

        Args:
            reward (npt.NDArray[np.float64]): Immediate reward per step.
            step_per_episode (int, optional): Steps per episode. Defaults to 500.
            window_size (int, optional): Window size for moving average. Defaults to 5.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        average_reward = reward.reshape(-1, step_per_episode).mean(axis=1)
        moving_averages = np.convolve(
            a=average_reward, v=np.ones(window_size) / window_size, mode="valid"
        )

        plt.plot(average_reward, label="Average reward")
        plt.plot(moving_averages, label="Moving average")

        plt.xlabel("Episode")
        plt.ylabel("$r_n$")

        plt.legend()
        if title:
            plt.title(title)

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def average_reward_plot(
        reward: npt.NDArray[np.float64],
        step_per_episode: int = 500,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the average reward. Save the plot if save_path is provided.

        Args:
            reward (npt.NDArray[np.float64]): Immediate reward per step.
            step_per_episode (int, optional): Steps per episode. Defaults to 500.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        average_reward = reward.reshape(-1, step_per_episode).mean(axis=1)
        plt.plot(average_reward)

        plt.xlabel("Episode")
        plt.ylabel("$r_n$")

        if title:
            plt.title(title)

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def target_plot_1d(
        x_range: Tuple[float, float, int],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            x_range (Tuple[float, float, int]): x range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function for 1D.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        x = np.linspace(*x_range)
        res = np.exp([log_target_pdf(np.array(i, dtype=np.float64)) for i in x])

        plt.plot(x, res)
        plt.title("Target distribution")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def target_plot_2d(
        x_mesh_range: Tuple[float, float, int],
        y_mesh_range: Tuple[float, float, int],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            x_mesh_range (Tuple[float, float, int]): x mesh range.
            y_mesh_range (Tuple[float, float, int]): y mesh range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function for 2D.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(*x_mesh_range), np.linspace(*y_mesh_range)
        )
        x, y = mesh_x.reshape(1, -1), mesh_y.reshape(1, -1)
        data = np.concatenate([x, y], axis=0).T

        res = np.exp(
            np.array([log_target_pdf(np.array(i, dtype=np.float64)) for i in data])
        )

        plt.contourf(mesh_x, mesh_y, res.reshape(x_mesh_range[2], y_mesh_range[2]))
        plt.colorbar()

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    @classmethod
    def target_plot_multi(
        cls,
        data_range: Tuple[Tuple[float, float, int], ...],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            data_range (Tuple[Tuple[float, float, int], ...]): Data range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        for i in data_range:
            cls.target_plot_1d(i, log_target_pdf, save_path)

    @classmethod
    def target_plot(
        cls,
        data_range: Tuple[Tuple[float, float, int], ...],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            data_range (Tuple[Tuple[float, float, int], ...]): Data range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        sample_dim = len(data_range)
        match sample_dim:
            case 1:
                cls.target_plot_1d(data_range[0], log_target_pdf, save_path)
            case 2:
                cls.target_plot_2d(
                    data_range[0], data_range[1], log_target_pdf, save_path
                )
            case _:
                cls.target_plot_multi(data_range, log_target_pdf, save_path)
