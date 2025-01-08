from abc import ABC, abstractmethod
from typing import Callable, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from numpy import typing as npt
from toolz import curry, pipe
from torch.nn import functional as F

from ..envs import MCMCEnvBase
from ..utils import Toolbox
from .learning import LearningInterface

T = TypeVar("T", bound=Tuple[int, int, float])
clear = Toolbox.get_clear_function()


class PlotterBase(ABC):
    def __init__(self, learning_instance: LearningInterface) -> None:
        """
        Callback for plotting the policy of the learning instance.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
        """
        self.learning_instance = learning_instance

    def _get_current_step(self, env: MCMCEnvBase) -> npt.NDArray[np.float64]:
        """
        Get the current step of the environment.

        Args:
            env (MCMCEnvBase): The environment.

        Returns:
            npt.NDArray[np.float64]: The current step.
        """
        return env.envs[0].get_wrapper_attr("current_step")

    def _get_store_accepted_sample(self, env: MCMCEnvBase) -> npt.NDArray[np.float64]:
        """
        Get the stored accepted sample of the environment.

        Args:
            env (MCMCEnvBase): The environment.

        Returns:
            npt.NDArray[np.float64]: The stored accepted sample.
        """
        return env.envs[0].get_wrapper_attr("store_accepted_sample")

    def _slice_to_current_step(
        self, store_sample: npt.NDArray[np.float64], current_step: int
    ) -> npt.NDArray[np.float64]:
        """
        Slice the stored sample to the current step.

        Args:
            store_sample (npt.NDArray[np.float64]): The stored sample.
            current_step (int): The current step.

        Returns:
            npt.NDArray[np.float64]: The sliced sample.
        """
        return store_sample[0 : current_step - 1]

    @abstractmethod
    def plot(self, step: int, axes: Axes) -> None:
        """
        Plot the policy of the learning instance.

        Args:
            step (int): Current training step.
            axes (Axes): The axes to plot on.
        """
        raise NotImplementedError("Method `plot` must be implemented.")


class PolicyPlotter(PlotterBase):
    def __init__(
        self,
        learning_instance: LearningInterface,
        num_of_mesh: int = 10,
    ) -> None:
        """
        Callback for plotting the policy of the learning instance in 2D. Only works for 2D environments.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
            num_of_mesh (int, optional): The number of mesh points. Defaults to 10.
        """
        super().__init__(learning_instance)
        self.num_of_mesh = num_of_mesh

    def _convert_to_torch(
        self, accepted_sample_np: npt.NDArray[np.float64]
    ) -> Float[torch.Tensor, "accepted sample"]:
        """
        Convert the accepted sample to a torch tensor.

        Args:
            accepted_sample_np (npt.NDArray[np.float64]): The accepted sample.

        Returns:
            Float[torch.Tensor, "accepted sample"]: The accepted sample as a torch tensor.
        """
        return torch.from_numpy(accepted_sample_np)

    def _get_min_max(
        self, accepted_sample_torch: Float[torch.Tensor, "accepted sample"]
    ) -> Tuple[
        Tuple[Float[torch.Tensor, "min x"], Float[torch.Tensor, "min y"]],
        Tuple[Float[torch.Tensor, "max x"], Float[torch.Tensor, "max y"]],
    ]:
        """
        Get the minimum and maximum values of the accepted sample.

        Args:
            accepted_sample_torch (Float[torch.Tensor, "accepted sample"]): The accepted sample.

        Returns:
            Tuple[
                Tuple[Float[torch.Tensor, "min x"], Float[torch.Tensor, "min y"]],
                Tuple[Float[torch.Tensor, "max x"], Float[torch.Tensor, "max y"]]
            ]: The minimum and maximum values.
        """
        x_min, y_min = torch.min(accepted_sample_torch, dim=0)[0]
        x_max, y_max = torch.max(accepted_sample_torch, dim=0)[0]
        return (x_min, y_min), (x_max, y_max)

    def _compute_ranges(
        self,
        x_min: Float[torch.Tensor, "min x"],
        y_min: Float[torch.Tensor, "min y"],
        x_max: Float[torch.Tensor, "max x"],
        y_max: Float[torch.Tensor, "max y"],
        nums_of_range: int,
    ) -> Tuple[Float[torch.Tensor, "x range"], Float[torch.Tensor, "y range"]]:
        """
        Compute the ranges for the heatmap plot.

        Args:
            x_min (Float[torch.Tensor, "min x"]): The minimum x value.
            y_min (Float[torch.Tensor, "min y"]): The minimum y value.
            x_max (Float[torch.Tensor, "max x"]): The maximum x value.
            y_max (Float[torch.Tensor, "max y"]): The maximum y value.
            nums_of_range (int): The number of mesh points.

        Returns:
            Tuple[Float[torch.Tensor, "x range"], Float[torch.Tensor, "y range"]]: The x and y ranges.
        """
        x_range_min, y_range_min = torch.floor(x_min), torch.floor(y_min)
        x_range_max, y_range_max = torch.floor(x_max), torch.floor(y_max)
        x_range = (
            torch.linspace(x_range_min, x_range_max, nums_of_range)
            if x_range_min != x_range_max
            else torch.linspace(x_range_min, x_range_max + 1.0, nums_of_range)
        )
        y_range = (
            torch.linspace(y_range_min, y_range_max, nums_of_range)
            if y_range_min != y_range_max
            else torch.linspace(y_range_min, y_range_max + 1.0, nums_of_range)
        )
        return x_range, y_range

    @curry
    def _heatmap_plot(
        self,
        policy: Callable[
            [Float[torch.Tensor, "sample"]], Float[torch.Tensor, "action"]
        ],
        x_range: Float[torch.Tensor, "x range"],
        y_range: Float[torch.Tensor, "y range"],
        step: int,
        axes: Axes,
        softplus_mode: bool = True,
    ) -> None:
        """
        Plot the heatmap of the policy. Helper function.

        Args:
            policy (Callable[[Float[torch.Tensor, "sample"]], Float[torch.Tensor, "action"]]): The policy function.
            x_range (Float[torch.Tensor, "x range"]): The x range.
            y_range (Float[torch.Tensor, "y range"]): The y range.
            step (int): The current step.
            axes (Axes): The axes to plot on.
            softplus_mode (bool, optional): Whether to use softplus mode. Defaults to True.

        Raises:
            ValueError: If the policy is not 2D.
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
                lambda ranges: Toolbox.imbalanced_mash_2d(*ranges),
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
                lambda ranges: Toolbox.imbalanced_mash_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )

        axes.set_title(f"Policy Heatmap Step: {step}")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        cbar = plt.colorbar(axes.images[0], ax=axes, shrink=0.8)
        cbar.set_label("Action")

    def plot(self, step: int, axes: Axes) -> None:
        """
        Helper function to plot a 2D policy heatmap.

        Args:
            step (int): Current training step.
        """
        policy = lambda x: self.learning_instance.actor(x.double())

        pipe(
            self.learning_instance.env,
            lambda env: (
                self._get_current_step(env),
                self._get_store_accepted_sample(env),
            ),
            lambda tpl: (tpl[0], self._slice_to_current_step(tpl[1], tpl[0])),
            lambda tpl: self._convert_to_torch(tpl[1]),
            lambda torch_data: self._get_min_max(torch_data),
            lambda min_max: self._compute_ranges(
                *min_max[0], *min_max[1], self.num_of_mesh
            ),
            lambda ranges: self._heatmap_plot(policy, *ranges, step, axes),
        )


class ScatterPlotter(PlotterBase):
    def __init__(
        self,
        learning_instance: LearningInterface,
    ) -> None:
        """
        Callback for plotting the sample scatter of the learning instance in 2D. Only works for 2D environments.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
        """
        super().__init__(learning_instance)

    def _scatter_plot(
        self, axes: Axes, data: npt.NDArray[np.float64], step: int, alpha: float
    ) -> None:
        axes.plot(data[:, 0], data[:, 1], "o-", alpha=alpha)

        axes.set_title(f"Scatter Plot Step: {step}")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

    def plot(self, step: int, axes: Axes, alpha: float = 1.0) -> None:
        """
        Helper function to plot a 2D sample scatter plot.

        Args:
            step (int): Current training step.
        """
        pipe(
            self.learning_instance.env,
            lambda env: (
                step,
                self._get_store_accepted_sample(env),
            ),
            lambda tpl: (tpl[0], self._slice_to_current_step(tpl[1], tpl[0])),
            lambda tpl: self._scatter_plot(axes, tpl[1], tpl[0], alpha),
        )


class TrainingVisualizer:
    def __init__(
        self,
        learning_instance: LearningInterface,
        plot_frequency: int = 10,
        num_of_mesh: int = 10,
    ) -> None:
        """
        Callback for plotting the policy of the learning instance in 2D. Only works for 2D environments.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
            ranges (Tuple[Tuple[int, int, float], Tuple[int, int, float]]): The ranges for the 2D plot.
            plot_frequency (int, optional): The frequency of plotting the policy. Defaults to 10.
        """
        self.learning_instance = learning_instance
        self.plot_frequency = plot_frequency
        self.num_of_mesh = num_of_mesh
        self.sample_dim = self._get_sample_dim()

        self.learning_instance.callback = self.plot

    def _get_sample_dim(self) -> int:
        """
        Get the sample dimension of the environment.

        Returns:
            int: The sample dimension.

        Raises:
            AttributeError: If the environment does not have `sample_dim`.
        """
        sample_dim = getattr(self.learning_instance.env.unwrapped, "get_attr", None)
        if sample_dim:
            return self.learning_instance.env.unwrapped.get_attr("sample_dim")[0]
        raise AttributeError("Environment must provide `sample_dim`.")

    def _policy_plot(self, step: int, axes: Axes):
        policy_plot = PolicyPlotter(self.learning_instance, self.num_of_mesh)
        return policy_plot.plot(step, axes)

    def _scatter_plot(self, step: int, axes: Axes):
        scatter_plot = ScatterPlotter(self.learning_instance)
        return scatter_plot.plot(step, axes)

    def plot(self) -> None:
        """
        Plot the policy of the learning instance in 2D.

        Raises:
            ValueError: Can only plot 2D policies.
        """
        if self.sample_dim != 2:
            raise ValueError("Policy plotting is supported only for 2D environments.")

        current_step = self.learning_instance.env.get_attr("current_step")[0]
        if (current_step + 1) % self.plot_frequency == 0:
            clear(wait=True)
            plt.close("all")

            _, axs = plt.subplots(1, 2, figsize=(13, 5))
            self._policy_plot(current_step + 1, axs[0])
            self._scatter_plot(current_step + 1, axs[1])

            plt.show()

    def train(self) -> None:
        """
        Train the Learning Instance.
        """
        self.learning_instance.train()
