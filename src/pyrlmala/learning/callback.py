import os
from typing import Any, Callable, Tuple, TypeVar

import torch

from ..utils import Toolbox
from .learning import LearningInterface

T = TypeVar("T", bound=Tuple[int, int, float])


def detect_environment() -> str:
    """
    Detect the current execution environment.

    Returns:
        str: The detected environment type: 'jupyter', 'ipython', or 'terminal'.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter Notebook or JupyterLab
            return "jupyter"
        elif shell == "TerminalInteractiveShell":  # IPython terminal
            return "ipython"
        else:
            return "terminal"
    except NameError:
        return "terminal"


def get_clear_function() -> Callable[[bool], Any]:
    """
    Get the appropriate clear function based on the environment.

    Returns:
        callable: The function to clear output in the terminal or notebook.
    """
    if detect_environment() == "jupyter":
        from IPython.display import clear_output as clear

        return clear
    else:
        return lambda wait=False: os.system("cls" if os.name == "nt" else "clear")


clear = get_clear_function()


class PolicyPlotter:
    def __init__(
        self,
        learning_instance: LearningInterface,
        ranges: Tuple[T, ...],
        plot_frequency: int = 10,
    ) -> None:
        """
        Callback for plotting the policy of the learning instance in 2D. Only works for 2D environments.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
            ranges (Tuple[Tuple[int, int, float], Tuple[int, int, float]]): The ranges for the 2D plot.
            plot_frequency (int, optional): The frequency of plotting the policy. Defaults to 10.
        """
        if len(ranges) != 2 or not all(len(r) == 3 for r in ranges):
            raise ValueError("`ranges` must contain exactly two 3-tuple ranges.")

        self.learning_instance = learning_instance
        self.plot_frequency = plot_frequency
        self.ranges = ranges
        self.sample_dim = self._get_sample_dim()

        self.learning_instance.callback = self.plot_policy

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

    def plot_policy(self) -> None:
        """
        Plot the policy of the learning instance in 2D.

        Raises:
            ValueError: Can only plot 2D policies.
        """
        if self.sample_dim != 2:
            raise ValueError("Policy plotting is supported only for 2D environments.")

        current_step = self.learning_instance.env.get_attr("current_step")[0]
        if (current_step + 1) % self.plot_frequency == 0:
            self._plot_policy_2D(current_step + 1)

    def _plot_policy_2D(self, step: int) -> None:
        """
        Helper function to plot a 2D policy heatmap.

        Args:
            step (int): Current training step.
        """
        policy = lambda x: self.learning_instance.actor(x.double())
        clear(wait=True)
        Toolbox.policy_plot_2D_heatmap(
            policy,
            torch.arange(*self.ranges[0]),
            torch.arange(*self.ranges[1]),
            title_addition=f"Step: {step}",
        )

    def train(self) -> None:
        """
        Train the Learning Instance.
        """
        self.learning_instance.train()
