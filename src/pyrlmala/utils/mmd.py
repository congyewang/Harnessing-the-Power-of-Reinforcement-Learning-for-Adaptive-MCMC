import warnings
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy
from jaxtyping import Float
from scipy.spatial.distance import pdist


class MedianTrick:
    """
    Class to calculate the median trick for Maximum Mean Discrepancy (MMD).
    """

    @staticmethod
    def median_trick_numpy(x: npt.NDArray[np.floating]) -> float:
        """
        Compute the median trick.

        Args:
            x (npt.NDArray[np.floating]): Input array.

        Returns:
            float: Median trick.
        """
        return (0.5 * np.median(pdist(x))).item()

    @staticmethod
    def torch_pdist(x: Float[torch.Tensor, "x"]) -> Float[torch.Tensor, "pdist"]:
        """
        PyTorch version of scipy's pdist function

        Args:
            x (Float[torch.Tensor, "x"]): Tensor of shape (n, d) where n is the number of samples and d is dimensionality

        Returns:
            Float[torch.Tensor, "pdist"]: Pairwise distances between points in x
        """
        n = x.shape[0]

        # Use cdist to compute the Euclidean distance between all pairs of points
        distances = torch.cdist(x, x, p=2)

        # Extract the upper triangular portion (excluding the diagonal) to match the pdist behavior
        indices = torch.triu_indices(n, n, offset=1)

        return distances[indices[0], indices[1]]

    @classmethod
    def median_trick_torch(cls, x: Float[torch.Tensor, "x"]) -> float:
        """
        Compute the median trick.

        Args:
            x (Float[torch.Tensor, "x"]): Input tensor.

        Returns:
            float: Median trick.
        """
        return (0.5 * torch.median(cls.torch_pdist(x))).item()


class KernelFunctions:
    """
    Class to compute kernel functions.
    """

    @staticmethod
    def gaussian_kernel_numpy(
        x: npt.NDArray[np.floating], y: npt.NDArray[np.floating], sigma: float
    ) -> npt.NDArray[np.floating]:
        return np.exp(-np.subtract.outer(x, y) ** 2 / (2 * sigma**2))

    @staticmethod
    def gaussian_kernel_torch(
        x: Float[torch.Tensor, "x"], y: Float[torch.Tensor, "y"], sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the RBF (Gaussian) kernel between x and y.

        Args:
            x (Float[torch.Tensor, "x"]): Input tensor x.
            y (Float[torch.Tensor, "y"]): Input tensor y.
            sigma (float, optional): Sigma. Defaults to 1.0.

        Returns:
            torch.Tensor: RBF kernel.
        """

        beta = 1.0 / (2.0 * sigma**2)
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-beta * dist_sq)


@runtime_checkable
class CalculateMMD(Protocol):
    @staticmethod
    def calculate(*args: Any, **kwargs: Dict[str, Any]) -> float: ...


class CalculateMMDTorch:
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two distributions.
    """

    @staticmethod
    def eval_step(engine: Engine, batch: List[Any]) -> List[Any]:
        """
        Evaluation step for the Ignite Engine.

        Args:
            engine (Engine): Ignite Engine instance.
            batch (List[Any]): Input batch containing data.

        Returns:
            List[Any]: The same batch, passed through.
        """
        return batch

    @staticmethod
    def calculate(
        x: Float[torch.Tensor, "x"],
        y: Float[torch.Tensor, "y"],
        **kwargs: Dict[str, Any],
    ) -> float:
        """
        Calculate the Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            x (Float[torch.Tensor, "x"]): Input tensor x.
            y (Float[torch.Tensor, "y"]): Input tensor y.
            **kwargs (Dict[str, Any]): Additional arguments for the MMD calculation.

        Returns:
            float: Maximum Mean Discrepancy (MMD) between the two distributions.
        """
        default_evaluator = Engine(CalculateMMDTorch.eval_step)

        if len(y) > len(x):
            warnings.warn(
                f"Length of y ({len(y)}) is greater than length of x ({len(x)}). "
                f"Using only the last {len(x)} elements of y.",
                UserWarning,
            )
            y = x[-len(y) :]

        metric = MaximumMeanDiscrepancy(**kwargs)
        metric.attach(default_evaluator, "mmd")
        state = default_evaluator.run([[x, y]])

        return state.metrics["mmd"]


class BatchedCalculateMMDTorch:
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two distributions in batches.
    """

    @staticmethod
    def calculate(
        x: Float[torch.Tensor, "x"],
        y: Float[torch.Tensor, "y"],
        sigma: float = 1.0,
        batch_size: int = 100,
    ) -> float:
        """
        Compute the Maximum Mean Discrepancy (MMD) between x and y.

        Args:
            x (Float[torch.Tensor, "x"]): Input tensor x.
            y (Float[torch.Tensor, "y"]): Input tensor y.
            sigma (float, optional): Sigma. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 100.

        Returns:
            Float[torch.Tensor, "mmd"]: MMD estimate.
        """

        m = x.size(0)
        n = y.size(0)
        mmd_estimate_xx, mmd_estimate_yy, mmd_estimate_xy = 0.0, 0.0, 0.0

        # Compute the MMD estimate in mini-batches
        for i in range(0, m, batch_size):
            x_batch = x[i : i + batch_size]
            for j in range(0, n, batch_size):
                y_batch = y[j : j + batch_size]

                xx_kernel = KernelFunctions.gaussian_kernel_torch(
                    x_batch, x_batch, sigma
                )
                yy_kernel = KernelFunctions.gaussian_kernel_torch(
                    y_batch, y_batch, sigma
                )
                xy_kernel = KernelFunctions.gaussian_kernel_torch(
                    x_batch, y_batch, sigma
                )

                # Compute the MMD estimate for this mini-batch
                mmd_estimate_xx += xx_kernel.sum()
                mmd_estimate_yy += yy_kernel.sum()
                mmd_estimate_xy += xy_kernel.sum()

        # Normalize the MMD estimate
        mmd_estimate = (
            mmd_estimate_xx / m**2
            + mmd_estimate_yy / n**2
            - 2 * mmd_estimate_xy / (m * n)
        )

        return mmd_estimate.item()


class CalculateMMDNumpy:
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two distributions using NumPy.
    """

    @staticmethod
    def calculate(
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        sigma: float = 1.0,
    ) -> float:
        """
        Calculate the Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            x (npt.NDArray[np.floating]): Input array x.
            y (npt.NDArray[np.floating]): Input array y.
            sigma (float, optional): Standard deviation. Defaults to 1.0.

        Returns:
            float: Maximum Mean Discrepancy (MMD) between the two distributions.
        """

        Kxx = KernelFunctions.gaussian_kernel_numpy(x, x, sigma)
        Kyy = KernelFunctions.gaussian_kernel_numpy(y, y, sigma)
        Kxy = KernelFunctions.gaussian_kernel_numpy(x, y, sigma)

        m = x.shape[0]
        n = y.shape[0]

        mmd = Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)

        return mmd.item()
