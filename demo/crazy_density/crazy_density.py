import numpy as np
from numpy import typing as npt


class CrazyDensity:
    @staticmethod
    def multiquadratic(
        x: npt.NDArray[np.floating], b: float, c: float
    ) -> npt.NDArray[np.floating]:
        """
        Applies the multiquadratic function element-wise to a NumPy array.

        Args:
            x (npt.NDArray[np.floating]): A NumPy array.
            b (float): The scaling factor.
            c (float): The constant offset.

        Returns:
            npt.NDArray[np.floating]: A NumPy array with the multiquadratic function applied element-wise.
        """
        return (c**2 + (x**2)) ** b

    @staticmethod
    def log_den_imbalanced(
        x: npt.NDArray[np.floating],
        power: float = 1,
        scale: float = 10.0,
        bias: float = 0.01,
    ) -> float:
        """
        2d density with the first ldimension divided by scale.

        Args:
            x (npt.NDArray[np.floating]): A NumPy array.
            power (float): The power.
            scale (float): The scaling factor.
            bias (float): The constant offset.

        Returns:
            float: The log density.
        """
        scale_matrix = np.diag([1.0 / scale, 1.0])
        return -CrazyDensity.multiquadratic(scale_matrix @ x, power, bias).sum()
        # return -multiquadratic(((scale_matrix @x)**2).sum(), power, bias)

    @staticmethod
    def log_den_imbalanced_rotate(
        x: npt.NDArray[np.floating],
        angle: float = np.pi / 4,
        power: float = 1.0,
        scale: float = 10,
    ) -> float:
        """
        rotate version of log_den_imbalanced.

        Args:
            x (npt.NDArray[np.floating]): A NumPy array.
            angle (float): The angle.
            power (float): The power.
            scale (float): The scaling factor.

        Returns:
            float: The log density.
        """
        rotate_matrix = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        x_ = rotate_matrix @ x
        return CrazyDensity.log_den_imbalanced(x_, power=power, scale=scale)

    @staticmethod
    def log_den_n_mixture(x: npt.NDArray[np.floating], n: int = 4):
        """
        mixture of n imbalanced densities.

        Args:
            x (npt.NDArray[np.floating]): A NumPy array.
            n (int): The number of mixtures.

        Returns:
            float: The log density.
        """
        log_mixture_eval = 0.0
        for i in range(0, n):
            power = 0.4 ** (i + 1)
            angle = i * np.pi / 4
            scale = 1.5 ** (n - i)
            log_mixture_eval += np.exp(
                CrazyDensity.log_den_imbalanced_rotate(
                    x, angle=angle, power=power, scale=scale
                )
            )
        log_mixture_eval = np.log(log_mixture_eval)
        return log_mixture_eval
