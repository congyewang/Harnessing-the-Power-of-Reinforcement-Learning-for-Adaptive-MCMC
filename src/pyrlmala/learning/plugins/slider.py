import threading
import warnings
from abc import abstractmethod
from typing import Any, Dict

import ipywidgets as widgets
import tomli_w
import tomllib
from filelock import FileLock
from IPython.display import display

from ...utils import Toolbox
from .base import PluginBase


class SliderBase(PluginBase):
    """
    A base class to create a slider to change the runtime configuration parameters.

    Attributes:
        runtime_config_path (str): Path to the runtime configuration file.
    """

    def __init__(self, runtime_config_path: str) -> None:
        """
        Create a slider to change the runtime configuration parameters.

        Args:
            runtime_config_path (str): Path to the runtime configuration file.
            learning_instance (LearningInterface): The learning instance.
        """
        super().__init__(learning_instance=None)
        self.runtime_config_path = runtime_config_path
        self.lock = FileLock(f"{runtime_config_path}.lock")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the runtime configuration file.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the configuration file cannot be decoded.

        Returns:
            Dict[str, Any]: The configuration parameters.
        """
        try:
            with open(self.runtime_config_path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file '{self.runtime_config_path}' not found."
            )
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error decoding TOML file: {e}")

    def _update_config(self, new_parameters: Dict[str, Any]) -> None:
        """
        Update the runtime configuration file with new parameters.

        Args:
            new_parameters (Dict[str, Any]): The new parameters to update.
        """
        config = self._load_config()
        config.update(new_parameters)

        with self.lock:
            with open(self.runtime_config_path, "wb") as f:
                tomli_w.dump(config, f)

    def _slider_update_thread(self, change: Dict[str, Any]) -> None:
        threading.Thread(target=self._on_slider_change, args=(change,)).start()

    @abstractmethod
    def _on_slider_change(self, change: Dict[str, Any]) -> None:
        """
        Update the runtime configuration parameters when the slider changes.

        Args:
            change (Dict[str, Any]): The change event.
        """
        raise NotImplementedError("Method '_on_slider_change' must be implemented.")

    @abstractmethod
    def _create_slider(self) -> None:
        """
        Create a slider to change the runtime configuration parameters.
        """
        raise NotImplementedError("Method '_create_slider' must be implemented.")

    def execute(self) -> None:
        """
        Execute the plugin.
        """
        if Toolbox.detect_environment() == "jupyter":
            self._create_slider()
        else:
            warnings.warn(
                "Slider plugin only works in Jupyter Notebook environment. Skipping..."
            )


class ActorLearningRateSlider(SliderBase):
    """
    A plugin to create a slider to change the actor learning rate.

    Attributes:
        runtime_config_path (str): Path to the runtime configuration file.
    """

    def _on_slider_change(self, change: Dict[str, Any]) -> None:
        """
        Update the actor learning rate when the slider changes.

        Args:
            change (Dict[str, Any]): The change event.
        """
        new_lr = change["new"]
        self._update_config({"actor_learning_rate": new_lr})

    def _create_slider(self) -> None:
        """
        Create a slider to change the actor learning rate.
        """
        config = self._load_config()
        current_lr = config.get("actor_learning_rate", 1e-5)

        slider = widgets.FloatLogSlider(
            value=current_lr,
            base=10,
            min=-10,
            max=-1,
            step=0.1,
            description="Actor LR:",
            disabled=False,
            continuous_update=True,
            readout_format=".2e",
        )

        slider.observe(self._slider_update_thread, names="value")
        display(slider)
