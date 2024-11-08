from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import Any, Dict, List, Type, Union

import tomllib

from ._config_tamplate import Config, PolicyNetworkConfig, QNetworkConfig


class BaseConfigParser(ABC):
    def __init__(self, config_file: str):
        """Initializing the parser and load the configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.config_file = config_file

    @staticmethod
    def load_toml(file_path: str) -> Dict[str, Union[str, int, float, bool, List[int]]]:
        """Loading a TOML file as a dictionary.

        Args:
            file_path (str): File path to the TOML file.

        Returns:
            Dictionary of the TOML file.
        """
        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{file_path}' not found.")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error decoding TOML file: {e}")

    def dict_to_dataclass(
        self,
        config: Type[Any],
        data: Dict[str, Union[str, int, float, bool, List[int]]],
    ) -> Any:
        """Converts a dictionary recursively to the specified data class.

        Args:
            config (Type[Any]): Data class on Config.
            data (Dict[str, Union[str, int, float, bool]]): Configuration dictionary.

        Returns:
            Any: Data class instance.
        """
        if not is_dataclass(config):
            raise TypeError(f"{config} is not a dataclass.")

        return config(
            **{
                field.name: (
                    self.dict_to_dataclass(field.type, data[field.name])
                    if is_dataclass(field.type) and field.name in data
                    else data.get(field.name)
                )
                for field in config.__dataclass_fields__.values()
            }
        )

    @abstractmethod
    def parse_toml_to_dataclass(
        self,
    ) -> Union[Config, PolicyNetworkConfig, QNetworkConfig]:
        """Parsing and converting from TOML files to Config data classes.

        Returns:
            Instance of config data class.
        """
        raise NotImplementedError(
            "Method 'parse_toml_to_dataclass' must be implemented."
        )


class HyperparameterConfigParser(BaseConfigParser):
    def __init__(self, config_file: str):
        """Initializing the parser and load the configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def parse_toml_to_dataclass(self) -> Config:
        """Parsing and converting from TOML files to Config data classes.

        Returns:
            Instance of config data class.
        """
        config_dict = self.load_toml(self.config_file)

        return self.dict_to_dataclass(Config, config_dict)


class PolicyNetworkConfigParser(BaseConfigParser):
    def __init__(self, config_file: str):
        """Initializing the parser and load the configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def parse_toml_to_dataclass(self) -> PolicyNetworkConfig:
        """Parsing and converting from TOML files to Config data classes.

        Returns:
            Instance of config data class.
        """
        config_dict = self.load_toml(self.config_file)

        return self.dict_to_dataclass(PolicyNetworkConfig, config_dict)


class QNetworkConfigParser(BaseConfigParser):
    def __init__(self, config_file: str):
        """Initializing the parser and load the configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file)

    def parse_toml_to_dataclass(self) -> QNetworkConfig:
        """Parsing and converting from TOML files to Config data classes.

        Returns:
            Instance of config data class.
        """
        config_dict = self.load_toml(self.config_file)

        return self.dict_to_dataclass(QNetworkConfig, config_dict)
