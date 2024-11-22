from abc import ABC
from dataclasses import is_dataclass
from typing import Any, Dict, List, Protocol, Type, TypeVar, Union

import tomllib

from .config_tamplate import Config, PolicyNetworkConfig, QNetworkConfig

# Define Generic Type
T = TypeVar("T", covariant=True)
TOMLData = Dict[str, str | int | float | bool | List[int]]


class BaseConfigParser(Protocol[T]):
    def __call__(self) -> T: ...


class AbstractConfigParser(ABC):
    def __init__(self, config_file: str):
        self.config_file = config_file

    @staticmethod
    def load_toml(file_path: str) -> TOMLData:
        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{file_path}' not found.")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error decoding TOML file: {e}")

    def dict_to_field(self, field_type: Type[Any], value: Any) -> Any:
        if is_dataclass(field_type):
            return self.dict_to_dataclass(field_type, value)
        return value

    def dict_to_dataclass(self, config: Type[T], data: TOMLData) -> T:
        if not is_dataclass(config):
            raise TypeError(f"{config} is not a dataclass.")

        return config(
            **{
                field.name: self.dict_to_field(field.type, data.get(field.name))
                for field in config.__dataclass_fields__.values()
            }
        )

    def parse_toml_to_dataclass(self, config_type: Type[T]) -> T:
        config_dict = self.load_toml(self.config_file)
        return self.dict_to_dataclass(config_type, config_dict)


class ConfigParser(AbstractConfigParser, BaseConfigParser[T]):
    def __init__(self, config_file: str, config_type: Type[T]):
        super().__init__(config_file)
        self.config_type = config_type

    def __call__(self) -> T:
        return self.parse_toml_to_dataclass(self.config_type)


class ConfigParserFactory:
    PARSER_MAPPING: Dict[
        str, Type[Union[Config, PolicyNetworkConfig, QNetworkConfig]]
    ] = {
        "config": Config,
        "policy_network": PolicyNetworkConfig,
        "q_network": QNetworkConfig,
    }

    @staticmethod
    def create_parser(config_file: str, config_type: str) -> Any:
        """Create and return the parsed config instance.

        Args:
            config_file (str): Path to the configuration file.
            config_type (str): Type of the configuration.

        Returns:
            Parsed configuration instance.
        """
        if config_type not in ConfigParserFactory.PARSER_MAPPING:
            raise ValueError(f"Unknown config type: {config_type}")

        config_class = ConfigParserFactory.PARSER_MAPPING[config_type]
        parser = ConfigParser(config_file, config_class)
        return parser()  # Directly return the parsed instance


class BaseInstanceConfigParser:
    def __init__(self, config_file: str, config_type: str):
        """Initializing the parser and load the configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            Instance of config data class.
        """
        self.config_file = config_file
        self.config_type = config_type
        self.config_instance = ConfigParserFactory.create_parser(
            self.config_file, self.config_type
        )

    def get_config(self):
        """Access the parsed configuration instance.

        Returns:
            The parsed configuration instance.
        """
        return self.config_instance

    def __getattr__(self, item: str) -> List[int] | str | None:
        """Delegate attribute access to the config instance."""
        return getattr(self.config_instance, item)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the object.

        Returns:
            str: A string representing the configuration parser and its content.
        """
        return repr(self.config_instance)


class HyperparameterConfigParser(BaseInstanceConfigParser):
    def __init__(self, config_file: str):
        super().__init__(config_file, "config")


class PolicyNetworkConfigParser(BaseInstanceConfigParser):
    def __init__(self, config_file: str):
        super().__init__(config_file, "policy_network")


class QNetworkConfigParser(BaseInstanceConfigParser):
    def __init__(self, config_file: str):
        super().__init__(config_file, "q_network")
