from abc import ABC
from dataclasses import is_dataclass
from typing import Any, Dict, List, Protocol, Type, TypeVar, Union

import tomllib

from .config_tamplate import Config, PolicyNetworkConfig, QNetworkConfig

# Define Generic Type
T = TypeVar("T", covariant=True)
TOMLData = Dict[str, str | int | float | bool | List[int]]


class BaseConfigParser(Protocol[T]):
    """
    Base class for the configuration parser.

    Attributes:
        config_file (str): Path to the configuration file.
        config_type (Type[T]): Type of the configuration data class.
    """

    def __call__(self) -> T:
        """Parse the configuration file and return the data class instance.

        Returns:
            T: The parsed configuration data class instance.
        """
        ...


class AbstractConfigParser(ABC):
    """
    Abstract class for the configuration parser.

    Attributes:
        config_file (str): Path to the configuration file.
    """

    def __init__(self, config_file: str):
        """
        Initialize the configuration parser.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.config_file = config_file

    @staticmethod
    def load_toml(file_path: str) -> TOMLData:
        """
        Load the TOML file and return the data as a dictionary.

        Args:
            file_path (str): Path to the TOML file.

        Raises:
            FileNotFoundError: Raised when the file is not found.
            ValueError: Raised when the file cannot be decoded.

        Returns:
            TOMLData: The data loaded from the TOML file.
        """
        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{file_path}' not found.")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error decoding TOML file: {e}")

    def dict_to_field(self, field_type: Type[Any], value: Any) -> Any:
        """
        Convert the dictionary value to the field type.

        Args:
            field_type (Type[Any]): Class type of the field.
            value (Any): Value to be converted.

        Returns:
            Any: The converted value.
        """
        if is_dataclass(field_type):
            return self.dict_to_dataclass(field_type, value)
        return value

    def dict_to_dataclass(self, config: Type[T], data: TOMLData) -> T:
        """
        Convert the dictionary to the data class instance.

        Args:
            config (Type[T]): Data class type.
            data (TOMLData): TOML data to be converted.

        Raises:
            TypeError: Raised when the config is not a dataclass.

        Returns:
            T: The data class instance.
        """
        if not is_dataclass(config):
            raise TypeError(f"{config} is not a dataclass.")

        return config(
            **{
                field.name: self.dict_to_field(field.type, data.get(field.name))
                for field in config.__dataclass_fields__.values()
            }
        )

    def parse_toml_to_dataclass(self, config_type: Type[T]) -> T:
        """
        Parse the TOML file and return the data class instance.

        Args:
            config_type (Type[T]): Data class type.

        Returns:
            T: The parsed data class instance.
        """
        config_dict = self.load_toml(self.config_file)
        return self.dict_to_dataclass(config_type, config_dict)


class ConfigParser(AbstractConfigParser, BaseConfigParser[T]):
    """
    Configuration parser class.

    Attributes:
        config_file (str): Path to the configuration file.
        config_type (Type[T]): Type of the configuration data class.
    """

    def __init__(self, config_file: str, config_type: Type[T]):
        """
        Initialize the configuration parser.

        Args:
            config_file (str): Path to the configuration file.
            config_type (Type[T]): Type of the configuration data class.
        """
        super().__init__(config_file)
        self.config_type = config_type

    def __call__(self) -> T:
        """
        Parse the configuration file and return the data class instance.

        Returns:
            T: The parsed configuration data class instance.
        """
        return self.parse_toml_to_dataclass(self.config_type)


class ConfigParserFactory:
    """
    Factory class for creating configuration parsers.

    Attributes:
        PARSER_MAPPING (Dict[str, Type[Union[Config, PolicyNetworkConfig, QNetworkConfig]]):
            Mapping of configuration type to the corresponding data class type.
    """

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
    """
    Base class for the configuration parser.

    Attributes:
        config_file (str): Path to the configuration file.
        config_type (str): Type of the configuration.
        config_instance (Any): Parsed configuration instance.
    """
    def __init__(self, config_file: str, config_type: str):
        """
        Initializing the parser and load the configuration file.

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
        """
        Access the parsed configuration instance.

        Returns:
            The parsed configuration instance.
        """
        return self.config_instance

    def __getattr__(self, item: str) -> List[int] | str | None:
        """
        Delegate attribute access to the config instance.

        Args:
            item (str): Attribute name.

        Returns:
            Any: The value of the attribute.
        """
        return getattr(self.config_instance, item)

    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the object.

        Returns:
            str: A string representing the configuration parser and its content.
        """
        return repr(self.config_instance)


class HyperparameterConfigParser(BaseInstanceConfigParser):
    """
    Configuration parser for hyperparameters.

    Attributes:
        config_file (str): Path to the configuration file.
        config_type (str): Type of the configuration.
        config_instance (Any): Parsed configuration instance.
    """
    def __init__(self, config_file: str):
        """
        Initialize the configuration parser.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file, "config")


class PolicyNetworkConfigParser(BaseInstanceConfigParser):
    """
    Configuration parser for the policy.

    Attributes:
        config_file (str): Path to the configuration file.
        config_type (str): Type of the configuration.
        config_instance (Any): Parsed configuration instance.
    """
    def __init__(self, config_file: str):
        """
        Initialize the configuration parser.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file, "policy_network")


class QNetworkConfigParser(BaseInstanceConfigParser):
    """
    Configuration parser for the Q-network.

    Attributes:
        config_file (str): Path to the configuration file.
        config_type (str): Type of the configuration.
        config_instance (Any): Parsed configuration instance.
    """
    def __init__(self, config_file: str):
        """
        Initialize the configuration parser.

        Args:
            config_file (str): Path to the configuration file.
        """
        super().__init__(config_file, "q_network")
