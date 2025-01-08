from abc import ABC, abstractmethod


class PluginBase(ABC):
    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError("Method not implemented")
