from enum import Enum
from typing import Any, Callable, Dict, List


class TrainEvents(Enum):
    """
    Enum class for training events.
    """

    BEFORE_TRAIN = "before_train"
    WITHIN_TRAIN = "within_train"
    AFTER_TRAIN = "after_train"


class EventManager:
    """
    A simple event manager that allows registering and triggering events.

    Attributes:
        events (Dict[str, List[Callable[..., None]]): Dictionary of events and their callbacks.
    """

    def __init__(self) -> None:
        """
        Initialize the EventManager.
        """
        self.events: Dict[str, List[Callable[..., None]]] = {}

    def register(self, event_name: TrainEvents, callback: Callable[..., None]) -> None:
        """
        Rigister a callback function for the specified event.

        Args:
            event_name (TrainEvents): Name of the event.
            callback (Callable[..., None]): Callback function to be registered.
        """
        event_name_str = event_name.value
        self.events.setdefault(event_name_str, []).append(callback)

    def unregister(
        self, event_name: TrainEvents, callback: Callable[..., None]
    ) -> bool:
        """
        Unregister a callback function for the specified event.

        Args:
            event_name (TrainEvents): Name of the event.
            callback (Callable[..., None]): Callback function to be unregistered.
        """
        event_name_str = event_name.value
        if event_name_str in self.events and callback in self.events[event_name_str]:
            self.events[event_name_str].remove(callback)
            if not self.events[event_name_str]:
                # If no callbacks are left, remove the event
                del self.events[event_name_str]
            return True
        return False

    def trigger(self, event_name: TrainEvents, *args: Any, **kwargs: Any) -> None:
        """
        Trigger the specified event with the given arguments.

        Args:
            event_name (TrainEvents): Name of the event.
            *args (Any): Positional arguments to be passed to the callback.
            **kwargs (Any): Keyword arguments to be passed to the callback.
        """
        event_name_str = event_name.value
        for callback in self.events.get(event_name_str, []):
            if callback:
                callback(*args, **kwargs)
