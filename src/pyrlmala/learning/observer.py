import os
from typing import Any, Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ConfigChangeHandler(FileSystemEventHandler):
    """
    Event handler for configuration file changes.

    Attributes:
        callback (Callable): Function to call when the file changes.
    """

    def __init__(self, callback: Callable[..., Any]):
        """
        Event handler for configuration file changes.

        Args:
            callback (Callable[..., Any]): Function to call when the file changes.
        """
        self.callback = callback

    def on_modified(self, event: Any):
        """
        Handle the file modification event.

        Args:
            event (Any): The event object.
        """
        if event.is_directory:
            return
        self.callback()


class ConfigObserver:
    """
    Observer class that uses watchdog to monitor configuration files.

    Attributes:
        file_path (str): The path to the configuration file.
        callback (Callable): Function to call when the file changes.
    """

    def __init__(self, file_path: str, callback: Callable[..., Any]) -> None:
        """
        Observer class that uses watchdog to monitor configuration files.

        Args:
            file_path (str): The path to the configuration file.
            callback (Callable[..., Any]): Function to call when the file changes.
        """
        self.file_path = file_path
        self.callback = callback
        self.event_handler = ConfigChangeHandler(self.callback)
        self.observer = Observer()

    def start(self) -> None:
        """
        Start the watchdog observer to monitor the configuration file.
        """
        directory = os.path.dirname(self.file_path)
        self.observer.schedule(self.event_handler, directory, recursive=False)
        self.observer.start()

    def stop(self) -> None:
        """
        Stop the watchdog observer.
        """
        self.observer.stop()
        self.observer.join()
