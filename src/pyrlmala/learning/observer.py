import json
import os
from typing import Any, Callable, Dict

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .preparation import PreparationInterface


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, file_path: str, callback: Callable[[Dict[Any, Any]], None]):
        self.file_path = file_path
        self.callback = callback

    def on_modified(self, event):
        if event.src_path == self.file_path:
            with open(self.file_path, "r") as f:
                updated_config = json.load(f)
            self.callback(updated_config)


class ConfigObserver:
    def __init__(
        self, config_path: str, update_callback: Callable[[Dict[Any, Any]], None]
    ):
        self.config_path = config_path
        self.update_callback = update_callback
        self.event_handler = ConfigChangeHandler(self.config_path, self.update_callback)
        self.observer = Observer()

    def start(self):
        config_dir = os.path.dirname(self.config_path)
        self.observer.schedule(self.event_handler, config_dir, recursive=False)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()


class ReflectionMixin:
    """
    Mixin class to enable reflection-based updates of runtime attributes.
    """

    def update_attributes(self, updated_config: Dict[Any, Any]):
        for key, value in updated_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: Attribute {key} does not exist.")


class PreparationInterfaceWithObserver(PreparationInterface, ReflectionMixin):
    def __init__(self, *args, hyperparameter_config_path: str = "", **kwargs):
        super().__init__(
            *args, hyperparameter_config_path=hyperparameter_config_path, **kwargs
        )

        # Initialize config observer
        self.config_observer = ConfigObserver(
            config_path=hyperparameter_config_path,
            update_callback=self.on_config_update,
        )
        self.config_observer.start()

    def on_config_update(self, updated_config: Dict[Any, Any]):
        print("Configuration file modified. Applying updates...")
        self.update_attributes(updated_config)

    def __del__(self):
        self.config_observer.stop()
