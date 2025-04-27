class DummyWriter:
    """
    Dummy writer for logging. Does nothing.
    """

    def add_scalars(self, *args, **kwargs):
        """
        Compatible with TensorBoard, but does nothing.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        pass

    def add_scalar(self, *args, **kwargs):
        """
        Compatible with TensorBoard, but does nothing.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        pass
