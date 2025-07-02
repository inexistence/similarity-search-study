class Processor:
    """
    A class to handle preprocessing tasks.
    """
    def name(self):
        """
        Return the name of the processor.
        """
        return self.__class__.__name__

    def process(self, input_path) -> list[str]:
        """
        Process the input data using the specified preprocessing methods.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")