from abc import ABC, abstractmethod


class Expert(ABC):
    """
    Abstract expert class that stores all seen data and provides
    standard interface for predictions and adding new data.
    """

    def __init__(self, batch, **kwargs):
        """
        Initialize the expert with initial data.

        Args:
            batch: batch data object
            kwargs: any additional hyperparameters
        """
        self.adaptive = False
        self.truth_mode = "y"

    @abstractmethod
    def initial_predict(self, x, **kwargs):
        """
        Initial prediction method for the expert.

        Args:
            x: inputs to predict [N, D]

        Returns:
            y_pred: predicted outputs [N, 1]
        """
        pass
    
    @abstractmethod
    def predict(self, x, time_token = None, **kwargs):
        """
        Predict outputs for new inputs.

        Args:
            x: new inputs [N_new, D]

        Returns:
            y_pred: predicted outputs [N_new, 1]
        """
        pass

    @abstractmethod
    def add_data(self, x_new, y_new, **kwargs):
        """
        Add new data to the expert's memory.

        Args:
            x_new: new inputs [N_new, D]
            y_new: new outputs [N_new, 1]
        """

        # fit...
        pass
