import abc

class Model(abc.ABC):
    """
    Abstract base class for all sequence models in OpenSeq.
    """

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the model.
        Specific parameters will vary by model.
        """
        pass

    @abc.abstractmethod
    def fit(self, X, **kwargs):
        """
        Fit/train the model on the provided data X.

        Args:
            X: Input data (e.g., MSA, sequences). Format depends on the model.
            **kwargs: Model-specific training parameters (e.g., learning rate, epochs, batch size).
        """
        pass

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions using the trained model.
        The nature of predictions depends on the model (e.g., contact map, probabilities).

        Args:
            X: Input data for which to make predictions.
            **kwargs: Model-specific prediction parameters.

        Returns:
            Predictions.
        """
        pass

    @abc.abstractmethod
    def sample(self, num_samples: int, **kwargs):
        """
        Generate new sequences from the trained model.

        Args:
            num_samples (int): The number of sequences to generate.
            **kwargs: Model-specific sampling parameters.

        Returns:
            A list or array of generated sequences.
        """
        pass

    @abc.abstractmethod
    def get_parameters(self):
        """
        Get the learned parameters of the model.

        Returns:
            A dictionary or custom object containing model parameters.
        """
        pass

    @abc.abstractmethod
    def load_parameters(self, params):
        """
        Load pre-trained parameters into the model.

        Args:
            params: A dictionary or custom object containing model parameters.
        """
        pass

    # Optional: Add methods for saving/loading entire model state if needed
    # def save(self, filepath: str):
    #     pass
    #
    # @classmethod
    # def load(cls, filepath: str):
    #     pass
