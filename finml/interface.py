from typing import Any, Protocol

import numpy.typing as npt


class TimeSeriesModel(Protocol):
    """
    An adapter for time series models.
    """

    def __init__(self, hyperparameters: dict = None) -> npt.NDArray[Any]:
        """
        Initialize the time series model.
        
        Parameters:
        - hyperparameters: A dictionary of hyperparameters for the model.
        """
        self.hyperparameters = hyperparameters 
        ...

    def fit(self, data, **kwargs) -> None:
        """
        Fit the model to the provided time series data.

        Parameters:
        - data: The time series data to fit the model to.
        - kwargs: Additional parameters for model fitting.
        """
        ...

    def predict(self, steps, past_data=None) -> npt.NDArray[Any]:
        """
        Predict future values based on the fitted model and the
        past data provided to it.

        `past_data` is assumed to be a contiguous sequence of points.
        The predictions are then made following the last point in `past_data`.

        If `past_data` is not provided...

        Parameters:
        - steps: The number of future time steps to predict.
        - past_data: Optional historical data to use for prediction.

        Returns:
        - Predicted values for the specified number of steps.
        """
        ...
