from typing import Any, Protocol

import numpy.typing as npt


class TimeSeriesModel(Protocol):
    """An adapter for time series models."""

    def __init__(self, hyperparameters: dict = None):
        """Initialize the time series model.

        Parameters
        ----------
        hyperparameters : dict, optional
            A dictionary of hyperparameters for the model, by default None.
        """
        self.hyperparameters = hyperparameters 
        ...

    def fit(self, data: npt.NDArray[Any], **kwargs) -> None:
        """Fit the model to the provided time series data.

        Parameters
        ----------
        data : npt.NDArray[Any]
            The time series data to fit the model to.
        """
        ...

    def predict(self, steps: int, past_data=None) -> npt.NDArray[Any]:
        """Predict future values based on the fitted model and the
        past data provided to it.

        `past_data` is assumed to be a contiguous sequence of points.
        The predictions are then made following the last point in `past_data`.

        Parameters
        ----------
        steps : int
            The number of future time steps to predict.
        past_data : _type_, optional
            Historical data to use for prediction, by default None

        Returns
        -------
        npt.NDArray[Any]
            The model predictions for the next `steps` time steps.
        """
        ...
