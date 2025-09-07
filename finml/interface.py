from typing import Any, Protocol

import numpy.typing as npt


class TimeSeriesModel(Protocol):
    """An adapter for time series models."""

    def fit(self, data: npt.NDArray[Any], **kwargs) -> None:
        """Fit the model to the provided time series data.

        Parameters
        ----------
        data : npt.NDArray[Any]
            The time series data to fit the model to.
        """
        ...

    def predict(self, steps: int, initial_condition: list | None = None) -> npt.NDArray[Any]:
        """Predict future values based on the fitted model and the past data provided to it.

        `initial_condition` is the minimum set of samples needed to calculate a prediction step.

        Parameters
        ----------
        steps : int
            The number of future time steps to predict.
        initial_condition : Any, optional
            Initial samples to calculate a model step. When `None` the model will predict data starting from the last 
            samples of the training data. 

        Returns
        -------
        npt.NDArray[Any]
            The model predictions for the next `steps` time steps.
        """
        ...
