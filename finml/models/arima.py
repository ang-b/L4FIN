import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.tsa.arima.model as smarima

from finml.interface import TimeSeriesModel


class PmdarimaAdapter(TimeSeriesModel):
    def __init__(self, arima_model: pm.ARIMA | pm.AutoARIMA):
        self.model = arima_model

    def fit(self, data, **kwargs) -> None:
        self.model.fit(data, kwargs.pop('X'), **kwargs)

    def predict(self, steps, initial_condition=None):
        prediction = self.model.predict(
            n_periods=steps,
            return_conf_int=False
            )
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if isinstance(prediction, pd.Series):
            return prediction.to_numpy()
        else:
            return prediction
    

    def to_statsmodels_arima_adapter(self) -> "StatsmodelsArimaAdapter":
        # TODO
        # - add conversion from pm.ARIMA to sm.ARIMA
        # - how do we store the results object?

        # for instance the following is not legal:
        # return StatsmodelsArimaAdapter(self.model)
        raise NotImplementedError("We don't know yet how to convert this")
        

class StatsmodelsArimaAdapter(TimeSeriesModel):
    def __init__(self, arima_model: smarima.ARIMA):
        self.model = arima_model
        self.fitted_results = None

    def fit(self, data, **kwargs) -> None:
        pass

    def predict(self, steps, initial_condition=None):
        return np.zeros(1)
    