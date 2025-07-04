import pandas as pd
import pmdarima as pm

from finml.interface import TimeSeriesModel


class PmdarimaAdapter(TimeSeriesModel):
    def __init__(self, arima_model: pm.ARIMA | pm.AutoARIMA):
        self.model = arima_model

    def fit(self, data, **kwargs) -> None:
        self.model.fit(data, kwargs.pop('X'), **kwargs)

    def predict(self, steps, past_data=None):
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