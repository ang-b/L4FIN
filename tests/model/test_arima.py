import pmdarima as pm

def test_statsmodels_arima_has_same_parameters():
    arima_model = pm.ARIMA(order=(2, 2, 2))
    