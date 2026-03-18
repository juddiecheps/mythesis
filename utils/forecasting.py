"""
Reusable forecasting utilities for Manufacturing Sector Forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def recursive_forecast(model, last_window, steps, target_idx, is_lstm=True):
    preds  = []
    window = last_window.copy()
    for _ in range(steps):
        if is_lstm:
            p = model.predict(window[np.newaxis, :, :], verbose=0)[0, 0]
        else:
            p = model.predict(window.reshape(1, -1), verbose=0)[0, 0]
        preds.append(p)
        window = np.roll(window, -1, axis=0)
        window[-1, target_idx] = p
    return np.array(preds)


def create_sequences(data, look_back, target_idx=0):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, target_idx])
    return np.array(X), np.array(y)


def generate_forecast_dates(last_date, steps, freq='MS'):
    return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]


def calculate_forecast_metrics(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2   = r2_score(actual, predicted)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def inverse_transform_predictions(predictions, scaler, target_idx, n_features):
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_idx] = predictions
    return scaler.inverse_transform(dummy)[:, target_idx]


def create_forecast_dataframe(dates, predictions, model_name="Model"):
    return pd.DataFrame({f'{model_name}_Forecast': predictions}, index=dates)


def prepare_mlp_data(X):
    if len(X.shape) == 3:
        return X.reshape(X.shape[0], -1)
    return X


def prepare_lstm_data(X):
    if len(X.shape) == 2:
        raise ValueError("Cannot automatically reshape 2D data for LSTM.")
    return X