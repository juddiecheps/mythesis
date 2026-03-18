#!/usr/bin/env python3
"""
Model training script for Manufacturing Sector Forecasting.
Trains ARIMA, MLP, and LSTM models on real KNBS data.
Target: Cement_Production_MT (primary manufacturing proxy)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = 'data/clean_data.csv'
LOOK_BACK = 12
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET_COL = 'Cement_Production_MT'
FEATURE_COLS = ['Cement_Production_MT', 'Sugar_Production_MT', 'Milk_Intake_MnLitres',
                'Soft_Drinks_000Litres', 'Assembled_Vehicles', 'Galvanized_Sheets_MT']

TRAIN_END   = '2023-12-01'
VAL_END     = '2024-06-01'
# Test: 2024-07 onwards

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df[FEATURE_COLS]
    df = df.asfreq('MS')
    return df

def create_sequences(data, look_back, target_idx=0):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, target_idx])
    return np.array(X), np.array(y)

def split_data(df):
    train = df[:TRAIN_END]
    val   = df[pd.Timestamp(TRAIN_END) + pd.DateOffset(months=1):VAL_END]
    test  = df[pd.Timestamp(VAL_END) + pd.DateOffset(months=1):]
    print(f"Train: {train.index[0].date()} – {train.index[-1].date()} ({len(train)} obs)")
    print(f"Val:   {val.index[0].date()} – {val.index[-1].date()} ({len(val)} obs)")
    print(f"Test:  {test.index[0].date()} – {test.index[-1].date()} ({len(test)} obs)")
    return train, val, test

def train_arima(df):
    print("\n=== ARIMA ===")
    target = df[TARGET_COL]
    train_target = target[:TRAIN_END]
    test_target  = target[pd.Timestamp(VAL_END) + pd.DateOffset(months=1):]

    model = ARIMA(train_target, order=(2, 1, 2))
    fitted = model.fit()
    print(fitted.summary())

    preds = fitted.forecast(steps=len(test_target))
    rmse = np.sqrt(mean_squared_error(test_target, preds))
    mae  = mean_absolute_error(test_target, preds)
    print(f"Test RMSE: {rmse:.1f}, MAE: {mae:.1f}")

    joblib.dump(fitted, 'models/arima_cement.pkl')
    print("✓ Saved models/arima_cement.pkl")
    return fitted, rmse, mae

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n=== MLP ===")
    Xt = X_train.reshape(X_train.shape[0], -1)
    Xv = X_val.reshape(X_val.shape[0], -1)
    Xs = X_test.reshape(X_test.shape[0], -1)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(Xt.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(Xt, y_train, validation_data=(Xv, y_val),
              epochs=200, batch_size=16, callbacks=[es], verbose=0)

    preds = model.predict(Xs, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    model.save('models/mlp_cement_model.keras')
    print("✓ Saved models/mlp_cement_model.keras")
    return model, rmse, mae

def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n=== LSTM ===")
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=200, batch_size=16, callbacks=[es], verbose=0)

    preds = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    model.save('models/lstm_cement_model.keras')
    print("✓ Saved models/lstm_cement_model.keras")
    return model, rmse, mae

def main():
    print("=" * 60)
    print("Manufacturing Sector Forecasting — Model Training")
    print("=" * 60)

    df = load_data()
    print(f"\nData loaded: {df.shape}, {df.index[0].date()} – {df.index[-1].date()}")

    train, val, test = split_data(df)

    # Fit scaler on training only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    joblib.dump(scaler, 'models/minmax_scaler.pkl')
    print("✓ Scaler fitted and saved")

    target_idx = FEATURE_COLS.index(TARGET_COL)

    # Build sequences using train + val for training context
    all_train_val_scaled = np.vstack([train_scaled, val_scaled])

    X_train, y_train = create_sequences(train_scaled, LOOK_BACK, target_idx)

    # For val sequences, we need preceding look_back from train
    combined_tv = np.vstack([train_scaled, val_scaled])
    X_val_full, y_val_full = create_sequences(combined_tv, LOOK_BACK, target_idx)
    X_val = X_val_full[len(X_train):]
    y_val = y_val_full[len(X_train):]

    # For test sequences
    combined_all = np.vstack([train_scaled, val_scaled, test_scaled])
    X_all, y_all = create_sequences(combined_all, LOOK_BACK, target_idx)
    n_train_val = len(combined_tv) - LOOK_BACK
    X_test = X_all[n_train_val:]
    y_test = y_all[n_train_val:]

    print(f"\nSequences — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train ARIMA
    arima_model, arima_rmse, arima_mae = train_arima(df)

    # Train MLP
    mlp_model, mlp_rmse, mlp_mae = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)

    # Train LSTM
    lstm_model, lstm_rmse, lstm_mae = train_lstm(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save metadata
    metadata = {
        'target': TARGET_COL,
        'columns': FEATURE_COLS,
        'look_back': LOOK_BACK,
        'train_end': TRAIN_END,
        'val_end': VAL_END,
        'model_performance': {
            'arima': {'rmse': arima_rmse, 'mae': arima_mae},
            'mlp':   {'rmse': float(mlp_rmse),  'mae': float(mlp_mae)},
            'lstm':  {'rmse': float(lstm_rmse), 'mae': float(lstm_mae)},
        }
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    print("\n✓ Metadata saved")

    print("\n" + "=" * 60)
    print("Training Summary (Test RMSE / MAE)")
    print("=" * 60)
    print(f"ARIMA : RMSE={arima_rmse:.1f}, MAE={arima_mae:.1f}")
    print(f"MLP   : RMSE={mlp_rmse:.4f}, MAE={mlp_mae:.4f}  (scaled)")
    print(f"LSTM  : RMSE={lstm_rmse:.4f}, MAE={lstm_mae:.4f}  (scaled)")
    print("=" * 60)

if __name__ == "__main__":
    main()