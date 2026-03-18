# CLAUDE.md — Manufacturing Sector Forecasting App

## YOUR TASK — READ THIS FIRST

Do not ask clarifying questions. Execute the following steps in order using the existing files in this project:

1. Read and merge `data/Manufacturing.xlsx` (2020–2024) and `data/Data.xlsx` (2025) into one clean dataset
2. Clean and preprocess the merged data
3. Retrain all three models (ARIMA, MLP, LSTM) using the real merged data
4. Save all retrained models and the new scaler/metadata to `models/`
5. Update `utils/forecasting.py` and `train_models.py` as needed so the app loads and runs correctly
6. Run `streamlit run app.py` to verify everything works end to end

All the code files already exist — `app.py`, `train_models.py`, `utils/forecasting.py`, `pages/2 Forecasting.py`, `pages/3 Explainable_AI.py`. Your job is to update them to work with the real data instead of the simulated data they were originally built on.

**Do not generate new data. Do not use simulated data. Use only the files in `data/`.**

---

## Project Overview

This Streamlit app forecasts **Kenya's manufacturing sector contribution to GDP** using ARIMA, MLP, and LSTM models. It was built for a thesis titled "Application of Artificial Neural Networks in Forecasting Growth of the Manufacturing Sector in Kenya".

The app is fully built and working — but all the trained models in `models/` were fitted on **simulated data**. They must be retrained on the real KNBS data now available in `data/`.

---

## Project Structure

```
cement-forecast-app/
├── data/
│   ├── Manufacturing.xlsx    # Real historical KNBS data: 2020–2024
│   ├── Data.xlsx             # Real KNBS data: 2025 monthly
│   └── clean_data.csv        # Overwrite this with the merged clean data
├── models/
│   ├── arima_cement.pkl          # Retrain — keep this exact filename
│   ├── lstm_cement_model.keras   # Retrain — keep this exact filename
│   ├── mlp_cement_model.keras    # Retrain — keep this exact filename
│   ├── minmax_scaler.pkl         # Refit — keep this exact filename
│   └── metadata.pkl              # Regenerate — keep this exact filename
├── pages/
│   ├── 2 Forecasting.py
│   └── 3 Explainable_AI.py
├── utils/
│   └── forecasting.py
├── app.py
└── train_models.py
```

---

## Data

### Two files to merge

| File | Period |
|---|---|
| `data/Manufacturing.xlsx` | 2020 – 2024 |
| `data/Data.xlsx` | 2025 (current year, monthly) |

### Merge instructions

1. Inspect both files — check sheet names, column names, date column format
2. Align column names if they differ between files
3. Concatenate into one dataframe, sort by date ascending
4. Set date as a monthly `DatetimeIndex`
5. Save merged+cleaned result to `data/clean_data.csv`

### Variables (KNBS Manufacturing Indicators)

- Cement Production (MT)
- Cement Consumption (MT)
- Soft Drinks Production ('000 Litres)
- Milk Intake (Million Litres)
- Sugar Production (MT)
- Assembled Vehicles (units)
- Galvanized Sheet Production (MT)

**Target variable:** Manufacturing sector GDP growth/contribution — identify the correct column from the data files. If there is no single GDP column, use the manufacturing indicators collectively as features and confirm with the existing `train_models.py` logic what target it expects.

### Train/Validation/Test split

| Subset | Period |
|---|---|
| Training | 2020 – end of 2023 |
| Validation | Jan 2024 – Jun 2024 |
| Testing | Jul 2024 – latest available |

---

## Preprocessing Steps

1. Parse date column as `datetime`, set as monthly index (`freq='MS'`)
2. Ensure all feature columns are `float64`
3. Handle missing values: interpolate short gaps, forward fill longer ones
4. Remove duplicate months if any
5. Run ADF test for stationarity — difference non-stationary series for ARIMA
6. Fit `MinMaxScaler` on training set only, transform all sets, save to `models/minmax_scaler.pkl`
7. Build LSTM sequences with 12-month lookback window
8. Save `metadata.pkl` with feature column names, target column, sequence length, and split dates

---

## Model Training

Keep all existing model file names — the app loads them by exact path.

### ARIMA
- Identify optimal (p, d, q) using ACF/PACF plots and AIC/BIC
- Fit on training set
- Save to `models/arima_cement.pkl`

### MLP
- Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, linear)
- Optimizer: Adam | Loss: MSE
- Early stopping on validation loss
- Save to `models/mlp_cement_model.keras`

### LSTM
- LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
- Input shape: (12, n_features)
- Optimizer: Adam | Loss: MSE
- Early stopping on validation loss
- Save to `models/lstm_cement_model.keras`

---

## Evaluation Metrics

Report for each model on the test set:
- MAE, RMSE, MAPE, R²
- Diebold-Mariano test between models

---

## Hard Constraints

1. Merge both data files before anything else
2. Use only real data from `data/` — no synthetic or simulated data
3. Keep existing model file names in `models/` unchanged
4. Fit scaler on training data only — never on validation or test
5. The forecast target is manufacturing sector GDP growth — not cement production specifically
6. Regenerate `metadata.pkl` after retraining
7. Verify `streamlit run app.py` runs without errors as the final step