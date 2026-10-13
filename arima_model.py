"""
arima_model.py
Module 3: ARIMA Forecasting Model using Statsmodels.
Performs Augmented Dickey-Fuller (ADF) stationarity testing, differencing,
model estimation, rolling 1-step test evaluation, and 7-day future forecasting.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from evaluation import calculate_metrics

warnings.filterwarnings("ignore")


def check_stationarity(series):
    """
    Applies the Augmented Dickey-Fuller (ADF) test to evaluate series stationarity.

    Returns:
        dict: ADF test statistic, p-value, critical values, and stationarity boolean.
    """
    clean_series = series.dropna()
    result = adfuller(clean_series, autolag="AIC")

    p_value = float(result[1])
    return {
        "adf_statistic": round(float(result[0]), 4),
        "p_value": round(p_value, 6),
        "used_lag": int(result[2]),
        "nobs": int(result[3]),
        "critical_values": {k: round(float(v), 4) for k, v in result[4].items()},
        "is_stationary": bool(p_value < 0.05)
    }


def find_best_arima_order(train_series, max_p=2, d=1, max_q=2):
    """
    Selects optimal (p, d, q) parameters by minimizing Akaike Information Criterion (AIC)
    over a constrained grid suitable for responsive execution on standard hardware.
    """
    best_aic = float("inf")
    best_order = (1, d, 1)

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(train_series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    return best_order


def run_arima_pipeline(train_df, test_df, order=None, future_days=7):
    """
    Executes the complete ARIMA forecasting pipeline:
    1. Checks stationarity of training series (ADF test).
    2. Identifies or applies chosen (p, d, q) parameters.
    3. Generates walk-forward 1-step predictions over the hold-out test set.
    4. Projects an out-of-sample short-term forecast (e.g. next 7 daily prices).
    5. Computes evaluation metrics (MAE, RMSE, MAPE) against true test values.

    Args:
        train_df (pd.DataFrame): Training slice with ['Date', 'Close']
        test_df (pd.DataFrame): Hold-out test slice with ['Date', 'Close']
        order (tuple, optional): (p, d, q) order. If None, grid-searched via AIC.
        future_days (int): Number of days to forecast into the future.

    Returns:
        dict: Pipeline results containing metrics, predictions, and diagnostic data.
    """
    train_prices = train_df["Close"].values
    test_prices = test_df["Close"].values

    # Step 1: Stationarity Diagnostics
    stationarity_diag = check_stationarity(train_df["Close"])

    # Step 2: Determine model order
    if order is None:
        # Use d=1 if series is non-stationary, else d=0
        d_param = 0 if stationarity_diag["is_stationary"] else 1
        chosen_order = find_best_arima_order(train_prices, max_p=2, d=d_param, max_q=2)
    else:
        chosen_order = order

    # Step 3: Rolling Walk-Forward Prediction over Test Set
    # We maintain history and predict one day ahead iteratively to simulate realistic daily deployment
    history = list(train_prices)
    test_predictions = []

    # Initial fit on full training set
    base_model = ARIMA(history, order=chosen_order)
    fitted_model = base_model.fit()

    for t in range(len(test_prices)):
        # Forecast 1 step ahead
        yhat = fitted_model.forecast(steps=1)[0]
        test_predictions.append(float(yhat))
        
        # In financial walk-forward testing, observe actual price at step t and update model state
        history.append(test_prices[t])
        # Fast update using statsmodels append/refit
        base_model = ARIMA(history, order=chosen_order)
        fitted_model = base_model.fit()

    test_predictions = np.array(test_predictions)

    # Step 4: Calculate Standardized Evaluation Metrics
    metrics = calculate_metrics(test_prices, test_predictions)

    # Step 5: Multi-day Future Forecast (next 7 days beyond test set)
    future_forecast = fitted_model.forecast(steps=future_days)
    future_forecast = [round(float(val), 2) for val in future_forecast]

    # Generate future dates
    last_date = pd.to_datetime(test_df["Date"].iloc[-1])
    future_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(future_days)
    ]

    # Format dates and arrays for JSON/web dashboard serialization
    test_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in test_df["Date"]]
    actual_test_prices = [round(float(p), 2) for p in test_prices]
    predicted_test_prices = [round(float(p), 2) for p in test_predictions]

    # Sample recent training data for context plotting (last 60 days of train)
    recent_train = train_df.iloc[-60:].copy()
    recent_train_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in recent_train["Date"]]
    recent_train_prices = [round(float(p), 2) for p in recent_train["Close"]]

    return {
        "model_name": "ARIMA",
        "order": chosen_order,
        "aic": round(float(fitted_model.aic), 2),
        "bic": round(float(fitted_model.bic), 2),
        "stationarity": stationarity_diag,
        "recent_train_dates": recent_train_dates,
        "recent_train_prices": recent_train_prices,
        "test_dates": test_dates,
        "actual_test_prices": actual_test_prices,
        "predicted_test_prices": predicted_test_prices,
        "future_dates": future_dates,
        "future_forecast": future_forecast,
        "metrics": metrics,
    }


def get_cached_or_run_arima(cache_path="data/arima_results.json", force_retrain=False):
    """
    Returns cached ARIMA model results or runs the pipeline if cache doesn't exist.
    """
    import json
    import os
    from data_loader import load_or_fetch_bitcoin_data, get_train_test_split

    if os.path.exists(cache_path) and not force_retrain:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    df = load_or_fetch_bitcoin_data()
    train_df, test_df = get_train_test_split(df, test_days=30)
    results = run_arima_pipeline(train_df, test_df, order=None, future_days=7)

    # Convert tuples to lists for JSON serialization
    serializable = results.copy()
    serializable["order"] = list(serializable["order"])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return serializable


if __name__ == "__main__":
    from data_loader import load_or_fetch_bitcoin_data, get_train_test_split

    print("Generating initial cached ARIMA results...")
    res = get_cached_or_run_arima(force_retrain=True)
    print("ARIMA Cached successfully with metrics:", res["metrics"])

