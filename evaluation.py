"""
evaluation.py
Standardized Evaluation Metrics for Time Series Forecasting Models (ARIMA & LSTM).
Calculates MAE, RMSE, and MAPE from actual vs predicted prices on identical test periods.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    Computes performance metrics comparing true vs predicted cryptocurrency prices.

    Args:
        y_true (array-like): Ground truth values over the test period.
        y_pred (array-like): Model predicted values over the test period.

    Returns:
        dict: {
            "mae": float,   # Mean Absolute Error in USD
            "rmse": float,  # Root Mean Squared Error in USD
            "mape": float   # Mean Absolute Percentage Error in %
        }
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}.")

    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Mean Absolute Percentage Error (MAPE)
    # Avoid zero division with small epsilon
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    else:
        mape = 0.0

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
    }


if __name__ == "__main__":
    sample_true = [60000, 61000, 62000]
    sample_pred = [59500, 61500, 61800]
    m = calculate_metrics(sample_true, sample_pred)
    print("Test Metrics:", m)
