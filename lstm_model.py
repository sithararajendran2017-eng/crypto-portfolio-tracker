"""
lstm_model.py
Module 4: LSTM Deep Learning Forecasting Model using TensorFlow / Keras.
Features chronological sliding-window sequences, strictly isolated scaling to prevent leakage,
early stopping, walk-forward test evaluation, and multi-step recursive future forecasting.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from evaluation import calculate_metrics

# Set random seeds for consistent reproducible student demonstrations
np.random.seed(42)
tf.random.set_seed(42)


def create_sequences(data, lookback=30):
    """
    Transforms a 1D array of prices into sliding window sequences:
    X[i] = [p_{i}, p_{i+1}, ..., p_{i+lookback-1}]
    y[i] = p_{i+lookback}
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def run_lstm_pipeline(train_df, test_df, lookback=30, epochs=20, batch_size=32, future_days=7):
    """
    Executes the complete LSTM forecasting pipeline:
    1. Fits MinMaxScaler strictly on training prices (preventing data leakage).
    2. Builds sliding-window sequences for training.
    3. Trains an LSTM architecture with Dropout and EarlyStopping.
    4. Evaluates predictions on the holdout test period.
    5. Generates a recursive multi-step out-of-sample forecast for `future_days`.
    6. Inverts scaling back to original USD price scale and computes evaluation metrics.

    Args:
        train_df (pd.DataFrame): Training slice with ['Date', 'Close']
        test_df (pd.DataFrame): Hold-out test slice with ['Date', 'Close']
        lookback (int): Number of prior days used to predict the next day
        epochs (int): Maximum training iterations
        batch_size (int): Gradient update batch size
        future_days (int): Forecast horizon (e.g., 1, 7, 14 days)

    Returns:
        dict: Pipeline results containing metrics, predictions, and future forecasts.
    """
    train_prices = train_df["Close"].values.reshape(-1, 1)
    test_prices = test_df["Close"].values.reshape(-1, 1)

    # Step 1: Strict normalization fitted ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_prices)

    # Step 2: Create supervised training sequences
    X_train, y_train = create_sequences(scaled_train, lookback=lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Chronological validation split from training slice (last 15% of train sequences)
    val_split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_tr, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

    # Step 3: Define lightweight LSTM Architecture for Student Hardware
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(units=32, return_sequences=False),
        Dropout(0.15),
        Dense(units=16, activation="relu"),
        Dense(units=1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Early stopping to prevent overfitting and ensure fast training
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Step 4: Generate Test Period Predictions
    # To predict test_prices[0], we need the last `lookback` prices from training data
    full_scaled_series = scaler.transform(
        np.concatenate([train_prices[-lookback:], test_prices])
    )

    X_test, _ = create_sequences(full_scaled_series, lookback=lookback)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    scaled_test_predictions = model.predict(X_test, verbose=0)
    test_predictions = scaler.inverse_transform(scaled_test_predictions).flatten()

    # Step 5: Compute Standardized Metrics against ground truth
    metrics = calculate_metrics(test_prices.flatten(), test_predictions)

    # Step 6: Multi-step Recursive Out-of-Sample Future Forecast
    # Start sequence: last `lookback` scaled points from the combined test data
    current_seq = full_scaled_series[-lookback:].copy()  # shape (lookback, 1)
    future_scaled_preds = []

    for _ in range(future_days):
        input_seq = current_seq.reshape((1, lookback, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        future_scaled_preds.append(pred_scaled)
        # Shift sequence forward: drop oldest, append newest prediction
        current_seq = np.vstack([current_seq[1:], [[pred_scaled]]])

    future_predictions = scaler.inverse_transform(
        np.array(future_scaled_preds).reshape(-1, 1)
    ).flatten()

    # Step 7: Format Dates and Arrays for Dashboard Output
    last_date = pd.to_datetime(test_df["Date"].iloc[-1])
    future_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(future_days)
    ]
    test_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in test_df["Date"]]

    # Final loss and epochs trained
    epochs_run = len(history.history["loss"])
    final_loss = round(float(history.history["loss"][-1]), 6)
    final_val_loss = round(float(history.history["val_loss"][-1]), 6)

    return {
        "model_name": "LSTM",
        "architecture": f"LSTM(32) -> Dropout(0.15) -> Dense(16) -> Dense(1)",
        "lookback_window": lookback,
        "epochs_run": epochs_run,
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        "test_dates": test_dates,
        "actual_test_prices": [round(float(p), 2) for p in test_prices.flatten()],
        "predicted_test_prices": [round(float(p), 2) for p in test_predictions],
        "future_dates": future_dates,
        "future_forecast": [round(float(p), 2) for p in future_predictions],
        "metrics": metrics,
    }


if __name__ == "__main__":
    from data_loader import fetch_crypto_history, get_train_test_split

    print("Testing LSTM Pipeline on Bitcoin...")
    df = fetch_crypto_history("BTC", period_years=2)
    train_df, test_df = get_train_test_split(df, test_days=30)
    res = run_lstm_pipeline(train_df, test_df, lookback=30, epochs=15, future_days=7)
    print("LSTM Completed in", res["epochs_run"], "epochs")
    print("Metrics:", res["metrics"])
    print("Future Forecast:", res["future_forecast"])
