"""
data_loader.py
Module 2: Historical Cryptocurrency Data Acquisition and Preprocessing.
Supports >10 cryptocurrencies via Yahoo Finance API, custom CSV file uploads,
data cleaning, chronological sorting, and strict train/test splitting.
"""

import os
import io
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "historical")
os.makedirs(DATA_DIR, exist_ok=True)

# Configurable list of supported cryptocurrencies (> 10 assets)
SUPPORTED_CRYPTOS = {
    "BTC": {"name": "Bitcoin", "ticker": "BTC-USD", "symbol": "BTC"},
    "ETH": {"name": "Ethereum", "ticker": "ETH-USD", "symbol": "ETH"},
    "SOL": {"name": "Solana", "ticker": "SOL-USD", "symbol": "SOL"},
    "BNB": {"name": "BNB", "ticker": "BNB-USD", "symbol": "BNB"},
    "XRP": {"name": "XRP", "ticker": "XRP-USD", "symbol": "XRP"},
    "ADA": {"name": "Cardano", "ticker": "ADA-USD", "symbol": "ADA"},
    "DOGE": {"name": "Dogecoin", "ticker": "DOGE-USD", "symbol": "DOGE"},
    "LTC": {"name": "Litecoin", "ticker": "LTC-USD", "symbol": "LTC"},
    "LINK": {"name": "Chainlink", "ticker": "LINK-USD", "symbol": "LINK"},
    "AVAX": {"name": "Avalanche", "ticker": "AVAX-USD", "symbol": "AVAX"},
    "DOT": {"name": "Polkadot", "ticker": "DOT-USD", "symbol": "DOT"},
    "POL": {"name": "Polygon", "ticker": "POL-USD", "symbol": "POL"},
}


def clean_and_format_price_df(raw_df):
    """
    Standardizes a DataFrame into clean ['Date', 'Close'] format.
    Ensures datetime conversion, drops nulls and duplicates, and sorts chronologically.
    """
    # Normalize column names
    col_mapping = {}
    for col in raw_df.columns:
        col_str = str(col).strip().lower()
        if col_str in ["date", "timestamp", "datetime", "time"]:
            col_mapping[col] = "Date"
        elif col_str in ["close", "price", "closing price", "adj close", "adj_close"]:
            col_mapping[col] = "Close"

    df = raw_df.rename(columns=col_mapping)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"Dataset must contain 'Date' and 'Close' columns. Found columns: {list(raw_df.columns)}")

    # Clean dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Clean numeric close prices
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    # Deduplicate and sort chronologically
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Insufficient historical data: only {len(df)} valid observations found. Need at least 50 days.")

    return df


def fetch_crypto_history(symbol="BTC", period_years=2, force_download=False):
    """
    Fetches historical daily prices for any supported cryptocurrency using Yahoo Finance.
    Caches to local CSV for fast re-access.

    Args:
        symbol (str): Asset symbol (e.g., 'BTC', 'ETH', 'SOL')
        period_years (int): Lookback history in years (1, 2, or 3)
        force_download (bool): If True, re-fetches from public API

    Returns:
        pd.DataFrame: Cleaned DataFrame with ['Date', 'Close']
    """
    symbol = symbol.strip().upper()
    if symbol not in SUPPORTED_CRYPTOS:
        raise ValueError(f"Cryptocurrency '{symbol}' is not in the supported list.")

    coin_info = SUPPORTED_CRYPTOS[symbol]
    ticker = coin_info["ticker"]
    cache_file = os.path.join(DATA_DIR, f"{symbol}_{period_years}y.csv")

    # Load from cache if valid and not forced
    if os.path.exists(cache_file) and not force_download:
        try:
            df = pd.read_csv(cache_file)
            return clean_and_format_price_df(df)
        except Exception:
            pass

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365 + 15)

    print(f"Downloading {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    try:
        raw = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
        if raw.empty:
            raise ValueError(f"No market data returned from API for {ticker}.")

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.levels[0]:
                close_series = raw["Close"].iloc[:, 0]
            else:
                close_series = raw.iloc[:, 0]
        else:
            close_series = raw["Close"]

        raw_df = pd.DataFrame({
            "Date": pd.to_datetime(close_series.index),
            "Close": close_series.values.astype(float)
        })

        clean_df = clean_and_format_price_df(raw_df)

        # Save cache
        to_save = clean_df.copy()
        to_save["Date"] = to_save["Date"].dt.strftime("%Y-%m-%d")
        to_save.to_csv(cache_file, index=False)

        return clean_df

    except Exception as e:
        # Fallback to cache if available
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            return clean_and_format_price_df(df)
        raise RuntimeError(f"Failed to fetch market data for {symbol} ({e}).")


def parse_uploaded_csv(file_storage):
    """
    Parses and cleans a user-uploaded CSV file.

    Args:
        file_storage (werkzeug.datastructures.FileStorage): Uploaded file

    Returns:
        pd.DataFrame: Cleaned DataFrame with ['Date', 'Close']
    """
    try:
        filename = file_storage.filename
        if not filename.lower().endswith(".csv"):
            raise ValueError("Uploaded file must be a .csv format.")

        content = file_storage.read().decode("utf-8")
        df_raw = pd.read_csv(io.StringIO(content))
        return clean_and_format_price_df(df_raw)
    except Exception as e:
        raise ValueError(f"Error parsing uploaded CSV: {str(e)}")


def get_train_test_split(df, test_days=30):
    """
    Chronological train/test split preventing data leakage.
    The last `test_days` form the test set; all prior days form the training set.
    """
    if len(df) <= test_days + 30:
        raise ValueError(f"Dataset has only {len(df)} rows; need at least {test_days + 30} rows for a {test_days}-day test period.")

    train_df = df.iloc[:-test_days].copy().reset_index(drop=True)
    test_df = df.iloc[-test_days:].copy().reset_index(drop=True)
    return train_df, test_df
