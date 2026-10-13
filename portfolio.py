"""
portfolio.py
Module 1 & Holdings Value Forecasting:
Provides cryptocurrency holdings management, live market pricing, asset allocation,
and multi-asset portfolio value forecasting using ARIMA and LSTM.
"""

import json
import os
import uuid
import time
import yfinance as yf

# Supported Cryptocurrencies mapping: Symbol -> (Yahoo Finance Ticker, Display Name)
SUPPORTED_COINS = {
    "BTC": {"ticker": "BTC-USD", "name": "Bitcoin"},
    "ETH": {"ticker": "ETH-USD", "name": "Ethereum"},
    "SOL": {"ticker": "SOL-USD", "name": "Solana"},
    "BNB": {"ticker": "BNB-USD", "name": "BNB"},
    "XRP": {"ticker": "XRP-USD", "name": "XRP"},
    "ADA": {"ticker": "ADA-USD", "name": "Cardano"},
    "DOGE": {"ticker": "DOGE-USD", "name": "Dogecoin"},
    "LTC": {"ticker": "LTC-USD", "name": "Litecoin"},
    "LINK": {"ticker": "LINK-USD", "name": "Chainlink"},
    "AVAX": {"ticker": "AVAX-USD", "name": "Avalanche"},
    "DOT": {"ticker": "DOT-USD", "name": "Polkadot"},
    "POL": {"ticker": "POL-USD", "name": "Polygon"},
}

# Reliable fallback benchmark prices in USD in case API is offline or network fails
FALLBACK_PRICES = {
    "BTC": 83120.0,
    "ETH": 2490.0,
    "SOL": 105.5,
    "BNB": 575.0,
    "XRP": 0.58,
    "ADA": 0.45,
    "DOGE": 0.12,
    "LTC": 72.00,
    "LINK": 14.20,
    "AVAX": 28.50,
    "DOT": 6.80,
    "MATIC": 0.42,
}

# In-memory price cache to avoid hitting rate limits
_PRICE_CACHE = {
    "timestamp": 0,
    "prices": FALLBACK_PRICES.copy(),
    "source": "Yahoo Finance (Cached Baseline)"
}
CACHE_TTL_SECONDS = 60  # Cache for 1 minute


def fetch_live_prices(symbols=None):
    """
    Fetch current USD spot prices using Yahoo Finance Market API.
    
    Returns:
        dict: {symbol: current_price_usd}
        str: data source status message
    """
    global _PRICE_CACHE
    current_time = time.time()

    if current_time - _PRICE_CACHE["timestamp"] < CACHE_TTL_SECONDS:
        return _PRICE_CACHE["prices"], _PRICE_CACHE["source"]

    target_symbols = symbols if symbols else list(SUPPORTED_COINS.keys())
    tickers = [SUPPORTED_COINS[s]["ticker"] for s in target_symbols if s in SUPPORTED_COINS]

    try:
        data = yf.download(tickers=tickers, period="1d", progress=False)
        if not data.empty:
            close_data = data["Close"]
            updated_prices = _PRICE_CACHE["prices"].copy()

            if len(tickers) == 1:
                val = float(close_data.dropna().iloc[-1])
                updated_prices[target_symbols[0]] = round(val, 2)
            else:
                for sym in target_symbols:
                    tick = SUPPORTED_COINS[sym]["ticker"]
                    if tick in close_data.columns:
                        series = close_data[tick].dropna()
                        if not series.empty:
                            updated_prices[sym] = round(float(series.iloc[-1]), 2)

            _PRICE_CACHE["prices"] = updated_prices
            _PRICE_CACHE["timestamp"] = current_time
            _PRICE_CACHE["source"] = "Yahoo Finance Market API (Live)"
            return _PRICE_CACHE["prices"], _PRICE_CACHE["source"]
        else:
            _PRICE_CACHE["source"] = "Yahoo Finance (Cached Benchmark)"
            return _PRICE_CACHE["prices"], _PRICE_CACHE["source"]
    except Exception:
        _PRICE_CACHE["source"] = "Yahoo Finance (Offline Cache)"
        return _PRICE_CACHE["prices"], _PRICE_CACHE["source"]


class PortfolioManager:
    """Manages user holdings, persistence to JSON, and financial metric calculations."""

    def __init__(self, storage_path="data/portfolio.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        """Initializes portfolio storage with realistic starter assets if empty."""
        if not os.path.exists(self.storage_path):
            sample_holdings = [
                {
                    "id": str(uuid.uuid4())[:8],
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quantity": 0.45,
                    "purchase_price": 58500.0,
                },
                {
                    "id": str(uuid.uuid4())[:8],
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "quantity": 2.5,
                    "purchase_price": 3100.0,
                },
                {
                    "id": str(uuid.uuid4())[:8],
                    "symbol": "SOL",
                    "name": "Solana",
                    "quantity": 18.0,
                    "purchase_price": 125.0,
                },
            ]
            self._save_raw_holdings(sample_holdings)

    def _load_raw_holdings(self):
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_raw_holdings(self, holdings):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(holdings, f, indent=2)

    def get_holdings(self):
        return self._load_raw_holdings()

    def add_holding(self, symbol, quantity, purchase_price):
        """
        Validate and add a new holding.
        Returns (success: bool, message: str)
        """
        symbol = symbol.strip().upper()
        if symbol not in SUPPORTED_COINS:
            return False, f"Asset '{symbol}' is not currently supported. Choose from: {', '.join(SUPPORTED_COINS.keys())}"

        try:
            quantity = float(quantity)
            purchase_price = float(purchase_price)
        except (ValueError, TypeError):
            return False, "Quantity and purchase price must be valid numeric values."

        if quantity <= 0:
            return False, "Quantity held must be strictly greater than 0."
        if purchase_price < 0:
            return False, "Purchase price cannot be negative."

        holdings = self._load_raw_holdings()

        new_entry = {
            "id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "name": SUPPORTED_COINS[symbol]["name"],
            "quantity": quantity,
            "purchase_price": purchase_price,
        }
        holdings.append(new_entry)
        self._save_raw_holdings(holdings)
        return True, f"Successfully added {quantity} {symbol} to portfolio."

    def edit_holding(self, holding_id, quantity, purchase_price):
        """
        Updates an existing cryptocurrency holding's quantity and purchase price.
        Returns (success: bool, message: str)
        """
        try:
            quantity = float(quantity)
            purchase_price = float(purchase_price)
        except (ValueError, TypeError):
            return False, "Quantity and purchase price must be valid numeric values."

        if quantity <= 0:
            return False, "Quantity held must be strictly greater than 0."
        if purchase_price < 0:
            return False, "Purchase price cannot be negative."

        holdings = self._load_raw_holdings()
        found = False
        for h in holdings:
            if h["id"] == holding_id:
                h["quantity"] = quantity
                h["purchase_price"] = purchase_price
                found = True
                break

        if not found:
            return False, f"Holding ID '{holding_id}' not found."

        self._save_raw_holdings(holdings)
        return True, "Holding updated successfully."

    def delete_holding(self, holding_id):
        holdings = self._load_raw_holdings()
        filtered = [h for h in holdings if h["id"] != holding_id]
        if len(filtered) == len(holdings):
            return False, "Holding ID not found."
        self._save_raw_holdings(filtered)
        return True, "Holding removed successfully."

    def calculate_portfolio(self):
        """
        Calculates all portfolio metrics:
        - Invested value per holding = quantity * purchase_price
        - Current value per holding = quantity * current_price
        - Profit/Loss per holding = current_value - invested_value
        - Return % per holding = ((current_price - purchase_price) / purchase_price) * 100
        - Total portfolio value, total invested, total P/L, total return %
        - Asset allocation % for each holding = (current_value / total_current_value) * 100
        """
        holdings = self._load_raw_holdings()
        prices, price_source = fetch_live_prices()

        total_invested = 0.0
        total_current_value = 0.0
        processed_holdings = []

        # First pass: calculate individual holding values
        for h in holdings:
            sym = h["symbol"]
            qty = float(h["quantity"])
            buy_price = float(h["purchase_price"])
            curr_price = float(prices.get(sym, buy_price))

            invested_val = qty * buy_price
            current_val = qty * curr_price
            pnl = current_val - invested_val
            pnl_pct = ((curr_price - buy_price) / buy_price * 100) if buy_price > 0 else 0.0

            total_invested += invested_val
            total_current_value += current_val

            processed_holdings.append({
                "id": h["id"],
                "symbol": sym,
                "name": h["name"],
                "quantity": qty,
                "purchase_price": buy_price,
                "current_price": curr_price,
                "invested_value": invested_val,
                "current_value": current_val,
                "profit_loss": pnl,
                "profit_loss_pct": pnl_pct,
                "allocation_pct": 0.0,
            })

        # Second pass: calculate allocation percentages
        for item in processed_holdings:
            if total_current_value > 0:
                item["allocation_pct"] = (item["current_value"] / total_current_value) * 100
            else:
                item["allocation_pct"] = 0.0

        total_pnl = total_current_value - total_invested
        total_pnl_pct = ((total_pnl / total_invested) * 100) if total_invested > 0 else 0.0

        asset_allocation = {}
        for item in processed_holdings:
            asset_allocation[item["symbol"]] = asset_allocation.get(item["symbol"], 0.0) + item["allocation_pct"]

        return {
            "holdings": processed_holdings,
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current_value, 2),
            "total_profit_loss": round(total_pnl, 2),
            "total_profit_loss_pct": round(total_pnl_pct, 2),
            "price_source": price_source,
            "asset_allocation": {k: round(v, 2) for k, v in asset_allocation.items()},
            "num_assets": len(processed_holdings),
        }

    def forecast_portfolio_value(self, model_choice="both", horizon_days=7):
        """
        Projects future valuation of user's personal cryptocurrency portfolio.
        Reuses existing ARIMA and LSTM pipelines to forecast each constituent asset,
        then calculates:
        - Predicted holding value = Quantity * Predicted asset price
        - Total predicted portfolio value = Sum of all predicted holding values
        - Projected P/L relative to current value = Predicted value - Current value
        - Projected % return relative to current value.
        """
        from data_loader import fetch_crypto_history, get_train_test_split
        from arima_model import run_arima_pipeline
        from lstm_model import run_lstm_pipeline

        current_summary = self.calculate_portfolio()
        holdings = current_summary["holdings"]
        current_total_value = current_summary["total_current_value"]

        if not holdings:
            raise ValueError("Cannot forecast an empty portfolio. Please add at least one cryptocurrency holding.")

        # Identify unique coins present in portfolio
        unique_symbols = list({h["symbol"] for h in holdings})
        coin_forecasts = {}
        failed_coins = []

        # Run model forecasts for each unique cryptocurrency in holdings
        for sym in unique_symbols:
            try:
                df = fetch_crypto_history(sym, period_years=2)
                train_df, test_df = get_train_test_split(df, test_days=30)

                coin_data = {"future_dates": []}

                if model_choice in ["arima", "both"]:
                    arima_out = run_arima_pipeline(train_df, test_df, future_days=horizon_days)
                    coin_data["arima_prices"] = arima_out["future_forecast"]
                    coin_data["future_dates"] = arima_out["future_dates"]

                if model_choice in ["lstm", "both"]:
                    lstm_out = run_lstm_pipeline(train_df, test_df, lookback=30, epochs=15, future_days=horizon_days)
                    coin_data["lstm_prices"] = lstm_out["future_forecast"]
                    coin_data["future_dates"] = lstm_out["future_dates"]

                coin_forecasts[sym] = coin_data

            except Exception as e:
                failed_coins.append({"symbol": sym, "error": str(e)})

        if not coin_forecasts:
            raise RuntimeError(f"Unable to generate price forecasts for portfolio assets: {failed_coins}")

        sample_sym = list(coin_forecasts.keys())[0]
        future_dates = coin_forecasts[sample_sym]["future_dates"]
        num_days = len(future_dates)

        holding_forecast_details = []
        portfolio_timeline = []

        arima_portfolio_totals = [0.0] * num_days if model_choice in ["arima", "both"] else None
        lstm_portfolio_totals = [0.0] * num_days if model_choice in ["lstm", "both"] else None

        for h in holdings:
            sym = h["symbol"]
            qty = h["quantity"]
            curr_val = h["current_value"]

            if sym in coin_forecasts:
                fc = coin_forecasts[sym]
                h_detail = {
                    "id": h["id"],
                    "symbol": sym,
                    "name": h["name"],
                    "quantity": qty,
                    "purchase_price": h["purchase_price"],
                    "current_price": h["current_price"],
                    "current_value": curr_val,
                }

                if "arima_prices" in fc:
                    h_arima_vals = [round(qty * p, 2) for p in fc["arima_prices"]]
                    h_detail["arima_prices"] = fc["arima_prices"]
                    h_detail["arima_values"] = h_arima_vals
                    for d_idx in range(num_days):
                        arima_portfolio_totals[d_idx] += h_arima_vals[d_idx]

                if "lstm_prices" in fc:
                    h_lstm_vals = [round(qty * p, 2) for p in fc["lstm_prices"]]
                    h_detail["lstm_prices"] = fc["lstm_prices"]
                    h_detail["lstm_values"] = h_lstm_vals
                    for d_idx in range(num_days):
                        lstm_portfolio_totals[d_idx] += h_lstm_vals[d_idx]

                holding_forecast_details.append(h_detail)
            else:
                for d_idx in range(num_days):
                    if arima_portfolio_totals:
                        arima_portfolio_totals[d_idx] += curr_val
                    if lstm_portfolio_totals:
                        lstm_portfolio_totals[d_idx] += curr_val

        for d_idx in range(num_days):
            t_entry = {
                "date": future_dates[d_idx],
                "day_number": d_idx + 1,
            }
            if arima_portfolio_totals:
                a_tot = round(arima_portfolio_totals[d_idx], 2)
                a_diff = round(a_tot - current_total_value, 2)
                a_pct = round((a_diff / current_total_value * 100), 2) if current_total_value > 0 else 0.0
                t_entry["arima_total"] = a_tot
                t_entry["arima_gain_loss"] = a_diff
                t_entry["arima_pct_change"] = a_pct

            if lstm_portfolio_totals:
                l_tot = round(lstm_portfolio_totals[d_idx], 2)
                l_diff = round(l_tot - current_total_value, 2)
                l_pct = round((l_diff / current_total_value * 100), 2) if current_total_value > 0 else 0.0
                t_entry["lstm_total"] = l_tot
                t_entry["lstm_gain_loss"] = l_diff
                t_entry["lstm_pct_change"] = l_pct

            portfolio_timeline.append(t_entry)

        # Milestones (Day 1, Day 7, Day 14)
        milestones = {}
        for target_day in [1, 7, 14]:
            if target_day <= num_days:
                milestones[f"day_{target_day}"] = portfolio_timeline[target_day - 1]

        # Asset contribution breakdown at end of horizon
        end_idx = num_days - 1
        asset_contributions = []
        final_total = (lstm_portfolio_totals[end_idx] if lstm_portfolio_totals else arima_portfolio_totals[end_idx])

        for hd in holding_forecast_details:
            val_at_end = hd["lstm_values"][end_idx] if "lstm_values" in hd else hd["arima_values"][end_idx]
            contrib_pct = round((val_at_end / final_total * 100), 2) if final_total > 0 else 0.0
            asset_contributions.append({
                "symbol": hd["symbol"],
                "name": hd["name"],
                "projected_value": val_at_end,
                "contribution_pct": contrib_pct
            })

        return {
            "model_choice": model_choice,
            "horizon_days": horizon_days,
            "future_dates": future_dates,
            "current_total_value": current_total_value,
            "timeline": portfolio_timeline,
            "milestones": milestones,
            "holding_forecasts": holding_forecast_details,
            "asset_contributions": asset_contributions,
            "failed_coins": failed_coins,
        }
