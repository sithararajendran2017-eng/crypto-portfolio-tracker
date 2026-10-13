"""
app.py
Interactive Web Application for Crypto Portfolio Tracker and Price Forecasting.
Features user-driven execution of ARIMA and LSTM models across >10 cryptocurrencies,
custom CSV uploads, configurable horizons, and empirical performance comparison.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from portfolio import PortfolioManager, SUPPORTED_COINS
from data_loader import (
    SUPPORTED_CRYPTOS,
    fetch_crypto_history,
    parse_uploaded_csv,
    get_train_test_split,
)
from arima_model import run_arima_pipeline
from lstm_model import run_lstm_pipeline

app = Flask(__name__)
app.secret_key = "crypto-portfolio-tracker-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

portfolio_mgr = PortfolioManager()

# Global in-memory store for user forecasts
LATEST_FORECAST_RESULT = None
LATEST_PORTFOLIO_FORECAST = None


@app.route("/")
def index():
    """Renders the primary Portfolio Dashboard with holdings, valuations, and portfolio value forecasts."""
    portfolio_data = portfolio_mgr.calculate_portfolio()
    return render_template(
        "portfolio.html",
        portfolio=portfolio_data,
        portfolio_forecast=LATEST_PORTFOLIO_FORECAST,
        supported_coins=SUPPORTED_COINS,
        active_page="portfolio"
    )


@app.route("/add_holding", methods=["POST"])
def add_holding():
    """Handles adding a new cryptocurrency holding."""
    symbol = request.form.get("symbol", "").strip()
    quantity = request.form.get("quantity", "").strip()
    purchase_price = request.form.get("purchase_price", "").strip()

    success, message = portfolio_mgr.add_holding(symbol, quantity, purchase_price)
    if success:
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for("index"))


@app.route("/edit_holding/<holding_id>", methods=["POST"])
def edit_holding(holding_id):
    """Handles editing an existing holding's quantity and purchase price."""
    quantity = request.form.get("quantity", "").strip()
    purchase_price = request.form.get("purchase_price", "").strip()

    success, message = portfolio_mgr.edit_holding(holding_id, quantity, purchase_price)
    if success:
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for("index"))


@app.route("/delete_holding/<holding_id>", methods=["POST"])
def delete_holding(holding_id):
    """Handles removal of a cryptocurrency position."""
    success, message = portfolio_mgr.delete_holding(holding_id)
    if success:
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for("index"))


@app.route("/forecast_portfolio", methods=["POST"])
def forecast_portfolio():
    """Calculates future valuation of the user's entire portfolio holdings using ARIMA & LSTM."""
    global LATEST_PORTFOLIO_FORECAST
    model_choice = request.form.get("model_choice", "both")
    horizon_days = int(request.form.get("horizon_days", 7))

    try:
        LATEST_PORTFOLIO_FORECAST = portfolio_mgr.forecast_portfolio_value(
            model_choice=model_choice,
            horizon_days=horizon_days
        )
        flash(f"Successfully calculated {horizon_days}-day portfolio value forecast!", "success")
    except Exception as e:
        flash(f"Portfolio Forecast Error: {str(e)}", "danger")

    return redirect(url_for("index") + "#portfolioForecastSection")


@app.route("/forecast", methods=["GET"])
def forecast_page():
    """
    Renders the interactive Forecasting Studio.
    Displays user configuration form and any generated forecast results.
    """
    global LATEST_FORECAST_RESULT
    return render_template(
        "forecast.html",
        supported_cryptos=SUPPORTED_CRYPTOS,
        result=LATEST_FORECAST_RESULT,
        active_page="forecast"
    )


@app.route("/generate_forecast", methods=["POST"])
def generate_forecast():
    """
    User-driven forecast execution route:
    Validates inputs, acquires data (API or uploaded CSV), trains selected model(s),
    and computes evaluation metrics.
    """
    global LATEST_FORECAST_RESULT

    crypto_symbol = request.form.get("crypto_symbol", "BTC").strip().upper()
    data_source = request.form.get("data_source", "api")
    period_years = int(request.form.get("period_years", 2))
    horizon_days = int(request.form.get("horizon_days", 7))
    model_choice = request.form.get("model_choice", "both")

    # Step 1: Acquire historical price series based on user data source
    try:
        if data_source == "csv":
            file = request.files.get("csv_file")
            if not file or file.filename == "":
                flash("Please select a valid CSV file containing 'Date' and 'Close' columns.", "danger")
                return redirect(url_for("forecast_page"))
            df = parse_uploaded_csv(file)
            dataset_name = f"Uploaded File ({file.filename})"
        else:
            if crypto_symbol not in SUPPORTED_CRYPTOS:
                flash(f"Invalid cryptocurrency selection: {crypto_symbol}", "danger")
                return redirect(url_for("forecast_page"))
            df = fetch_crypto_history(symbol=crypto_symbol, period_years=period_years)
            dataset_name = f"{SUPPORTED_CRYPTOS[crypto_symbol]['name']} ({crypto_symbol}) — Yahoo Finance"

        # Step 2: Chronological Train/Test Partitioning (30-day holdout)
        test_days = 30
        train_df, test_df = get_train_test_split(df, test_days=test_days)

    except Exception as e:
        flash(f"Data Acquisition Error: {str(e)}", "danger")
        return redirect(url_for("forecast_page"))

    # Step 3: Run selected forecasting pipeline(s)
    arima_res = None
    lstm_res = None

    try:
        if model_choice in ["arima", "both"]:
            arima_res = run_arima_pipeline(
                train_df=train_df,
                test_df=test_df,
                future_days=horizon_days
            )

        if model_choice in ["lstm", "both"]:
            lstm_res = run_lstm_pipeline(
                train_df=train_df,
                test_df=test_df,
                lookback=30,
                epochs=15,
                future_days=horizon_days
            )

    except Exception as e:
        flash(f"Forecasting Model Error: {str(e)}", "danger")
        return redirect(url_for("forecast_page"))

    # Step 4: Package comprehensive results
    LATEST_FORECAST_RESULT = {
        "dataset_name": dataset_name,
        "symbol": crypto_symbol,
        "total_records": len(df),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_start": train_df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "train_end": train_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "test_start": test_df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "test_end": test_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "horizon_days": horizon_days,
        "model_choice": model_choice,
        "test_dates": [pd.to_datetime(d).strftime("%Y-%m-%d") for d in test_df["Date"]],
        "actual_test_prices": [round(float(p), 2) for p in test_df["Close"].values],
        "arima": arima_res,
        "lstm": lstm_res,
    }

    flash(f"Successfully generated forecasts for {dataset_name} ({horizon_days}-day horizon)!", "success")
    return redirect(url_for("forecast_page"))


@app.route("/comparison")
def comparison_page():
    """
    Renders dedicated ARIMA vs. LSTM comparative matrix and in-depth academic evaluation.
    """
    global LATEST_FORECAST_RESULT
    return render_template(
        "comparison.html",
        result=LATEST_FORECAST_RESULT,
        active_page="comparison"
    )


@app.route("/about")
def about_page():
    """Overview of project scope, academic deliverables, and future roadmap."""
    return render_template("about.html", active_page="about")


if __name__ == "__main__":
    import pandas as pd
    print("=" * 65)
    print("Crypto Portfolio Tracker & Forecasting Studio Starting...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=True, host="127.0.0.1", port=5000)
