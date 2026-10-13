# Crypto Portfolio Tracker & Cryptocurrency Price Forecasting Module
## Final-Year College Project Prototype (Semester Review Deliverable)

An interactive, user-input-driven web application for cryptocurrency portfolio tracking and short-term price forecasting using **both ARIMA and LSTM models** with empirical performance benchmarking.

---

### 1. Semester Review Scope & Architecture

- **Supported Cryptocurrencies (>10 Assets):** Bitcoin (BTC), Ethereum (ETH), Solana (SOL), BNB, XRP, Cardano (ADA), Dogecoin (DOGE), Litecoin (LTC), Chainlink (LINK), Avalanche (AVAX), Polkadot (DOT), Polygon (MATIC).
- **Flexible Data Ingestion:**
  - Public Financial Market API (Yahoo Finance) with configurable 1–3 year observation windows.
  - Custom CSV dataset upload (`Date` and `Close` columns).
- **ARIMA Econometric Model (Statsmodels):**
  - Stationarity analysis via Augmented Dickey-Fuller (ADF) test.
  - First-order differencing ($d=1$) when $p \ge 0.05$.
  - Practical AIC parameter selection.
  - Rolling walk-forward 1-step test evaluation.
  - Multi-step forward projection (1, 7, 14, 30 days).
- **LSTM Deep Neural Network (TensorFlow/Keras):**
  - MinMaxScaler strictly fitted on training data to prevent lookahead data leakage.
  - Supervised sliding sequences with a 30-day lookback window.
  - Regularized recurrent architecture (`LSTM(32)` + `Dropout(0.15)` + `Dense(16)` + `Dense(1)`).
  - EarlyStopping callback on validation loss.
  - Multi-step recursive forecasting forward across chosen horizon.
- **Fair Empirical Comparison:**
  - Evaluated on the exact same 30-day chronological test partition.
  - Standardized metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Percentage Error (MAPE)**.
  - Side-by-side comparison tables, winning model badges, and error bar charts.
- **Scope Note for Evaluators:**
  - Advanced features like SHAP feature attribution and composite portfolio risk scoring ($V, C, \bar{\rho}$) are documented as subsequent semester extensions.

---

### 2. Project File Structure

```
final year project/
├── app.py                   # Main Flask application and interactive routes
├── data_loader.py           # Multi-crypto API ingestion, CSV parser, and train/test splits
├── arima_model.py           # ARIMA forecasting pipeline with ADF testing and AIC selection
├── lstm_model.py            # LSTM forecasting pipeline with sliding sequences and early stopping
├── evaluation.py            # Standardized MAE, RMSE, and MAPE calculations
├── portfolio.py             # Portfolio manager with Yahoo Finance live market valuation
├── requirements.txt         # Core project dependencies
├── data/
│   ├── historical/          # Cached historical crypto price datasets
│   ├── sample_crypto.csv    # Sample CSV for demonstration of file uploads
│   └── portfolio.json       # Persistent user holdings storage
├── templates/
│   ├── base.html            # Bootstrap 5 layout & navigation
│   ├── forecast.html        # Interactive Forecasting Studio (user controls & charts)
│   ├── comparison.html      # In-depth ARIMA vs. LSTM comparative benchmark
│   ├── portfolio.html       # Portfolio tracking dashboard
│   └── about.html           # Project specifications and methodology
└── static/
    └── css/
        └── style.css        # Clean financial dashboard styling
```

---

### 3. Step-by-Step Running Instructions

#### 1. Launch the Application
In your terminal, run:
```bash
python app.py
```

#### 2. Open in Your Browser
Navigate to:
```
http://127.0.0.1:5000
```

#### 3. How to Demonstrate for Semester Review:
1. **Portfolio Tracking (`/`):** View crypto positions, live valuations, and asset allocation doughnut chart.
2. **Forecasting Studio (`/forecast`):**
   - Select any cryptocurrency (e.g. **Bitcoin**, **Ethereum**, or **Solana**).
   - Select data source: **Public API Feed** or upload `data/sample_crypto.csv`.
   - Select forecast horizon: **7 Days Ahead** (or 1 / 14 / 30 days).
   - Select Model: **Both ARIMA & LSTM (Comparative Benchmark)**.
   - Click **Generate Forecast**.
   - Review the holdout test evaluation table, actual vs. predicted chart, and future 7-day projection table.
3. **Model Comparison (`/comparison`):**
   - Review side-by-side MAE, RMSE, and MAPE metrics.
   - Inspect grouped error bar charts.
   - Review academic discussion of linear vs. deep learning trade-offs.
