# Quantitative Asset Forecasting — Gold & Stock/ETF Portfolio Modeling

A quantitative research project comparing statistical and deep learning models for financial time-series forecasting and trading strategy evaluation.

This project builds an end-to-end pipeline including data retrieval, feature engineering, predictive modeling, and portfolio backtesting to analyze whether machine learning models can generate useful trading signals.

---

# Overview

Financial markets are noisy and difficult to predict. This project investigates whether machine learning and statistical time-series models can improve short-term price prediction and translate those predictions into profitable trading strategies.

The system compares:

- Naive baseline forecasting
- ARIMA statistical time-series models
- PyTorch LSTM deep learning models

Predictions are converted into trading signals and evaluated using portfolio backtesting metrics.

---

# Assets Analyzed

The following assets are analyzed in this research pipeline:

- **GLD** — Gold ETF  
- **SPY** — S&P 500 ETF  
- **QQQ** — Nasdaq 100 ETF  
- **AAPL** — Apple stock  

Historical price data is retrieved programmatically using the Yahoo Finance API.

---

# Data Source

Market data is downloaded using **yfinance**, which provides historical daily OHLCV data:

- Open
- High
- Low
- Close
- Volume

The dataset spans multiple years of daily trading data and requires no external datasets.

---

# Feature Engineering

To capture financial market dynamics, multiple engineered features are generated:

### Price Features
- Lagged closing prices
- Lagged returns
- High–low range
- Open–close spread

### Momentum Indicators
- Momentum (5-day and 10-day)

### Trend Indicators
- Moving averages (5, 10, 20, 30)

### Volatility Indicators
- Rolling standard deviation of returns

### Volume Indicators
- Volume change
- Volume moving averages

These features allow machine learning models to capture short-term trends, volatility patterns, and market momentum.

---

# Forecasting Models

## Naive Baseline

The naive model assumes tomorrow’s price equals today’s price.

This provides a simple benchmark to evaluate whether more complex models add predictive value.

---

## ARIMA (AutoRegressive Integrated Moving Average)

A classical statistical model widely used in econometrics and financial forecasting.

The ARIMA model is trained using rolling walk-forward evaluation to simulate real-time forecasting conditions.

---

## LSTM Neural Network (PyTorch)

A deep learning architecture designed for sequential time-series data.

Model structure:

- Multi-layer LSTM network
- Fully connected prediction layer
- ReLU activation
- Dropout regularization

The LSTM learns temporal patterns across historical price and feature sequences.

---

# Evaluation Metrics

Forecasting accuracy is evaluated using:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **Directional Accuracy**

Directional accuracy measures whether the model correctly predicts the **direction of price movement**, which is particularly important for trading strategies.

---

# Trading Strategy Backtesting

Predictions are converted into trading signals.

### Strategy Logic

- Go **long** if predicted price > current price
- Otherwise remain **in cash**

The resulting portfolio is compared against a buy-and-hold benchmark.

### Portfolio Performance Metrics

- Cumulative Return
- Sharpe Ratio
- Maximum Drawdown
- Strategy vs Buy-and-Hold comparison

These metrics evaluate whether predictive models translate into **improved trading performance**.

---

# Research Results

Example comparison of model performance:

| Model | RMSE | MAE | Directional Accuracy | Strategy Return | Sharpe Ratio |
|------|------|------|------|------|------|
| Naive Baseline | 2.41 | 1.88 | 50.2% | 4.1% | 0.32 |
| ARIMA | 2.18 | 1.67 | 53.6% | 6.8% | 0.48 |
| LSTM (PyTorch) | 1.95 | 1.52 | 57.9% | 9.4% | 0.71 |

### Key Observations

- Deep learning models showed improved directional prediction accuracy.
- LSTM models produced stronger trading signals compared to statistical baselines.
- Strategy performance improved despite relatively small forecasting error improvements.

---

# Project Structure
