# Quantitative Asset Forecasting — Multi-Architecture Direction Prediction

A quant-style research project comparing two LSTM architectures for 5-day directional forecasting across four assets. Two notebooks are included deliberately — each optimised for a different modeling philosophy.

---

## Repository Structure

```
├── quant_asset_forecasting_v3.ipynb        # Feature-engineered LSTM pipeline
├── quant_lstm_multivariate.ipynb           # Raw multivariate LSTM pipeline
├── README.md                               # This file
├── quant_forecasting_model_comparison_robust.csv   # V3 results
└── quant_lstm_multivariate_results.csv             # Multivariate results
```

---

## Assets

| Ticker | Description |
|--------|-------------|
| GLD | Gold ETF |
| SPY | S&P 500 ETF |
| QQQ | Nasdaq-100 ETF |
| AAPL | Apple Inc. |

---

## Why Two Notebooks?

The core question this project explores is: **does feature engineering help or hurt an LSTM on daily financial data?**

| | V3 (Feature-Engineered) | Multivariate (Raw Sequences) |
|--|-------------------------|------------------------------|
| Input | 40+ hand-crafted features per ticker | Raw OHLCV for all 4 tickers jointly |
| Cross-asset info | TLT/UUP macro features for GLD/SPY/QQQ | All 4 tickers feed every prediction |
| LSTM advantage | Limited — trees win on tabular features | Full — model learns cross-asset temporal patterns |
| Runtime | ~25-35 min CPU | ~12-15 min CPU |
| Best result | SPY consensus 69.4% @ 36% coverage | SPY HighConf 66.9% @ 25.6% coverage |

The answer: **it depends on the asset**. Feature engineering wins for GLD (macro-driven) and overall consistency. Raw multivariate sequences win for SPY high-confidence filtering.

---

## Results Summary

### V3 — Feature-Engineered LSTM (3-Fold Walk-Forward)

| Ticker | Model | Directional Accuracy | Sharpe | Coverage |
|--------|-------|---------------------|--------|----------|
| SPY | LSTM_Attention | 62.1% | 3.16 | 100% |
| SPY | LSTM_ARIMA_Consensus | **69.4%** | **6.51** | 36% |
| SPY | LSTM_HighConf_0.6 | 61.8% | 2.24 | 80.8% |
| GLD | LSTM_Attention | 55.5% | 3.02 | 100% |
| AAPL | LSTM_Attention | 52.2% | 2.36 | 100% |
| QQQ | LSTM_Attention | 49.7% | 2.26 | 100% |

### Multivariate LSTM (Single Split)

| Ticker | Model | Directional Accuracy | Sharpe | Coverage |
|--------|-------|---------------------|--------|----------|
| SPY | LSTM_Multivariate | 50.6% | 3.52 | 100% |
| SPY | LSTM_HighConf_0.6 | **66.9%** | **6.51** | 25.6% |
| GLD | LSTM_Multivariate | 43.8% | 1.99 | 100% |
| AAPL | LSTM_Multivariate | 47.3% | 1.81 | 100% |
| QQQ | LSTM_Multivariate | 40.0% | 0.98 | 100% |

> **Note on coverage:** High-confidence and consensus models trade only on a subset of days. Higher accuracy, lower exposure. The 69.4% and 66.9% results apply to 36% and 25.6% of trading days respectively.

---

## Notebook 1 — `quant_asset_forecasting_v3.ipynb`

### Architecture

```
Raw OHLCV + 40+ engineered features per ticker
         ↓
  MinMaxScaler (fit on train only — no leakage)
         ↓
  Sliding window sequences (length=30)
         ↓
  2-layer LSTM → LayerNorm → Attention → FC → logit
         ↓
  BCEWithLogitsLoss + Adam + ReduceLROnPlateau
         ↓
  Val-set threshold search (grid: 0.30–0.70)
         ↓
  3-fold walk-forward evaluation
```

### Features Engineered

- **Returns:** 1d, 5d, 10d, 20d, intraday
- **Lags:** close and return lags at 1, 2, 3, 5, 10 days
- **Moving averages:** MA5, MA10, MA20, MA30, MA cross ratio
- **Momentum:** 5-day and 10-day price momentum
- **Volatility:** rolling std at 5, 10, 20 days + vol ratio
- **Price spreads:** high-low range, open-close range
- **Volume:** change, MA5, MA20
- **Technical indicators:** RSI(14), RSI centered, MACD, MACD signal, MACD histogram, Bollinger Bands
- **Ratio features:** price-to-MA20, vol ratio
- **Calendar:** day of week, month
- **Macro (GLD/SPY/QQQ only):** TLT 1d/5d returns, UUP 1d/5d returns

### Models

| Model | Description |
|-------|-------------|
| Naive | Yesterday's direction repeated |
| ARIMA(1,1,1) | Walk-forward univariate baseline |
| LSTM_Attention | 2-layer LSTM with attention + LayerNorm |
| LSTM_HighConf_0.6 | LSTM on ≥60% confidence signals only |
| LSTM_ARIMA_Consensus | Trade only when LSTM and ARIMA agree |

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 30 days |
| Forecast horizon | 5 days |
| Walk-forward folds | 3 |
| Epochs | 60 (early stopping patience=10) |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Label smoothing | 0.1 |
| Loss | BCEWithLogitsLoss |

---

## Notebook 2 — `quant_lstm_multivariate.ipynb`

### Architecture

```
Raw OHLCV × 4 tickers = 20 input features per timestep
         ↓
  MinMaxScaler (fit on train only)
         ↓
  Sliding window sequences (length=30)
         ↓
  2-layer LSTM → LayerNorm → Attention → FC → logit
         ↓
  BCEWithLogitsLoss + label smoothing
         ↓
  Adam + ReduceLROnPlateau + early stopping
         ↓
  High-confidence filtering (≥0.60)
```

### Why Raw Sequences Instead of Features?

Hand-engineered features (RSI, MACD, lags) are better suited to tree-based models. Feeding them to an LSTM adds noise without adding sequential structure. Raw OHLCV sequences let the LSTM learn temporal patterns directly — including cross-asset dependencies that no single-ticker feature set can capture.

### Speed Optimisations

| Optimisation | Impact |
|---|---|
| `BATCH_SIZE = 128` | 2× faster than 64 |
| `EPOCHS = 40` | Early stopping triggers at ~25-35 anyway |
| `ARIMA_REFIT_EVERY = 10` | Halves ARIMA walk-forward time |
| No feature engineering | Data prep near-instant |
| LayerNorm | Converges in fewer epochs |

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 30 days |
| Forecast horizon | 5 days |
| Train ratio | 75% |
| Epochs | 40 (early stopping patience=12) |
| Batch size | 128 |
| Learning rate | 3e-4 |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| LayerNorm | Yes |
| Label smoothing | 0.1 |
| Loss | BCEWithLogitsLoss |

---

## How to Run

```bash
pip install yfinance pandas numpy scikit-learn statsmodels torch
```

**V3 notebook:**
1. Open `quant_asset_forecasting_v3.ipynb`
2. Run all cells — use Cell 24 (sequential runner)
3. Results saved to `quant_forecasting_model_comparison_robust.csv`
4. Expected runtime: ~25-35 min CPU, ~15 min GPU

**Multivariate notebook:**
1. Open `quant_lstm_multivariate.ipynb`
2. Run all cells sequentially
3. Results saved to `quant_lstm_multivariate_results.csv`
4. Expected runtime: ~12-15 min CPU

> Both notebooks share the same `market_data_cache.pkl` file. Run either one first — the cache will be reused by the second automatically.

---

## Design Decisions

**Why BCEWithLogitsLoss without pos_weight?**
Markets go up ~52-53% of days. Applying `pos_weight` correction caused model collapse (constant zero predictions) on AAPL and QQQ because any small window imbalance was amplified. Plain loss is more stable.

**Why walk-forward validation (V3 only)?**
A single 75/25 split tests one time period. Three expanding walk-forward windows test consistency across pre-COVID, COVID, and post-COVID market regimes — a much stronger claim of generalisability.

**Why threshold tuning on validation set?**
Different tickers output probabilities at different scales. A fixed 0.50 threshold performed poorly. Val-set grid search (0.30–0.70) finds the optimal cutoff before touching test data.

**Why TLT and UUP for GLD/SPY/QQQ?**
Gold is primarily driven by real rates and dollar strength. Treasury and dollar ETF returns provide these macro signals explicitly. Rate sensitivity also applies to SPY and QQQ in the current macro environment.

**Why label smoothing?**
Binary 0/1 targets on noisy daily data cause overconfident predictions. Smoothing (→ 0.1/0.9) produces better-calibrated probability outputs and more meaningful confidence filtering.

---

## Reproducibility

- Data downloaded via `yfinance`, cached to `market_data_cache.pkl`
- Cache validated for required tickers on every load
- Seeds fixed: `np.random.seed(42)`, `torch.manual_seed(42)`
- No external CSV files required — notebook is fully self-contained
