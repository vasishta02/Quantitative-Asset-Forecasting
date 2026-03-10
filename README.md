# Quantitative Asset Forecasting — Multi-Architecture Direction Prediction

A quant-style research project comparing two LSTM architectures for 5-day directional
forecasting across four assets. Two notebooks are included deliberately — each optimised
for a different modeling philosophy, with results compared honestly.

---

## Repository Structure

```
├── quant_asset_forecasting_v4_1.ipynb          # Feature-engineered LSTM pipeline (primary)
├── quant_lstm_multivariate.ipynb               # Raw multivariate LSTM pipeline
├── README.md                                   # This file
├── quant_forecasting_results_v4_1.csv          # V4.1 final results
└── quant_lstm_multivariate_results.csv         # Multivariate results
```

---

## Assets & Data Sources

| Ticker | Description | Used In |
|--------|-------------|---------|
| GLD | Gold ETF | Both notebooks |
| SPY | S&P 500 ETF | Both notebooks |
| QQQ | Nasdaq-100 ETF | Both notebooks |
| AAPL | Apple Inc. | Both notebooks |
| TLT | 20-Year Treasury ETF | V4.1 (GLD/SPY/QQQ) |
| UUP | US Dollar Index ETF | V4.1 (GLD/SPY/QQQ) |
| ^VIX | CBOE Volatility Index | V4.1 (all tickers) |
| XLK/XLF/XLE/XLV/XLI | Sector ETFs | V4.1 (QQQ only) |

> **Why sector ETFs for QQQ only?**
> SPY *is* the sectors — adding sector returns as SPY features creates near-multicollinearity.
> QQQ (Nasdaq-100) benefits from sector rotation signals because tech vs financials leadership
> genuinely predicts Nasdaq direction. This distinction was discovered empirically across runs.

---

## Why Two Notebooks?

The core research question: **does feature engineering help or hurt an LSTM on daily financial data?**

| | V4.1 (Feature-Engineered) | Multivariate (Raw Sequences) |
|--|---------------------------|------------------------------|
| Input | 50+ hand-crafted features | Raw OHLCV × 4 tickers jointly |
| Cross-asset info | TLT/UUP/VIX/Sectors explicit | All 4 tickers implicit |
| LSTM advantage | Limited — trees win on tabular | Full — learns cross-asset patterns |
| Consistency | Strong across all 4 tickers | SPY HighConf outstanding, others weak |
| Runtime | ~30-40 min CPU | ~12-15 min CPU |

**Answer: it depends on the asset.** Feature engineering wins for GLD (macro-driven) and
provides consistent results across all tickers. Raw multivariate sequences produced the
single best SPY result (66.9% HighConf) but failed on GLD and QQQ.

---

## Results

### V4.1 — Feature-Engineered LSTM (3-Fold Walk-Forward)

| Ticker | Model | Dir. Accuracy | Sharpe | Coverage | Notes |
|--------|-------|--------------|--------|----------|-------|
| SPY | LSTM_Attention | 64.5% | 3.53 | 100% | Best full-coverage SPY |
| SPY | LSTM_ARIMA_Consensus | **68–71%** | **6.6–6.9** | 29–36% | Consensus signal across runs |
| SPY | LSTM_HighConf_0.6 | 64.2% | 5.24 | 61% | High-conviction days only |
| QQQ | LSTM_Attention | **63.0%** | 3.21 | 100% | Best QQQ result — sectors helped |
| QQQ | LSTM_ARIMA_Consensus | 65.8% | 5.69 | 31% | Strong consensus signal |
| QQQ | LSTM_HighConf_0.6 | 65.5% | 5.88 | 39% | |
| GLD | LSTM_Attention | 51.4% | 3.17 | 100% | |
| GLD | LSTM_HighConf_0.6 | **63.9%** | **9.15** | 59% | Best Sharpe across all runs |
| AAPL | LSTM_Attention | 53.9% | 2.31 | 100% | |
| AAPL | LSTM_HighConf_0.6 | 54.5% | 2.85 | 60% | Earnings flag contributed |



> Across multiple runs and model variants (V3–V4.1), the SPY LSTM–ARIMA consensus
> signal produced **68–71% directional accuracy with Sharpe ratios between 6.6 and 6.9**
> at ~29–36% signal coverage. The 71% result occurred in the V3 configuration prior
> to adding sector features, demonstrating the robustness of the consensus approach
> across model iterations.


### Multivariate LSTM (Single Split)

| Ticker | Model | Dir. Accuracy | Sharpe | Coverage | Notes |
|--------|-------|--------------|--------|----------|-------|
| SPY | LSTM_HighConf_0.6 | **66.9%** | **6.51** | 25.6% | Best single SPY result |
| SPY | LSTM_Multivariate | 50.6% | 3.52 | 100% | Full model weak |
| QQQ | LSTM_Multivariate | 40.0% | 0.98 | 100% | Sector noise hurts QQQ |
| GLD | LSTM_Multivariate | 43.8% | 1.99 | 100% | Macro signal drowned out |
| AAPL | LSTM_Multivariate | 47.3% | 1.81 | 100% | |

### What is Sharpe Ratio?

Sharpe measures return per unit of risk: `(Avg Daily Return / Std Dev) × √252`

| Sharpe | Interpretation |
|--------|---------------|
| 0 – 1 | Poor |
| 1 – 2 | Acceptable |
| 2 – 3 | Good |
| 3+ | Very good |
| 5+ | Exceptional |

> **Caveat:** These Sharpe numbers assume costless execution and no slippage.
> Realistic deflation of ~30-40% applies. SPY consensus 6.64 → ~4.0-4.5 in live trading.
> Results are from backtesting only — not live trading performance.

---

## Notebook 1 — `quant_asset_forecasting_v4_1.ipynb`

### Architecture

```
Raw OHLCV
  + 40 technical features (RSI, MACD, Bollinger Bands, momentum, volatility)
  + TLT/UUP macro returns (GLD/SPY/QQQ)
  + VIX level, returns, MA, regime flags (all tickers)
  + Sector ETF rotation features XLK/XLF/XLE/XLV/XLI (QQQ only)
  + AAPL earnings calendar flags (days_to_earnings, earnings_week, earnings_month)
         ↓
  MinMaxScaler (fit on train only — no leakage)
         ↓
  Sliding window sequences (length=30)
         ↓
  2-layer LSTM → LayerNorm → Attention → FC → logit
         ↓
  BCEWithLogitsLoss + label smoothing (0/1 → 0.1/0.9)
         ↓
  Adam + ReduceLROnPlateau + early stopping (patience=10)
         ↓
  Val-set threshold search (grid: 0.30–0.70)
         ↓
  3-fold expanding walk-forward evaluation
         ↓
  Signal generation: HighConf filter + LSTM-ARIMA Consensus
```

### Signal Generation

Two signal filters are applied after the base LSTM:

**High-Confidence Filter**
```python
# Trade only when model conviction exceeds threshold in either direction
high_conf_mask = (lstm_probs >= 0.60) | (lstm_probs <= 0.40)
```
Trades ~39-61% of days depending on ticker. Higher accuracy, lower exposure.

**LSTM-ARIMA Consensus**
```python
# Trade only when LSTM and ARIMA agree on direction
consensus_mask = (lstm_pred == arima_direction) & (abs(lstm_probs - 0.5) >= 0.05)
```
Trades ~29-37% of days. Both models must independently agree before a signal fires.
This is the strongest accuracy result — 68% on SPY, 65.8% on QQQ.

### Feature Engineering (50+ features)

**Per-ticker base features:**
- Returns: 1d, 5d, 10d, 20d, intraday
- Lags: close and return at 1, 2, 3, 5, 10 days
- Moving averages: MA5, MA10, MA20, MA30, MA cross ratio
- Momentum: 5d and 10d price momentum
- Volatility: rolling std at 5, 10, 20 days + vol ratio
- Price spreads: high-low range, open-close range
- Volume: change, MA5, MA20
- RSI(14), RSI centered, MACD, MACD signal, MACD histogram
- Bollinger Bands: upper, lower, width
- Price-to-MA20 ratio
- Calendar: day of week, month

**Macro / cross-asset (ticker-specific):**
- TLT 1d/5d returns → GLD, SPY, QQQ (rate sensitivity)
- UUP 1d/5d returns → GLD, SPY, QQQ (dollar strength)
- VIX level, 1d return, MA5, above-20 flag, above-30 flag → all tickers
- XLK/XLF/XLE/XLV/XLI 1d return, 5d return, relative strength vs SPY → QQQ only
- days_to_earnings, earnings_week flag, earnings_month flag → AAPL only

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 30 days |
| Forecast horizon | 5 days |
| Walk-forward folds | 3 (expanding window) |
| Min train size | 900 rows |
| Min test size | 180 rows |
| Epochs | 60 (early stopping patience=10) |
| Batch size | 64 |
| Learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| LayerNorm | Yes — stabilises gradients |
| Label smoothing | 0.1 (0/1 → 0.1/0.9) |
| Loss | BCEWithLogitsLoss (no pos_weight) |
| Threshold grid | 0.30–0.70 (val-set tuned) |
| Recency weighting | WeightedRandomSampler (exponential) |

---

## Notebook 2 — `quant_lstm_multivariate.ipynb`

### Architecture

```
Raw OHLCV × 4 tickers = 20 input features per timestep
  + VIX level, returns, MA5, above-20 flag
         ↓
  MinMaxScaler (fit on train only)
         ↓
  Sliding window sequences (length=30)
         ↓
  2-layer LSTM → LayerNorm → Attention → FC → logit
         ↓
  BCEWithLogitsLoss + label smoothing
         ↓
  Adam + ReduceLROnPlateau + early stopping (patience=12)
         ↓
  High-confidence filtering (≥0.60 threshold)
```

### Why Raw Sequences Instead of Features?

Hand-engineered features (RSI, MACD, lags) are better suited to tree-based models.
Feeding them to an LSTM adds tabular noise without adding sequential structure.
Raw OHLCV sequences let the LSTM learn temporal patterns directly — including
cross-asset dependencies: when SPY drops 3 days in a row, QQQ tends to follow.
That pattern is learnable from raw sequences without explicit feature engineering.

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
| ARIMA refit | Every 10 steps |

---

## How to Run

```bash
pip install yfinance pandas numpy scikit-learn statsmodels torch
```

**V4.1 notebook:**
1. Delete `market_data_cache.pkl` if it exists (forces fresh download with all tickers)
2. Open `quant_asset_forecasting_v4_1.ipynb`
3. Run Cell 24 (sequential runner — safer in Jupyter/Colab)
4. Results saved to `quant_forecasting_results_v4_1.csv`
5. Expected runtime: ~30-40 min CPU, ~15-20 min GPU

**Multivariate notebook:**
1. Open `quant_lstm_multivariate.ipynb`
2. Run all cells sequentially
3. Results saved to `quant_lstm_multivariate_results.csv`
4. Expected runtime: ~12-15 min CPU

> Both notebooks share `market_data_cache.pkl`. Run V4.1 first —
> it downloads all required tickers. The multivariate notebook will reuse the cache.

---

## Design Decisions

**Why BCEWithLogitsLoss without pos_weight?**
Markets go up ~52-53% of days — nearly balanced. Applying pos_weight correction
caused model collapse (constant zero predictions) on AAPL and QQQ. Removed entirely.

**Why walk-forward instead of a single split?**
A single 75/25 split tests one time period. Three expanding walk-forward windows
test consistency across pre-COVID, COVID, and post-COVID market regimes — a much
stronger claim of generalisability.

**Why threshold tuning on validation set?**
Different tickers output probabilities at different scales. Fixed 0.50 threshold
performed poorly. Val-set grid search (0.30–0.70) finds the optimal cutoff without
touching test data. Caveat: minor overfitting to the validation window is possible.

**Why TLT and UUP for GLD/SPY/QQQ?**
Gold is driven by real rates and dollar strength. Treasury (TLT) and dollar (UUP)
returns provide these macro signals explicitly. Rate sensitivity also applies to SPY
and QQQ in the current macro environment.

**Why sector ETFs for QQQ only, not SPY?**
SPY is composed of the same sectors — adding sector returns as SPY features creates
near-multicollinearity. QQQ (Nasdaq-100) benefits because tech vs financials
leadership genuinely predicts Nasdaq direction independently of SPY.

**Why label smoothing?**
Binary 0/1 targets on noisy daily data cause overconfident predictions. Smoothing
(→ 0.1/0.9) produces better-calibrated probability outputs and more meaningful
confidence filtering.

**Why LayerNorm after LSTM?**
Normalising LSTM outputs before attention stabilises gradient flow and speeds
convergence, typically achieving the same accuracy in fewer epochs.

**Why VIX as a feature?**
VIX is the market's forward-looking fear gauge. High VIX = high volatility regime
where direction prediction is harder. Low VIX = trending market. Including VIX
level and regime flags (above 20, above 30) lets the model weight signals
differently based on market conditions.

---

## Reproducibility

- Data downloaded via `yfinance`, cached to `market_data_cache.pkl`
- Cache validated for required tickers on every load
- VIX downloaded separately (avoids yfinance column naming issues with `^` prefix)
- Seeds fixed: `np.random.seed(42)`, `torch.manual_seed(42)`
- No external CSV files required — fully self-contained

---

## Limitations & Interview Notes

- **No transaction costs or slippage modelled** — realistic Sharpe ~30-40% lower
- **Threshold tuned on validation** — slight optimism in HighConf/Consensus results
- **Past performance** — walk-forward reduces but does not eliminate backtest bias
- **Coverage tradeoff** — consensus models trade 29-37% of days; must be willing to sit flat
- **AAPL hardest to predict** — single-stock idiosyncratic risk dominates systematic signals
