# 📈 Reliance Stock Price Prediction

A machine learning project to predict **Reliance Industries (NSE)** stock movement using 26 years of historical data (2000–2026), three modelling approaches, and a full data pipeline from PostgreSQL to trained models.

---

##  Project Structure

```
Reliance_stock_price_prediction/
│
├── Reliance_price_prediction.ipynb   # Main notebook — all 3 models
├── README.md
└── requirements.txt
```

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Data collection | `yfinance` |
| Storage | PostgreSQL + SQLAlchemy |
| Feature engineering | `pandas`, `numpy` |
| ML models | `Prophet`, `XGBoost`, `Optuna` |
| DL model | `TensorFlow / Keras` (LSTM) |
| Evaluation | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |

---

##  Pipeline Overview

```
yfinance API  →  PostgreSQL  →  Feature Engineering  →  Model Training  →  Evaluation
```

1. Raw OHLCV data downloaded via `yfinance` for `RELIANCE.NS` (2000–2026)
2. Stored in a local PostgreSQL database (`reliance_price` table)
3. 15 technical indicators engineered as features
4. Three models trained on a **strict chronological split** (train ≤ 2024-12-31, test > 2024-12-31)
5. All models evaluated on the same held-out test set

---

##  Feature Engineering

15 features were engineered from raw OHLCV data:

| Category | Features |
|---|---|
| Lag features | `lag_1`, `lag_5`, `lag_10`, `lag_21` |
| Moving averages | `ma_7`, `ma_21`, `ma_50` |
| Volatility | `volatile` (14-day rolling std) |
| Returns | `daily_return`, `weekly_return` |
| Momentum | `RSI` (14-day) |
| Price ratios | `price vs ma_7`, `price vs ma_21`, `price vs ma_50` |
| Volume | `volume_MA7`, `volume_ratio` |

**Target variable:** `pct_change().shift(-1)` — next day's percentage return (not raw price).
Predicting returns instead of price levels avoids autocorrelation inflation of R².

---

##  Models

### 1. Prophet (Baseline)
Facebook's time-series forecasting library. Used as a simple trend baseline with no feature engineering.

- Train: data up to 2024-01-01
- Test: 2024 onwards

### 2. XGBoost (with Optuna tuning)
Gradient boosting on engineered features. Includes:
- Chronological validation split (last 20% of train set)
- Feature importance analysis + dropping low-importance features
- Hyperparameter tuning with **Optuna** (50 trials, `TimeSeriesSplit` CV)

### 3. LSTM (Deep Learning)
Sequence model on 60-day sliding windows of scaled close prices.
- Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16) → Dense(1)
- Chronological validation split (no random shuffling)
- Scaler fitted **only on training data**, applied to test data — no leakage
- Callbacks: `EarlyStopping(patience=25)`, `ReduceLROnPlateau`

---

## 📊 Results

> **Note on metrics:** Target is next-day % return, not raw price.  
> R² on return prediction is a stricter and more honest measure than R² on price levels.  
> A naive baseline (predict 0 return every day) scores R² ≈ 0.

| Model | RMSE | R² | MAPE |
|---|---|---|---|
| Prophet | ₹162.25 | -1.73 | 10.47% |
| XGBoost (tuned) | ₹0.0130 | -0.0042 | - |
| LSTM | ₹39.49 | 0.8557 | 2.28% |

*Run the notebook to see exact figures — results depend on your local data fetch date.*

---

##  What I Learned (Honest Post-Review Notes)

During development and peer review, several issues were identified and fixed:

**1. Random validation split → fixed**
Early versions used `validation_split=0.2` in Keras, which shuffles data randomly and leaks future prices into training. Fixed by using a chronological split (last 20% of training data as validation).

**2. Predicting raw price → fixed**
Early versions predicted raw closing price (₹). Because stock prices are autocorrelated (today ≈ yesterday), any model can score R²=0.90+ by just predicting "tomorrow = today". Fixed by predicting **next-day % return** instead.

**3. Scaler leakage → verified clean**
`MinMaxScaler` is fit only on training data. The scaler is never re-fit on test data.

These fixes mean the reported results are **honest** — lower numbers than inflated versions but actually meaningful.

---

##  How to Run

### Prerequisites
```bash
pip install yfinance prophet xgboost optuna tensorflow scikit-learn sqlalchemy psycopg2 matplotlib seaborn
```

### PostgreSQL setup
1. Create a database: `reliance_price`
2. Update credentials in the notebook (cell 4)
3. Run the data ingestion cell **once** (it's commented out to prevent duplicate inserts)

### Running the notebook
```bash
jupyter notebook Reliance_price_prediction.ipynb
```
Run cells top to bottom. Each section is labelled: **EDA → Prophet → XGBoost → LSTM → Comparison**.

---

##  Data

- **Source:** NSE via `yfinance` (`RELIANCE.NS`)
- **Range:** January 2000 – April 2026
- **Frequency:** Daily (business days)
- **Rows:** ~6,500 trading days
- **Storage:** Local PostgreSQL (not included in repo for size reasons)

---

## 👤 Author

**Kuldip Lakhtariya**  
B.Tech Student | ML & Data Science  
📍 Ahmedabad, Gujarat

[![GitHub](https://img.shields.io/badge/GitHub-Kuldip--Lakhtariya-black)](https://github.com/Kuldip-Lakhtariya)
