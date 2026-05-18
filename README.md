#  Reliance Stock Price Prediction

End-to-end machine learning project predicting next-day closing 
price of Reliance Industries (NSE) using 26 years of historical data.

##  Results

| Model           | RMSE   | R²    | MAPE  |
|-----------------|--------|-------|-------|
| Prophet         | ₹162   | 0.596 | 8.2%  |
| XGBoost (tuned) | ₹24    | 0.943 | 1.4%  |
| LSTM            | ₹54    | 0.72  | 3.18% |

**Winner: XGBoost** — engineered technical indicators outperform 
raw sequence learning for daily stock prediction.

##  Tech Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Fetch NSE stock data via Yahoo Finance API |
| `PostgreSQL` | Store and query historical price data |
| `SQLAlchemy` | Python ↔ PostgreSQL connection |
| `XGBoost` | Primary prediction model |
| `Prophet` | Trend forecasting baseline |
| `TensorFlow/Keras` | LSTM deep learning model |
| `Optuna` | Hyperparameter tuning |

## Project Structure

```text
├── data_pipeline.ipynb              # Scrape + store in PostgreSQL
├── Reliance_price_prediction.ipynb  # EDA + all 3 models
├── lstm_stock.ipynb                 # LSTM deep learning model
└── README.md
```

## 📌 Conclusion

XGBoost with engineered technical indicators (R²=0.942) outperforms 
LSTM (R²=0.73), demonstrating that domain knowledge adds more 
predictive signal than raw sequence learning for daily NSE stocks.

## 👤 Author

**Kuldip Lakhtariya**  
B.Tech Student | ML & Data Science  
📍 Ahmedabad, Gujarat

[![GitHub](https://img.shields.io/badge/GitHub-Kuldip--Lakhtariya-black)](https://github.com/Kuldip-Lakhtariya)
