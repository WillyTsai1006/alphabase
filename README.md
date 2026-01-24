# 📈 AlphaBase: Institutional Quant Trading System (V2.1)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-AI_Engine-green.svg)
![HMM](https://img.shields.io/badge/HMM-Risk_Control-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)

AlphaBase is an event-driven, institutional-grade quantitative trading system designed for multi-asset portfolios. It integrates **PostgreSQL** for high-performance data engineering, **LightGBM** for Alpha generation, and **Hidden Markov Models (HMM)** for macro-level risk management.

## ✨ System Architecture (系統架構)

The system adopts a "Push-down Computing" and modularized architecture:

1. **Data Engineering (PostgreSQL + Python)**
   - **Infrastructure as Code (IaC)**: Automated schema initialization, indexing, and primary key constraints.
   - **Incremental ETL**: Daily updates with < 10 seconds execution time.
   - **SQL Feature View**: Utilizes Window Functions to calculate technical indicators (RSI, MA, Bollinger Bands) directly within the database, reducing Python memory consumption by 90%.

2. **AI Engine (Machine Learning)**
   - **Triple Barrier Method**: Path-dependent dynamic labeling based on market volatility.
   - **Purged K-Fold Cross Validation**: Strictly eliminates "Look-ahead Bias" during model evaluation.
   - **Bayesian Optimization**: Optuna-driven hyperparameter tuning.

3. **Execution & Risk Management**
   - **T+1 Realistic Execution**: Signals generated at $T_{close}$ are executed at $T+1_{open}$, including 0.1% slippage and commission costs.
   - **HMM Market Radar**: Identifies market crashes and automatically triggers the "Go-to-Cash" mechanism to protect capital.
   - **Kelly Criterion**: Dynamic position sizing (10% - 25%) based on AI confidence.

## 📊 Backtest Performance (回測績效)

Under rigorous conditions (No Look-ahead Bias, T+1 Execution, Transaction Costs Included), the portfolio composed of tech giants (AAPL, MSFT, NVDA) demonstrated exceptional resilience.

- **Total Return**: 1241.31%
- **Sharpe Ratio**: 2.48
- **Max Drawdown**: -12.70% (Successfully suppressed by HMM logic)
- **Win Rate**: 65.22%

### 🖥️ Interactive Dashboard
![AlphaBase Dashboard Screenshot](./assets/dashboard.png)

## 🚀 How to Run (本地運行)

1. Install dependencies:
```bash
   pip install -r requirements.txt
 ```
2. Configure your PostgreSQL credentials in config.py.

3. Fetch data and train models:
```bash
    python src/data_loader.py
    python src/quant_engine.py
    python src/hmm_engine.py
```
4. Launch the Dashboard:
```bash
streamlit run src/app.py
```

## 🔮 Future Work

[ ] Implement Meta-Labeling for dual-model calibration.

[ ] Connect with Interactive Brokers (IBKR) API for live automated trading.

---
