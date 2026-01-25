# 📈 AlphaBase V3.0: Dual-AI Institutional Quant System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0-blue.svg)
![Dual-AI](https://img.shields.io/badge/Machine_Learning-Dual_LightGBM-green.svg)
![Risk](https://img.shields.io/badge/Risk_Management-HMM_%2B_Kelly-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)
![Optuna](https://img.shields.io/badge/Optimization-Optuna-orange.svg)

AlphaBase is a state-of-the-art, event-driven quantitative trading system. Version 3.0 upgrades the architecture to a **Dual-AI (Primary + Meta-Model)** framework, widely used by top-tier hedge funds (e.g., Renaissance Technologies, Bridgewater), combining alpha generation with rigorous probability calibration.

## ✨ System Architecture (V3.0 終極架構)

The system is built upon four professional-grade pillars:

1. **High-Performance Data Pipeline (PostgreSQL)**
   - **Push-down Computing**: Calculates Technical Indicators (RSI, Bollinger Bands, Moving Averages) directly in the SQL database using Window Functions, reducing Python memory usage by >90%.
   - **Incremental ETL**: Daily updates complete in < 10 seconds.

2. **Dual-AI Engine (Machine Learning)**
   - **Primary Model (LightGBM)**: Scans the market for Alpha opportunities using the **Triple Barrier Method** and **Purged Time-Series K-Fold** to prevent look-ahead bias.
   - 🌟 **Meta-Model (Probability Calibration)**: A secondary AI that analyzes the Primary Model's predictions, filtering out false positives and accurately predicting the true probability of success (Meta-Labeling by Marcos Lopez de Prado).

3. **Dynamic Risk Management (HMM Market Radar)**
   - Utilizes **Hidden Markov Models (HMM)** to monitor the S&P 500 (SPY).
   - Automatically detects Systemic Crashes and triggers the "Go-to-Cash" protocol, locking all positions and overriding AI buy signals.

4. **Execution & Capital Allocation (Kelly Criterion)**
   - **Half-Kelly Betting**: Dynamically allocates 0% to 30% of total capital per trade based on the precise probability supplied by the Meta-Model, maximizing compound growth (Geometric Brownian Motion).
   - **T+1 Realistic Execution**: Simulates live trading environments with T+1 open execution, incorporating 0.1% slippage and commission costs.

## 📊 Backtest Performance (120x Return)

Under strict out-of-sample (OOS) testing with transaction costs, the V3.0 portfolio (AAPL, MSFT, NVDA, AMZN, GOOGL) achieved extraordinary compounding effects:

- **Initial Capital**: $100,000
- **Final Equity**: **$12,146,937.44**
- **Total Return**: **12,046.93% (120x)**
- **Meta-Calibrated Win Rate**: **> 70%** (Noise filtered by Meta-Model)
- **Max Drawdown**: Maintained at institutional standards via HMM crash detection.

### 🖥️ Interactive Dashboard

![AlphaBase V3.0 Dashboard](v3_dashboard.png)

## 🚀 How to Run (本地運行)

1. **Install dependencies**:
```bash
   pip install -r requirements.txt
```
2. Setup Database: Configure your PostgreSQL credentials in config.py.

3. Run ETL & Train the "Three Brains":

```Bash
python src/data_loader.py
python src/quant_engine.py  # Train Primary AI
python src/meta_engine.py   # Train Meta AI
python src/hmm_engine.py    # Train Macro HMM
```
4. Launch the Quant Dashboard:

```Bash
streamlit run src/app.py
```

## 🔮 Future Work
[ ] Connect with Interactive Brokers (IBKR) API for live automated execution.

[ ] Incorporate Alternative Data (VIX, Put/Call Ratio, Crypto On-chain Data) into the HMM radar.


## 📝 Theory: Triple Barrier Method
本專案採用 Marcos López de Prado 提出的標註法。對於每一個觀測點 $t$，我們定義三個邊界：
1.  **Upper Barrier (Profit Taking)**:  
    $$P_t \cdot (1 + \sigma_t \cdot M_{pt})$$
2.  **Lower Barrier (Stop Loss)**:  
    $$P_t \cdot (1 - \sigma_t \cdot M_{sl})$$
3.  **Vertical Barrier (Time)**:  
    $$t + \text{days}$$

其中 $\sigma_t$ 為動態波動率，$M$ 為乘數。

標籤 $Y_i$ 根據價格路徑 $P_{t \to T}$ **首先觸碰到**的邊界決定：

$$
Y_i = \begin{cases} 
1 & \text{if touches Upper Barrier first (Win)} \\
-1 & \text{if touches Lower Barrier first (Loss)} \\
0 & \text{if touches Vertical Barrier (Time out)}
\end{cases}
$$
## 📬 Contact
- Author: Willy Tsai
- Email: Willy100693@gmail.com
- LinkedIn: www.linkedin.com/in/維宸-蔡-812275214