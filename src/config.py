# config.py
import os

# 1. 基礎設施配置 (Infrastructure)
# 建議未來改用 os.getenv('DB_PASSWORD') 讀取環境變數，目前先集中管理
DB_CONFIG = {
    'user': 'quant',
    'password': 'password', # 請替換為你的密碼
    'host': 'localhost',
    'port': '5432',
    'dbname': 'alphabase'
}
DB_URI = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
# 2. 交易標的與回測配置 (Trading & Backtest)
TARGET_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'SPY']
BENCHMARK_SYMBOL = 'SPY'
BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'tc': 0.001,           # 0.1% 手續費
    'slippage': 0.001,     # 0.1% 滑價
    'sl_mult': 2.0,        # 止損倍數
    'tp_mult': 4.0,        # 止盈倍數
    'horizon_days': 5,     # 持倉天數
    'threshold': 0.55      # 買入信心門檻
}
# 3. AI 模型特徵 (Features)
FEATURES = ['rsi_14', 'bollinger_upper', 'bollinger_lower', 'log_return', 'ma_20', 'volatility']
# 模型存檔路徑
MODEL_PATHS = {
    'lgbm': 'alphabase_lgbm.pkl',
    'hmm': 'hmm_model.pkl', # 預留給未來的 HMM 模型
    'meta': 'alphabase_meta.pkl'
}