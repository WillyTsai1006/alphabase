import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

# 1. 基礎設施配置 (Infrastructure)
load_dotenv()

DB_CONFIG = {
    'user': os.getenv('DB_USER', 'quant'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'dbname': os.getenv('DB_NAME', 'alphabase')
}
DB_URI = (
    f"postgresql://{quote_plus(DB_CONFIG['user'])}:"
    f"{quote_plus(DB_CONFIG['password'])}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)
# 2. 交易標的與回測配置 (Trading & Backtest)
TARGET_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'SPY', 'INTC', 'PYPL', 'PFE', 'ZM']
BENCHMARK_SYMBOL = 'SPY'
BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'tc': 0.001,           # 0.1% 手續費
    'slippage': 0.001,     # 0.1% 滑價
    'sl_mult': 2.0,        # 止損倍數
    'tp_mult': 4.0,        # 止盈倍數
    'horizon_days': 5,     # 持倉天數
    'threshold': 0.55,     # 主模型買入信心門檻
    'meta_threshold': 0.50 # Meta 模型預設過濾門檻
}
LABEL_PARAMS = {
    'horizon_days': BACKTEST_PARAMS['horizon_days'],
    'pt_sl': [BACKTEST_PARAMS['tp_mult'], BACKTEST_PARAMS['sl_mult']],
}
# 3. AI 模型特徵 (Features)
FEATURES = ['rsi_14', 'bollinger_upper', 'bollinger_lower', 'log_return', 'ma_20', 'volatility']
# 模型存檔路徑
MODEL_PATHS = {
    'lgbm': 'alphabase_lgbm.pkl',
    'hmm': 'hmm_model.pkl', # 預留給未來的 HMM 模型
    'meta': 'alphabase_meta.pkl'
}