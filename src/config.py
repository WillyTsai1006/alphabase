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
TRADABLE_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'INTC', 'PYPL', 'PFE', 'ZM']
BENCHMARK_SYMBOL = 'SPY'
TARGET_SYMBOLS = TRADABLE_SYMBOLS + [BENCHMARK_SYMBOL]
BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'tc': 0.001,           # 0.1% 手續費
    'slippage': 0.001,     # 0.1% 滑價
    'sl_mult': 1.0,        # 止損倍數
    'tp_mult': 2.0,        # 止盈倍數
    'horizon_days': 15,    # 持倉天數
    'threshold': 0.55,     # 主模型買入信心門檻
    'meta_threshold': 0.50, # Meta 模型預設過濾門檻
    'kelly_requires_calibration': True,
    'fallback_position_fraction': 0.10,
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
RESEARCH_CONFIG = {
    'research_id': 'alphabase_v3_research_2026_05',
    'data_start': '2016-01-01',
    'data_end': '2025-12-31',
    'train_start': '2016-01-01',
    'walk_forward_start': '2020-01-01',
    'walk_forward_end': '2025-12-31',
    'hmm_train_end': '2022-12-31',
    'hmm_oos_start': '2023-01-01',
    'universe': TRADABLE_SYMBOLS,
    'universe_rule': (
        'Fixed large-cap US equities plus SPY benchmark, selected before the '
        'research run and not changed after seeing backtest results. This is '
        'not a survivorship-free institutional universe.'
    ),
    'data_cleaning_rules': [
        'Keep rows inside the fixed data_start/data_end interval.',
        'Require positive open/high/low/close and non-negative volume.',
        'Require high >= max(open, close) and low <= min(open, close).',
        'Drop duplicate (time, symbol) rows after sorting.',
        'Drop rows with missing engineered features before model training.',
    ],
    'walk_forward': {
        'train_years': 3,
        'test_months': 6,
        'step_months': 6,
        'embargo_days': BACKTEST_PARAMS['horizon_days'],
    },
    'calibration': {
        'n_bins': 10,
        'max_brier_score_for_kelly': 0.20,
        'max_expected_calibration_error_for_kelly': 0.05,
    },
    'quality_gates': {
        'min_primary_mean_auc': 0.55,
        'min_primary_fold_auc': 0.50,
    },
    'artifact_paths': {
        'primary_model': 'artifacts/research/alphabase_v3_research_2026_05/primary_lgbm.pkl',
        'meta_model': 'artifacts/research/alphabase_v3_research_2026_05/meta_lgbm.pkl',
        'hmm_model': 'artifacts/research/alphabase_v3_research_2026_05/hmm_model.pkl',
        'primary_walk_forward_metrics': 'artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_metrics.csv',
        'primary_walk_forward_predictions': 'artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_predictions.csv',
        'ranker_walk_forward_predictions': 'artifacts/research/alphabase_v3_research_2026_05/ranker_walk_forward_predictions.csv',
        'fold_strategy_metrics': 'artifacts/research/alphabase_v3_research_2026_05/fold_strategy_metrics.csv',
        'report': 'docs/research/backtest_report.md',
    },
}