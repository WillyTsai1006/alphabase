import yfinance as yf
import pandas as pd
from sqlalchemy import text
import time
from datetime import datetime, timedelta
# 導入共用配置與工具
from config import TARGET_SYMBOLS
from utils import get_logger, db_manager

logger = get_logger("DataLoader")
DEFAULT_START_DATE = "2016-01-01"

def init_db_schema():
    """自動建立主鍵、索引與特徵視圖 (IaC)"""
    logger.info("🛠️ 正在初始化與優化資料庫架構...")
    sql_statements = [
        "CREATE TABLE IF NOT EXISTS symbols (symbol VARCHAR(20) PRIMARY KEY, asset_type VARCHAR(20), is_active BOOLEAN);",
        "CREATE TABLE IF NOT EXISTS market_data (time TIMESTAMP WITHOUT TIME ZONE, symbol VARCHAR(20), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume BIGINT);",
        "DO $$ BEGIN IF NOT EXISTS (SELECT constraint_name FROM information_schema.table_constraints WHERE table_name = 'market_data' AND constraint_type = 'PRIMARY KEY') THEN ALTER TABLE market_data ADD PRIMARY KEY (time, symbol); END IF; END $$;",
        "CREATE INDEX IF NOT EXISTS idx_market_data_time ON market_data (time DESC);",
        "CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol);",
        """
        CREATE OR REPLACE VIEW features_view AS
        WITH base_data AS (
            SELECT time, symbol, close,
            LN(close / NULLIF(LAG(close, 1) OVER (PARTITION BY symbol ORDER BY time), 0)) as log_return,
            AVG(close) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma_20,
            STDDEV(close) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as std_20,
            close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY time) as price_diff
            FROM market_data
        ),
        rsi_data AS (
            SELECT *,
            ma_20 + (2 * std_20) as bollinger_upper, ma_20 - (2 * std_20) as bollinger_lower,
            AVG(CASE WHEN price_diff > 0 THEN price_diff ELSE 0 END) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as avg_gain,
            AVG(CASE WHEN price_diff < 0 THEN ABS(price_diff) ELSE 0 END) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as avg_loss
            FROM base_data
        )
        SELECT time, symbol, close, log_return, ma_20, bollinger_upper, bollinger_lower,
        100.0 - (100.0 / (1.0 + (avg_gain / NULLIF(avg_loss, 0)))) as rsi_14
        FROM rsi_data;
        """
    ]
    with db_manager.engine.connect() as conn:
        for sql in sql_statements:
            conn.execute(text(sql))
        conn.commit()
    logger.info("✅ 資料庫架構與視圖 (View) 初始化成功！")

def init_symbols():
    """將配置檔中的目標股票寫入資料庫"""
    with db_manager.engine.connect() as conn:
        existing = {row[0] for row in conn.execute(text("SELECT symbol FROM symbols"))}
    new_symbols = [t for t in TARGET_SYMBOLS if t not in existing]
    if new_symbols:
        pd.DataFrame([{'symbol': t, 'asset_type': 'Stock', 'is_active': True} for t in new_symbols]).to_sql('symbols', db_manager.engine, if_exists='append', index=False)
        logger.info(f"✅ 新增 {len(new_symbols)} 檔股票至 symbols 表")

def fetch_incremental(symbol):
    """增量下載邏輯"""
    logger.info(f"🔍 檢查 {symbol} 的數據狀態...")
    with db_manager.engine.connect() as conn:
        last_date = conn.execute(text("SELECT MAX(time) FROM market_data WHERE symbol = :symbol"), {"symbol": symbol}).scalar()
    start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else DEFAULT_START_DATE
    if start_date >= datetime.now().strftime('%Y-%m-%d'):
        return True
    df = yf.download(symbol, start=start_date, progress=False)
    if df.empty: return True
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.rename(columns={k: v for k, v in {'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}.items() if k in df.columns}, inplace=True)
    if df['time'].dt.tz is not None: df['time'] = df['time'].dt.tz_localize(None)
    df['symbol'] = symbol
    df = df[[col for col in ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol'] if col in df.columns]]
    df = df[df['volume'] > 0]
    if last_date:
        df = df[df['time'] > pd.to_datetime(last_date)]
    if not df.empty:
        df.to_sql('market_data', db_manager.engine, if_exists='append', index=False, chunksize=2000)
        logger.info(f"✅ 成功寫入 {len(df)} 筆 {symbol} 新數據。")
    else:
        logger.info(f"ℹ️ {symbol} 過濾後無新數據需寫入。")
    return True

if __name__ == "__main__":
    init_db_schema()
    init_symbols()
    success = sum(1 for ticker in TARGET_SYMBOLS if fetch_incremental(ticker))
    logger.info(f"🎉 數據更新任務完成! 成功/總數: {success}/{len(TARGET_SYMBOLS)}")