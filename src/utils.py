# utils.py
import logging
from sqlalchemy import create_engine
import pandas as pd
from config import DB_URI

# 1. 日誌系統 (Logger) - 解決「日誌不夠詳細」
def get_logger(name):
    """標準化日誌生成器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 輸出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # 未來可以加一個 FileHandler 輸出到 log 檔案
    return logger

# 2. 資料庫引擎 (Database) - 解決「SQL注入與代碼重複」
class DatabaseManager:
    _instance = None
    def __new__(cls):
        """單例模式 (Singleton)，確保整個程式只建立一個資料庫連線"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.engine = create_engine(DB_URI)
            cls._instance.logger = get_logger("Database")
        return cls._instance

    def safe_query(self, query, params=None):
        """安全查詢：使用綁定變數防止 SQL 注入"""
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            self.logger.error(f"SQL Query Failed: {e}")
            return pd.DataFrame()
# 實例化 DB 供其他模組導入
db_manager = DatabaseManager()