import pandas as pd
import numpy as np
import joblib
import warnings
from hmmlearn.hmm import GaussianHMM
from sqlalchemy import text
# 導入 V2 共用配置與工具
from config import BENCHMARK_SYMBOL, MODEL_PATHS, RESEARCH_CONFIG
from utils import get_logger, db_manager
from pathlib import Path
warnings.filterwarnings('ignore')
logger = get_logger("HMMEngine")

class MarketRegimeModel:
    def __init__(self, n_components=3):
        """初始化隱馬爾可夫模型 (HMM)"""
        self.n_components = n_components
        # n_iter: 迭代次數, random_state: 固定隨機種子保證每次訓練結果一致
        self.model = GaussianHMM(n_components=n_components, covariance_type='full', n_iter=1000, random_state=42)
        self.regime_map = {}

    def fetch_spy_data(self):
        """從 PostgreSQL 獲取 SPY (大盤) 數據並計算總經特徵"""
        logger.info(f"📥 正在從資料庫獲取 {BENCHMARK_SYMBOL} (大盤) 數據...")
        # 安全地透過 SQL 獲取收盤價與計算日報酬率
        query = """
        SELECT time, close, 
               LN(close / NULLIF(LAG(close, 1) OVER (ORDER BY time), 0)) as log_return
        FROM market_data 
        WHERE symbol = :symbol
        ORDER BY time ASC
        """
        df = pd.read_sql(text(query), db_manager.engine, params={'symbol': BENCHMARK_SYMBOL})
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        # 特徵工程：計算 5 日波動率
        df['volatility_5d'] = df['log_return'].rolling(window=5).std()
        return df.dropna()

    def train_and_identify(self, df):
        """訓練 HMM 並自動為機器分出的 3 種狀態進行命名"""
        X = df[['log_return', 'volatility_5d']].values
        logger.info(f"🧠 開始訓練 HMM 隱馬爾可夫模型 (樣本數: {len(X)})...")
        self.model.fit(X)
        # 讓機器預測歷史狀態 (0, 1, or 2)
        df['regime'] = self.model.predict(X)
        # 統計每個狀態的「平均報酬率」與「平均波動率」
        stats = df.groupby('regime')[['log_return', 'volatility_5d']].mean()
        # 智慧命名邏輯：Crash 不只看波動，也懲罰正報酬，避免高波動上漲期被誤命名。
        vol_span = stats['volatility_5d'].max() - stats['volatility_5d'].min()
        ret_span = stats['log_return'].max() - stats['log_return'].min()
        vol_score = (stats['volatility_5d'] - stats['volatility_5d'].min()) / (vol_span if vol_span else 1)
        ret_score = (stats['log_return'] - stats['log_return'].min()) / (ret_span if ret_span else 1)
        risk_score = vol_score - ret_score
        crash_regime = risk_score.idxmax()
        # 報酬率最高的狀態 = Bull (牛市)，但不可與 Crash 重疊
        bull_candidates = stats.drop(index=crash_regime)
        bull_regime = bull_candidates['log_return'].idxmax()
        # 剩下的就是 Sideways (震盪市)
        sideways_regime = [r for r in range(self.n_components) if r not in [crash_regime, bull_regime]][0]
        # 建立狀態映射表
        self.regime_map = {
            int(bull_regime): 'Bull (牛市)', 
            int(sideways_regime): 'Sideways (震盪)', 
            int(crash_regime): 'Crash (崩盤/高波動)'
        }
        logger.info("📊 --- HMM 大盤狀態識別報告 ---")
        for r, name in self.regime_map.items():
            logger.info(f"Regime {r} ({name}): 平均報酬 {stats.loc[r, 'log_return']:.4f}, 平均波動 {stats.loc[r, 'volatility_5d']:.4f}")
        return df

    def save_model(self):
        """保存模型與狀態映射表"""
        data = {
            'model': self.model,
            'map': self.regime_map,
            'train_end': RESEARCH_CONFIG['hmm_train_end'],
            'oos_start': RESEARCH_CONFIG['hmm_oos_start'],
        }
        joblib.dump(data, MODEL_PATHS['hmm'])
        research_path = RESEARCH_CONFIG['artifact_paths']['hmm_model']
        Path(research_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, research_path)
        logger.info(f"💾 HMM 模型已保存至 {MODEL_PATHS['hmm']}")

if __name__ == "__main__":
    hmm = MarketRegimeModel()
    spy_df = hmm.fetch_spy_data()
    # 為了防止「未來函數」，我們只用 2023 年以前的歷史數據來訓練大盤規律
    # 這樣回測 2023-2025 年時，HMM 才是用「未知的眼光」在看盤
    train_df = spy_df[spy_df.index <= RESEARCH_CONFIG['hmm_train_end']]
    if len(train_df) > 100:
        hmm.train_and_identify(train_df)
        hmm.save_model()
    else:
        logger.error(f"❌ {BENCHMARK_SYMBOL} 歷史數據不足，無法訓練 HMM。")
        logger.info(f"💡 提示: 請確保 config.py 的 TARGET_SYMBOLS 有包含 '{BENCHMARK_SYMBOL}'，並執行 data_loader.py 抓取數據。")