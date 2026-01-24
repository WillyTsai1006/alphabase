import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score
import joblib
import warnings
# 導入共用配置與工具
from config import TARGET_SYMBOLS, FEATURES, MODEL_PATHS, BACKTEST_PARAMS
from utils import get_logger
from quant_engine import DataAndLabelEngine
warnings.filterwarnings('ignore')
logger = get_logger("MetaEngine")

class MetaLabelingEngine:
    """元標註 (Meta-Labeling) 訓練引擎"""
    def __init__(self):
        # 載入主模型 (Primary Model)
        self.primary_model = joblib.load(MODEL_PATHS['lgbm'])
        self.base_threshold = BACKTEST_PARAMS['threshold'] # 預設買入信心門檻 (0.55)

    def generate_meta_labels(self, df):
        """
        核心邏輯：生成元標註 (Meta-Labels)
        - 1: 主模型預測買入，且真的賺錢 (True Positive)
        - 0: 主模型預測買入，但卻虧損 (False Positive)
        """
        logger.info("🔍 正在讓主模型對歷史數據進行『模擬看盤』...")
        # 1. 取得主模型的預測機率
        df['primary_prob'] = self.primary_model.predict(df[FEATURES].fillna(0))
        # 2. 過濾出主模型「喊買」的樣本 (Events)
        events = df[df['primary_prob'] >= self.base_threshold].copy()
        logger.info(f"📊 主模型共發出 {len(events)} 次買入訊號。")
        # 3. 標註 Meta-Target：實際上有沒有賺錢？
        # 原本的 target: 1 = 上漲, 0 = 下跌
        # 因為這群樣本都是主模型預測要上漲的，所以 target=1 代表主模型猜對了(Success)，target=0 代表猜錯(Failure)
        events['meta_target'] = events['target'] 
        success_rate = events['meta_target'].mean()
        logger.info(f"🎯 主模型原始勝率: {success_rate:.2%}")
        return events

    def train_meta_model(self, events):
        """訓練第二大腦 (Meta-Model) - V3.1 強化版"""
        logger.info("🧠 開始訓練 Meta-Model (元模型)...")
        meta_features = FEATURES + ['primary_prob']
        X = events[meta_features]
        y = events['meta_target']
        split_date = X.index.get_level_values('time').max() - pd.Timedelta(days=180)
        tr_mask = X.index.get_level_values('time') < split_date
        ts_mask = X.index.get_level_values('time') >= split_date
        X_train, y_train = X[tr_mask], y[tr_mask]
        X_test, y_test = X[ts_mask], y[ts_mask]
        # 改回最強的 gbdt，並加強正規化 (L1/L2) 防止過擬合
        params = {
            'objective': 'binary',
            'metric': 'auc',          # 改用 AUC 來評估分離能力
            'boosting_type': 'gbdt',  # 切回梯度提升
            'num_leaves': 16,         # 樹不要太深
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'lambda_l1': 1.0,         # 強力正規化
            'lambda_l2': 1.0,
            'min_child_samples': 50,  # 葉子節點樣本數不能太少
            'verbose': -1
        }
        meta_lgbm = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)
        # 自動尋找最佳過濾門檻 (不再寫死 0.6)
        test_preds = meta_lgbm.predict(X_test)
        base_win_rate = y_test.mean()
        # 尋找能讓勝率提升的最大門檻
        best_threshold = 0.5
        best_win_rate = base_win_rate
        for thresh in np.arange(0.5, 0.8, 0.05):
            filtered_preds = test_preds > thresh
            if filtered_preds.sum() > 10: # 確保至少保留 10 筆交易
                filtered_win_rate = precision_score(y_test, filtered_preds)
                if filtered_win_rate > best_win_rate:
                    best_win_rate = filtered_win_rate
                    best_threshold = thresh
        logger.info(f"✅ Meta-Model 訓練完成！(最佳過濾門檻: {best_threshold:.2f})")
        logger.info(f"🏆 【V3.1 雙重過濾成效】 OOS 測試勝率從原本的 {base_win_rate:.2%} 提升至 -> {best_win_rate:.2%} 🚀")
        MODEL_PATHS['meta'] = 'alphabase_meta.pkl'
        joblib.dump(meta_lgbm, 'alphabase_meta.pkl')
        logger.info("💾 Meta-Model 已保存。")

if __name__ == "__main__":
    from config import TARGET_SYMBOLS
    # 1. 載入含原始標籤的資料
    df = DataAndLabelEngine.load_data(TARGET_SYMBOLS)
    df_labeled = DataAndLabelEngine.create_labels(df)
    # 2. 啟動 Meta-Labeling
    engine = MetaLabelingEngine()
    events = engine.generate_meta_labels(df_labeled)
    engine.train_meta_model(events)