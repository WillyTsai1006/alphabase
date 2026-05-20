import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from pathlib import Path
# 導入共用配置與工具
from config import TARGET_SYMBOLS, FEATURES, MODEL_PATHS, BACKTEST_PARAMS, RESEARCH_CONFIG
from utils import get_logger
from quant_engine import DataAndLabelEngine
from research import compute_calibration_report, is_kelly_sizing_approved, load_primary_oos_predictions
warnings.filterwarnings('ignore')
logger = get_logger("MetaEngine")

class CalibratedMetaModel:
    """Wrap the raw meta model with an optional probability calibrator."""
    def __init__(self, model, calibrator=None):
        self.model = model
        self.calibrator = calibrator

    def predict(self, X):
        raw_prob = np.asarray(self.model.predict(X), dtype=float)
        if self.calibrator is None:
            return raw_prob
        if hasattr(self.calibrator, 'predict_proba'):
            return np.asarray(self.calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1], dtype=float)
        return np.asarray(self.calibrator.predict(raw_prob), dtype=float)

def save_meta_model(model, threshold, path=MODEL_PATHS['meta'], calibration_report=None, kelly_sizing_approved=False):
    """保存 Meta 模型與最佳過濾門檻，避免訓練結果只留在 log。"""
    raw_model = model.model if isinstance(model, CalibratedMetaModel) else model
    calibrator = model.calibrator if isinstance(model, CalibratedMetaModel) else getattr(model, 'calibrator', None)
    artifact = {
        'model': raw_model,
        'threshold': float(threshold),
        'features': FEATURES + ['primary_prob'],
        'base_threshold': BACKTEST_PARAMS['threshold'],
        'calibration_report': calibration_report,
        'kelly_sizing_approved': bool(kelly_sizing_approved),
        'calibrator': calibrator,
    }
    joblib.dump(artifact, path)
    if path == MODEL_PATHS['meta']:
        research_path = RESEARCH_CONFIG['artifact_paths']['meta_model']
        Path(research_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, research_path)

def load_meta_artifact(path=MODEL_PATHS['meta']):
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and 'model' in artifact:
        if artifact.get('calibrator') is not None and not isinstance(artifact['model'], CalibratedMetaModel):
            artifact['model'] = CalibratedMetaModel(artifact['model'], artifact['calibrator'])
        return artifact
    return {
        'model': artifact,
        'threshold': BACKTEST_PARAMS['meta_threshold'],
        'features': FEATURES + ['primary_prob'],
        'base_threshold': BACKTEST_PARAMS['threshold'],
        'calibration_report': None,
        'kelly_sizing_approved': False,
    }

def load_meta_model(path=MODEL_PATHS['meta']):
    """載入新版 artifact；舊版純模型 pkl 會使用預設門檻。"""
    artifact = load_meta_artifact(path)
    return artifact['model'], artifact.get('threshold', BACKTEST_PARAMS['meta_threshold'])

class MetaLabelingEngine:
    """元標註 (Meta-Labeling) 訓練引擎"""
    def __init__(self):
        # 載入主模型 (Primary Model)
        self.primary_model = joblib.load(MODEL_PATHS['lgbm'])
        self.base_threshold = BACKTEST_PARAMS['threshold'] # 預設買入信心門檻 (0.55)

    def generate_meta_labels(self, df, primary_predictions=None, require_oos_predictions=True):
        """
        核心邏輯：生成元標註 (Meta-Labels)
        - 1: 主模型預測買入，且真的賺錢 (True Positive)
        - 0: 主模型預測買入，但卻虧損 (False Positive)
        """
        df = df.copy()
        if primary_predictions is None and require_oos_predictions:
            primary_predictions = load_primary_oos_predictions()
        if primary_predictions is not None:
            logger.info("🔍 使用 walk-forward OOS primary predictions 生成 Meta-Labels...")
            pred_df = primary_predictions.copy()
            if isinstance(pred_df.index, pd.MultiIndex):
                pred_df = pred_df.reset_index()
            pred_df['time'] = pd.to_datetime(pred_df['time'])
            base_df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
            df = base_df.drop(columns=['primary_prob'], errors='ignore').merge(
                pred_df[['time', 'symbol', 'primary_prob']],
                on=['time', 'symbol'],
                how='inner',
                validate='one_to_one',
            ).set_index(['time', 'symbol'])
        elif require_oos_predictions:
            raise ValueError("Meta-Labeling 需要 walk-forward OOS primary predictions。")
        else:
            logger.warning("⚠️ 使用樣本內 Primary predict 生成 Meta-Labels，僅供開發測試。")
            if df[FEATURES].isna().any().any():
                raise ValueError("Meta-Labeling 特徵含缺值，請先清理資料。")
            df['primary_prob'] = self.primary_model.predict(df[FEATURES])
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
        if events.empty:
            raise ValueError("Meta-Model 無可訓練事件，請降低主模型門檻或檢查資料。")
        meta_features = FEATURES + ['primary_prob']
        X = events[meta_features]
        y = events['meta_target']
        event_dates = X.index.get_level_values('time').unique().sort_values()
        if len(event_dates) < 5:
            raise ValueError("Meta-Model 事件日期不足，無法建立訓練/調校/評估切分。")
        tune_start = event_dates[int(len(event_dates) * 0.60)]
        eval_start = event_dates[int(len(event_dates) * 0.80)]
        tr_mask = X.index.get_level_values('time') < tune_start
        tune_mask = (X.index.get_level_values('time') >= tune_start) & (X.index.get_level_values('time') < eval_start)
        eval_mask = X.index.get_level_values('time') >= eval_start
        X_train, y_train = X[tr_mask], y[tr_mask]
        X_tune, y_tune = X[tune_mask], y[tune_mask]
        X_eval, y_eval = X[eval_mask], y[eval_mask]
        if X_train.empty or X_tune.empty or X_eval.empty:
            raise ValueError("Meta-Model 訓練/調校/評估切分資料不足，請增加歷史資料。")
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
        raw_tune_preds = np.asarray(meta_lgbm.predict(X_tune), dtype=float)
        raw_eval_preds = np.asarray(meta_lgbm.predict(X_eval), dtype=float)
        calibrator = LogisticRegression(solver='lbfgs')
        calibrator.fit(raw_tune_preds.reshape(-1, 1), y_tune)
        calibrated_meta = CalibratedMetaModel(meta_lgbm, calibrator)
        # 自動尋找最佳過濾門檻；調校集與最終校準報告分離。
        val_preds = calibrated_meta.predict(X_tune)
        eval_preds = calibrated_meta.predict(X_eval)
        calibration = compute_calibration_report(
            y_eval,
            eval_preds,
        )
        kelly_approved = is_kelly_sizing_approved(calibration)
        base_win_rate = y_tune.mean()
        # 校準後的成功機率可能低於 0.5；以分位數搜尋可交易門檻，而不是寫死 0.5+。
        min_events = max(20, int(len(val_preds) * 0.10))
        candidate_thresholds = np.unique(np.quantile(val_preds, np.linspace(0.0, 0.90, 19)))
        best_threshold = float(candidate_thresholds[0])
        best_win_rate = -1.0
        best_event_count = 0
        for thresh in candidate_thresholds:
            filtered_preds = val_preds >= thresh
            event_count = int(filtered_preds.sum())
            if event_count >= min_events:
                filtered_win_rate = precision_score(y_tune, filtered_preds, zero_division=0)
                if (
                    filtered_win_rate > best_win_rate
                    or (filtered_win_rate == best_win_rate and event_count > best_event_count)
                ):
                    best_win_rate = filtered_win_rate
                    best_threshold = float(thresh)
                    best_event_count = event_count
        if best_win_rate < 0:
            best_win_rate = base_win_rate
            best_event_count = len(val_preds)
        logger.info(f"✅ Meta-Model 訓練完成！(最佳過濾門檻: {best_threshold:.2f})")
        logger.info(f"🏆 【V3.1 雙重過濾成效】 OOS 調校勝率從原本的 {base_win_rate:.2%} 提升至 -> {best_win_rate:.2%}，保留 {best_event_count} 筆事件 🚀")
        logger.info(
            "📏 Calibration: "
            f"Brier={calibration['brier_score']:.4f}, "
            f"ECE={calibration['expected_calibration_error']:.4f}, "
            f"Kelly sizing approved={kelly_approved}"
        )
        save_meta_model(calibrated_meta, best_threshold, calibration_report=calibration, kelly_sizing_approved=kelly_approved)
        logger.info("💾 Meta-Model 已保存。")
        return calibrated_meta, best_threshold, calibration

if __name__ == "__main__":
    from config import TARGET_SYMBOLS
    # 1. 載入含原始標籤的資料
    df = DataAndLabelEngine.load_data(TARGET_SYMBOLS)
    df_labeled = DataAndLabelEngine.create_labels(df)
    # 2. 啟動 Meta-Labeling
    engine = MetaLabelingEngine()
    events = engine.generate_meta_labels(df_labeled, primary_predictions=load_primary_oos_predictions())
    engine.train_meta_model(events)