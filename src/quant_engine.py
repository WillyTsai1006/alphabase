import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import warnings
from sqlalchemy import text, bindparam

# 導入共用配置與工具
from config import TARGET_SYMBOLS, FEATURES, MODEL_PATHS, LABEL_PARAMS, RESEARCH_CONFIG
from utils import get_logger, db_manager
from research import build_walk_forward_splits, clean_market_data

warnings.filterwarnings('ignore')
np.random.seed(42)
logger = get_logger("QuantEngine")

class DataAndLabelEngine:
    @staticmethod
    def load_data(symbols):
        """安全加載多資產數據 (防止 SQL 注入)"""
        logger.info(f"📥 正在加載 {len(symbols)} 檔股票數據...")
        if not symbols:
            return pd.DataFrame()
        if all(s.isalnum() for s in symbols):
            query = text("""
            SELECT f.time, f.symbol, f.close, f.log_return, f.ma_20, f.rsi_14, 
                f.bollinger_upper, f.bollinger_lower, m.open, m.high, m.low, m.volume
            FROM features_view f
            JOIN market_data m ON f.time = m.time AND f.symbol = m.symbol
            WHERE f.symbol IN :symbols
            ORDER BY f.time ASC
            """).bindparams(bindparam("symbols", expanding=True))
            df = pd.read_sql(query, db_manager.engine, params={"symbols": tuple(symbols)})
        else:
            raise ValueError("股票代碼包含非法字符")
        df['time'] = pd.to_datetime(df['time'])
        return clean_market_data(df.set_index(['time', 'symbol']))

    @staticmethod
    def create_labels(df, horizon_days=None, pt_sl=None):
        """三重屏障標註法，使用與回測一致的止盈/止損與持倉天數。"""
        horizon_days = LABEL_PARAMS['horizon_days'] if horizon_days is None else horizon_days
        pt_sl = LABEL_PARAMS['pt_sl'] if pt_sl is None else pt_sl
        logger.info(f"🔄 開始計算三重屏障標註 (Horizon: {horizon_days} days)...")
        df['volatility'] = df.groupby(level='symbol')['close'].pct_change().ewm(span=100).std()
        out_df = pd.DataFrame(index=df.index)
        out_df['ret'] = 0
        out_df['exit_time'] = pd.NaT
        for sym in df.index.get_level_values('symbol').unique():
            sub = df.xs(sym, level='symbol')
            c, h, l, v = sub['close'], sub['high'], sub['low'], sub['volatility']
            
            for i, time in enumerate(sub.index):
                if i + horizon_days >= len(sub) or pd.isna(v.iloc[i]): continue
                ub, lb = c.iloc[i] * (1 + pt_sl[0] * v.iloc[i]), c.iloc[i] * (1 - pt_sl[1] * v.iloc[i])
                f_h, f_l = h.iloc[i+1 : i+horizon_days+1], l.iloc[i+1 : i+horizon_days+1]
                
                pt, sl = f_h >= ub, f_l <= lb
                lbl, ext = 0, sub.index[i+horizon_days]
                if pt.any() and sl.any(): lbl, ext = (1, pt.idxmax()) if pt.idxmax() < sl.idxmax() else (0, sl.idxmax())
                elif pt.any(): lbl, ext = 1, pt.idxmax()
                elif sl.any(): lbl, ext = 0, sl.idxmax()
                
                out_df.loc[(time, sym), ['ret', 'exit_time']] = [lbl, ext]
        df['target'], df['exit_time'] = out_df['ret'], out_df['exit_time']
        return df.dropna(subset=['target', 'exit_time'])

class PurgedTimeSeriesSplit:
    """淨化時間序列交叉驗證 (防未來函數)"""
    def __init__(self, n_splits=3): self.n_splits = n_splits
    def split(self, X, exit_times):
        times = X.index.get_level_values('time').unique().sort_values()
        indices = np.array_split(np.arange(len(times)), self.n_splits + 1)
        for i in range(self.n_splits):
            tr_mask = X.index.get_level_values('time').isin(times[np.concatenate(indices[:i+1])])
            vl_mask = X.index.get_level_values('time').isin(times[indices[i+1]])
            tr_idx, vl_idx = np.where(tr_mask)[0], np.where(vl_mask)[0]
            yield tr_idx[exit_times.iloc[tr_idx] < times[indices[i+1]][0]], vl_idx

class ModelTrainer:
    def __init__(self, df, features):
        self.X, self.y, self.exits = df[features], df['target'], df['exit_time']

    def objective(self, trial):
        p = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
             'num_leaves': trial.suggest_int('num_leaves', 16, 128),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
             'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
             'min_child_samples': trial.suggest_int('min_child_samples', 20, 100), 'is_unbalance': True}
        scores = []
        for tr, vl in PurgedTimeSeriesSplit().split(self.X, self.exits):
            m = lgb.train(p, lgb.Dataset(self.X.iloc[tr], label=self.y.iloc[tr]), valid_sets=[lgb.Dataset(self.X.iloc[vl], label=self.y.iloc[vl])], callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(roc_auc_score(self.y.iloc[vl], m.predict(self.X.iloc[vl])))
        return np.mean(scores)

    def train(self, best_params):
        logger.info("🚀 訓練最終模型 (OOS Split)...")
        split = self.X.index.get_level_values('time').max() - pd.Timedelta(days=180)
        tr, ts = self.X.index.get_level_values('time') < split, self.X.index.get_level_values('time') >= split
        m = lgb.train(best_params, lgb.Dataset(self.X[tr], label=self.y[tr]), num_boost_round=500)
        logger.info(f"✅ OOS AUC: {roc_auc_score(self.y[ts], m.predict(self.X[ts])):.4f}")
        joblib.dump(m, MODEL_PATHS['lgbm'])
        research_path = RESEARCH_CONFIG['artifact_paths']['primary_model']
        from pathlib import Path

        Path(research_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, research_path)
        logger.info(f"💾 模型已保存至 {MODEL_PATHS['lgbm']}")
        return m

    def walk_forward_train(self, params, artifact_dir=None, num_boost_round=300):
        """Train one model per fixed rolling window and return OOS predictions."""
        rows, preds = [], []
        dates = self.X.index.get_level_values('time')
        splits = build_walk_forward_splits(dates)
        if not splits:
            raise ValueError("No valid walk-forward splits for the configured research window.")
        if artifact_dir:
            from pathlib import Path

            artifact_dir = Path(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
        for idx, split in enumerate(splits, start=1):
            tr = (
                (dates >= split["train_start"])
                & (dates <= split["train_end"])
                & (pd.to_datetime(self.exits) < split["test_start"])
            )
            ts = (dates >= split["test_start"]) & (dates <= split["test_end"])
            if tr.sum() == 0 or ts.sum() == 0:
                continue
            model = lgb.train(params, lgb.Dataset(self.X[tr], label=self.y[tr]), num_boost_round=num_boost_round)
            fold_pred = model.predict(self.X[ts])
            auc = roc_auc_score(self.y[ts], fold_pred) if self.y[ts].nunique() > 1 else np.nan
            rows.append({**split, "train_rows": int(tr.sum()), "test_rows": int(ts.sum()), "auc": auc})
            fold_df = pd.DataFrame({"target": self.y[ts].values, "primary_prob": fold_pred}, index=self.X[ts].index)
            fold_df["fold"] = idx
            preds.append(fold_df)
            if artifact_dir:
                joblib.dump(model, artifact_dir / f"primary_lgbm_fold_{idx:02d}.pkl")
        return pd.DataFrame(rows), pd.concat(preds).sort_index() if preds else pd.DataFrame()

if __name__ == "__main__":
    df = DataAndLabelEngine.load_data(TARGET_SYMBOLS)
    df_lab = DataAndLabelEngine.create_labels(df)
    trainer = ModelTrainer(df_lab, FEATURES)
    logger.info("🤖 開始 Optuna 優化...")
    study = optuna.create_study(direction='maximize')
    study.optimize(trainer.objective, n_trials=30)
    trainer.train(study.best_params)