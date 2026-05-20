import pandas as pd
import numpy as np
import warnings
import joblib
from datetime import timedelta
from sqlalchemy import text
# 導入共用配置與工具
from config import BACKTEST_PARAMS, BENCHMARK_SYMBOL, MODEL_PATHS, FEATURES
from utils import get_logger, db_manager
warnings.filterwarnings('ignore')
logger = get_logger("Backtester_V3")

class InstitutionalBacktester:
    """整合 HMM 風控、LightGBM 雙重校準 (Meta-Labeling) 與正宗凱利公式的終極回測引擎"""
    def __init__(self, data_df, primary_model, meta_model, hmm_model_data, meta_threshold=None):
        self.data = data_df.copy()
        self.primary_model = primary_model
        if isinstance(meta_model, dict) and 'model' in meta_model:
            meta_threshold = meta_model.get('threshold', meta_threshold)
            meta_model = meta_model['model']
        self.meta_model = meta_model
        self.features = FEATURES
        self.params = BACKTEST_PARAMS.copy()
        self.meta_threshold = BACKTEST_PARAMS['meta_threshold'] if meta_threshold is None else meta_threshold
        # HMM 總經模型設定
        self.hmm_model = hmm_model_data['model']
        self.hmm_map = hmm_model_data['map']
        self.crash_regime = [k for k, v in self.hmm_map.items() if 'Crash' in v][0]
        # 狀態追蹤
        self.cash = self.params['initial_capital']
        self.positions, self.entry_prices = {}, {}
        self.stop_losses, self.take_profits, self.days_held = {}, {}, {}
        self.trade_log, self.equity_curve = [], []

    def load_spy_data(self):
        """讀取 SPY 並計算每日的 HMM 狀態"""
        logger.info("📥 載入 SPY 數據並預測歷史大盤狀態...")
        query = """
        SELECT time, close, LN(close / NULLIF(LAG(close, 1) OVER (ORDER BY time), 0)) as log_return
        FROM market_data
        WHERE symbol = :symbol
        """
        spy = pd.read_sql(text(query), db_manager.engine, params={'symbol': BENCHMARK_SYMBOL})
        spy['time'] = pd.to_datetime(spy['time'])
        spy['volatility_5d'] = spy['log_return'].rolling(window=5).std()
        spy = spy.dropna().set_index('time')
        spy['regime'] = self.hmm_model.predict(spy[['log_return', 'volatility_5d']].values)
        return spy

    def generate_signals(self):
        logger.info("🤖 [階段一] 主模型掃描全歷史訊號...")
        self.data['primary_prob'] = self.primary_model.predict(self.data[self.features].fillna(0))
        logger.info("🧠 [階段二] 第二大腦 (Meta-Model) 進行勝率校準...")
        # 準備 Meta-Model 需要的特徵：原本的特徵 + 主模型信心度
        meta_features = self.features + ['primary_prob']
        self.data['meta_prob'] = self.meta_model.predict(self.data[meta_features].fillna(0))
        return self.data

    def run_backtest(self, max_positions=3):
        logger.info("📈 開始執行 [V3.0 雙重 AI + HMM + 真實凱利公式] 回測...")
        spy_df = self.load_spy_data()
        daily_groups = self.data.groupby('time')
        prev_signals = {}
        # 計算賠率 b (Profit-to-Loss Ratio)，設定 tp=4, sl=2，賠率約為 2:1
        b = self.params['tp_mult'] / self.params['sl_mult'] 
        for current_time, group in daily_groups:
            daily_data = group.set_index('symbol')
            current_syms = daily_data.index.tolist()
            today_spy = spy_df.loc[current_time] if current_time in spy_df.index else None
            is_crash = (today_spy is not None and today_spy['regime'] == self.crash_regime)
            # 1. T+1 進場執行
            if current_time in prev_signals:
                # Meta-Model 的信心度 (meta_prob) 來排序，找出勝率最高的標的
                sig_df = prev_signals[current_time].sort_values('meta_prob', ascending=False)
                avail_slots = 0 if is_crash else max_positions - len(self.positions)
                for _, sig in sig_df.iterrows():
                    if avail_slots <= 0: break
                    sym = sig['symbol']
                    if sym in self.positions or sym not in current_syms: continue
                    exec_price = daily_data.loc[sym]['open'] * (1 + self.params['slippage'])
                    curr_equity = self.cash + sum([self.positions[s] * daily_data.loc[s]['close'] for s in self.positions if s in current_syms])
                    # 正宗凱利公式 (Kelly Criterion)
                    # f* = p - (1-p)/b 
                    # p = Meta-Model 算出的精準勝率, b = 賠率 (約 2.0)
                    p = sig['meta_prob']
                    kelly_f = p - (1 - p) / b
                    # 安全機制：使用半凱利 (Half-Kelly) 降低波動，並設定部位上限 30%，小於 0% 則過濾不買
                    kelly_fraction = max(0, min(kelly_f * 0.5, 0.30))
                    # 只有凱利算出來大於 5% 資金的才值得買 (過濾雜訊)
                    if kelly_fraction > 0.05:
                        shares = int(min(curr_equity * kelly_fraction, self.cash) / (exec_price * (1 + self.params['tc'])))
                        if shares > 0:
                            self.cash -= shares * exec_price * (1 + self.params['tc'])
                            self.positions[sym], self.entry_prices[sym], self.days_held[sym] = shares, exec_price, 0
                            vol = sig['volatility']
                            self.stop_losses[sym] = exec_price * (1 - vol * self.params['sl_mult'])
                            self.take_profits[sym] = exec_price * (1 + vol * self.params['tp_mult'])
                            avail_slots -= 1
            # 2. 持倉監控與出場 
            closed_syms = []
            for sym, shares in self.positions.items():
                if sym in current_syms:
                    row = daily_data.loc[sym]
                    exit_p, reason = None, ""
                    dynamic_sl = self.stop_losses[sym] * 1.01 if is_crash else self.stop_losses[sym]
                    if row['open'] <= dynamic_sl: exit_p, reason = row['open'] * (1 - self.params['slippage']), "Gap Down SL"
                    elif row['low'] <= dynamic_sl: exit_p, reason = dynamic_sl * (1 - self.params['slippage']), "Stop Loss"
                    elif row['high'] >= self.take_profits[sym]: exit_p, reason = self.take_profits[sym] * (1 - self.params['slippage']), "Take Profit"
                    elif self.days_held[sym] >= self.params['horizon_days']: exit_p, reason = row['close'] * (1 - self.params['slippage']), "Time Exit"
                    if exit_p:
                        self.cash += shares * exit_p * (1 - self.params['tc'])
                        self.trade_log.append({'symbol': sym, 'exit_date': current_time, 'pnl': (exit_p - self.entry_prices[sym]) / self.entry_prices[sym], 'reason': reason})
                        closed_syms.append(sym)
                    else: self.days_held[sym] += 1
            for s in closed_syms: [d.pop(s) for d in (self.positions, self.entry_prices, self.stop_losses, self.take_profits, self.days_held)]
            # 3. 收集今日信號
            # 主模型篩選第一層 (>=threshold)
            primary_pass = daily_data[daily_data['primary_prob'] >= self.params['threshold']].reset_index()
            # Meta-Model 篩選第二層：使用訓練時保存的最佳門檻
            if not primary_pass.empty:
                meta_pass = primary_pass[primary_pass['meta_prob'] >= self.meta_threshold]
                if not meta_pass.empty:
                    prev_signals[current_time + timedelta(days=1)] = meta_pass
            # 4. 記錄資產
            self.equity_curve.append({'time': current_time, 'equity': self.cash + sum([self.positions[s] * daily_data.loc[s]['close'] for s in self.positions if s in current_syms])})
        self.equity_df = pd.DataFrame(self.equity_curve).set_index('time')
        self.trades_df = pd.DataFrame(self.trade_log)

if __name__ == "__main__":
    from quant_engine import DataAndLabelEngine
    from config import TARGET_SYMBOLS
    # 載入資料與三個模型 (LGBM, Meta-LGBM, HMM)
    df = DataAndLabelEngine.load_data(TARGET_SYMBOLS)
    df['volatility'] = df.groupby(level='symbol')['close'].pct_change().ewm(span=100).std()
    df = df.dropna().reset_index()
    lgbm_model = joblib.load(MODEL_PATHS['lgbm'])
    from meta_engine import load_meta_model
    meta_model, meta_threshold = load_meta_model(MODEL_PATHS['meta'])
    hmm_data = joblib.load(MODEL_PATHS['hmm'])
    bt = InstitutionalBacktester(df, lgbm_model, meta_model, hmm_data, meta_threshold=meta_threshold)
    bt.generate_signals()
    bt.run_backtest()
    logger.info(f"✅ V3.0 回測完成！最終總資產: ${bt.equity_df['equity'].iloc[-1]:,.2f}")