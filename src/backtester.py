import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings

from config import BACKTEST_PARAMS, BENCHMARK_SYMBOL
from utils import get_logger, db_manager
import joblib

warnings.filterwarnings('ignore')
logger = get_logger("Backtester")

class InstitutionalBacktester:
    """整合 HMM 風控與凱利資金控管的機構級回測引擎"""
    def __init__(self, data_df, model, features, hmm_model_data):
        self.data = data_df.copy()
        self.model = model
        self.features = features
        self.params = BACKTEST_PARAMS
        # HMM 總經模型
        self.hmm_model = hmm_model_data['model']
        self.hmm_map = hmm_model_data['map']
        # 找出代表崩盤 (Crash) 的 Regime 代號
        self.crash_regime = [k for k, v in self.hmm_map.items() if 'Crash' in v][0]
        # 狀態追蹤
        self.cash = self.params['initial_capital']
        self.positions, self.entry_prices = {}, {}
        self.stop_losses, self.take_profits, self.days_held = {}, {}, {}
        self.trade_log, self.equity_curve = [], []

    def load_spy_data(self):
        """讀取 SPY 並計算每日的 HMM 狀態"""
        logger.info("📥 載入 SPY 數據並預測歷史大盤狀態...")
        query = f"SELECT time, close, LN(close / NULLIF(LAG(close, 1) OVER (ORDER BY time), 0)) as log_return FROM market_data WHERE symbol = '{BENCHMARK_SYMBOL}'"
        spy = pd.read_sql(query, db_manager.engine)
        spy['time'] = pd.to_datetime(spy['time'])
        spy['volatility_5d'] = spy['log_return'].rolling(window=5).std()
        spy = spy.dropna().set_index('time')
        # 預測大盤狀態
        spy['regime'] = self.hmm_model.predict(spy[['log_return', 'volatility_5d']].values)
        return spy

    def generate_signals(self):
        logger.info("🤖 AI 正在掃描全歷史 Alpha 信號...")
        self.data['prob'] = self.model.predict(self.data[self.features].fillna(0))
        return self.data

    def run_backtest(self, max_positions=3):
        logger.info("📈 開始執行 [HMM 防禦] + [凱利資金控管] 的 T+1 回測...")
        spy_df = self.load_spy_data()
        daily_groups = self.data.groupby('time')
        prev_signals = {}
        for current_time, group in daily_groups:
            daily_data = group.set_index('symbol')
            current_syms = daily_data.index.tolist()
            # 0. 獲取今日大盤氣象 
            today_spy = spy_df.loc[current_time] if current_time in spy_df.index else None
            is_crash = (today_spy is not None and today_spy['regime'] == self.crash_regime)
            # 1. T+1 進場執行 (HMM & Kelly 邏輯) 
            if current_time in prev_signals:
                sig_df = prev_signals[current_time].sort_values('prob', ascending=False)
                avail_slots = max_positions - len(self.positions)
                # HMM 防禦：大盤崩盤時，剝奪 AI 買入權力
                if is_crash:
                    avail_slots = 0 # 鎖死倉位，現金為王
                for _, sig in sig_df.iterrows():
                    if avail_slots <= 0: break
                    sym = sig['symbol']
                    if sym in self.positions or sym not in current_syms: continue
                    exec_price = daily_data.loc[sym]['open'] * (1 + self.params['slippage'])
                    curr_equity = self.cash + sum([self.positions[s] * daily_data.loc[s]['close'] for s in self.positions if s in current_syms])
                    # 凱利公式改良：根據 AI 信心度調整資金
                    # 信心越高，買越多 (範圍 10% ~ 25% 之間)
                    prob = sig['prob']
                    kelly_fraction = 0.10 + (prob - self.params['threshold']) * 1.5 
                    kelly_fraction = min(max(kelly_fraction, 0.10), 0.25) # 限制最大 25%，最小 10%
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
                    # 若身處崩盤市，所有持倉的止損位自動上移 (收緊)
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
            todays_sigs = daily_data[daily_data['prob'] > self.params['threshold']].reset_index()
            if not todays_sigs.empty: prev_signals[current_time + timedelta(days=1)] = todays_sigs
            # 4. 記錄資產
            self.equity_curve.append({'time': current_time, 'equity': self.cash + sum([self.positions[s] * daily_data.loc[s]['close'] for s in self.positions if s in current_syms])})
        self.equity_df = pd.DataFrame(self.equity_curve).set_index('time')
        self.trades_df = pd.DataFrame(self.trade_log)

if __name__ == "__main__":
    from quant_engine import DataAndLabelEngine
    from config import TARGET_SYMBOLS, FEATURES, MODEL_PATHS
    # 確保已經有 SPY 的 HMM 模型，沒有的話請先跑 python src/hmm_engine.py
    try:
        hmm_data = joblib.load(MODEL_PATHS['hmm'])
        lgbm_model = joblib.load(MODEL_PATHS['lgbm'])
    except FileNotFoundError:
        logger.error("❌ 找不到模型檔案！請確保已經執行 quant_engine.py 和 hmm_engine.py")
        exit()
    df = DataAndLabelEngine.load_data(TARGET_SYMBOLS)
    df['volatility'] = df.groupby(level='symbol')['close'].pct_change().ewm(span=100).std()
    df = df.dropna().reset_index()
    bt = InstitutionalBacktester(df, lgbm_model, FEATURES, hmm_data)
    bt.generate_signals()
    bt.run_backtest()
    logger.info(f"✅ 回測完成！最終總資產: ${bt.equity_df['equity'].iloc[-1]:,.2f}")