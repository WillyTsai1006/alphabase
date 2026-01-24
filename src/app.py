import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sqlalchemy import text
st.set_page_config(page_title="AlphaBase Quant V3.0", page_icon="📈", layout="wide")
from config import MODEL_PATHS, BACKTEST_PARAMS
from utils import get_logger, db_manager
from quant_engine import DataAndLabelEngine
from backtester import InstitutionalBacktester
logger = get_logger("StreamlitApp")

@st.cache_data(ttl=3600)
def get_available_symbols():
    query = "SELECT DISTINCT symbol FROM market_data ORDER BY symbol"
    with db_manager.engine.connect() as conn:
        return [row[0] for row in conn.execute(text(query))]
available_symbols = get_available_symbols()

@st.cache_resource(show_spinner=False)
def load_models_and_data(selected_symbols):
    df = DataAndLabelEngine.load_data(selected_symbols)
    df['volatility'] = df.groupby(level='symbol')['close'].pct_change().ewm(span=100).std()
    df = df.dropna().reset_index()
    # [V3.0 升級] 載入三重大腦：主模型、次模型、HMM 總經模型
    lgbm_model = joblib.load(MODEL_PATHS['lgbm'])
    meta_model = joblib.load(MODEL_PATHS['meta'])
    hmm_data = joblib.load(MODEL_PATHS['hmm'])
    return df, lgbm_model, meta_model, hmm_data

@st.cache_data(show_spinner=False)
def run_backtest_cached(selected_symbols, threshold, sl_mult, tp_mult):
    df, lgbm_model, meta_model, hmm_data = load_models_and_data(tuple(selected_symbols))
    # [V3.0 升級] 傳入 meta_model
    bt = InstitutionalBacktester(df, lgbm_model, meta_model, hmm_data)
    bt.params['threshold'] = threshold
    bt.params['sl_mult'] = sl_mult
    bt.params['tp_mult'] = tp_mult
    bt.generate_signals()
    bt.run_backtest(max_positions=3)
    return bt.equity_df, bt.trades_df
# UI 介面
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
st.sidebar.title("AlphaBase V3.0 控制")
st.sidebar.subheader("🎯 選擇投資組合")
default_selections = available_symbols[:3] if len(available_symbols) >= 3 else available_symbols
selected_symbols = st.sidebar.multiselect("股票池", options=available_symbols, default=default_selections)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 模型微調")
threshold_val = st.sidebar.slider("主模型初篩門檻", 0.50, 0.70, BACKTEST_PARAMS['threshold'], 0.01)
st.sidebar.markdown("---")
st.sidebar.success("✅ **Meta-Labeling 次模型**: 啟動中\n\n✅ **HMM 崩盤防禦**: 啟動中\n\n✅ **半凱利公式 (Half-Kelly)**: 啟動中")
st.title("AlphaBase 量化戰情室 V3.0 📊")
st.markdown("全球頂級對沖基金架構：**主模型找機會 ➜ Meta 模型算勝率 ➜ 凱利公式定注碼 ➜ HMM 避股災**")
if not selected_symbols: st.stop()
with st.spinner('🚀 正在運行雙重 AI 與凱利動態回測...'):
    equity_df, trades_df = run_backtest_cached(selected_symbols, threshold_val, BACKTEST_PARAMS['sl_mult'], BACKTEST_PARAMS['tp_mult'])
if equity_df.empty or trades_df.empty: st.stop()
final_cap = equity_df['equity'].iloc[-1]
total_ret = (final_cap / BACKTEST_PARAMS['initial_capital']) - 1
cum_max = equity_df['equity'].cummax()
max_dd = ((equity_df['equity'] - cum_max) / cum_max).min()
wins = trades_df[trades_df['pnl'] > 0]
win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
col1, col2, col3, col4 = st.columns(4)
col1.metric("最終資產", f"${final_cap:,.0f}", f"{total_ret:.2%}")
col2.metric("最大回撤", f"{max_dd:.2%}", delta_color="inverse")
col3.metric("Meta 校準勝率", f"{win_rate:.2%}")
col4.metric("總交易次數", f"{len(trades_df)} 次")
st.markdown("---")
fig_equity = px.line(equity_df, x=equity_df.index, y='equity', title=f'極致複利：凱利公式資金曲線 ({", ".join(selected_symbols)})', template='plotly_dark')
st.plotly_chart(fig_equity, use_container_width=True)
st.caption("AlphaBase V3.0 | 雙重 AI 機器學習架構 | Tainan Local Time: 03:00 AM")