import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sqlalchemy import text
# 頁面基本設定
st.set_page_config(page_title="AlphaBase Quant V2.1", page_icon="📈", layout="wide")
# 導入企業級模組
from config import MODEL_PATHS, BACKTEST_PARAMS
from utils import get_logger, db_manager
from quant_engine import DataAndLabelEngine
from backtester import InstitutionalBacktester
logger = get_logger("StreamlitApp")
# 0. 動態獲取股票清單
@st.cache_data(ttl=3600) # 快取 1 小時
def get_available_symbols():
    """從資料庫獲取目前有數據的股票清單"""
    query = "SELECT DISTINCT symbol FROM market_data ORDER BY symbol"
    with db_manager.engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result]
available_symbols = get_available_symbols()
# 1. 核心運算區 (根據選擇的股票動態載入)
@st.cache_resource(show_spinner=False)
def load_models_and_data(selected_symbols):
    """載入特定股票數據與 AI/HMM 模型"""
    # 根據使用者選擇的股票載入數據
    df = DataAndLabelEngine.load_data(selected_symbols)
    df['volatility'] = df.groupby(level='symbol')['close'].pct_change().ewm(span=100).std()
    df = df.dropna().reset_index()
    # 載入預訓練模型
    lgbm_model = joblib.load(MODEL_PATHS['lgbm'])
    hmm_data = joblib.load(MODEL_PATHS['hmm'])
    return df, lgbm_model, hmm_data

@st.cache_data(show_spinner=False)
def run_backtest_cached(selected_symbols, threshold, sl_mult, tp_mult):
    """執行回測 (當選擇的股票變動時，快取會自動更新)"""
    df, lgbm_model, hmm_data = load_models_and_data(tuple(selected_symbols)) # tuple供快取識別
    # 初始化回測器
    bt = InstitutionalBacktester(df, lgbm_model, ['rsi_14', 'bollinger_upper', 'bollinger_lower', 'log_return', 'ma_20', 'volatility'], hmm_data)
    # 動態參數
    bt.params['threshold'] = threshold
    bt.params['sl_mult'] = sl_mult
    bt.params['tp_mult'] = tp_mult
    # 執行回測
    bt.generate_signals()
    bt.run_backtest(max_positions=3)
    return bt.equity_df, bt.trades_df

# 2. 側邊欄控制面板
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
st.sidebar.title("AlphaBase 參數控制")
# 動態股票選擇器
st.sidebar.subheader("🎯 選擇投資組合")
# 預設選前 3 檔，若資料庫少於 3 檔則全選
default_selections = available_symbols[:3] if len(available_symbols) >= 3 else available_symbols
selected_symbols = st.sidebar.multiselect(
    "勾選你想回測的股票池",
    options=available_symbols,
    default=default_selections,
    help="只會顯示資料庫中已有的股票。"
)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ AI 模型敏感度")
threshold_val = st.sidebar.slider("買入信心門檻", min_value=0.50, max_value=0.70, value=BACKTEST_PARAMS['threshold'], step=0.01)
st.sidebar.subheader("🛡️ 動態風控倍數")
sl_val = st.sidebar.slider("止損倍數 (SL)", 1.0, 3.0, BACKTEST_PARAMS['sl_mult'], 0.1)
tp_val = st.sidebar.slider("止盈倍數 (TP)", 2.0, 6.0, BACKTEST_PARAMS['tp_mult'], 0.1)
st.sidebar.markdown("---")
st.sidebar.info("💡 HMM 總經防禦：已開啟\n\n💡 凱利公式注碼：已開啟")
# 3. 執行與計算 KPI (含防呆機制)
st.title("AlphaBase 量化戰情室 V2.1 📊")
st.markdown("基於 **PostgreSQL + LightGBM + HMM 總經防禦** 的機構級量化系統")
# 沒選股票就暫停
if not selected_symbols:
    st.warning("⚠️ 請在左側欄位至少選擇一檔股票進行回測。")
    st.stop()
with st.spinner('🚀 正在運行 AI 選股與投資組合回測...'):
    equity_df, trades_df = run_backtest_cached(selected_symbols, threshold_val, sl_val, tp_val)
if equity_df.empty or trades_df.empty:
    st.error("❌ 回測無交易紀錄。這通常是因為「買入信心門檻」設太高，或測試期間正好是大盤崩盤日（被HMM過濾）。請調低門檻試試。")
    st.stop()
initial_cap = BACKTEST_PARAMS['initial_capital']
final_cap = equity_df['equity'].iloc[-1]
total_ret = (final_cap / initial_cap) - 1
cum_max = equity_df['equity'].cummax()
max_dd = ((equity_df['equity'] - cum_max) / cum_max).min()
wins = trades_df[trades_df['pnl'] > 0]
losses = trades_df[trades_df['pnl'] <= 0]
win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
# 4. 主畫面 Dashboard
# 頂部 KPI 卡片
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("最終資產", f"${final_cap:,.0f}", f"{total_ret:.2%}")
col2.metric("最大回撤", f"{max_dd:.2%}", delta_color="inverse")
col3.metric("勝率", f"{win_rate:.2%}")
col4.metric("獲利因子", f"{profit_factor:.2f}")
col5.metric("總交易次數", f"{len(trades_df)} 次")
st.markdown("---")
# 圖表區
tab1, tab2, tab3 = st.tabs(["📈 資金與回撤", "🔍 股票貢獻度", "📜 交易明細"])
with tab1:
    fig_equity = px.line(equity_df, x=equity_df.index, y='equity', title=f'投資組合淨值曲線 ({", ".join(selected_symbols)})', template='plotly_dark')
    fig_equity.update_layout(yaxis_title="總資產 ($)")
    st.plotly_chart(fig_equity, width='stretch')
with tab2:
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        pnl_by_sym = trades_df.groupby('symbol')['pnl'].sum().reset_index().sort_values('pnl', ascending=False)
        fig_bar = px.bar(pnl_by_sym, x='symbol', y='pnl', color='pnl', color_continuous_scale=px.colors.diverging.RdYlGn, title="各股票總報酬貢獻度", template='plotly_dark')
        st.plotly_chart(fig_bar, width='stretch')
    with col_chart2:
        reason_counts = trades_df['reason'].value_counts().reset_index()
        fig_pie = px.pie(reason_counts, values='count', names='reason', hole=0.4, title='出場原因分佈 (含 HMM 動態止損)', template='plotly_dark')
        st.plotly_chart(fig_pie, width='stretch')
with tab3:
    st.dataframe(trades_df.sort_values('exit_date', ascending=False).style.format({'pnl': '{:.2%}'}), width='stretch')
st.caption("AlphaBase System V2.1 | Interactive Quant Dashboard | Powered by Streamlit")