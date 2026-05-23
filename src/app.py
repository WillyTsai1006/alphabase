import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sqlalchemy import text
st.set_page_config(page_title="AlphaBase Quant V3.0", page_icon="📈", layout="wide")
from config import MODEL_PATHS, BACKTEST_PARAMS, RESEARCH_CONFIG
from utils import get_logger, db_manager
from quant_engine import DataAndLabelEngine
from meta_engine import load_meta_artifact
from backtester import InstitutionalBacktester
from research import load_primary_oos_predictions, missing_required_artifacts, research_quality_status
logger = get_logger("StreamlitApp")

@st.cache_data(ttl=3600)
def get_available_symbols():
    query = "SELECT DISTINCT symbol FROM market_data ORDER BY symbol"
    try:
        with db_manager.engine.connect() as conn:
            return [row[0] for row in conn.execute(text(query))]
    except Exception as exc:
        logger.error(f"無法取得股票清單: {exc}")
        return []
available_symbols = get_available_symbols()

@st.cache_resource(show_spinner=False)
def load_models_and_data(selected_symbols):
    df = DataAndLabelEngine.load_data(selected_symbols)
    df['volatility'] = df.groupby(level='symbol')['close'].pct_change().ewm(span=100).std()
    df = df.dropna().reset_index()
    # [V3.0 升級] 載入三重大腦：主模型、次模型、HMM 總經模型
    lgbm_model = joblib.load(MODEL_PATHS['lgbm'])
    meta_artifact = load_meta_artifact(MODEL_PATHS['meta'])
    hmm_data = joblib.load(MODEL_PATHS['hmm'])
    primary_predictions = load_primary_oos_predictions()
    return df, lgbm_model, meta_artifact, hmm_data, primary_predictions

@st.cache_data(show_spinner=False)
def run_backtest_cached(selected_symbols, threshold, sl_mult, tp_mult):
    df, lgbm_model, meta_artifact, hmm_data, primary_predictions = load_models_and_data(tuple(selected_symbols))
    # [V3.0 升級] 傳入 meta_model
    bt = InstitutionalBacktester(
        df,
        lgbm_model,
        meta_artifact,
        hmm_data,
        primary_predictions=primary_predictions,
        require_oos_predictions=True,
    )
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
if not available_symbols:
    st.error("無法連線資料庫或尚未初始化 market_data。請確認 DB 已啟動，並先執行 ETL 與模型訓練流程。")
    st.stop()
missing_artifacts = missing_required_artifacts()
if missing_artifacts:
    st.error(
        "研究 artifacts 尚未完整產出，暫停績效儀表板以避免發布未驗證數字。"
        f" 缺少: {', '.join(missing_artifacts)}。請先執行 README 的研究流程。"
    )
    st.stop()
quality_status = research_quality_status()
default_selections = available_symbols[:3] if len(available_symbols) >= 3 else available_symbols
selected_symbols = st.sidebar.multiselect("股票池", options=available_symbols, default=default_selections)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 模型微調")
threshold_val = st.sidebar.slider("主模型初篩門檻", 0.50, 0.70, BACKTEST_PARAMS['threshold'], 0.01)
st.sidebar.markdown("---")
st.sidebar.success("✅ **Ranker 策略**: 已接入\n\n✅ **HMM 崩盤防禦**: 啟動中\n\n✅ **倉位**: Ranker 使用 top-k 固定倉位；Kelly 僅適用舊 binary/meta path")
st.title("AlphaBase 量化戰情室 V3.0 📊")
st.markdown("互動檢視：**主模型找機會 ➜ Meta 模型算勝率 ➜ 校準通過才使用 Kelly ➜ HMM 避股災**")
st.info(
    "此頁可調整參數做互動回測，正式績效請以固定研究報告為準："
    f"`{RESEARCH_CONFIG['artifact_paths']['report']}`。"
)
if quality_status.get("formal_strategy_approved", False):
    st.success(
        "正式 Ranker 策略已通過 baseline gate："
        f"ML 相對 momentum = {quality_status.get('strategy_summary', {}).get('ml_minus_momentum', 0):.4f}，"
        f"勝出 folds = {quality_status.get('strategy_summary', {}).get('ml_beats_momentum_folds', 0)}/"
        f"{quality_status.get('strategy_summary', {}).get('fold_count', 0)}。"
    )
elif not quality_status.get("strategy_edge_approved", False):
    st.warning(
        "Ranker strategy 尚未通過正式 baseline gate；本頁結果僅供探索。"
    )
if not quality_status.get("primary_edge_approved", False):
    st.caption(
        "診斷：legacy binary classifier AUC gate 未通過；正式策略以 ranker baseline gate 為準。"
    )
if not selected_symbols: st.stop()
with st.spinner('🚀 正在運行雙重 AI 與凱利動態回測...'):
    try:
        equity_df, trades_df = run_backtest_cached(selected_symbols, threshold_val, BACKTEST_PARAMS['sl_mult'], BACKTEST_PARAMS['tp_mult'])
    except FileNotFoundError as exc:
        st.error(f"找不到必要模型檔：{exc.filename}。請依序執行 quant_engine.py、meta_engine.py、hmm_engine.py。")
        st.stop()
    except Exception as exc:
        logger.error(f"回測執行失敗: {exc}")
        st.error(f"回測執行失敗：{exc}")
        st.stop()
if equity_df.empty:
    st.warning("此參數組合沒有可顯示的資產曲線。")
    st.stop()
if trades_df.empty:
    final_cap = equity_df['equity'].iloc[-1]
    total_ret = (final_cap / BACKTEST_PARAMS['initial_capital']) - 1
    st.warning("此互動回測在目前校準門檻與風控設定下沒有產生交易。")
    col1, col2 = st.columns(2)
    col1.metric("最終資產", f"${final_cap:,.0f}", f"{total_ret:.2%}")
    col2.metric("總交易次數", "0 次")
    st.plotly_chart(
        px.line(equity_df, x=equity_df.index, y='equity', title='資金曲線 (無交易)', template='plotly_dark'),
        use_container_width=True,
    )
    st.stop()
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
st.markdown("---")
st.subheader("🔍 歸因分析：誰是 MVP？")

col_a, col_b = st.columns(2)

with col_a:
    # 1. 各股累積獲利 (Bar Chart)
    # 把每支股票的 PnL 加總，看誰貢獻最多
    pnl_by_symbol = trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=True)
    
    fig_pnl = px.bar(
        x=pnl_by_symbol.values,
        y=pnl_by_symbol.index,
        orientation='h',
        title='各股累積報酬貢獻 (Sum of PnL %)',
        labels={'x': '累積 PnL (1.0 = 100%)', 'y': '股票代碼'},
        color=pnl_by_symbol.values,
        color_continuous_scale=['#FF4B4B', '#1C83E1', '#00CC96'] # 紅 -> 藍 -> 綠
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

with col_b:
    # 2. 各股勝率與交易次數 (Scatter Chart)
    # X軸=交易次數, Y軸=勝率, 點的大小=總獲利絕對值
    symbol_stats = trades_df.groupby('symbol').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        total_pnl=('pnl', 'sum')
    ).reset_index()

    fig_stats = px.scatter(
        symbol_stats,
        x='trades',
        y='win_rate',
        size=symbol_stats['total_pnl'].abs() + 0.1, # 加 0.1 防止點太小看不到
        color='total_pnl',
        hover_name='symbol',
        title='交易頻率 vs. 勝率分佈',
        labels={'trades': '交易次數 (頻率)', 'win_rate': '勝率 (Win Rate)'},
        color_continuous_scale='RdYlGn',
        range_y=[0, 1.1]
    )
    # 加一條 50% 勝率基準線
    fig_stats.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% 勝率")
    st.plotly_chart(fig_stats, use_container_width=True)
st.markdown("---")
st.subheader("⚖️ 專業量化評估：風險 vs. 報酬")

# 1. 計算高階量化指標
daily_returns = equity_df['equity'].pct_change().dropna()
# 年化標準差 (風險)
ann_std = daily_returns.std() * np.sqrt(252)
# 夏普值 (Sharpe Ratio) - 每承受一單位風險，能帶來多少超額報酬
sharpe_ratio = (daily_returns.mean() * 252) / ann_std if ann_std != 0 else 0
# 索提諾值 (Sortino Ratio) - 只懲罰向下的波動 (虧損)，更適合凱利公式這種爆發性策略
downside_returns = daily_returns[daily_returns < 0]
sortino_ratio = (daily_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

# 獲利因子 (Profit Factor) - 總獲利金額 / 總虧損金額
gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
avg_win = wins['pnl'].mean()
avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
payoff_ratio = avg_win / abs(avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss < 0 else None

# 2. 顯示高階指標卡片
met1, met2, met3, met4 = st.columns(4)
met1.metric("夏普指標 (Sharpe)", f"{sharpe_ratio:.2f}", help=">1.5為優，>2.0為極神")
met2.metric("索提諾指標 (Sortino)", f"{sortino_ratio:.2f}", help=">2.0為優，衡量下檔風險")
met3.metric("獲利因子 (Profit Factor)", "N/A" if np.isinf(profit_factor) else f"{profit_factor:.2f}", help=">1.5為穩健，>2.0代表賺多賠少")
met4.metric("平均盈虧比 (Payoff Ratio)", "N/A" if payoff_ratio is None else f"{payoff_ratio:.2f}", help="每筆獲利/每筆虧損的大小")

col_c, col_d = st.columns(2)

with col_c:
    # 3. 水下視圖 (Underwater Plot / Drawdown Chart)
    # 這張圖是策略的「照妖鏡」，用來顯示策略「痛」的時候有多痛，以及多久才能恢復
    drawdown_df = ((equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()).reset_index()
    fig_dd = px.area(drawdown_df, x='time', y='equity', title='水下視圖 (Drawdown Chart)', color_discrete_sequence=['red'])
    fig_dd.update_layout(yaxis_title="回撤幅度 (%)", yaxis_tickformat='.1%')
    st.plotly_chart(fig_dd, use_container_width=True)

with col_d:
    # 4. 交易損益分佈直方圖 (Trade PnL Distribution)
    # 檢查是否符合凱利公式的「肥尾效應 (Fat Tail)」：右邊大賺的尾巴應該要很長
    fig_dist = px.histogram(
        trades_df, x='pnl', nbins=50, 
        title='單筆交易損益分佈 (Trade PnL Distribution)',
        color=trades_df['pnl'] > 0,
        color_discrete_map={True: '#00CC96', False: '#FF4B4B'},
        labels={'pnl': '單筆報酬率', 'count': '交易次數'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)
st.caption("AlphaBase V3.0 | 雙重 AI 機器學習架構 | Tainan Local Time: 03:00 AM")