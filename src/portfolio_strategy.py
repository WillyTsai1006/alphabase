import numpy as np
import pandas as pd


def month_end_rows(frame):
    return frame.groupby(frame.index.to_period("M"), group_keys=False).tail(1)


def build_inverse_volatility_weights(
    closes,
    assets=("QQQ", "GLD"),
    lookback=63,
    signal_start=None,
    signal_end=None,
):
    missing = set(assets) - set(closes.columns)
    if missing:
        raise ValueError(f"Missing close columns: {sorted(missing)}")
    volatility = closes[list(assets)].pct_change(fill_method=None).rolling(lookback).std() * np.sqrt(252)
    monthly_volatility = month_end_rows(volatility)
    if signal_start is not None:
        monthly_volatility = monthly_volatility[monthly_volatility.index >= pd.Timestamp(signal_start)]
    if signal_end is not None:
        monthly_volatility = monthly_volatility[monthly_volatility.index <= pd.Timestamp(signal_end)]
    inverse = 1.0 / monthly_volatility
    weights = inverse.div(inverse.sum(axis=1), axis=0).dropna()
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("Portfolio weights do not sum to one")
    return weights


def next_trading_day(index, signal_date):
    position = index.searchsorted(signal_date, side="right")
    return index[position] if position < len(index) else pd.NaT


def backtest_t1_open(opens, weights, one_way_cost=0.002, benchmark="SPY"):
    required = set(weights.columns) | {benchmark}
    missing = required - set(opens.columns)
    if missing:
        raise ValueError(f"Missing open columns: {sorted(missing)}")
    entry_dates = pd.Series(
        [next_trading_day(opens.index, date) for date in weights.index],
        index=weights.index,
    )
    rows = []
    pre_trade_weights = pd.Series(0.0, index=weights.columns)
    for position in range(len(weights) - 1):
        signal_date = weights.index[position]
        entry_date = entry_dates.iloc[position]
        exit_date = entry_dates.iloc[position + 1]
        if pd.isna(entry_date) or pd.isna(exit_date):
            continue
        current_weights = weights.iloc[position]
        asset_returns = opens.loc[exit_date, weights.columns] / opens.loc[entry_date, weights.columns] - 1.0
        turnover = float((current_weights - pre_trade_weights).abs().sum())
        gross_return = float((current_weights * asset_returns).sum())
        net_return = (1.0 + gross_return) * (1.0 - turnover * one_way_cost) - 1.0
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "net_return": net_return,
                "benchmark_return": float(opens.loc[exit_date, benchmark] / opens.loc[entry_date, benchmark] - 1.0),
            }
        )
        end_values = current_weights * (1.0 + asset_returns)
        pre_trade_weights = end_values / end_values.sum()
    return pd.DataFrame(rows).set_index("signal_date")


def summarize_returns(returns, periods_per_year=12):
    returns = pd.Series(returns).dropna()
    equity = (1.0 + returns).cumprod()
    years = len(returns) / periods_per_year
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return {
        "periods": len(returns),
        "total_return": equity.iloc[-1] - 1.0,
        "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "annual_volatility": volatility,
        "sharpe": returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year),
        "monthly_max_drawdown": (equity / equity.cummax() - 1.0).min(),
        "positive_period_rate": (returns > 0).mean(),
    }


def daily_equity(opens, closes, weights, one_way_cost=0.002):
    entry_dates = [next_trading_day(opens.index, date) for date in weights.index]
    shares = pd.Series(0.0, index=weights.columns)
    capital = 1.0
    equity = []
    for position in range(len(weights) - 1):
        entry_date = entry_dates[position]
        exit_date = entry_dates[position + 1]
        if position:
            pre_trade_values = shares * opens.loc[entry_date, weights.columns]
            capital = float(pre_trade_values.sum())
            pre_trade_weights = pre_trade_values / capital
        else:
            pre_trade_weights = pd.Series(0.0, index=weights.columns)
        target = weights.iloc[position]
        turnover = float((target - pre_trade_weights).abs().sum())
        capital *= 1.0 - turnover * one_way_cost
        shares = capital * target / opens.loc[entry_date, weights.columns]
        dates = closes.loc[entry_date:exit_date].index
        for date in dates[dates < exit_date]:
            equity.append((date, float((shares * closes.loc[date, weights.columns]).sum())))
    final_date = entry_dates[-1]
    equity.append((final_date, float((shares * opens.loc[final_date, weights.columns]).sum())))
    return pd.Series(dict(equity), name="strategy_equity")
