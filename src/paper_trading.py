from copy import deepcopy

import pandas as pd

from portfolio_strategy import build_inverse_volatility_weights, next_trading_day


STRATEGY_VERSION = "qqq_gld_inverse_volatility_63_v1"
ASSETS = ("QQQ", "GLD")


def completed_month_ends(index, as_of):
    as_of = pd.Timestamp(as_of)
    dates = pd.DatetimeIndex(index)
    dates = dates[(dates <= as_of) & (dates.to_period("M") < as_of.to_period("M"))]
    if not len(dates):
        return dates
    frame = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(frame.groupby(frame.index.to_period("M")).tail(1).index)


def initialize_paper_state(as_of, baseline_signal_date, starting_cash=100000.0):
    return {
        "strategy_version": STRATEGY_VERSION,
        "initialized_at": pd.Timestamp(as_of).date().isoformat(),
        "last_signal_date": pd.Timestamp(baseline_signal_date).date().isoformat(),
        "cash": float(starting_cash),
        "shares": {asset: 0.0 for asset in ASSETS},
        "fills": [],
    }


def advance_paper_state(opens, closes, state, as_of, one_way_cost=0.002):
    if state["strategy_version"] != STRATEGY_VERSION:
        raise ValueError("Paper state belongs to a different strategy version")
    as_of = pd.Timestamp(as_of)
    result = deepcopy(state)
    signal_dates = completed_month_ends(closes.index, as_of)
    last_signal = pd.Timestamp(result["last_signal_date"])
    signal_dates = signal_dates[signal_dates > last_signal]

    for signal_date in signal_dates:
        fill_date = next_trading_day(opens.index, signal_date)
        if pd.isna(fill_date) or fill_date > as_of:
            break
        target = build_inverse_volatility_weights(
            closes.loc[:signal_date],
            assets=ASSETS,
            lookback=63,
            signal_start=signal_date,
            signal_end=signal_date,
        ).iloc[-1]
        fill_prices = opens.loc[fill_date, list(ASSETS)]
        old_shares = pd.Series(result["shares"], dtype=float)
        current_values = old_shares * fill_prices
        nav_before_cost = float(result["cash"] + current_values.sum())
        current_weights = current_values / nav_before_cost
        turnover = float((target - current_weights).abs().sum())
        cost = nav_before_cost * turnover * one_way_cost
        investable = nav_before_cost - cost
        new_shares = investable * target / fill_prices
        result["cash"] = float(investable - (new_shares * fill_prices).sum())
        result["shares"] = {asset: float(new_shares[asset]) for asset in ASSETS}
        result["last_signal_date"] = signal_date.date().isoformat()
        result["fills"].append(
            {
                "signal_date": signal_date.date().isoformat(),
                "fill_date": fill_date.date().isoformat(),
                "turnover": turnover,
                "cost": cost,
                "nav_before_cost": nav_before_cost,
                "target_weights": {asset: float(target[asset]) for asset in ASSETS},
                "fill_prices": {asset: float(fill_prices[asset]) for asset in ASSETS},
            }
        )

    mark_dates = closes.index[closes.index <= as_of]
    if not len(mark_dates):
        raise ValueError("No close is available on or before as_of")
    mark_date = mark_dates[-1]
    mark_prices = closes.loc[mark_date, list(ASSETS)]
    shares = pd.Series(result["shares"], dtype=float)
    nav = float(result["cash"] + (shares * mark_prices).sum())
    status = {
        "as_of": as_of.date().isoformat(),
        "mark_date": mark_date.date().isoformat(),
        "strategy_version": STRATEGY_VERSION,
        "last_signal_date": result["last_signal_date"],
        "nav": nav,
        "cash": result["cash"],
        "shares": result["shares"],
        "fill_count": len(result["fills"]),
    }
    return result, status
