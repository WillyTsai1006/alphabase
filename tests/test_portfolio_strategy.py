import numpy as np
import pandas as pd
import pytest

from portfolio_strategy import backtest_t1_open, build_inverse_volatility_weights


def synthetic_closes():
    dates = pd.bdate_range("2020-01-01", periods=320)
    phase = np.arange(len(dates))
    qqq_returns = 0.001 + 0.020 * np.sin(phase)
    gld_returns = 0.0005 + 0.005 * np.sin(phase)
    return pd.DataFrame(
        {
            "QQQ": 100 * np.cumprod(1 + qqq_returns),
            "GLD": 100 * np.cumprod(1 + gld_returns),
        },
        index=dates,
    )


def test_inverse_volatility_weights_are_long_only_and_normalized():
    weights = build_inverse_volatility_weights(synthetic_closes())

    assert (weights >= 0).all().all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights["GLD"] > weights["QQQ"]).all()


def test_weights_do_not_use_prices_after_signal_date():
    closes = synthetic_closes()
    original = build_inverse_volatility_weights(closes)
    signal_date = original.index[len(original) // 2]
    changed = closes.copy()
    changed.loc[changed.index > signal_date, "QQQ"] *= 10

    recalculated = build_inverse_volatility_weights(changed)

    pd.testing.assert_series_equal(original.loc[signal_date], recalculated.loc[signal_date])


def test_backtest_enters_after_signal_and_charges_initial_turnover():
    dates = pd.bdate_range("2024-01-02", periods=6)
    opens = pd.DataFrame(
        {"QQQ": [100, 101, 102, 104, 105, 106], "GLD": [100] * 6, "SPY": [100] * 6},
        index=dates,
    )
    weights = pd.DataFrame(
        {"QQQ": [1.0, 1.0], "GLD": [0.0, 0.0]},
        index=[dates[0], dates[3]],
    )

    periods = backtest_t1_open(opens, weights, one_way_cost=0.002)

    row = periods.iloc[0]
    expected_gross = opens.loc[dates[4], "QQQ"] / opens.loc[dates[1], "QQQ"] - 1.0
    assert row["entry_date"] == dates[1]
    assert row["entry_date"] > periods.index[0]
    assert row["turnover"] == pytest.approx(1.0)
    assert row["net_return"] == pytest.approx((1 + expected_gross) * 0.998 - 1)


def test_backtest_turnover_uses_drifted_pre_trade_weights():
    dates = pd.bdate_range("2024-01-02", periods=9)
    opens = pd.DataFrame(
        {
            "QQQ": [100, 100, 150, 150, 200, 200, 200, 200, 200],
            "GLD": [100] * 9,
            "SPY": [100] * 9,
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {"QQQ": [0.5, 0.5, 0.5], "GLD": [0.5, 0.5, 0.5]},
        index=[dates[0], dates[3], dates[6]],
    )

    periods = backtest_t1_open(opens, weights)

    assert periods.iloc[1]["turnover"] == pytest.approx(1.0 / 3.0)
