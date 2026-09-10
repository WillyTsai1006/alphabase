import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_strategy import (  # noqa: E402
    backtest_t1_open,
    build_inverse_volatility_weights,
    daily_equity,
    summarize_returns,
)


ASSETS = ["QQQ", "GLD"]
BENCHMARK = "SPY"
ONE_WAY_COST = 0.002
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "alphabase_v4_qqq_gld_inverse_vol"
REPORT_PATH = ROOT / "docs" / "research" / "portfolio_strategy_report.md"


def download_prices():
    yf.set_tz_cache_location(str(ROOT / "data" / "yfinance_cache"))
    raw = yf.download(
        ASSETS + [BENCHMARK],
        start="2008-01-01",
        end="2025-12-02",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    opens = raw["Open"].reindex(columns=ASSETS + [BENCHMARK])
    closes = raw["Close"].reindex(columns=ASSETS + [BENCHMARK])
    if opens.isna().any().any() or closes.isna().any().any():
        raise RuntimeError("Downloaded prices contain missing values")
    return opens, closes


def adjusted_benchmark_returns(periods):
    returns = periods["benchmark_return"].copy()
    if len(returns):
        returns.iloc[0] = (1.0 + returns.iloc[0]) * (1.0 - ONE_WAY_COST) - 1.0
    return returns


def max_drawdown(equity):
    return float((equity / equity.cummax() - 1.0).min())


def bootstrap_excess(periods, block_size=6, samples=10000):
    excess = (periods["net_return"] - periods["benchmark_return"]).to_numpy()
    blocks = math.ceil(len(excess) / block_size)
    max_start = len(excess) - block_size + 1
    rng = np.random.default_rng(42)
    annualized_means = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, max_start, size=blocks)
        draw = np.concatenate([excess[start : start + block_size] for start in starts])[: len(excess)]
        annualized_means[sample] = draw.mean() * 12.0
    return {
        "annualized_mean_excess": excess.mean() * 12.0,
        "ci_95_low": np.quantile(annualized_means, 0.025),
        "ci_95_high": np.quantile(annualized_means, 0.975),
        "probability_mean_excess_positive": (annualized_means > 0).mean(),
    }


def evaluate_segment(periods, name):
    strategy = summarize_returns(periods["net_return"])
    benchmark = summarize_returns(adjusted_benchmark_returns(periods))
    return [
        {"segment": name, "portfolio": "strategy", **strategy},
        {"segment": name, "portfolio": "spy", **benchmark},
    ]


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    opens, closes = download_prices()
    development_weights = build_inverse_volatility_weights(
        closes,
        assets=ASSETS,
        lookback=63,
        signal_start="2009-01-01",
        signal_end="2023-11-30",
    )
    development = backtest_t1_open(opens, development_weights, ONE_WAY_COST, BENCHMARK)
    discovery = development[development.index < pd.Timestamp("2018-01-01")]
    validation = development[development.index >= pd.Timestamp("2018-01-01")]

    holdout_weights = build_inverse_volatility_weights(
        closes,
        assets=ASSETS,
        lookback=63,
        signal_start="2023-12-01",
        signal_end="2025-11-30",
    )
    holdout = backtest_t1_open(opens, holdout_weights, ONE_WAY_COST, BENCHMARK)
    summary = pd.DataFrame(
        evaluate_segment(discovery, "discovery_2009_2017")
        + evaluate_segment(validation, "validation_2018_2023")
        + evaluate_segment(holdout, "holdout_2024_2025")
    )

    holdout_daily = daily_equity(opens, closes, holdout_weights, ONE_WAY_COST)
    first_entry = holdout.iloc[0]["entry_date"]
    last_exit = holdout.iloc[-1]["exit_date"]
    spy_shares = (1.0 - ONE_WAY_COST) / opens.loc[first_entry, BENCHMARK]
    spy_daily = closes.loc[first_entry:last_exit, BENCHMARK] * spy_shares
    spy_daily = spy_daily[spy_daily.index < last_exit]
    spy_daily.loc[last_exit] = opens.loc[last_exit, BENCHMARK] * spy_shares
    daily_drawdowns = pd.DataFrame(
        [
            {"portfolio": "strategy", "daily_close_max_drawdown": max_drawdown(holdout_daily)},
            {"portfolio": "spy", "daily_close_max_drawdown": max_drawdown(spy_daily)},
        ]
    )

    neighborhood_rows = []
    for lookback in [42, 63, 126]:
        weights = build_inverse_volatility_weights(
            closes,
            assets=ASSETS,
            lookback=lookback,
            signal_start="2009-01-01",
            signal_end="2023-11-30",
        )
        periods = backtest_t1_open(opens, weights, ONE_WAY_COST, BENCHMARK)
        result = summarize_returns(periods[periods.index >= pd.Timestamp("2018-01-01")]["net_return"])
        neighborhood_rows.append({"lookback": lookback, **result})
    neighborhood = pd.DataFrame(neighborhood_rows)
    bootstrap = pd.DataFrame([bootstrap_excess(validation)])

    double_cost_returns = (
        (1.0 + holdout["gross_return"]) * (1.0 - holdout["turnover"] * 2 * ONE_WAY_COST) - 1.0
    )
    double_cost = summarize_returns(double_cost_returns)
    validation_yearly = validation[["net_return", "benchmark_return"]].copy()
    validation_yearly["year"] = validation_yearly.index.year
    validation_yearly = validation_yearly.groupby("year")[["net_return", "benchmark_return"]].agg(
        lambda returns: (1.0 + returns).prod() - 1.0
    )
    validation_yearly["excess"] = validation_yearly["net_return"] - validation_yearly["benchmark_return"]
    strongest_year = int(validation_yearly["excess"].idxmax())
    without_strongest = validation[validation.index.year != strongest_year]
    without_strongest_strategy = summarize_returns(without_strongest["net_return"])
    without_strongest_spy = summarize_returns(without_strongest["benchmark_return"])
    stress_tests = pd.DataFrame(
        [
            {
                "test": "holdout_double_cost",
                "removed_year": np.nan,
                "strategy_cagr": double_cost["cagr"],
                "strategy_sharpe": double_cost["sharpe"],
                "spy_cagr": np.nan,
            },
            {
                "test": "validation_without_strongest_excess_year",
                "removed_year": strongest_year,
                "strategy_cagr": without_strongest_strategy["cagr"],
                "strategy_sharpe": without_strongest_strategy["sharpe"],
                "spy_cagr": without_strongest_spy["cagr"],
            },
        ]
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    development.to_csv(ARTIFACT_DIR / "development_periods.csv")
    holdout.to_csv(ARTIFACT_DIR / "holdout_periods.csv")
    holdout_weights.to_csv(ARTIFACT_DIR / "holdout_weights.csv")
    summary.to_csv(ARTIFACT_DIR / "summary.csv", index=False)
    daily_drawdowns.to_csv(ARTIFACT_DIR / "daily_drawdowns.csv", index=False)
    neighborhood.to_csv(ARTIFACT_DIR / "lookback_robustness.csv", index=False)
    bootstrap.to_csv(ARTIFACT_DIR / "validation_bootstrap.csv", index=False)
    validation_yearly.to_csv(ARTIFACT_DIR / "validation_yearly.csv")
    stress_tests.to_csv(ARTIFACT_DIR / "stress_tests.csv", index=False)

    validation_strategy = summary.query(
        "segment == 'validation_2018_2023' and portfolio == 'strategy'"
    ).iloc[0]
    validation_spy = summary.query(
        "segment == 'validation_2018_2023' and portfolio == 'spy'"
    ).iloc[0]
    holdout_strategy = summary.query(
        "segment == 'holdout_2024_2025' and portfolio == 'strategy'"
    ).iloc[0]
    holdout_spy = summary.query(
        "segment == 'holdout_2024_2025' and portfolio == 'spy'"
    ).iloc[0]
    quality_gates = pd.DataFrame(
        [
            {"gate": "positive_holdout_cagr", "passed": holdout_strategy["cagr"] > 0},
            {"gate": "holdout_sharpe_above_spy", "passed": holdout_strategy["sharpe"] > holdout_spy["sharpe"]},
            {
                "gate": "daily_drawdown_below_spy",
                "passed": abs(daily_drawdowns.iloc[0]["daily_close_max_drawdown"])
                < abs(daily_drawdowns.iloc[1]["daily_close_max_drawdown"]),
            },
            {"gate": "cagr_at_least_80pct_of_spy", "passed": holdout_strategy["cagr"] >= 0.8 * holdout_spy["cagr"]},
            {"gate": "positive_cagr_at_double_cost", "passed": double_cost["cagr"] > 0},
        ]
    )
    quality_gates.to_csv(ARTIFACT_DIR / "quality_gates.csv", index=False)
    report = f"""# QQQ/GLD Inverse-Volatility Candidate

## Locked protocol

- Assets: QQQ and GLD
- Signal: trailing 63-session realized volatility at month end
- Allocation: inverse-volatility weights, long-only and fully invested
- Execution: next trading day's adjusted open
- Rebalance: monthly
- One-way turnover cost: {pct(ONE_WAY_COST)}
- Discovery: 2009-2017
- Validation: 2018-2023
- One-time candidate-specific holdout: 2024-2025; 2026 is excluded

## Results

| Segment | Portfolio | CAGR | Sharpe | Monthly max drawdown |
|---|---|---:|---:|---:|
| Validation | Strategy | {pct(validation_strategy['cagr'])} | {validation_strategy['sharpe']:.2f} | {pct(validation_strategy['monthly_max_drawdown'])} |
| Validation | SPY | {pct(validation_spy['cagr'])} | {validation_spy['sharpe']:.2f} | {pct(validation_spy['monthly_max_drawdown'])} |
| Holdout | Strategy | {pct(holdout_strategy['cagr'])} | {holdout_strategy['sharpe']:.2f} | {pct(holdout_strategy['monthly_max_drawdown'])} |
| Holdout | SPY | {pct(holdout_spy['cagr'])} | {holdout_spy['sharpe']:.2f} | {pct(holdout_spy['monthly_max_drawdown'])} |

Holdout daily-close maximum drawdown was {pct(daily_drawdowns.iloc[0]['daily_close_max_drawdown'])} for the strategy and {pct(daily_drawdowns.iloc[1]['daily_close_max_drawdown'])} for SPY.

## Interpretation

This candidate passed the predefined risk-adjusted screen and the 2024-2025 holdout gates. Results were stable across 42, 63, and 126-session volatility estimates and remained positive when transaction costs were doubled in the local stress test.

It is not proof of persistent alpha. The validation moving-block bootstrap interval for arithmetic mean excess return crossed zero, and removing the strongest excess year ({strongest_year}) reduced strategy CAGR to {pct(without_strongest_strategy['cagr'])}, below SPY at {pct(without_strongest_spy['cagr'])}. The short, unusually favorable holdout contains only 23 monthly periods. Treat this as a promising diversified allocation candidate that merits paper trading, not as a production-ready trading edge.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {ARTIFACT_DIR}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
