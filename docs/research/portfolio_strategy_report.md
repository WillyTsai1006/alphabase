# QQQ/GLD Inverse-Volatility Candidate

## Locked protocol

- Assets: QQQ and GLD
- Signal: trailing 63-session realized volatility at month end
- Allocation: inverse-volatility weights, long-only and fully invested
- Execution: next trading day's adjusted open
- Rebalance: monthly
- One-way turnover cost: 0.20%
- Discovery: 2009-2017
- Validation: 2018-2023
- One-time candidate-specific holdout: 2024-2025; 2026 is excluded

## Results

| Segment | Portfolio | CAGR | Sharpe | Monthly max drawdown |
|---|---|---:|---:|---:|
| Validation | Strategy | 10.78% | 0.85 | -17.81% |
| Validation | SPY | 10.41% | 0.61 | -23.31% |
| Holdout | Strategy | 34.16% | 3.83 | -0.77% |
| Holdout | SPY | 22.09% | 1.95 | -6.96% |

Holdout daily-close maximum drawdown was -8.76% for the strategy and -18.76% for SPY.

## Interpretation

This candidate passed the predefined risk-adjusted screen and the 2024-2025 holdout gates. Results were stable across 42, 63, and 126-session volatility estimates and remained positive when transaction costs were doubled in the local stress test.

It is not proof of persistent alpha. The validation moving-block bootstrap interval for arithmetic mean excess return crossed zero, and removing the strongest excess year (2020) reduced strategy CAGR to 7.76%, below SPY at 9.00%. The short, unusually favorable holdout contains only 23 monthly periods. Treat this as a promising diversified allocation candidate that merits paper trading, not as a production-ready trading edge.
