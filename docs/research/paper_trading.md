# QQQ/GLD Paper-Trading Protocol

The paper portfolio freezes `qqq_gld_inverse_volatility_63_v1`. It does not retrain or select parameters from 2026 data.

## Lifecycle

1. The first run initializes a USD 100,000 cash account and records the latest completed month as its baseline. It does not create a retrospective fill.
2. After a later month has completed, the runner calculates QQQ/GLD inverse-volatility weights using only information available at that month end.
3. The simulated rebalance fills at the next available adjusted open and charges 0.20% of one-way asset turnover.
4. Repeated runs are idempotent: a processed signal cannot create a second fill.
5. State and current status are stored under `artifacts/paper/`, which is intentionally excluded from Git because it is prospective runtime state.

Run manually with:

```bash
python scripts/run_paper_portfolio.py
```

Use `--as-of YYYY-MM-DD` only for deterministic operational replay. Historical replay is not additional holdout evidence and must not be used to change the locked strategy.
