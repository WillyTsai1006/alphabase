# AlphaBase V3 Reproducible Research Report

Generated: 2026-05-20 11:08:42 UTC

## Research identity

- Research ID: `alphabase_v3_research_2026_05`
- Fixed data interval: `2016-01-01` to `2025-12-31`
- Walk-forward interval: `2020-01-01` to `2025-12-31`
- Feature set: `rsi_14, bollinger_upper, bollinger_lower, log_return, ma_20, volatility`

## Universe rule

Fixed large-cap US equities plus SPY benchmark, selected before the research run and not changed after seeing backtest results.

Fixed universe:

`AAPL, MSFT, NVDA, GOOGL, AMZN, SPY, INTC, PYPL, PFE, ZM`

## Data cleaning rules

- Keep rows inside the fixed data_start/data_end interval.
- Require positive open/high/low/close and non-negative volume.
- Require high >= max(open, close) and low <= min(open, close).
- Drop duplicate (time, symbol) rows after sorting.
- Drop rows with missing engineered features before model training.

## Fixed backtest parameters

```json
{
  "fallback_position_fraction": 0.1,
  "horizon_days": 5,
  "initial_capital": 100000,
  "kelly_requires_calibration": true,
  "meta_threshold": 0.5,
  "sl_mult": 2.0,
  "slippage": 0.001,
  "tc": 0.001,
  "threshold": 0.55,
  "tp_mult": 4.0
}
```

## Walk-forward / rolling retrain protocol

```json
{
  "embargo_days": 5,
  "step_months": 6,
  "test_months": 6,
  "train_years": 3
}
```

Each fold trains only on data before the test window, applies an embargo equal to
`horizon_days`, and records out-of-sample predictions for the following fixed
test window. Single split OOS results are not treated as sufficient evidence.

## Research artifacts

| Artifact | Path | Status | SHA256 |
| --- | --- | --- | --- |
| primary_model | `artifacts/research/alphabase_v3_research_2026_05/primary_lgbm.pkl` | missing | `N/A` |
| meta_model | `artifacts/research/alphabase_v3_research_2026_05/meta_lgbm.pkl` | missing | `N/A` |
| hmm_model | `artifacts/research/alphabase_v3_research_2026_05/hmm_model.pkl` | missing | `N/A` |
| primary_walk_forward_metrics | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_metrics.csv` | missing | `N/A` |
| primary_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_predictions.csv` | missing | `N/A` |

## Calibration and Kelly sizing decision

```json
{
  "max_brier_score_for_kelly": 0.2,
  "max_expected_calibration_error_for_kelly": 0.05,
  "n_bins": 10
}
```

Saved calibration report:

```json
{
  "status": "missing"
}
```

Decision: Not approved in this report. Kelly sizing requires a saved Meta artifact with calibration metrics inside the configured limits.

## Backtest result status

Missing artifacts: primary_model, meta_model, hmm_model, primary_walk_forward_metrics, primary_walk_forward_predictions.

No return, win-rate, or drawdown claim should be published unless this report is
generated from the fixed artifact set above and includes walk-forward fold
metrics plus calibration evidence.

## Reproduction commands

```bash
cp .env.example .env
docker compose up -d
python3 src/data_loader.py
python3 scripts/run_walk_forward_research.py
python3 src/quant_engine.py
python3 src/meta_engine.py
python3 src/hmm_engine.py
python3 scripts/generate_research_report.py
```
