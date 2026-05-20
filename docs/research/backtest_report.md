# AlphaBase V3 Reproducible Research Report

Generated: 2026-05-20 14:55:49 UTC

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
  "horizon_days": 15,
  "initial_capital": 100000,
  "kelly_requires_calibration": true,
  "meta_threshold": 0.5,
  "sl_mult": 1.0,
  "slippage": 0.001,
  "tc": 0.001,
  "threshold": 0.55,
  "tp_mult": 2.0
}
```

## Walk-forward / rolling retrain protocol

```json
{
  "embargo_days": 15,
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
| primary_model | `artifacts/research/alphabase_v3_research_2026_05/primary_lgbm.pkl` | present | `d3614af0947b04728c037a6f0029c19ca38fe0a3a66372f9961d39c639c20d12` |
| meta_model | `artifacts/research/alphabase_v3_research_2026_05/meta_lgbm.pkl` | present | `ca856ad6a5137df34bf77fd31557b2aaec2712647392a23bb8127223823e71f4` |
| hmm_model | `artifacts/research/alphabase_v3_research_2026_05/hmm_model.pkl` | present | `bd0563634319e23550ddc00b93e1c4a4734b9c9da398fdf056015368c6fdf78d` |
| primary_walk_forward_metrics | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_metrics.csv` | present | `8a89594a96dddf57a8f0301a6581f64dc321ee8c01198f07da0ebb1b762ae637` |
| primary_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_predictions.csv` | present | `e236b1c3729ef6aba7f48637c6de16366463f7d6317bb15a47895f00a781c9d8` |

## Calibration and Kelly sizing decision

Walk-forward metrics summary:

```json
{
  "fold_count": 12,
  "max_auc": 0.5615405999261232,
  "mean_auc": 0.5158297856981718,
  "min_auc": 0.4720880173466381
}
```

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
  "bins": [
    {
      "abs_error": null,
      "bin": 0,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 1,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 2,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": 0.020019392362268684,
      "bin": 3,
      "count": 937,
      "empirical_rate": 0.3799359658484525,
      "mean_prob": 0.35991657348618383
    },
    {
      "abs_error": null,
      "bin": 4,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 5,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 6,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 7,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 8,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    },
    {
      "abs_error": null,
      "bin": 9,
      "count": 0,
      "empirical_rate": null,
      "mean_prob": null
    }
  ],
  "brier_score": 0.23604267783084368,
  "expected_calibration_error": 0.020019392362268684,
  "max_calibration_error": 0.020019392362268684,
  "sample_count": 937
}
```

Decision: Not approved in this report. Kelly sizing requires a saved Meta artifact with calibration metrics inside the configured limits.

## Backtest result status

All fixed artifacts are present in this environment. Review fold-level metrics and calibration evidence before publishing performance numbers.

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
