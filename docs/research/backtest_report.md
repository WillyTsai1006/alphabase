# AlphaBase V3 Reproducible Research Report

Generated: 2026-05-20 14:30:51 UTC

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
| primary_model | `artifacts/research/alphabase_v3_research_2026_05/primary_lgbm.pkl` | present | `46bb95446d47639eb581eeb4851c8bc423fdc993cedd6cb0745df159917f77b5` |
| meta_model | `artifacts/research/alphabase_v3_research_2026_05/meta_lgbm.pkl` | present | `4b96063b526304aa86f85c65800167c1fd4a3ca91c96bc16ab4884db60626f0a` |
| hmm_model | `artifacts/research/alphabase_v3_research_2026_05/hmm_model.pkl` | present | `5b074a49f763de1c6bdd5955d1d5184c27a0dbc26469b2d1a6066ba4505ea88a` |
| primary_walk_forward_metrics | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_metrics.csv` | present | `8f962edf2b5fa752bb7f7d1fc80a3e184e04cf97c6eef339d7aacee07f195fe7` |
| primary_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_predictions.csv` | present | `f5326afc217747c26ce5aff3836fffccfa46796d956f15a2423f9dc7a7757df1` |

## Calibration and Kelly sizing decision

Walk-forward metrics summary:

```json
{
  "fold_count": 12,
  "max_auc": 0.7532090132090132,
  "mean_auc": 0.5988808897408778,
  "min_auc": 0.4376029492509216
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
      "abs_error": 0.0649256015031899,
      "bin": 0,
      "count": 21,
      "empirical_rate": 0.0,
      "mean_prob": 0.0649256015031899
    },
    {
      "abs_error": 0.13953447327603635,
      "bin": 1,
      "count": 16,
      "empirical_rate": 0.0,
      "mean_prob": 0.13953447327603635
    },
    {
      "abs_error": 0.25486148770233114,
      "bin": 2,
      "count": 2,
      "empirical_rate": 0.0,
      "mean_prob": 0.25486148770233114
    },
    {
      "abs_error": 0.3013552042458563,
      "bin": 3,
      "count": 1,
      "empirical_rate": 0.0,
      "mean_prob": 0.3013552042458563
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
  "brier_score": 0.015987798531125485,
  "expected_calibration_error": 0.1101766845908522,
  "max_calibration_error": 0.3013552042458563,
  "sample_count": 40
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
