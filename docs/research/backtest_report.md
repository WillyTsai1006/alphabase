# AlphaBase V3 Reproducible Research Report

Generated: 2026-05-23 05:44:36 UTC

## Research identity

- Research ID: `alphabase_v3_research_2026_05`
- Fixed data interval: `2016-01-01` to `2025-12-31`
- Walk-forward interval: `2020-01-01` to `2025-12-31`
- Feature set: `rsi_14, bollinger_upper, bollinger_lower, log_return, ma_20, volatility`

## Universe rule

Fixed large-cap US equities plus SPY benchmark, selected before the research run and not changed after seeing backtest results. This is not a survivorship-free institutional universe.

Fixed universe:

`AAPL, MSFT, NVDA, GOOGL, AMZN, INTC, PYPL, PFE, ZM`

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
| primary_model | `artifacts/research/alphabase_v3_research_2026_05/primary_lgbm.pkl` | present | `72c088730a5bf8f4c4275980921039d7b22fa80024fe84d46aa5162538f65b7e` |
| meta_model | `artifacts/research/alphabase_v3_research_2026_05/meta_lgbm.pkl` | present | `f3d7cf2b5807bf5b1110408e00d330a7d8ea13ac2e999d9343bdca783d273104` |
| hmm_model | `artifacts/research/alphabase_v3_research_2026_05/hmm_model.pkl` | present | `9c49516cdac642fc3c5a258a6f3697cb8a05bfb88f3f584952e17d63b45d6ade` |
| primary_walk_forward_metrics | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_metrics.csv` | present | `e4772e6ad99221377a6ca6ac023f868379cf9ee8b0f4fe4a75e5ebb05e8697ae` |
| primary_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_05/primary_walk_forward_predictions.csv` | present | `39c2131a082a407a0668a0e7cd368d3adffc4c05d481fe2ca6cb60c25e937549` |
| ranker_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_05/ranker_walk_forward_predictions.csv` | present | `e1f7efeafbffed94961c4287c6dfb78b2db0241b6bfd5a7d3cecc6d5c6467516` |
| fold_strategy_metrics | `artifacts/research/alphabase_v3_research_2026_05/fold_strategy_metrics.csv` | present | `0180dc2795100f60ceaed2fe0e65a4b68ff424c8c0f2813d03cd375726e83139` |

## Calibration and Kelly sizing decision

Walk-forward metrics summary:

```json
{
  "fold_count": 12,
  "max_auc": 0.570510297783025,
  "mean_auc": 0.4953117294324798,
  "min_auc": 0.4249193040729678,
  "primary_edge_approved": false,
  "required_mean_auc": 0.55,
  "required_min_fold_auc": 0.5,
  "status": "present",
  "strategy_edge_approved": true,
  "strategy_summary": {
    "equal_weight_mean_relative_return": 0.0033059570022440908,
    "fold_count": 12,
    "ml_beats_momentum_folds": 8,
    "ml_mean_relative_return": 0.008809376170524142,
    "ml_minus_momentum": 0.0012978504196869162,
    "ml_vs_momentum_approved": true,
    "momentum_mean_relative_return": 0.007511525750837225,
    "status": "present"
  }
}
```

Primary edge decision: Not approved. Treat dashboard output as exploratory until primary mean/min fold AUC clear the gates.

ML vs baseline summary:

```json
{
  "equal_weight_mean_relative_return": 0.0033059570022440908,
  "fold_count": 12,
  "ml_beats_momentum_folds": 8,
  "ml_mean_relative_return": 0.008809376170524142,
  "ml_minus_momentum": 0.0012978504196869162,
  "ml_vs_momentum_approved": true,
  "momentum_mean_relative_return": 0.007511525750837225,
  "status": "present"
}
```

Strategy edge decision: Approved

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
      "abs_error": 0.06456105120329797,
      "bin": 3,
      "count": 828,
      "empirical_rate": 0.40217391304347827,
      "mean_prob": 0.3376128618401803
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
  "brier_score": 0.24436715028252215,
  "expected_calibration_error": 0.06456105120329797,
  "max_calibration_error": 0.06456105120329797,
  "sample_count": 828
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
