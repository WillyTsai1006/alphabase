# AlphaBase V3 Reproducible Research Report

Generated: 2026-09-06 17:52:04 UTC

## Research identity

- Research ID: `alphabase_v3_research_2026_09_purged`
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
test window. Ranker fit rows are retained only when their forward-return label
ends before validation starts; final training and validation rows are retained
only when their labels end before the test window starts. Single split OOS
results are not treated as sufficient evidence.

## Research artifacts

| Artifact | Path | Status | SHA256 |
| --- | --- | --- | --- |
| primary_model | `artifacts/research/alphabase_v3_research_2026_09_purged/primary_lgbm.pkl` | present | `1c245e682d5503da1746394a2db21b1a3742d6317d2670000b6520a4e494cc93` |
| meta_model | `artifacts/research/alphabase_v3_research_2026_09_purged/meta_lgbm.pkl` | present | `1f3423ac57c4f4f246c96547347f327f1418cd4f77d6b0681067d56f72fe6c9b` |
| hmm_model | `artifacts/research/alphabase_v3_research_2026_09_purged/hmm_model.pkl` | present | `c608fcc30512dbcc6617d962edeaaaa6ed7545d44fe8b126f538f17dde62c839` |
| primary_walk_forward_metrics | `artifacts/research/alphabase_v3_research_2026_09_purged/primary_walk_forward_metrics.csv` | present | `3bd8f4cf3263fea5ca66f56a5e3219db08c84642287b344d43f9bae82bd57ea2` |
| primary_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_09_purged/primary_walk_forward_predictions.csv` | present | `a2bf27d8aaa70db0d50061f5a339e015a67b77c534f2245aa4ef1a6e62d9a873` |
| ranker_walk_forward_predictions | `artifacts/research/alphabase_v3_research_2026_09_purged/ranker_walk_forward_predictions.csv` | present | `9276ff2e6df66b3a8ca9e4e611cbd296bd50574a4cb6009bcbd15076b3ec3a2f` |
| fold_strategy_metrics | `artifacts/research/alphabase_v3_research_2026_09_purged/fold_strategy_metrics.csv` | present | `3d91448e7fcdc0476c16d0237baad835dd3817ecb80e64bf3c361093179115e3` |

## Research gates, diagnostics, and Kelly decision

Walk-forward metrics summary:

```json
{
  "fold_count": 12,
  "max_auc": 0.5727516350892974,
  "mean_auc": 0.4972807562381358,
  "min_auc": 0.425963998232056,
  "primary_edge_approved": false,
  "required_mean_auc": 0.55,
  "required_min_fold_auc": 0.5,
  "status": "present",
  "strategy_candidate_gate_passed": false,
  "strategy_summary": {
    "candidate_gate_passed": false,
    "equal_weight_mean_relative_return": 0.003303174651380325,
    "fold_count": 12,
    "ml_beats_momentum_folds": 8,
    "ml_mean_relative_return": 0.006687041651611783,
    "ml_minus_momentum": -0.0008239756182131328,
    "momentum_mean_relative_return": 0.007511017269824917,
    "status": "present"
  }
}
```

Binary classifier diagnostic: AUC gate not passed.

ML vs baseline summary:

```json
{
  "candidate_gate_passed": false,
  "equal_weight_mean_relative_return": 0.003303174651380325,
  "fold_count": 12,
  "ml_beats_momentum_folds": 8,
  "ml_mean_relative_return": 0.006687041651611783,
  "ml_minus_momentum": -0.0008239756182131328,
  "momentum_mean_relative_return": 0.007511017269824917,
  "status": "present"
}
```

Ranker candidate gate: Not passed.

This is an internal ranking diagnostic, not a deployable-strategy approval. It
uses overlapping forward-return observations and does not model concurrent
positions, capital constraints, portfolio turnover, or transaction costs.

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
      "abs_error": 0.2380945621732935,
      "bin": 2,
      "count": 49,
      "empirical_rate": 0.5306122448979592,
      "mean_prob": 0.2925176827246657
    },
    {
      "abs_error": 0.06521919670733306,
      "bin": 3,
      "count": 793,
      "empirical_rate": 0.39722572509457754,
      "mean_prob": 0.3320065283872445
    },
    {
      "abs_error": 0.4060590552887142,
      "bin": 4,
      "count": 3,
      "empirical_rate": 0.0,
      "mean_prob": 0.4060590552887142
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
  "brier_score": 0.24710399401668637,
  "expected_calibration_error": 0.07645400438020432,
  "max_calibration_error": 0.4060590552887142,
  "sample_count": 845
}
```

Kelly decision: Not approved for the legacy binary/meta path. The ranker diagnostic uses fixed top-k selections and does not establish approval for live position sizing.

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
