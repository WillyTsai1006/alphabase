import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import BACKTEST_PARAMS, FEATURES, RESEARCH_CONFIG  # noqa: E402
from research import artifact_status, research_quality_status  # noqa: E402

try:
    import joblib
except Exception:  # pragma: no cover - report still works without joblib
    joblib = None


def _format_json(data):
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def collect_artifacts(config=RESEARCH_CONFIG):
    return artifact_status(config, ROOT)


def render_report(config=RESEARCH_CONFIG):
    artifacts = collect_artifacts(config)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    missing = [row["name"] for row in artifacts if row["status"] == "missing"]
    metrics_summary = research_quality_status(config, ROOT)
    meta_path = ROOT / config["artifact_paths"]["meta_model"]
    calibration_report = None
    kelly_approved = False
    if joblib and meta_path.exists():
        meta_artifact = joblib.load(meta_path)
        calibration_report = meta_artifact.get("calibration_report") if isinstance(meta_artifact, dict) else None
        kelly_approved = bool(meta_artifact.get("kelly_sizing_approved")) if isinstance(meta_artifact, dict) else False
    kelly_status = (
        "Approved by saved Meta artifact calibration report."
        if kelly_approved
        else "Not approved in this report. Kelly sizing requires a saved Meta artifact with calibration metrics inside the configured limits."
    )

    artifact_table = "\n".join(
        f"| {row['name']} | `{row['path']}` | {row['status']} | `{row['sha256']}` |"
        for row in artifacts
    )
    return f"""# AlphaBase V3 Reproducible Research Report

Generated: {generated_at}

## Research identity

- Research ID: `{config['research_id']}`
- Fixed data interval: `{config['data_start']}` to `{config['data_end']}`
- Walk-forward interval: `{config['walk_forward_start']}` to `{config['walk_forward_end']}`
- Feature set: `{', '.join(FEATURES)}`

## Universe rule

{config['universe_rule']}

Fixed universe:

`{', '.join(config['universe'])}`

## Data cleaning rules

{chr(10).join(f'- {rule}' for rule in config['data_cleaning_rules'])}

## Fixed backtest parameters

```json
{_format_json(BACKTEST_PARAMS)}
```

## Walk-forward / rolling retrain protocol

```json
{_format_json(config['walk_forward'])}
```

Each fold trains only on data before the test window, applies an embargo equal to
`horizon_days`, and records out-of-sample predictions for the following fixed
test window. Single split OOS results are not treated as sufficient evidence.

## Research artifacts

| Artifact | Path | Status | SHA256 |
| --- | --- | --- | --- |
{artifact_table}

## Calibration and Kelly sizing decision

Walk-forward metrics summary:

```json
{_format_json(metrics_summary)}
```

Primary edge decision: {"Approved" if metrics_summary.get("primary_edge_approved") else "Not approved. Treat dashboard output as exploratory until primary mean/min fold AUC clear the gates."}

ML vs baseline summary:

```json
{_format_json(metrics_summary.get('strategy_summary', {'status': 'missing'}))}
```

Strategy edge decision: {"Approved" if metrics_summary.get('strategy_edge_approved') else "Not approved. ML top-k must beat the momentum baseline on mean relative return."}

```json
{_format_json(config['calibration'])}
```

Saved calibration report:

```json
{_format_json(calibration_report or {'status': 'missing'})}
```

Decision: {kelly_status}

## Backtest result status

{"Missing artifacts: " + ", ".join(missing) + "." if missing else "All fixed artifacts are present in this environment. Review fold-level metrics and calibration evidence before publishing performance numbers."}

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
"""


def main():
    report = render_report()
    output = ROOT / RESEARCH_CONFIG["artifact_paths"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
