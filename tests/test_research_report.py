from scripts.generate_research_report import collect_artifacts, render_report


def test_research_report_contains_fixed_protocol():
    report = render_report()

    assert "Fixed data interval: `2016-01-01` to `2025-12-31`" in report
    assert "Walk-forward / rolling retrain protocol" in report
    assert "Walk-forward metrics summary" in report
    assert "ML vs baseline summary" in report
    assert "Calibration and Kelly sizing decision" in report
    assert "No return, win-rate, or drawdown claim should be published" in report


def test_research_report_marks_missing_artifacts():
    artifacts = collect_artifacts()

    assert {row["name"] for row in artifacts} == {
        "primary_model",
        "meta_model",
        "hmm_model",
        "primary_walk_forward_metrics",
        "primary_walk_forward_predictions",
        "ranker_walk_forward_predictions",
        "fold_strategy_metrics",
    }
    assert all(row["status"] in {"present", "missing"} for row in artifacts)
