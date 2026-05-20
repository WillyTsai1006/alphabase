import pandas as pd

from research import (
    build_walk_forward_splits,
    clean_market_data,
    compute_calibration_report,
    is_kelly_sizing_approved,
)


def test_clean_market_data_applies_research_contract():
    df = pd.DataFrame(
        [
            {"time": "2017-01-03", "symbol": "AAPL", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"time": "2017-01-04", "symbol": "AAPL", "open": 10, "high": 9, "low": 8, "close": 10, "volume": 100},
            {"time": "2030-01-03", "symbol": "AAPL", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        ]
    )

    cleaned = clean_market_data(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["time"] == "2017-01-03"


def test_walk_forward_splits_are_rolling_and_embargoed():
    dates = pd.date_range("2016-01-01", "2025-12-31", freq="B")

    splits = build_walk_forward_splits(dates)

    assert splits
    assert splits[0]["test_start"] == pd.Timestamp("2020-01-01")
    assert splits[0]["train_end"] < splits[0]["test_start"]
    assert (splits[0]["test_start"] - splits[0]["train_end"]).days > 5


def test_calibration_report_controls_kelly_approval():
    report = compute_calibration_report([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99], n_bins=2)

    assert report["sample_count"] == 4
    assert report["brier_score"] < 0.02
    assert is_kelly_sizing_approved(report) is True


def test_bad_calibration_rejects_kelly_sizing():
    report = compute_calibration_report([0, 0, 1, 1], [0.95, 0.90, 0.10, 0.05], n_bins=2)

    assert is_kelly_sizing_approved(report) is False
