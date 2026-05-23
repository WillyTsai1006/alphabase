import pandas as pd

from research import (
    add_forward_returns,
    build_walk_forward_splits,
    clean_market_data,
    compute_calibration_report,
    evaluate_topk_strategies,
    is_kelly_sizing_approved,
    select_hybrid_alpha,
    summarize_strategy_metrics,
    apply_hybrid_score,
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


def test_clean_market_data_handles_multiindex():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2017-01-03"), "AAPL"),
            (pd.Timestamp("2030-01-03"), "AAPL"),
        ],
        names=["time", "symbol"],
    )
    df = pd.DataFrame(
        {
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10, 10],
            "volume": [100, 100],
        },
        index=index,
    )

    cleaned = clean_market_data(df)

    assert len(cleaned) == 1
    assert cleaned.index[0] == (pd.Timestamp("2017-01-03"), "AAPL")


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


def test_primary_oos_prediction_loader_requires_columns(tmp_path):
    from research import load_primary_oos_predictions

    path = tmp_path / "predictions.csv"
    pd.DataFrame({"time": ["2024-01-02"], "symbol": ["AAPL"]}).to_csv(path, index=False)

    try:
        load_primary_oos_predictions("predictions.csv", root=tmp_path)
    except ValueError as exc:
        assert "primary_prob" in str(exc)
    else:
        raise AssertionError("Expected missing primary_prob to fail")


def test_strategy_evaluation_compares_ml_to_momentum():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
    returns = pd.DataFrame(
        {
            "time": dates,
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "close": [100, 100, 110, 90],
            "return_20": [0.2, 0.1, 0.2, 0.1],
            "forward_return": [0.05, -0.02, 0.04, -0.01],
            "relative_forward_return": [0.04, -0.03, 0.03, -0.02],
        }
    )
    predictions = pd.DataFrame(
        {
            "time": dates,
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "primary_prob": [0.9, 0.1, 0.8, 0.2],
        }
    )
    folds = pd.DataFrame(
        [{"fold": 1, "test_start": pd.Timestamp("2024-01-01"), "test_end": pd.Timestamp("2024-01-02")}]
    )

    metrics = evaluate_topk_strategies(predictions, returns, folds, top_k=1)
    summary = summarize_strategy_metrics(metrics)

    assert set(metrics["strategy"]) == {"ml_topk", "momentum_topk", "equal_weight"}
    assert summary["ml_vs_momentum_approved"] is False
    assert summary["fold_count"] == 1


def test_add_forward_returns_adds_relative_returns():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "symbol": ["AAPL", "AAPL", "SPY", "SPY"],
            "open": [100, 110, 100, 105],
            "high": [101, 111, 101, 106],
            "low": [99, 109, 99, 104],
            "close": [100, 110, 100, 105],
            "volume": [1, 1, 1, 1],
        }
    )

    result = add_forward_returns(df, horizon_days=1, benchmark_symbol="SPY")
    aapl = result[(result["symbol"] == "AAPL") & (result["time"] == pd.Timestamp("2024-01-01"))].iloc[0]

    assert round(aapl["forward_return"], 2) == 0.10
    assert round(aapl["relative_forward_return"], 2) == 0.05


def test_hybrid_score_selects_ranker_when_validation_improves():
    validation = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "rank_score": [0.9, 0.1, 0.8, 0.2],
            "return_20": [0.1, 0.2, 0.1, 0.2],
            "relative_forward_return": [0.05, -0.01, 0.04, -0.02],
        }
    )

    alpha = select_hybrid_alpha(validation, alpha_grid=[0, 2], top_k=1)
    scored = apply_hybrid_score(validation, alpha)

    assert alpha == 2.0
    assert "hybrid_score" in scored.columns
