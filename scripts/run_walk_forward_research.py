import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import BENCHMARK_SYMBOL, FEATURES, RESEARCH_CONFIG, TARGET_SYMBOLS  # noqa: E402
from quant_engine import DataAndLabelEngine, ModelTrainer  # noqa: E402
from research import (  # noqa: E402
    add_forward_returns,
    apply_hybrid_score,
    build_purged_ranker_masks,
    evaluate_topk_strategies,
    select_hybrid_alpha,
)


FIXED_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "min_child_samples": 50,
    "is_unbalance": True,
}

RANKER_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 16,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "min_child_samples": 80,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
}


def build_ranker_predictions(feature_df, returns_df, folds, top_k=3):
    data = returns_df[returns_df["symbol"].isin(RESEARCH_CONFIG["universe"])].copy()
    data = data.sort_values(["symbol", "time"])
    data["return_5"] = data.groupby("symbol")["close"].pct_change(5)
    data["return_10"] = data.groupby("symbol")["close"].pct_change(10)
    data["volatility_ranker_20"] = data.groupby("symbol")["log_return"].rolling(20).std().reset_index(level=0, drop=True)
    ranker_features = FEATURES + ["return_5", "return_10", "return_20", "volatility_ranker_20"]
    data = data.dropna(subset=ranker_features + ["relative_forward_return", "forward_time"])
    data = data.set_index(["time", "symbol"], drop=False)
    X = data[ranker_features]
    y = data["relative_forward_return"]
    dates = X.index.get_level_values("time")
    label_end_times = data["forward_time"]
    rows = []

    for _, fold in folds.iterrows():
        train_mask = (dates >= fold["train_start"]) & (dates <= fold["train_end"])
        train_dates = dates[train_mask].unique().sort_values()
        if len(train_dates) < 5 or train_mask.sum() == 0:
            continue
        validation_start = train_dates[int(len(train_dates) * 0.80)]
        masks = build_purged_ranker_masks(dates, label_end_times, fold, validation_start)
        fit_mask = masks["fit"]
        validation_mask = masks["validation"]
        final_train_mask = masks["final_train"]
        test_mask = masks["test"]
        if fit_mask.sum() == 0 or validation_mask.sum() == 0 or final_train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        validation_model = lgb.train(
            RANKER_PARAMS,
            lgb.Dataset(X[fit_mask], label=y[fit_mask]),
            num_boost_round=200,
        )
        validation_df = data.loc[X[validation_mask].index][
            ["time", "symbol", "relative_forward_return", "return_20"]
        ].copy()
        validation_df["rank_score"] = validation_model.predict(X[validation_mask])
        alpha = select_hybrid_alpha(validation_df.reset_index(drop=True), top_k=top_k)

        model = lgb.train(
            RANKER_PARAMS,
            lgb.Dataset(X[final_train_mask], label=y[final_train_mask]),
            num_boost_round=200,
        )
        test_df = data.loc[X[test_mask].index][["time", "symbol", "return_20"]].copy()
        test_df["rank_score"] = model.predict(X[test_mask])
        scored = apply_hybrid_score(test_df.reset_index(drop=True), alpha)
        scored["fold"] = int(fold["fold"])
        scored["hybrid_alpha"] = alpha
        rows.append(scored[["time", "symbol", "rank_score", "hybrid_score", "hybrid_alpha", "fold"]])

    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["time", "symbol", "rank_score", "hybrid_score", "hybrid_alpha", "fold"])
    )


def main():
    df = DataAndLabelEngine.load_data(RESEARCH_CONFIG["universe"])
    df = DataAndLabelEngine.create_labels(df)
    trainer = ModelTrainer(df, FEATURES)
    metrics, predictions = trainer.walk_forward_train(
        FIXED_LGBM_PARAMS,
        artifact_dir=None,
        num_boost_round=300,
    )
    metrics = metrics.reset_index(drop=True)
    metrics["fold"] = metrics.index + 1
    predictions = predictions.reset_index()
    returns_source = DataAndLabelEngine.load_data(TARGET_SYMBOLS)
    returns_source["volatility"] = returns_source.groupby(level="symbol")["close"].pct_change().ewm(span=100).std()
    returns_df = add_forward_returns(
        returns_source,
        horizon_days=RESEARCH_CONFIG["walk_forward"]["embargo_days"],
        benchmark_symbol=BENCHMARK_SYMBOL,
    )
    ranker_predictions = build_ranker_predictions(df, returns_df, metrics)
    strategy_predictions = ranker_predictions.rename(columns={"hybrid_score": "primary_prob"})
    strategy_metrics = evaluate_topk_strategies(strategy_predictions, returns_df, metrics)
    metrics_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_metrics"]
    predictions_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_predictions"]
    ranker_predictions_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["ranker_walk_forward_predictions"]
    strategy_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["fold_strategy_metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    ranker_predictions.to_csv(ranker_predictions_path, index=False)
    strategy_metrics.to_csv(strategy_path, index=False)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {ranker_predictions_path}")
    print(f"Wrote {strategy_path}")


if __name__ == "__main__":
    main()
