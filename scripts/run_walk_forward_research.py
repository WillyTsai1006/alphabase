import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import BENCHMARK_SYMBOL, FEATURES, RESEARCH_CONFIG, TARGET_SYMBOLS  # noqa: E402
from quant_engine import DataAndLabelEngine, ModelTrainer  # noqa: E402
from research import add_forward_returns, evaluate_topk_strategies  # noqa: E402


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
    returns_df = add_forward_returns(
        returns_source,
        horizon_days=RESEARCH_CONFIG["walk_forward"]["embargo_days"],
        benchmark_symbol=BENCHMARK_SYMBOL,
    )
    strategy_metrics = evaluate_topk_strategies(predictions, returns_df, metrics)
    metrics_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_metrics"]
    predictions_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_predictions"]
    strategy_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["fold_strategy_metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    strategy_metrics.to_csv(strategy_path, index=False)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {strategy_path}")


if __name__ == "__main__":
    main()
