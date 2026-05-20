import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import FEATURES, RESEARCH_CONFIG  # noqa: E402
from quant_engine import DataAndLabelEngine, ModelTrainer  # noqa: E402


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
    artifact_dir = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_model"]
    artifact_dir = artifact_dir.parent
    metrics, predictions = trainer.walk_forward_train(
        FIXED_LGBM_PARAMS,
        artifact_dir=artifact_dir,
        num_boost_round=300,
    )
    metrics_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_metrics"]
    predictions_path = ROOT / RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_predictions"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    predictions.reset_index().to_csv(predictions_path, index=False)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
