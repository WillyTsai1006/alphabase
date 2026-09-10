import pandas as pd

import quant_engine
from config import FEATURES
from quant_engine import DataAndLabelEngine, ModelTrainer


class FakeModel:
    def predict(self, X):
        return [0.6] * len(X)


def make_labeled_frame():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["time", "symbol"])
    df = pd.DataFrame(index=index)
    for feature in FEATURES:
        df[feature] = 1.0
    df["target"] = [0, 1, 0, 1]
    df["exit_time"] = dates
    return df


def test_create_labels_times_out_as_failure():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["time", "symbol"])
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 102.5, 103.5],
            "low": [99.5, 100.5, 101.5, 102.5],
        },
        index=index,
    )

    labeled = DataAndLabelEngine.create_labels(df, horizon_days=1, pt_sl=[10000.0, 10000.0])

    assert labeled.iloc[0]["target"] == 0


def test_walk_forward_train_purges_exit_times(monkeypatch):
    df = make_labeled_frame()
    trainer = ModelTrainer(df, FEATURES)
    test_start = pd.Timestamp("2024-01-03")
    trainer.exits.iloc[1] = pd.Timestamp("2024-01-03")
    monkeypatch.setattr(
        quant_engine,
        "build_walk_forward_splits",
        lambda dates: [
            {
                "train_start": pd.Timestamp("2024-01-01"),
                "train_end": pd.Timestamp("2024-01-02"),
                "test_start": test_start,
                "test_end": pd.Timestamp("2024-01-04"),
            }
        ],
    )
    monkeypatch.setattr(quant_engine.lgb, "train", lambda *args, **kwargs: FakeModel())

    metrics, predictions = trainer.walk_forward_train({"objective": "binary"}, num_boost_round=1)

    assert metrics.iloc[0]["train_rows"] == 1
    assert len(predictions) == 2


def test_final_primary_holdout_purges_cross_boundary_labels(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["time", "symbol"])
    df = pd.DataFrame(index=index)
    for feature in FEATURES:
        df[feature] = 1.0
    df["target"] = [i % 2 for i in range(len(df))]
    df["exit_time"] = dates
    split = dates[-1] - pd.Timedelta(days=180)
    df.loc[(split - pd.Timedelta(days=1), "AAPL"), "exit_time"] = split
    trainer = ModelTrainer(df, FEATURES)
    trained_rows = {}

    def train_model(params, dataset, **kwargs):
        trained_rows["count"] = len(dataset.data)
        return FakeModel()

    monkeypatch.setattr(quant_engine.lgb, "train", train_model)
    monkeypatch.setattr(quant_engine.joblib, "dump", lambda *args, **kwargs: None)

    trainer.train({"objective": "binary"})

    assert trained_rows["count"] == int((dates < split).sum()) - 1
