import pandas as pd

import meta_engine
from config import FEATURES
from meta_engine import MetaLabelingEngine


class ConstantPrimary:
    def predict(self, X):
        return [0.0] * len(X)


class FakeMetaModel:
    def predict(self, X):
        return [0.8 if i % 2 else 0.2 for i in range(len(X))]


def make_meta_frame(periods=260):
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["time", "symbol"])
    df = pd.DataFrame(index=index)
    for feature in FEATURES:
        df[feature] = 1.0
    df["target"] = [i % 2 for i in range(len(df))]
    df["exit_time"] = dates
    return df


def test_generate_meta_labels_requires_oos_predictions(monkeypatch):
    engine = MetaLabelingEngine.__new__(MetaLabelingEngine)
    engine.primary_model = ConstantPrimary()
    engine.base_threshold = 0.55
    monkeypatch.setattr(meta_engine, "load_primary_oos_predictions", lambda: (_ for _ in ()).throw(FileNotFoundError()))

    try:
        engine.generate_meta_labels(make_meta_frame(), primary_predictions=None, require_oos_predictions=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected missing OOS predictions to fail")


def test_generate_meta_labels_uses_oos_predictions():
    df = make_meta_frame(periods=4)
    predictions = df.reset_index()[["time", "symbol"]]
    predictions["primary_prob"] = [0.1, 0.8, 0.8, 0.1]
    engine = MetaLabelingEngine.__new__(MetaLabelingEngine)
    engine.primary_model = ConstantPrimary()
    engine.base_threshold = 0.55

    events = engine.generate_meta_labels(df, primary_predictions=predictions)

    assert len(events) == 2
    assert (events["primary_prob"] == 0.8).all()


def test_train_meta_model_separates_threshold_and_calibration(monkeypatch):
    events = make_meta_frame()
    events["primary_prob"] = 0.8
    events["meta_target"] = events["target"]
    engine = MetaLabelingEngine.__new__(MetaLabelingEngine)
    monkeypatch.setattr(meta_engine.lgb, "train", lambda *args, **kwargs: FakeMetaModel())
    saved = {}
    monkeypatch.setattr(
        meta_engine,
        "save_meta_model",
        lambda model, threshold, calibration_report, kelly_sizing_approved: saved.update(
            threshold=threshold,
            calibration_report=calibration_report,
            kelly_sizing_approved=kelly_sizing_approved,
        ),
    )

    _, threshold, calibration = engine.train_meta_model(events)

    assert 0 <= threshold <= 1
    assert calibration["sample_count"] > 0
    assert saved["calibration_report"] == calibration
