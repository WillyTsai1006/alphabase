import joblib

from config import BACKTEST_PARAMS, LABEL_PARAMS
from meta_engine import load_meta_model, save_meta_model


def test_label_params_follow_backtest_params():
    assert LABEL_PARAMS['horizon_days'] == BACKTEST_PARAMS['horizon_days']
    assert LABEL_PARAMS['pt_sl'] == [BACKTEST_PARAMS['tp_mult'], BACKTEST_PARAMS['sl_mult']]


def test_meta_model_artifact_round_trip(tmp_path):
    path = tmp_path / "meta.pkl"
    model = {"name": "stub-model"}

    save_meta_model(model, 0.65, path)
    loaded_model, threshold = load_meta_model(path)

    assert loaded_model == model
    assert threshold == 0.65


def test_legacy_meta_model_uses_default_threshold(tmp_path):
    path = tmp_path / "legacy_meta.pkl"
    joblib.dump({"legacy": True}, path)

    loaded_model, threshold = load_meta_model(path)

    assert loaded_model == {"legacy": True}
    assert threshold == BACKTEST_PARAMS['meta_threshold']
