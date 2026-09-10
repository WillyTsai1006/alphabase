import joblib

from config import BACKTEST_PARAMS, LABEL_PARAMS
from meta_engine import load_meta_artifact, load_meta_model, save_meta_model


def test_label_params_follow_backtest_params():
    assert LABEL_PARAMS['horizon_days'] == BACKTEST_PARAMS['horizon_days']
    assert LABEL_PARAMS['pt_sl'] == [BACKTEST_PARAMS['tp_mult'], BACKTEST_PARAMS['sl_mult']]


def test_meta_model_artifact_round_trip(tmp_path):
    path = tmp_path / "meta.pkl"
    model = {"name": "stub-model"}

    calibration = {"brier_score": 0.1, "expected_calibration_error": 0.02}
    save_meta_model(model, 0.65, path, calibration_report=calibration, kelly_sizing_approved=True)
    loaded_model, threshold = load_meta_model(path)
    artifact = load_meta_artifact(path)

    assert loaded_model == model
    assert threshold == 0.65
    assert artifact["calibration_report"] == calibration
    assert artifact["kelly_sizing_approved"] is True


def test_legacy_meta_model_uses_default_threshold(tmp_path):
    path = tmp_path / "legacy_meta.pkl"
    joblib.dump({"legacy": True}, path)

    loaded_model, threshold = load_meta_model(path)

    assert loaded_model == {"legacy": True}
    assert threshold == BACKTEST_PARAMS['meta_threshold']
