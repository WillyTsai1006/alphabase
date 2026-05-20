import pandas as pd

from backtester import InstitutionalBacktester
from config import FEATURES


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


class CrashFreeHMM:
    def predict(self, X):
        return [0] * len(X)


def make_data():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rows = []
    for date in dates:
        row = {
            "time": date,
            "symbol": "AAPL",
            "open": 100.0,
            "high": 110.0,
            "low": 99.0,
            "close": 105.0,
        }
        row.update({feature: 1.0 for feature in FEATURES})
        row["volatility"] = 0.01
        rows.append(row)
    return pd.DataFrame(rows)


def test_meta_threshold_filters_out_low_confidence_signals(monkeypatch):
    data = make_data()
    hmm_data = {"model": CrashFreeHMM(), "map": {0: "Bull", 1: "Crash"}}
    backtester = InstitutionalBacktester(
        data,
        ConstantModel(0.8),
        ConstantModel(0.4),
        hmm_data,
        meta_threshold=0.7,
    )
    spy = pd.DataFrame({"regime": [0, 0]}, index=data["time"])
    monkeypatch.setattr(backtester, "load_spy_data", lambda: spy)

    backtester.generate_signals()
    backtester.run_backtest(max_positions=1)

    assert backtester.positions == {}


def test_meta_threshold_allows_high_confidence_signals(monkeypatch):
    data = make_data()
    hmm_data = {"model": CrashFreeHMM(), "map": {0: "Bull", 1: "Crash"}}
    backtester = InstitutionalBacktester(
        data,
        ConstantModel(0.8),
        ConstantModel(0.8),
        hmm_data,
        meta_threshold=0.7,
    )
    spy = pd.DataFrame({"regime": [0, 0]}, index=data["time"])
    monkeypatch.setattr(backtester, "load_spy_data", lambda: spy)

    backtester.generate_signals()
    backtester.run_backtest(max_positions=1)

    assert "AAPL" in backtester.positions
