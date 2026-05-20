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


class CrashHMM:
    def predict(self, X):
        return [1] * len(X)


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


def make_friday_monday_data():
    data = make_data()
    data["time"] = pd.to_datetime(["2024-01-05", "2024-01-08"])
    return data


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

    assert backtester.trades_df.iloc[0]["symbol"] == "AAPL"
    assert backtester.trades_df.iloc[0]["reason"] == "Take Profit"


def test_t_plus_one_uses_next_trading_day_not_calendar_day(monkeypatch):
    data = make_friday_monday_data()
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

    assert backtester.trades_df.iloc[0]["exit_date"] == pd.Timestamp("2024-01-08")


def test_backtester_uses_oos_primary_predictions(monkeypatch):
    data = make_data()
    predictions = data[["time", "symbol"]].copy()
    predictions["primary_prob"] = 0.8
    hmm_data = {"model": CrashFreeHMM(), "map": {0: "Bull", 1: "Crash"}}
    backtester = InstitutionalBacktester(
        data,
        ConstantModel(0.0),
        ConstantModel(0.8),
        hmm_data,
        meta_threshold=0.7,
        primary_predictions=predictions,
        require_oos_predictions=True,
    )
    spy = pd.DataFrame({"regime": [0, 0]}, index=data["time"])
    monkeypatch.setattr(backtester, "load_spy_data", lambda: spy)

    backtester.generate_signals()

    assert (backtester.data["primary_prob"] == 0.8).all()


def test_hmm_crash_filter_only_applies_after_oos_start(monkeypatch):
    data = make_data()
    hmm_data = {"model": CrashHMM(), "map": {0: "Bull", 1: "Crash"}, "oos_start": "2025-01-01"}
    backtester = InstitutionalBacktester(
        data,
        ConstantModel(0.8),
        ConstantModel(0.8),
        hmm_data,
        meta_threshold=0.7,
    )
    spy = pd.DataFrame({"regime": [1, 1]}, index=data["time"])
    monkeypatch.setattr(backtester, "load_spy_data", lambda: spy)

    backtester.generate_signals()
    backtester.run_backtest(max_positions=1)

    assert not backtester.trades_df.empty
