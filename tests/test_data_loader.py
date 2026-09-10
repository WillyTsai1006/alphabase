import pandas as pd

import data_loader


class FakeResult:
    def scalar(self):
        return None


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return FakeResult()


class FakeEngine:
    def connect(self):
        return FakeConnection()


def test_fetch_incremental_reports_empty_download_as_failure(monkeypatch):
    monkeypatch.setattr(data_loader.db_manager, "engine", FakeEngine())
    monkeypatch.setattr(data_loader.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    assert data_loader.fetch_incremental("BAD") is False
