import pandas as pd
import pytest

from paper_trading import advance_paper_state, initialize_paper_state


def paper_prices():
    dates = pd.bdate_range("2023-09-01", "2024-03-04")
    closes = pd.DataFrame(index=dates)
    closes["QQQ"] = 100.0 + pd.Series(range(len(dates)), index=dates) * 0.4
    closes["GLD"] = 100.0 + pd.Series(range(len(dates)), index=dates) * 0.1
    closes.loc[:, "QQQ"] += (pd.Series(range(len(dates)), index=dates) % 3) * 0.2
    closes.loc[:, "GLD"] += (pd.Series(range(len(dates)), index=dates) % 5) * 0.1
    opens = closes - 0.1
    return opens, closes


def test_initial_state_stays_in_cash_without_a_new_completed_month():
    opens, closes = paper_prices()
    state = initialize_paper_state("2024-02-15", "2024-01-31")

    updated, status = advance_paper_state(opens, closes, state, "2024-02-15")

    assert updated["fills"] == []
    assert status["nav"] == pytest.approx(100000.0)


def test_new_month_end_signal_fills_at_next_open_and_is_idempotent():
    opens, closes = paper_prices()
    state = initialize_paper_state("2024-02-15", "2024-01-31")

    updated, status = advance_paper_state(opens, closes, state, "2024-03-04")
    repeated, repeated_status = advance_paper_state(opens, closes, updated, "2024-03-04")

    assert len(updated["fills"]) == 1
    assert updated["fills"][0]["signal_date"] == "2024-02-29"
    assert updated["fills"][0]["fill_date"] == "2024-03-01"
    assert sum(updated["fills"][0]["target_weights"].values()) == pytest.approx(1.0)
    assert updated["fills"][0]["cost"] == pytest.approx(200.0)
    assert status["fill_count"] == 1
    assert repeated["fills"] == updated["fills"]
    assert repeated_status["fill_count"] == 1
