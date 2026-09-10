import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_trading import (  # noqa: E402
    ASSETS,
    advance_paper_state,
    completed_month_ends,
    initialize_paper_state,
)


DEFAULT_STATE_PATH = ROOT / "artifacts" / "paper" / "qqq_gld_state.json"
DEFAULT_STATUS_PATH = ROOT / "artifacts" / "paper" / "qqq_gld_latest_status.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Advance the locked QQQ/GLD paper portfolio")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    return parser.parse_args()


def download_prices(as_of):
    start = (as_of - timedelta(days=500)).isoformat()
    end = (as_of + timedelta(days=1)).isoformat()
    yf.set_tz_cache_location(str(ROOT / "data" / "yfinance_cache"))
    raw = yf.download(
        list(ASSETS),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no paper-trading data")
    opens = raw["Open"].reindex(columns=ASSETS)
    closes = raw["Close"].reindex(columns=ASSETS)
    return opens.dropna(), closes.dropna()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    as_of = pd.Timestamp(args.as_of).date()
    opens, closes = download_prices(as_of)
    if args.state_path.exists():
        state = json.loads(args.state_path.read_text(encoding="utf-8"))
    else:
        month_ends = completed_month_ends(closes.index, as_of)
        if not len(month_ends):
            raise RuntimeError("No completed month is available for paper-state initialization")
        state = initialize_paper_state(as_of, month_ends[-1])
    state, status = advance_paper_state(opens, closes, state, as_of)
    write_json(args.state_path, state)
    write_json(DEFAULT_STATUS_PATH, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
