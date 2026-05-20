import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from config import RESEARCH_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_market_data(df, config=RESEARCH_CONFIG):
    """Apply the research data-quality contract before features or labels."""
    if df.empty:
        return df.copy()

    cleaned = df.copy()
    time_values = (
        cleaned.index.get_level_values("time")
        if isinstance(cleaned.index, pd.MultiIndex) and "time" in cleaned.index.names
        else cleaned["time"]
    )
    mask = pd.to_datetime(time_values).between(config["data_start"], config["data_end"])
    cleaned = cleaned.loc[mask].copy()

    required = ["open", "high", "low", "close"]
    for column in required:
        cleaned = cleaned[cleaned[column] > 0]
    if "volume" in cleaned.columns:
        cleaned = cleaned[cleaned["volume"] >= 0]
    cleaned = cleaned[
        (cleaned["high"] >= cleaned[["open", "close"]].max(axis=1))
        & (cleaned["low"] <= cleaned[["open", "close"]].min(axis=1))
    ]

    if isinstance(cleaned.index, pd.MultiIndex):
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    else:
        cleaned = cleaned.drop_duplicates(subset=["time", "symbol"], keep="last")
        cleaned = cleaned.sort_values(["time", "symbol"])
    return cleaned


def build_walk_forward_splits(dates, config=RESEARCH_CONFIG):
    """Create fixed rolling train/test windows for reproducible OOS research."""
    unique_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if unique_dates.empty:
        return []

    wf = config["walk_forward"]
    current_start = pd.Timestamp(config["walk_forward_start"])
    final_end = pd.Timestamp(config["walk_forward_end"])
    splits = []

    while current_start <= final_end:
        train_start = current_start - pd.DateOffset(years=wf["train_years"])
        train_end = current_start - pd.Timedelta(days=wf["embargo_days"] + 1)
        test_end = min(current_start + pd.DateOffset(months=wf["test_months"]) - pd.Timedelta(days=1), final_end)
        if train_start >= pd.Timestamp(config["train_start"]):
            train_mask = (unique_dates >= train_start) & (unique_dates <= train_end)
            test_mask = (unique_dates >= current_start) & (unique_dates <= test_end)
            if train_mask.any() and test_mask.any():
                splits.append(
                    {
                        "train_start": train_start,
                        "train_end": train_end,
                        "test_start": current_start,
                        "test_end": test_end,
                    }
                )
        current_start = current_start + pd.DateOffset(months=wf["step_months"])
    return splits


def compute_calibration_report(y_true, y_prob, n_bins=None):
    """Return Brier score and calibration-bin errors for probability sizing."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y_true.size == 0 or y_true.size != y_prob.size:
        raise ValueError("y_true and y_prob must be non-empty arrays of equal length")

    n_bins = n_bins or RESEARCH_CONFIG["calibration"]["n_bins"]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    max_error = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": idx, "count": 0, "mean_prob": None, "empirical_rate": None, "abs_error": None})
            continue
        mean_prob = float(y_prob[mask].mean())
        empirical_rate = float(y_true[mask].mean())
        abs_error = abs(mean_prob - empirical_rate)
        weight = count / y_true.size
        ece += weight * abs_error
        max_error = max(max_error, abs_error)
        rows.append(
            {
                "bin": idx,
                "count": count,
                "mean_prob": mean_prob,
                "empirical_rate": empirical_rate,
                "abs_error": float(abs_error),
            }
        )

    return {
        "sample_count": int(y_true.size),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(max_error),
        "bins": rows,
    }


def is_kelly_sizing_approved(calibration_report, config=RESEARCH_CONFIG):
    limits = config["calibration"]
    return (
        calibration_report["brier_score"] <= limits["max_brier_score_for_kelly"]
        and calibration_report["expected_calibration_error"]
        <= limits["max_expected_calibration_error_for_kelly"]
    )


def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_status(config=RESEARCH_CONFIG, root=PROJECT_ROOT):
    rows = []
    for name, path in config["artifact_paths"].items():
        if name == "report":
            continue
        full_path = Path(root) / path
        checksum = sha256_file(full_path)
        rows.append(
            {
                "name": name,
                "path": path,
                "status": "present" if checksum else "missing",
                "sha256": checksum or "N/A",
            }
        )
    return rows


def missing_required_artifacts(config=RESEARCH_CONFIG, root=PROJECT_ROOT):
    return [row["name"] for row in artifact_status(config, root) if row["status"] != "present"]


def load_primary_oos_predictions(path=None, root=PROJECT_ROOT):
    path = Path(root) / (path or RESEARCH_CONFIG["artifact_paths"]["primary_walk_forward_predictions"])
    predictions = pd.read_csv(path, parse_dates=["time"])
    required = {"time", "symbol", "primary_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Primary OOS predictions missing columns: {sorted(missing)}")
    return predictions[["time", "symbol", "primary_prob"]].copy()
