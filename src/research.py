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
    time_index = pd.to_datetime(time_values)
    mask = (time_index >= pd.Timestamp(config["data_start"])) & (time_index <= pd.Timestamp(config["data_end"]))
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


def add_forward_returns(df, horizon_days, benchmark_symbol=None):
    """Add absolute and benchmark-relative forward returns for ranking evaluation."""
    data = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values(["symbol", "time"]).copy()
    if "return_20" not in data.columns:
        data["return_20"] = data.groupby("symbol")["close"].pct_change(20)
    data["forward_time"] = data.groupby("symbol")["time"].shift(-horizon_days)
    data["forward_close"] = data.groupby("symbol")["close"].shift(-horizon_days)
    data["forward_return"] = (data["forward_close"] / data["close"]) - 1.0
    if benchmark_symbol and benchmark_symbol in set(data["symbol"]):
        benchmark = data[data["symbol"] == benchmark_symbol][["time", "forward_return"]].rename(
            columns={"forward_return": "benchmark_forward_return"}
        )
        data = data.merge(benchmark, on="time", how="left")
        data["relative_forward_return"] = data["forward_return"] - data["benchmark_forward_return"]
    else:
        data["benchmark_forward_return"] = np.nan
        data["relative_forward_return"] = data["forward_return"]
    return data


def labels_end_before(label_end_times, boundary):
    """Return a mask for labels fully observed before a validation boundary."""
    return pd.DatetimeIndex(pd.to_datetime(label_end_times)) < pd.Timestamp(boundary)


def build_purged_ranker_masks(dates, label_end_times, fold, validation_start):
    """Build ranker masks without allowing forward-return labels across boundaries."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(fold["test_start"])

    train_window = (dates >= pd.Timestamp(fold["train_start"])) & (dates <= pd.Timestamp(fold["train_end"]))
    test = (dates >= test_start) & (dates <= pd.Timestamp(fold["test_end"]))
    fit = train_window & (dates < validation_start) & labels_end_before(label_end_times, validation_start)
    validation = train_window & (dates >= validation_start) & labels_end_before(label_end_times, test_start)
    final_train = train_window & labels_end_before(label_end_times, test_start)
    return {
        "fit": fit,
        "validation": validation,
        "final_train": final_train,
        "test": test,
    }


def evaluate_topk_strategies(predictions, returns_df, folds, top_k=3, score_column="primary_prob"):
    """Compare ML top-k picks with simple momentum and equal-weight baselines per fold."""
    pred = predictions.copy()
    pred["time"] = pd.to_datetime(pred["time"])
    returns = returns_df.copy()
    returns["time"] = pd.to_datetime(returns["time"])
    merged = pred.merge(
        returns[["time", "symbol", "forward_return", "relative_forward_return", "return_20"]],
        on=["time", "symbol"],
        how="left",
    ).dropna(subset=["forward_return", "relative_forward_return"])

    rows = []
    for _, fold in folds.iterrows():
        fold_data = merged[
            (merged["time"] >= pd.Timestamp(fold["test_start"]))
            & (merged["time"] <= pd.Timestamp(fold["test_end"]))
        ].copy()
        if fold_data.empty:
            rows.append({"fold": int(fold.get("fold", len(rows) + 1)), "strategy": "ml_topk", "trade_count": 0})
            continue

        daily = fold_data.groupby("time", group_keys=False)
        strategies = {
            "ml_topk": daily.apply(lambda g: g.nlargest(top_k, score_column), include_groups=False),
            "momentum_topk": daily.apply(lambda g: g.nlargest(top_k, "return_20"), include_groups=False),
            "equal_weight": fold_data,
        }
        for name, picks in strategies.items():
            picks = picks.dropna(subset=["forward_return"])
            rows.append(
                {
                    "fold": int(fold.get("fold", len(rows) + 1)),
                    "strategy": name,
                    "trade_count": int(len(picks)),
                    "mean_forward_return": float(picks["forward_return"].mean()) if not picks.empty else np.nan,
                    "mean_relative_return": float(picks["relative_forward_return"].mean()) if not picks.empty else np.nan,
                    "hit_rate": float((picks["forward_return"] > 0).mean()) if not picks.empty else np.nan,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                }
            )
    return pd.DataFrame(rows)


def daily_zscore(df, column):
    grouped = df.groupby("time")[column]
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[column] - grouped.transform("mean")) / std).fillna(0)


def topk_mean_relative_return(df, score_column, top_k=3):
    if df.empty:
        return np.nan
    picks = df.groupby("time", group_keys=False).apply(
        lambda group: group.nlargest(top_k, score_column),
        include_groups=False,
    )
    if picks.empty:
        return np.nan
    return float(picks["relative_forward_return"].mean())


def select_hybrid_alpha(validation_df, alpha_grid=None, top_k=3):
    alpha_grid = alpha_grid or [-2, -1, -0.5, 0, 0.5, 1, 2]
    if validation_df.empty:
        return 0.0
    validation = validation_df.copy()
    validation["rank_z"] = daily_zscore(validation, "rank_score")
    validation["momentum_z"] = daily_zscore(validation, "return_20")
    best_alpha, best_score = 0.0, -np.inf
    for alpha in alpha_grid:
        validation["hybrid_score"] = validation["momentum_z"] + (alpha * validation["rank_z"])
        score = topk_mean_relative_return(validation, "hybrid_score", top_k=top_k)
        if pd.notna(score) and score > best_score:
            best_alpha = float(alpha)
            best_score = score
    return best_alpha


def apply_hybrid_score(df, alpha):
    scored = df.copy()
    scored["rank_z"] = daily_zscore(scored, "rank_score")
    scored["momentum_z"] = daily_zscore(scored, "return_20")
    scored["hybrid_score"] = scored["momentum_z"] + (alpha * scored["rank_z"])
    return scored


def summarize_strategy_metrics(strategy_metrics):
    if strategy_metrics is None or strategy_metrics.empty:
        return {"status": "missing", "candidate_gate_passed": False}
    pivot = strategy_metrics.pivot_table(
        index="fold",
        columns="strategy",
        values="mean_relative_return",
        aggfunc="mean",
    )
    if "ml_topk" not in pivot or "momentum_topk" not in pivot:
        return {"status": "missing", "candidate_gate_passed": False}
    spread = pivot["ml_topk"] - pivot["momentum_topk"]
    valid_spread = spread.dropna()
    winning_folds = int((valid_spread > 0).sum())
    return {
        "status": "present",
        "fold_count": int(valid_spread.shape[0]),
        "ml_mean_relative_return": float(pivot["ml_topk"].mean()),
        "momentum_mean_relative_return": float(pivot["momentum_topk"].mean()),
        "equal_weight_mean_relative_return": float(pivot.get("equal_weight", pd.Series(dtype=float)).mean()),
        "ml_minus_momentum": float(valid_spread.mean()),
        "ml_beats_momentum_folds": winning_folds,
        "candidate_gate_passed": bool(
            not valid_spread.empty
            and valid_spread.mean() > 0
            and winning_folds > len(valid_spread) / 2
        ),
    }


def research_quality_status(config=RESEARCH_CONFIG, root=PROJECT_ROOT):
    metrics_path = Path(root) / config["artifact_paths"]["primary_walk_forward_metrics"]
    strategy_path = Path(root) / config["artifact_paths"].get("fold_strategy_metrics", "")
    if not metrics_path.exists():
        return {"status": "missing", "primary_edge_approved": False}
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {"status": "empty", "primary_edge_approved": False}
    mean_auc = float(metrics["auc"].mean())
    min_auc = float(metrics["auc"].min())
    gates = config["quality_gates"]
    approved = mean_auc >= gates["min_primary_mean_auc"] and min_auc >= gates["min_primary_fold_auc"]
    strategy_summary = (
        summarize_strategy_metrics(pd.read_csv(strategy_path))
        if strategy_path and strategy_path.exists()
        else {"status": "missing", "candidate_gate_passed": False}
    )
    primary_edge_approved = bool(approved)
    return {
        "status": "present",
        "fold_count": int(len(metrics)),
        "mean_auc": mean_auc,
        "min_auc": min_auc,
        "max_auc": float(metrics["auc"].max()),
        "primary_edge_approved": primary_edge_approved,
        "required_mean_auc": gates["min_primary_mean_auc"],
        "required_min_fold_auc": gates["min_primary_fold_auc"],
        "strategy_summary": strategy_summary,
        "strategy_candidate_gate_passed": bool(strategy_summary.get("candidate_gate_passed", False)),
    }
