import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

RAW_FEATURES = [
    "active_days_7d",
    "no_bet_streak_days",
    "bet_frequency_day_median",
    "bet_amount_ratio_today_vs_history",
    "bet_count_ratio_today_vs_history",
    "avg_bet_one_time_today_log",
    "rtp_7d",
    "profit_today_to_bet_today_ratio",
    "max_consecutive_loss_count_today",
    "current_balance_to_avg_bet_ratio",
]


def resolve(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def prepare(input_csv, output_csv, stats_json):
    df = pd.read_csv(resolve(input_csv), parse_dates=["bet_date"])

    # Fast-export caveat: this column is a pressure proxy in the core fast export.
    df["loss_bullet_count_today_proxy"] = df["max_consecutive_loss_count_today"]

    work = df[["user_id", "bet_date"] + RAW_FEATURES + ["loss_bullet_count_today_proxy"]].copy()

    work["bet_frequency_day_median"] = work["bet_frequency_day_median"].fillna(0)
    work["bet_amount_ratio_today_vs_history"] = work["bet_amount_ratio_today_vs_history"].fillna(1)
    work["bet_count_ratio_today_vs_history"] = work["bet_count_ratio_today_vs_history"].fillna(1)
    work["current_balance_to_avg_bet_ratio"] = work["current_balance_to_avg_bet_ratio"].fillna(0)

    stats = {}
    for col in RAW_FEATURES:
        series = work[col].replace([np.inf, -np.inf], np.nan)
        fill_value = float(series.median(skipna=True)) if series.notna().any() else 0.0
        series = series.fillna(fill_value)
        p01 = float(series.quantile(0.01))
        p99 = float(series.quantile(0.99))
        clipped = series.clip(p01, p99)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0)) or 1.0
        work[col] = series
        work[f"{col}_winsor"] = clipped
        work[f"{col}_z"] = (clipped - mean) / std
        stats[col] = {"fill_value": fill_value, "p01": p01, "p99": p99, "mean": mean, "std": std}

    z_cols = ["user_id", "bet_date"] + [f"{col}_z" for col in RAW_FEATURES]
    output = resolve(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    work[z_cols].to_csv(output, index=False)

    stats_path = resolve(stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"rows": len(work), "output_csv": str(output), "stats_json": str(stats_path)}


def main():
    parser = argparse.ArgumentParser(description="Prepare model-ready normalized HMM core features.")
    parser.add_argument("--input-csv", default="outputs/hmm_core_features_fm01_cny_2026_to_may_fast.csv")
    parser.add_argument("--output-csv", default="outputs/hmm_core_features_model_ready_2026_to_may_fast.csv")
    parser.add_argument("--stats-json", default="outputs/hmm_core_features_model_ready_stats_2026_to_may_fast.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_csv, args.output_csv, args.stats_json), ensure_ascii=False))


if __name__ == "__main__":
    main()

