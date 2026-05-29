import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

HMM_FEATURES = [
    "bet_days_7d",
    "no_bet_streak_days",
    "bet_interval_day",
    "bet_amount_ratio_today_vs_history",
    "bet_count_ratio_today_vs_history",
    "avg_bet_one_time_today_log",
    "rtp_7_bet_days",
    "rtp_day",
    "max_consecutive_loss_count_today",
    "current_balance_max_to_avg_bet_ratio",
]


def resolve(path):
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def fmt(value, digits=4):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def make_feature_summary(df):
    rows = []
    for col in HMM_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        finite = s.replace([np.inf, -np.inf], np.nan)
        non_null = finite.dropna()
        rows.append(
            {
                "feature": col,
                "rows": len(s),
                "missing_count": int(finite.isna().sum()),
                "missing_rate": float(finite.isna().mean()),
                "zero_count": int((finite == 0).sum(skipna=True)),
                "zero_rate": float((finite == 0).mean(skipna=True)),
                "mean": float(non_null.mean()) if len(non_null) else np.nan,
                "std": float(non_null.std()) if len(non_null) else np.nan,
                "min": float(non_null.min()) if len(non_null) else np.nan,
                "p01": float(non_null.quantile(0.01)) if len(non_null) else np.nan,
                "p05": float(non_null.quantile(0.05)) if len(non_null) else np.nan,
                "p25": float(non_null.quantile(0.25)) if len(non_null) else np.nan,
                "median": float(non_null.quantile(0.50)) if len(non_null) else np.nan,
                "p75": float(non_null.quantile(0.75)) if len(non_null) else np.nan,
                "p95": float(non_null.quantile(0.95)) if len(non_null) else np.nan,
                "p99": float(non_null.quantile(0.99)) if len(non_null) else np.nan,
                "max": float(non_null.max()) if len(non_null) else np.nan,
                "skew": float(non_null.skew()) if len(non_null) > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_daily_summary(df):
    daily = (
        df.groupby("bet_date")
        .agg(
            user_days=("user_id", "size"),
            users=("user_id", "nunique"),
            bet_amount=("bet_amount_today", "sum"),
            payout=("payout_today", "sum"),
            profit=("profit_today", "sum"),
            avg_bet_days_7d=("bet_days_7d", "mean"),
            avg_no_bet_streak_days=("no_bet_streak_days", "mean"),
        )
        .reset_index()
    )
    daily["rtp"] = daily["payout"] / daily["bet_amount"].replace(0, np.nan)
    return daily


def make_report(df, summary, corr, out_dir):
    date_min = df["bet_date"].min()
    date_max = df["bet_date"].max()
    lines = [
        "# HMM Feature EDA",
        "",
        "## Dataset",
        "",
        f"- Rows: `{len(df):,}` user-days",
        f"- Users: `{df['user_id'].nunique():,}`",
        f"- Date range: `{date_min}` to `{date_max}`",
        "",
        "## Feature Summary",
        "",
        "| Feature | Missing | Zero | Mean | P50 | P95 | P99 | Max | Skew |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {feature} | {missing_rate:.2%} | {zero_rate:.2%} | {mean} | {median} | {p95} | {p99} | {max} | {skew} |".format(
                feature=row["feature"],
                missing_rate=row["missing_rate"],
                zero_rate=row["zero_rate"],
                mean=fmt(row["mean"]),
                median=fmt(row["median"]),
                p95=fmt(row["p95"]),
                p99=fmt(row["p99"]),
                max=fmt(row["max"]),
                skew=fmt(row["skew"]),
            )
        )

    corr_pairs = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1 :]:
            value = corr.loc[col1, col2]
            if pd.notna(value):
                corr_pairs.append((col1, col2, value))
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:15]

    lines += [
        "",
        "## Top Absolute Correlations",
        "",
        "| Feature A | Feature B | Pearson Corr |",
        "|---|---|---:|",
    ]
    for a, b, value in corr_pairs:
        lines.append(f"| {a} | {b} | {value:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Ratio features have missing values on each user's first betting day because no prior betting-day history exists.",
        "- `rtp_7_bet_days` is rolling over the latest 7 betting days, not 7 calendar days.",
        "- `bet_interval_day` is the daily interval proxy based on first/last bullet timestamp and bullet count.",
        "- `current_balance_max_to_avg_bet_ratio` can be very right-skewed because it divides max daily balance by historical average single bet.",
    ]

    (out_dir / "hmm_feature_eda_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EDA for HMM feature CSV.")
    parser.add_argument("--input-csv", default="data/raw/hmm_core_features_fm01_cny_2026_to_may_fast.csv")
    parser.add_argument("--output-dir", default="docs/eda_hmm_features")
    args = parser.parse_args()

    input_csv = resolve(args.input_csv)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["bet_date"])
    summary = make_feature_summary(df)
    daily = make_daily_summary(df)
    corr = df[HMM_FEATURES].replace([np.inf, -np.inf], np.nan).corr()

    summary.to_csv(out_dir / "feature_summary.csv", index=False)
    daily.to_csv(out_dir / "daily_summary.csv", index=False)
    corr.to_csv(out_dir / "feature_correlation.csv")

    make_report(df, summary, corr, out_dir)

    metadata = {
        "input_csv": str(input_csv),
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "date_min": str(df["bet_date"].min().date()),
        "date_max": str(df["bet_date"].max().date()),
        "features": HMM_FEATURES,
    }
    (out_dir / "eda_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

