import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai-player-hmm")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_missing_zero(summary, out_dir):
    data = summary[["feature", "missing_rate", "zero_rate"]].copy()
    long = data.melt(id_vars="feature", var_name="metric", value_name="rate")
    long["metric"] = long["metric"].map({"missing_rate": "Missing", "zero_rate": "Zero"})

    plt.figure(figsize=(12, 6))
    sns.barplot(data=long, y="feature", x="rate", hue="metric", palette=["#c84c4c", "#4c78a8"])
    plt.title("Missing and Zero Rate by Feature")
    plt.xlabel("Rate")
    plt.ylabel("")
    plt.xlim(0, min(1.0, max(0.05, long["rate"].max() * 1.15)))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    savefig(out_dir / "missing_zero_rates.png")


def plot_distribution_grid(df, out_dir):
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.ravel()

    for ax, col in zip(axes, HMM_FEATURES):
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            ax.set_title(col)
            continue
        lo, hi = s.quantile([0.01, 0.99])
        clipped = s.clip(lo, hi)
        sns.histplot(clipped, bins=60, ax=ax, color="#4c78a8", edgecolor=None)
        ax.axvline(s.median(), color="#c84c4c", linestyle="--", linewidth=1)
        ax.set_title(f"{col} (P1-P99 clipped)")
        ax.set_xlabel("")
        ax.set_ylabel("Rows")

    savefig(out_dir / "feature_distributions_p1_p99.png")


def plot_log_distribution_grid(df, out_dir):
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.ravel()

    for ax, col in zip(axes, HMM_FEATURES):
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s >= 0]
        if s.empty:
            ax.set_title(col)
            continue
        transformed = np.log1p(s)
        lo, hi = transformed.quantile([0.01, 0.99])
        sns.histplot(transformed.clip(lo, hi), bins=60, ax=ax, color="#59a14f", edgecolor=None)
        ax.axvline(transformed.median(), color="#c84c4c", linestyle="--", linewidth=1)
        ax.set_title(f"log1p({col})")
        ax.set_xlabel("")
        ax.set_ylabel("Rows")

    savefig(out_dir / "feature_distributions_log1p.png")


def plot_correlation(corr, out_dir):
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Feature Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    savefig(out_dir / "feature_correlation_heatmap.png")


def plot_daily_trends(daily, out_dir):
    daily = daily.sort_values("bet_date")
    daily["bet_date"] = pd.to_datetime(daily["bet_date"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    axes[0].plot(daily["bet_date"], daily["user_days"], color="#4c78a8")
    axes[0].set_title("Daily User-Days")
    axes[0].set_ylabel("Rows")

    axes[1].plot(daily["bet_date"], daily["bet_amount"], color="#f58518")
    axes[1].set_title("Daily Bet Amount")
    axes[1].set_ylabel("Bet")

    axes[2].plot(daily["bet_date"], daily["rtp"], color="#54a24b")
    axes[2].axhline(1.0, color="#777", linestyle="--", linewidth=1)
    axes[2].set_title("Daily RTP")
    axes[2].set_ylabel("RTP")

    axes[3].plot(daily["bet_date"], daily["avg_no_bet_streak_days"], color="#b279a2")
    axes[3].set_title("Average No-Bet Streak Days")
    axes[3].set_ylabel("Days")
    axes[3].set_xlabel("Bet Date")

    savefig(out_dir / "daily_trends.png")


def plot_ratio_scatter(df, out_dir, sample_size=50000):
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    x = sample["bet_amount_ratio_today_vs_history"].replace([np.inf, -np.inf], np.nan)
    y = sample["bet_count_ratio_today_vs_history"].replace([np.inf, -np.inf], np.nan)
    plot_df = pd.DataFrame({"amount_ratio": x, "count_ratio": y}).dropna()
    plot_df["amount_ratio_log"] = np.log1p(plot_df["amount_ratio"].clip(lower=0))
    plot_df["count_ratio_log"] = np.log1p(plot_df["count_ratio"].clip(lower=0))

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=plot_df,
        x="amount_ratio_log",
        y="count_ratio_log",
        s=8,
        alpha=0.2,
        color="#4c78a8",
        edgecolor=None,
    )
    plt.title("Bet Amount Ratio vs Bet Count Ratio (log1p)")
    plt.xlabel("log1p(bet_amount_ratio_today_vs_history)")
    plt.ylabel("log1p(bet_count_ratio_today_vs_history)")
    savefig(out_dir / "bet_amount_vs_count_ratio_scatter.png")


def plot_rtp_relationship(df, out_dir, sample_size=50000):
    sample = df.sample(min(sample_size, len(df)), random_state=43)
    plot_df = sample[["rtp_day", "rtp_7_bet_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[(plot_df["rtp_day"] <= plot_df["rtp_day"].quantile(0.99)) & (plot_df["rtp_7_bet_days"] <= plot_df["rtp_7_bet_days"].quantile(0.99))]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=plot_df, x="rtp_day", y="rtp_7_bet_days", s=8, alpha=0.2, color="#54a24b", edgecolor=None)
    plt.axline((0, 0), slope=1, color="#777", linestyle="--", linewidth=1)
    plt.title("RTP Day vs RTP 7 Betting Days (P99 clipped sample)")
    plt.xlabel("rtp_day")
    plt.ylabel("rtp_7_bet_days")
    savefig(out_dir / "rtp_day_vs_rtp_7_bet_days.png")


def plot_hmm_state_profile(project_root, out_dir):
    profile_path = project_root / "model_outputs" / "hmm_core_model_k4" / "state_profile_readable.csv"
    if not profile_path.exists():
        return
    profile = pd.read_csv(profile_path)
    english_labels = {
        0: "Stable Active",
        1: "High-Churn Decline",
        2: "Low-Bet Active",
        3: "Reactivation",
    }
    profile["state_label"] = profile["hmm_state"].map(lambda state: f"S{state} {english_labels.get(state, 'State')}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=profile, x="state_label", y="row_share_pct", ax=axes[0], color="#4c78a8")
    axes[0].set_title("HMM State Row Share")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Row Share (%)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=profile, x="state_label", y="churn7_rate_pct", ax=axes[1], color="#c84c4c")
    axes[1].set_title("HMM State Churn 7D Proxy")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Churn Proxy (%)")
    axes[1].tick_params(axis="x", rotation=30)
    savefig(out_dir / "hmm_state_share_and_churn.png")


def update_report(report_path):
    image_block = """

## Visualizations

![Missing and zero rates](figures/missing_zero_rates.png)

![Feature distributions clipped to P1-P99](figures/feature_distributions_p1_p99.png)

![Feature log distributions](figures/feature_distributions_log1p.png)

![Feature correlation heatmap](figures/feature_correlation_heatmap.png)

![Daily trends](figures/daily_trends.png)

![Bet amount ratio vs count ratio](figures/bet_amount_vs_count_ratio_scatter.png)

![RTP relationship](figures/rtp_day_vs_rtp_7_bet_days.png)

![HMM state share and churn proxy](figures/hmm_state_share_and_churn.png)
"""
    text = report_path.read_text(encoding="utf-8")
    if "## Visualizations" not in text:
        report_path.write_text(text.rstrip() + image_block + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create visualizations for HMM feature EDA.")
    parser.add_argument("--input-csv", default="data/raw/hmm_core_features_fm01_cny_2026_to_may_fast.csv")
    parser.add_argument("--eda-dir", default="docs/eda_hmm_features")
    args = parser.parse_args()

    input_csv = resolve(args.input_csv)
    eda_dir = resolve(args.eda_dir)
    out_dir = eda_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["bet_date"])
    summary = pd.read_csv(eda_dir / "feature_summary.csv")
    daily = pd.read_csv(eda_dir / "daily_summary.csv")
    corr = pd.read_csv(eda_dir / "feature_correlation.csv", index_col=0)

    sns.set_theme(style="whitegrid", context="notebook")
    plot_missing_zero(summary, out_dir)
    plot_distribution_grid(df, out_dir)
    plot_log_distribution_grid(df, out_dir)
    plot_correlation(corr, out_dir)
    plot_daily_trends(daily, out_dir)
    plot_ratio_scatter(df, out_dir)
    plot_rtp_relationship(df, out_dir)
    plot_hmm_state_profile(PROJECT_ROOT, out_dir)
    update_report(eda_dir / "hmm_feature_eda_report.md")

    print(f"saved figures to {out_dir}")


if __name__ == "__main__":
    main()
