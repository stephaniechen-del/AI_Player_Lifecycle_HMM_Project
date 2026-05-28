import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_hmm_features_2026 import connect, to_utc_bound


ROOT = Path(__file__).resolve().parent


DAILY_SQL_TEMPLATE = """
WITH base AS (
    SELECT
        CAST(user_id AS VARCHAR) AS user_id,
        created_at + interval '8 hour' AS ts_bj,
        DATE(created_at + interval '8 hour') AS bet_date,
        bet::DOUBLE PRECISION AS bet,
        payout::DOUBLE PRECISION AS payout,
        COALESCE(profit::DOUBLE PRECISION, payout::DOUBLE PRECISION - bet::DOUBLE PRECISION) AS profit,
        curr_balance::DOUBLE PRECISION AS curr_balance,
        fish_value::VARCHAR AS fish_value,
        killed::INTEGER AS killed,
        bullet_level::DOUBLE PRECISION AS bullet_level
    FROM "transform-agfish-game".public.bullet
    WHERE op_code NOT IN ('B26','TST','TSB','TSO')
      AND currency_type = 'CNY'
      AND game_id = 'FM01'
      AND created_at >= '{start_utc}'
      AND created_at <  '{end_utc}'
),
daily AS (
    SELECT
        user_id,
        bet_date,
        COUNT(*) AS bet_count_today,
        SUM(bet) AS bet_amount_today,
        AVG(bet) AS avg_bet_one_time_today,
        SUM(payout) AS payout_today,
        SUM(profit) AS profit_today,
        SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) AS loss_bullet_count_today,
        CASE WHEN SUM(bet) > 0 THEN SUM(payout) / SUM(bet) END AS rtp_today,
        CASE WHEN SUM(bet) > 0 THEN SUM(profit) / SUM(bet) END AS profit_today_to_bet_today_ratio,
        CASE
            WHEN COUNT(*) > 1 THEN DATEDIFF(second, MIN(ts_bj), MAX(ts_bj))::DOUBLE PRECISION / NULLIF(COUNT(*) - 1, 0)
        END AS bet_frequency_day_mean_approx,
        AVG(bullet_level) AS avg_bullet_level_today,
        SUM(CASE WHEN killed = 1 THEN 1 ELSE 0 END)::DOUBLE PRECISION / NULLIF(COUNT(*), 0) AS weighted_kill_rate_today
    FROM base
    GROUP BY 1, 2
),
last_ts AS (
    SELECT user_id, bet_date, MAX(ts_bj) AS last_ts_bj
    FROM base
    GROUP BY 1, 2
),
last_balance AS (
    SELECT b.user_id, b.bet_date, MAX(b.curr_balance) AS current_balance_end_day
    FROM base b
    JOIN last_ts l
      ON b.user_id = l.user_id
     AND b.bet_date = l.bet_date
     AND b.ts_bj = l.last_ts_bj
    GROUP BY 1, 2
),
fish_counts AS (
    SELECT user_id, bet_date, fish_value, COUNT(*) AS n
    FROM base
    WHERE fish_value IS NOT NULL
    GROUP BY 1, 2, 3
),
fish_probs AS (
    SELECT
        user_id,
        bet_date,
        fish_value,
        n,
        SUM(n) OVER (PARTITION BY user_id, bet_date) AS total_n,
        COUNT(*) OVER (PARTITION BY user_id, bet_date) AS fish_class_count
    FROM fish_counts
),
fish_entropy AS (
    SELECT
        user_id,
        bet_date,
        CASE
            WHEN MAX(fish_class_count) <= 1 THEN 0
            ELSE -SUM((n::DOUBLE PRECISION / total_n)
                * LN(n::DOUBLE PRECISION / total_n)) / LN(MAX(fish_class_count)::DOUBLE PRECISION)
        END AS target_selection_entropy_today
    FROM fish_probs
    GROUP BY user_id, bet_date
)
SELECT
    d.*,
    lb.current_balance_end_day,
    fe.target_selection_entropy_today
FROM daily d
LEFT JOIN last_balance lb
  ON d.user_id = lb.user_id AND d.bet_date = lb.bet_date
LEFT JOIN fish_entropy fe
  ON d.user_id = fe.user_id AND d.bet_date = fe.bet_date
ORDER BY d.user_id, d.bet_date
"""


def export_daily_base(output_csv, start_date, end_date):
    start_utc = to_utc_bound(start_date).strftime("%Y-%m-%d %H:%M:%S")
    end_utc = to_utc_bound(end_date).strftime("%Y-%m-%d %H:%M:%S")
    sql = DAILY_SQL_TEMPLATE.format(start_utc=start_utc, end_utc=end_utc)

    tunnel, conn = connect()
    rows = 0
    output = Path(output_csv)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with conn.cursor() as header_cur:
            header_cur.execute(f"SELECT * FROM ({sql}) q LIMIT 0")
            header = [desc[0] for desc in header_cur.description]

        with conn.cursor(name="hmm_daily_fast_cursor") as cur:
            cur.itersize = 50000
            cur.execute(sql)
            with output.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                for row in cur:
                    writer.writerow(row)
                    rows += 1
                    if rows % 100000 == 0:
                        print(json.dumps({"daily_rows_exported": rows, "daily_csv": str(output)}), flush=True)
    finally:
        conn.close()
        tunnel.stop()

    return output, rows


def build_features(daily_csv, feature_csv):
    df = pd.read_csv(daily_csv, parse_dates=["bet_date"])
    df = df.sort_values(["user_id", "bet_date"]).reset_index(drop=True)

    g = df.groupby("user_id", group_keys=False)
    df["prev_bet_date"] = g["bet_date"].shift(1)
    df["no_bet_streak_days"] = (df["bet_date"] - df["prev_bet_date"]).dt.days.sub(1).fillna(0).clip(lower=0)

    # Calendar 7-day active count, computed on the compact user-day table.
    active_counts = []
    for _, part in df[["user_id", "bet_date"]].groupby("user_id", sort=False):
        dates = part["bet_date"].to_numpy()
        left = 0
        counts = []
        for right, current in enumerate(dates):
            while dates[left] < current - np.timedelta64(6, "D"):
                left += 1
            counts.append(right - left + 1)
        active_counts.extend(counts)
    df["active_days_7d"] = active_counts

    hist_bet_amount = g["bet_amount_today"].transform(lambda s: s.expanding().mean().shift(1))
    hist_bet_count = g["bet_count_today"].transform(lambda s: s.expanding().mean().shift(1))
    hist_avg_bet = g["avg_bet_one_time_today"].transform(lambda s: s.expanding().mean().shift(1))

    df["bet_amount_ratio_today_vs_history"] = df["bet_amount_today"] / hist_bet_amount.replace(0, np.nan)
    df["bet_count_ratio_today_vs_history"] = df["bet_count_today"] / hist_bet_count.replace(0, np.nan)
    df["avg_bet_one_time_today_log"] = np.log1p(df["avg_bet_one_time_today"])
    df["rtp_7d"] = (
        g["payout_today"].transform(lambda s: s.rolling(7, min_periods=1).sum())
        / g["bet_amount_today"].transform(lambda s: s.rolling(7, min_periods=1).sum()).replace(0, np.nan)
    )
    df["current_balance_to_avg_bet_ratio"] = df["current_balance_end_day"] / hist_avg_bet.replace(0, np.nan)

    # Fast V1 approximations:
    # - exact median interval is expensive on the full event table, so we use mean interval from daily min/max/count.
    # - exact max consecutive loss is expensive on the full event table, so we use daily loss bullet count as a pressure proxy.
    df["bet_frequency_day_median"] = df["bet_frequency_day_mean_approx"]
    df["max_consecutive_loss_count_today"] = df["loss_bullet_count_today"]

    threshold = df.groupby("user_id")["avg_bullet_level_today"].transform(lambda s: s.quantile(0.75))
    df["high_level_bullet_ratio_today"] = (df["avg_bullet_level_today"] >= threshold).astype(float)

    feature_cols = [
        "user_id",
        "bet_date",
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
        "high_level_bullet_ratio_today",
        "target_selection_entropy_today",
        "weighted_kill_rate_today",
        "bet_count_today",
        "bet_amount_today",
        "payout_today",
        "profit_today",
        "current_balance_end_day",
    ]
    out = Path(feature_csv)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df[feature_cols].to_csv(out, index=False)
    return out, len(df)


def main():
    parser = argparse.ArgumentParser(description="Fast export of 2026 user-day HMM features from Redshift bullet table.")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2027-01-01")
    parser.add_argument("--daily-csv", default="outputs/hmm_daily_base_fm01_cny_2026.csv")
    parser.add_argument("--feature-csv", default="outputs/hmm_features_fm01_cny_2026_fast.csv")
    args = parser.parse_args()

    daily_csv, daily_rows = export_daily_base(args.daily_csv, args.start_date, args.end_date)
    feature_csv, feature_rows = build_features(daily_csv, args.feature_csv)
    print(json.dumps({
        "daily_csv": str(daily_csv),
        "daily_rows": daily_rows,
        "feature_csv": str(feature_csv),
        "feature_rows": feature_rows,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

