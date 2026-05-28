import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_hmm_features_2026 import connect, to_utc_bound


ROOT = Path(__file__).resolve().parent


CORE_DAILY_SQL_TEMPLATE = """
SELECT
    CAST(user_id AS VARCHAR) AS user_id,
    DATE(created_at + interval '8 hour') AS bet_date,
    COUNT(*) AS bet_count_today,
    SUM(bet::DOUBLE PRECISION) AS bet_amount_today,
    AVG(bet::DOUBLE PRECISION) AS avg_bet_one_time_today,
    SUM(payout::DOUBLE PRECISION) AS payout_today,
    SUM(COALESCE(profit::DOUBLE PRECISION, payout::DOUBLE PRECISION - bet::DOUBLE PRECISION)) AS profit_today,
    SUM(CASE WHEN COALESCE(profit::DOUBLE PRECISION, payout::DOUBLE PRECISION - bet::DOUBLE PRECISION) < 0 THEN 1 ELSE 0 END) AS loss_bullet_count_today,
    CASE WHEN SUM(bet::DOUBLE PRECISION) > 0 THEN SUM(payout::DOUBLE PRECISION) / SUM(bet::DOUBLE PRECISION) END AS rtp_today,
    CASE WHEN SUM(bet::DOUBLE PRECISION) > 0 THEN SUM(COALESCE(profit::DOUBLE PRECISION, payout::DOUBLE PRECISION - bet::DOUBLE PRECISION)) / SUM(bet::DOUBLE PRECISION) END AS profit_today_to_bet_today_ratio,
    CASE
        WHEN COUNT(*) > 1 THEN DATEDIFF(second, MIN(created_at + interval '8 hour'), MAX(created_at + interval '8 hour'))::DOUBLE PRECISION / NULLIF(COUNT(*) - 1, 0)
    END AS bet_frequency_day_mean_approx,
    MAX(curr_balance::DOUBLE PRECISION) AS current_balance_day_max
FROM "transform-agfish-game".public.bullet
WHERE op_code NOT IN ('B26','TST','TSB','TSO')
  AND currency_type = 'CNY'
  AND game_id = 'FM01'
  AND created_at >= '{start_utc}'
  AND created_at <  '{end_utc}'
GROUP BY 1, 2
ORDER BY 1, 2
"""


def export_core_daily(output_csv, start_date, end_date):
    sql = CORE_DAILY_SQL_TEMPLATE.format(
        start_utc=to_utc_bound(start_date).strftime("%Y-%m-%d %H:%M:%S"),
        end_utc=to_utc_bound(end_date).strftime("%Y-%m-%d %H:%M:%S"),
    )
    output = Path(output_csv)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    tunnel, conn = connect()
    rows = 0
    try:
        with conn.cursor() as header_cur:
            header_cur.execute(f"SELECT * FROM ({sql}) q LIMIT 0")
            header = [desc[0] for desc in header_cur.description]
        with conn.cursor(name="hmm_core_daily_cursor") as cur:
            cur.itersize = 50000
            cur.execute(sql)
            with output.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                for row in cur:
                    writer.writerow(row)
                    rows += 1
                    if rows % 100000 == 0:
                        print(json.dumps({"core_daily_rows_exported": rows, "daily_csv": str(output)}), flush=True)
    finally:
        conn.close()
        tunnel.stop()
    return output, rows


def build_core_features(daily_csv, feature_csv):
    df = pd.read_csv(daily_csv, parse_dates=["bet_date"])
    df = df.sort_values(["user_id", "bet_date"]).reset_index(drop=True)
    g = df.groupby("user_id", group_keys=False)

    df["prev_bet_date"] = g["bet_date"].shift(1)
    df["no_bet_streak_days"] = (df["bet_date"] - df["prev_bet_date"]).dt.days.sub(1).fillna(0).clip(lower=0)

    active_counts = []
    for _, part in df[["user_id", "bet_date"]].groupby("user_id", sort=False):
        dates = part["bet_date"].to_numpy()
        left = 0
        for right, current in enumerate(dates):
            while dates[left] < current - np.timedelta64(6, "D"):
                left += 1
            active_counts.append(right - left + 1)
    df["active_days_7d"] = active_counts

    hist_bet_amount = g["bet_amount_today"].transform(lambda s: s.expanding().mean().shift(1))
    hist_bet_count = g["bet_count_today"].transform(lambda s: s.expanding().mean().shift(1))
    hist_avg_bet = g["avg_bet_one_time_today"].transform(lambda s: s.expanding().mean().shift(1))

    df["bet_frequency_day_median"] = df["bet_frequency_day_mean_approx"]
    df["bet_amount_ratio_today_vs_history"] = df["bet_amount_today"] / hist_bet_amount.replace(0, np.nan)
    df["bet_count_ratio_today_vs_history"] = df["bet_count_today"] / hist_bet_count.replace(0, np.nan)
    df["avg_bet_one_time_today_log"] = np.log1p(df["avg_bet_one_time_today"])
    df["rtp_7d"] = (
        g["payout_today"].transform(lambda s: s.rolling(7, min_periods=1).sum())
        / g["bet_amount_today"].transform(lambda s: s.rolling(7, min_periods=1).sum()).replace(0, np.nan)
    )
    df["max_consecutive_loss_count_today"] = df["loss_bullet_count_today"]
    df["current_balance_to_avg_bet_ratio"] = df["current_balance_day_max"] / hist_avg_bet.replace(0, np.nan)

    cols = [
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
        "bet_count_today",
        "bet_amount_today",
        "payout_today",
        "profit_today",
        "current_balance_day_max",
    ]
    out = Path(feature_csv)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out, index=False)
    return out, len(df)


def main():
    parser = argparse.ArgumentParser(description="Fast core user-day HMM features for 2026 FM01 CNY bullet data.")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2027-01-01")
    parser.add_argument("--daily-csv", default="outputs/hmm_core_daily_base_fm01_cny_2026.csv")
    parser.add_argument("--feature-csv", default="outputs/hmm_core_features_fm01_cny_2026_fast.csv")
    args = parser.parse_args()

    daily_csv, daily_rows = export_core_daily(args.daily_csv, args.start_date, args.end_date)
    feature_csv, feature_rows = build_core_features(daily_csv, args.feature_csv)
    print(json.dumps({
        "daily_csv": str(daily_csv),
        "daily_rows": daily_rows,
        "feature_csv": str(feature_csv),
        "feature_rows": feature_rows,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

