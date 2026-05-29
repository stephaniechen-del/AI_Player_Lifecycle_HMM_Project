import argparse
import csv
import json
import os
import socket
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import sshtunnel


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_ENV_FILES = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "connection_template" / ".env",
    Path("/Users/stephaniechen/Desktop/weekly_report_dashboard_share/.env"),
    Path("/Users/stephaniechen/Documents/Playground/weekly_report_dashboard_share/.env"),
]


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_key_path():
    raw_path = required_env("SSH_PRIVATE_KEY_PATH")
    candidates = [Path(raw_path).expanduser()]
    if not candidates[0].is_absolute():
        candidates.append(PROJECT_ROOT / candidates[0])
        candidates.append(PROJECT_ROOT / "connection_template" / candidates[0])
    candidates.append(Path("/Users/stephaniechen/Desktop/weekly_report_dashboard_share/oceanhunter-prod-bastion-ec2.pem"))
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"SSH key not found. Checked: {', '.join(str(p) for p in candidates)}")


def to_utc_bound(local_date):
    dt = datetime.combine(date.fromisoformat(local_date), datetime.min.time())
    return dt - timedelta(hours=8)


def connect():
    for env_file in DEFAULT_ENV_FILES:
        load_env_file(env_file)

    tunnel = sshtunnel.SSHTunnelForwarder(
        (required_env("SSH_TUNNEL_HOST"), int(os.environ.get("SSH_TUNNEL_PORT", "22"))),
        ssh_username=required_env("SSH_TUNNEL_USER"),
        ssh_pkey=str(resolve_key_path()),
        remote_bind_address=(required_env("REDSHIFT_HOST"), int(os.environ.get("REDSHIFT_PORT", "5439"))),
        local_bind_address=("127.0.0.1", find_free_port()),
    )
    tunnel.start()
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        dbname=required_env("REDSHIFT_DATABASE"),
        user=required_env("REDSHIFT_USER"),
        password=required_env("REDSHIFT_PASSWORD"),
        sslmode="require" if os.environ.get("REDSHIFT_SSL", "true").lower() == "true" else "prefer",
        connect_timeout=20,
    )
    return tunnel, conn


CORE_DAILY_SQL_TEMPLATE = """
WITH base AS (
    SELECT
        CAST(user_id AS VARCHAR) AS user_id,
        DATE(created_at + interval '8 hour') AS bet_date,
        created_at + interval '8 hour' AS ts_bj,
        COALESCE(event_id::VARCHAR, bullet_id::VARCHAR, '') AS event_sort_id,
        bet::DOUBLE PRECISION AS bet,
        payout::DOUBLE PRECISION AS payout,
        COALESCE(profit::DOUBLE PRECISION, payout::DOUBLE PRECISION - bet::DOUBLE PRECISION) AS profit,
        curr_balance::DOUBLE PRECISION AS curr_balance
    FROM "transform-agfish-game".public.bullet
    WHERE op_code NOT IN ('B26','TST','TSB','TSO')
      AND currency_type = 'CNY'
      AND game_id = 'FM01'
      AND created_at >= '{start_utc}'
      AND created_at <  '{end_utc}'
),
ordered AS (
    SELECT
        *,
        CASE WHEN profit < 0 THEN 1 ELSE 0 END AS is_loss,
        SUM(CASE WHEN profit < 0 THEN 0 ELSE 1 END)
            OVER (
                PARTITION BY user_id, bet_date
                ORDER BY ts_bj, event_sort_id
                ROWS UNBOUNDED PRECEDING
            ) AS loss_group
    FROM base
),
loss_runs AS (
    SELECT
        user_id,
        bet_date,
        loss_group,
        COUNT(*) AS loss_run_length
    FROM ordered
    WHERE is_loss = 1
    GROUP BY 1, 2, 3
),
loss_daily AS (
    SELECT
        user_id,
        bet_date,
        MAX(loss_run_length) AS max_consecutive_loss_count_today
    FROM loss_runs
    GROUP BY 1, 2
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
        CASE WHEN SUM(bet) > 0 THEN SUM(payout) / SUM(bet) END AS rtp_day,
        CASE
            WHEN COUNT(*) > 1 THEN DATEDIFF(second, MIN(ts_bj), MAX(ts_bj))::DOUBLE PRECISION / NULLIF(COUNT(*) - 1, 0)
        END AS bet_interval_day,
        MAX(curr_balance) AS current_balance_day_max
    FROM base
    GROUP BY 1, 2
)
SELECT
    d.*,
    COALESCE(l.max_consecutive_loss_count_today, 0) AS max_consecutive_loss_count_today
FROM daily d
LEFT JOIN loss_daily l
  ON d.user_id = l.user_id
 AND d.bet_date = l.bet_date
ORDER BY 1, 2
"""


def export_core_daily(output_csv, start_date, end_date):
    sql = CORE_DAILY_SQL_TEMPLATE.format(
        start_utc=to_utc_bound(start_date).strftime("%Y-%m-%d %H:%M:%S"),
        end_utc=to_utc_bound(end_date).strftime("%Y-%m-%d %H:%M:%S"),
    )
    output = Path(output_csv)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
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
    df["bet_days_7d"] = active_counts

    hist_bet_amount = g["bet_amount_today"].transform(lambda s: s.expanding().mean().shift(1))
    hist_bet_count = g["bet_count_today"].transform(lambda s: s.expanding().mean().shift(1))
    hist_avg_bet = g["avg_bet_one_time_today"].transform(lambda s: s.expanding().mean().shift(1))

    df["bet_amount_ratio_today_vs_history"] = df["bet_amount_today"] / hist_bet_amount.replace(0, np.nan)
    df["bet_count_ratio_today_vs_history"] = df["bet_count_today"] / hist_bet_count.replace(0, np.nan)
    df["avg_bet_one_time_today_log"] = np.log1p(df["avg_bet_one_time_today"])
    df["rtp_7_bet_days"] = (
        g["payout_today"].transform(lambda s: s.rolling(7, min_periods=1).sum())
        / g["bet_amount_today"].transform(lambda s: s.rolling(7, min_periods=1).sum()).replace(0, np.nan)
    )
    df["current_balance_max_to_avg_bet_ratio"] = df["current_balance_day_max"] / hist_avg_bet.replace(0, np.nan)

    cols = [
        "user_id",
        "bet_date",
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
        "bet_count_today",
        "bet_amount_today",
        "payout_today",
        "profit_today",
        "current_balance_day_max",
    ]
    out = Path(feature_csv)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
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
