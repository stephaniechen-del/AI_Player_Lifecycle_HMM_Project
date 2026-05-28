import argparse
import csv
import json
import os
import socket
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg2
import sshtunnel


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILES = [
    ROOT / ".env",
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
        candidates.append(ROOT / candidates[0])
    candidates.append(Path("/Users/stephaniechen/Desktop/weekly_report_dashboard_share/oceanhunter-prod-bastion-ec2.pem"))
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"SSH key not found. Checked: {', '.join(str(p) for p in candidates)}")


def connect():
    for env_file in DEFAULT_ENV_FILES:
        load_env_file(env_file)

    local_port = find_free_port()
    tunnel = sshtunnel.SSHTunnelForwarder(
        (required_env("SSH_TUNNEL_HOST"), int(os.environ.get("SSH_TUNNEL_PORT", "22"))),
        ssh_username=required_env("SSH_TUNNEL_USER"),
        ssh_pkey=str(resolve_key_path()),
        remote_bind_address=(required_env("REDSHIFT_HOST"), int(os.environ.get("REDSHIFT_PORT", "5439"))),
        local_bind_address=("127.0.0.1", local_port),
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


FEATURE_SQL_TEMPLATE = """
WITH base AS (
    SELECT
        CAST(user_id AS VARCHAR) AS user_id,
        created_at + interval '8 hour' AS ts_bj,
        DATE(created_at + interval '8 hour') AS bet_date,
        COALESCE(event_id::VARCHAR, bullet_id::VARCHAR, '') AS event_sort_id,
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
ordered AS (
    SELECT
        *,
        LAG(ts_bj) OVER (PARTITION BY user_id, bet_date ORDER BY ts_bj, event_sort_id) AS prev_ts_bj,
        CASE WHEN profit < 0 THEN 1 ELSE 0 END AS is_loss,
        SUM(CASE WHEN profit < 0 THEN 0 ELSE 1 END)
            OVER (PARTITION BY user_id, bet_date ORDER BY ts_bj, event_sort_id ROWS UNBOUNDED PRECEDING) AS loss_group,
        ROW_NUMBER() OVER (PARTITION BY user_id, bet_date ORDER BY ts_bj DESC, event_sort_id DESC) AS rn_desc
    FROM base
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
        CASE WHEN SUM(bet) > 0 THEN SUM(payout) / SUM(bet) END AS rtp_today,
        CASE WHEN SUM(bet) > 0 THEN SUM(profit) / SUM(bet) END AS profit_today_to_bet_today_ratio,
        MEDIAN(EXTRACT(EPOCH FROM (ts_bj - prev_ts_bj))) AS bet_frequency_day_median,
        AVG(bullet_level) AS avg_bullet_level_today,
        SUM(CASE WHEN killed = 1 THEN 1 ELSE 0 END)::DOUBLE PRECISION / NULLIF(COUNT(*), 0) AS weighted_kill_rate_today,
        MAX(CASE WHEN rn_desc = 1 THEN curr_balance END) AS current_balance_end_day
    FROM ordered
    GROUP BY 1, 2
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
fish_counts AS (
    SELECT
        user_id,
        bet_date,
        fish_value,
        COUNT(*) AS n
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
),
user_bullet_level_threshold AS (
    SELECT
        user_id,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bullet_level) AS p75_bullet_level
    FROM base
    WHERE bullet_level IS NOT NULL
    GROUP BY 1
),
high_level AS (
    SELECT
        b.user_id,
        b.bet_date,
        AVG(CASE WHEN b.bullet_level >= t.p75_bullet_level THEN 1.0 ELSE 0.0 END) AS high_level_bullet_ratio_today
    FROM base b
    JOIN user_bullet_level_threshold t
      ON b.user_id = t.user_id
    GROUP BY 1, 2
),
daily_with_lag AS (
    SELECT
        d.*,
        LAG(bet_date) OVER (PARTITION BY user_id ORDER BY bet_date) AS prev_bet_date,
        AVG(bet_amount_today) OVER (PARTITION BY user_id ORDER BY bet_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS hist_avg_bet_amount_before_today,
        AVG(bet_count_today) OVER (PARTITION BY user_id ORDER BY bet_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS hist_avg_bet_count_before_today,
        AVG(avg_bet_one_time_today) OVER (PARTITION BY user_id ORDER BY bet_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS hist_avg_bet_one_time_before_today,
        SUM(payout_today) OVER (PARTITION BY user_id ORDER BY bet_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS payout_7_bet_days,
        SUM(bet_amount_today) OVER (PARTITION BY user_id ORDER BY bet_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS bet_amount_7_bet_days
    FROM daily d
),
active_window AS (
    SELECT
        d1.user_id,
        d1.bet_date,
        COUNT(d2.bet_date) AS active_days_7d
    FROM daily d1
    JOIN daily d2
      ON d1.user_id = d2.user_id
     AND d2.bet_date BETWEEN d1.bet_date - interval '6 day' AND d1.bet_date
    GROUP BY 1, 2
)
SELECT
    d.user_id,
    d.bet_date,
    aw.active_days_7d,
    COALESCE(DATEDIFF(day, d.prev_bet_date, d.bet_date) - 1, 0) AS no_bet_streak_days,
    d.bet_frequency_day_median,
    CASE WHEN d.hist_avg_bet_amount_before_today > 0 THEN d.bet_amount_today / d.hist_avg_bet_amount_before_today END AS bet_amount_ratio_today_vs_history,
    CASE WHEN d.hist_avg_bet_count_before_today > 0 THEN d.bet_count_today::DOUBLE PRECISION / d.hist_avg_bet_count_before_today END AS bet_count_ratio_today_vs_history,
    LN(1 + d.avg_bet_one_time_today) AS avg_bet_one_time_today_log,
    CASE WHEN d.bet_amount_7_bet_days > 0 THEN d.payout_7_bet_days / d.bet_amount_7_bet_days END AS rtp_7d,
    d.profit_today_to_bet_today_ratio,
    COALESCE(ld.max_consecutive_loss_count_today, 0) AS max_consecutive_loss_count_today,
    CASE WHEN d.hist_avg_bet_one_time_before_today > 0 THEN d.current_balance_end_day / d.hist_avg_bet_one_time_before_today END AS current_balance_to_avg_bet_ratio,
    hl.high_level_bullet_ratio_today,
    fe.target_selection_entropy_today,
    d.weighted_kill_rate_today,
    d.bet_count_today,
    d.bet_amount_today,
    d.payout_today,
    d.profit_today,
    d.current_balance_end_day
FROM daily_with_lag d
LEFT JOIN active_window aw
  ON d.user_id = aw.user_id AND d.bet_date = aw.bet_date
LEFT JOIN loss_daily ld
  ON d.user_id = ld.user_id AND d.bet_date = ld.bet_date
LEFT JOIN high_level hl
  ON d.user_id = hl.user_id AND d.bet_date = hl.bet_date
LEFT JOIN fish_entropy fe
  ON d.user_id = fe.user_id AND d.bet_date = fe.bet_date
ORDER BY d.user_id, d.bet_date
"""


def to_utc_bound(local_date):
    dt = datetime.combine(date.fromisoformat(local_date), datetime.min.time())
    return dt - timedelta(hours=8)


def export_features(output_csv, start_date, end_date, limit=None):
    tunnel, conn = connect()
    rows = 0
    sql = FEATURE_SQL_TEMPLATE.format(
        start_utc=to_utc_bound(start_date).strftime("%Y-%m-%d %H:%M:%S"),
        end_utc=to_utc_bound(end_date).strftime("%Y-%m-%d %H:%M:%S"),
    )
    if limit:
        sql = f"SELECT * FROM ({sql}) q LIMIT {int(limit)}"
    try:
        output = Path(output_csv)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)

        with conn.cursor() as header_cur:
            header_cur.execute(f"SELECT * FROM ({sql}) q LIMIT 0")
            header = [desc[0] for desc in header_cur.description]

        with conn.cursor(name="hmm_feature_export_cursor") as cur:
            cur.itersize = 50000
            cur.execute(sql)
            with output.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                for row in cur:
                    writer.writerow(row)
                    rows += 1
                    if rows % 100000 == 0:
                        print(json.dumps({"exported_rows": rows, "output_csv": str(output)}), flush=True)
        return {"exported_rows": rows, "output_csv": str(output)}
    finally:
        conn.close()
        tunnel.stop()


def main():
    parser = argparse.ArgumentParser(description="Export 2026 Beijing-time user-day HMM features from Redshift bullet table.")
    parser.add_argument("--output-csv", default="outputs/hmm_features_fm01_cny_2026.csv")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2027-01-01")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(export_features(args.output_csv, args.start_date, args.end_date, args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
