# AI Player Lifecycle HMM Project

This folder contains the current Redshift export, feature engineering pipeline, and HMM modeling outputs for the FM01 CNY bullet data.

## Data Scope

- Source table: `"transform-agfish-game".public.bullet`
- Filters:
  - `op_code NOT IN ('B26','TST','TSB','TSO')`
  - `currency_type = 'CNY'`
  - `game_id = 'FM01'`
- Date timezone: Beijing time, `DATE(created_at + interval '8 hour')`
- Export range: `2026-01-01` to `2026-05-30`
- Final user-day rows: `368,333`
- Users: `167,265`

## Folder Layout

```text
data/raw/
  hmm_core_daily_base_fm01_cny_2026_to_may.csv
  hmm_core_features_fm01_cny_2026_to_may_fast.csv

data/model_ready/
  hmm_core_features_model_ready_2026_to_may_fast.csv
  hmm_core_features_model_ready_stats_2026_to_may_fast.json

model_outputs/hmm_core_model_k4/
  state_profile_readable.csv
  transition_matrix_readable_pct.csv
  user_day_with_hmm_state.csv
  gaussian_hmm_k4.joblib
  model_summary.json

scripts/
  export_hmm_core_features_2026_fast.py
  prepare_hmm_core_features.py
  run_hmm_core_model.py
  query_public_bullet_user.py

connection_template/
  .env.example
  requirements.txt
  secrets/

docs/
  redshift_connection_readme.md
  hmm_model_result_summary.md
```

## HMM Core Features

The model-ready file contains z-score normalized versions of:

```text
bet_days_7d
no_bet_streak_days
bet_interval_day
bet_amount_ratio_today_vs_history
bet_count_ratio_today_vs_history
avg_bet_one_time_today_log
rtp_7_bet_days
rtp_day
max_consecutive_loss_count_today
current_balance_max_to_avg_bet_ratio
```

## Feature Notes

- `bet_days_7d` is the count of betting days in the past 7 calendar days, including the current betting day.
- `bet_interval_day` is the daily interval proxy: `(last bullet timestamp - first bullet timestamp) / (bullet_count - 1)`.
- `rtp_7_bet_days` uses the latest 7 betting days, not 7 calendar days.
- `rtp_day` is current-day payout divided by current-day bet.
- `max_consecutive_loss_count_today` is now calculated exactly from event-level ordered bullets.
- `current_balance_max_to_avg_bet_ratio` uses the maximum current balance observed that day divided by historical average single bet.

## Reproduce

Install dependencies:

```bash
python3 -m pip install -r connection_template/requirements.txt
```

Export core features:

```bash
python3 scripts/export_hmm_core_features_2026_fast.py \
  --start-date 2026-01-01 \
  --end-date 2026-06-01 \
  --daily-csv data/raw/hmm_core_daily_base_fm01_cny_2026_to_may.csv \
  --feature-csv data/raw/hmm_core_features_fm01_cny_2026_to_may_fast.csv
```

Prepare model-ready features:

```bash
python3 scripts/prepare_hmm_core_features.py \
  --input-csv data/raw/hmm_core_features_fm01_cny_2026_to_may_fast.csv \
  --output-csv data/model_ready/hmm_core_features_model_ready_2026_to_may_fast.csv \
  --stats-json data/model_ready/hmm_core_features_model_ready_stats_2026_to_may_fast.json
```

Run HMM:

```bash
python3 scripts/run_hmm_core_model.py \
  --z-csv data/model_ready/hmm_core_features_model_ready_2026_to_may_fast.csv \
  --raw-csv data/raw/hmm_core_features_fm01_cny_2026_to_may_fast.csv \
  --states 4 \
  --min-train-days 3 \
  --output-dir model_outputs/hmm_core_model_k4
```

## Security

Do not store real `.env` files or private keys in this project folder unless you intentionally keep the folder local only. Use `connection_template/.env.example` as the template and put real credentials in a local `.env` when running exports.
