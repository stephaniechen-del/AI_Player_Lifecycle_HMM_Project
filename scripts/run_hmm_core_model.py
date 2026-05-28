import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


ROOT = Path(__file__).resolve().parent
Z_FEATURES = [
    "active_days_7d_z",
    "no_bet_streak_days_z",
    "bet_frequency_day_median_z",
    "bet_amount_ratio_today_vs_history_z",
    "bet_count_ratio_today_vs_history_z",
    "avg_bet_one_time_today_log_z",
    "rtp_7d_z",
    "profit_today_to_bet_today_ratio_z",
    "max_consecutive_loss_count_today_z",
    "current_balance_to_avg_bet_ratio_z",
]

RAW_FEATURES = [c.removesuffix("_z") for c in Z_FEATURES]


def resolve(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def state_label(row):
    if row["no_bet_streak_days_mean"] >= 3 or row["active_days_7d_mean"] <= 1.25:
        if row["churn7_rate"] >= 0.55:
            return "衰退/流失前状态"
        return "低频回访状态"
    if row["bet_amount_ratio_today_vs_history_median"] >= 1.4 or row["bet_count_ratio_today_vs_history_median"] >= 1.4:
        if row["profit_today_to_bet_today_ratio_mean"] < -0.25:
            return "高投入追损状态"
        return "高投入增长状态"
    if row["active_days_7d_mean"] >= 2.3 and row["churn7_rate"] <= 0.35:
        return "稳定活跃状态"
    return "低频试探状态"


def main():
    parser = argparse.ArgumentParser(description="Train Gaussian HMM on core user-day features and profile states.")
    parser.add_argument("--z-csv", default="outputs/hmm_core_features_model_ready_2026_to_may_fast.csv")
    parser.add_argument("--raw-csv", default="outputs/hmm_core_features_fm01_cny_2026_to_may_fast.csv")
    parser.add_argument("--output-dir", default="outputs/hmm_core_model_k4")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--min-train-days", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=150)
    args = parser.parse_args()

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = pd.read_csv(resolve(args.z_csv), parse_dates=["bet_date"])
    raw = pd.read_csv(resolve(args.raw_csv), parse_dates=["bet_date"])
    z = z.sort_values(["user_id", "bet_date"]).reset_index(drop=True)
    raw = raw.sort_values(["user_id", "bet_date"]).reset_index(drop=True)

    lengths_by_user = z.groupby("user_id", sort=False).size()
    train_users = lengths_by_user[lengths_by_user >= args.min_train_days].index
    train = z[z["user_id"].isin(train_users)].copy()
    train_lengths = train.groupby("user_id", sort=False).size().to_list()
    X_train = train[Z_FEATURES].to_numpy(dtype=float)

    model = GaussianHMM(
        n_components=args.states,
        covariance_type="diag",
        n_iter=args.max_iter,
        tol=1e-3,
        min_covar=1e-3,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, train_lengths)

    # Decode all sequences, including one-day users, for profiling.
    all_states = np.empty(len(z), dtype=int)
    start = 0
    for _, part in z.groupby("user_id", sort=False):
        end = start + len(part)
        all_states[start:end] = model.predict(part[Z_FEATURES].to_numpy(dtype=float))
        start = end

    result = raw.copy()
    result["hmm_state"] = all_states

    # Consecutive duration in the current HMM state for each user.
    result["state_change"] = (
        result.groupby("user_id")["hmm_state"].diff().fillna(1).ne(0).astype(int)
    )
    result["state_segment"] = result.groupby("user_id")["state_change"].cumsum()
    result["hmm_state_duration_days"] = (
        result.groupby(["user_id", "state_segment"]).cumcount() + 1
    )
    result = result.drop(columns=["state_change", "state_segment"])

    result["next_bet_date"] = result.groupby("user_id")["bet_date"].shift(-1)
    result["next_gap_days"] = (result["next_bet_date"] - result["bet_date"]).dt.days - 1
    max_date = result["bet_date"].max()
    observed_next = result["next_bet_date"].notna()
    enough_followup = result["bet_date"] <= max_date - pd.Timedelta(days=7)
    result["churn7_proxy"] = np.nan
    result.loc[observed_next, "churn7_proxy"] = (result.loc[observed_next, "next_gap_days"] >= 7).astype(float)
    result.loc[~observed_next & enough_followup, "churn7_proxy"] = 1.0

    profile = result.groupby("hmm_state").agg(
        rows=("user_id", "size"),
        users=("user_id", "nunique"),
        active_days_7d_mean=("active_days_7d", "mean"),
        no_bet_streak_days_mean=("no_bet_streak_days", "mean"),
        bet_frequency_day_median_mean=("bet_frequency_day_median", "mean"),
        bet_amount_ratio_today_vs_history_median=("bet_amount_ratio_today_vs_history", "median"),
        bet_count_ratio_today_vs_history_median=("bet_count_ratio_today_vs_history", "median"),
        avg_bet_one_time_today_log_mean=("avg_bet_one_time_today_log", "mean"),
        rtp_7d_mean=("rtp_7d", "mean"),
        profit_today_to_bet_today_ratio_mean=("profit_today_to_bet_today_ratio", "mean"),
        max_consecutive_loss_count_today_median=("max_consecutive_loss_count_today", "median"),
        current_balance_to_avg_bet_ratio_median=("current_balance_to_avg_bet_ratio", "median"),
        bet_count_today_median=("bet_count_today", "median"),
        bet_amount_today_median=("bet_amount_today", "median"),
        churn7_rate=("churn7_proxy", "mean"),
        next_gap_days_mean=("next_gap_days", "mean"),
        state_duration_median=("hmm_state_duration_days", "median"),
    ).reset_index()
    profile["row_share"] = profile["rows"] / len(result)
    profile["label"] = profile.apply(state_label, axis=1)
    profile = profile.sort_values(["churn7_rate", "no_bet_streak_days_mean"], ascending=[False, False])

    transition = pd.DataFrame(model.transmat_)
    transition.index = [f"state_{i}" for i in range(args.states)]
    transition.columns = [f"state_{i}" for i in range(args.states)]

    means_z = pd.DataFrame(model.means_, columns=Z_FEATURES)
    means_z.insert(0, "hmm_state", range(args.states))
    covars = model.covars_
    if covars.ndim == 3:
        covars = np.stack([np.diag(covars[i]) for i in range(covars.shape[0])])
    covars_z = pd.DataFrame(covars, columns=Z_FEATURES)
    covars_z.insert(0, "hmm_state", range(args.states))

    # 2-step transition probabilities for downstream use.
    trans2 = np.linalg.matrix_power(model.transmat_, 2)
    trans2_df = pd.DataFrame(trans2)
    trans2_df.index = [f"state_{i}" for i in range(args.states)]
    trans2_df.columns = [f"state_{i}" for i in range(args.states)]

    profile.to_csv(out_dir / "state_profile.csv", index=False)
    transition.to_csv(out_dir / "transition_matrix.csv")
    trans2_df.to_csv(out_dir / "transition_matrix_2step.csv")
    means_z.to_csv(out_dir / "state_means_z.csv", index=False)
    covars_z.to_csv(out_dir / "state_covars_z.csv", index=False)
    result.to_csv(out_dir / "user_day_with_hmm_state.csv", index=False)
    joblib.dump(model, out_dir / "gaussian_hmm_k4.joblib")

    summary = {
        "rows": int(len(result)),
        "users": int(result["user_id"].nunique()),
        "train_rows": int(len(train)),
        "train_users": int(len(train_users)),
        "states": args.states,
        "min_train_days": args.min_train_days,
        "converged": bool(model.monitor_.converged),
        "iterations": int(model.monitor_.iter),
        "log_likelihood": float(model.monitor_.history[-1]),
        "date_min": str(result["bet_date"].min().date()),
        "date_max": str(result["bet_date"].max().date()),
    }
    (out_dir / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
