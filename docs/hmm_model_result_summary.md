# HMM Model Result Summary

## Training Setup

- Model: Gaussian HMM
- States: `K=4`
- Training rows: `202,530`
- Training users: `22,691`
- Training condition: users with at least 3 betting days
- Full labeled rows: `365,156`
- Full users: `166,102`
- Converged: `true`
- Iterations: `96`

## State Interpretation

| State | Manual Label | Row Share | Users | Churn 7D Proxy |
|---|---|---:|---:|---:|
| S0 | High-churn low-frequency losing state | 48.01% | 123,373 | 73.33% |
| S1 | Stable active state | 21.95% | 36,388 | 31.53% |
| S2 | Long-gap return / reactivation state | 10.57% | 21,296 | 40.89% |
| S3 | Short-cycle positive-feedback state | 19.47% | 36,351 | 40.31% |

## Business Reading

### S0: High-Churn Low-Frequency Losing State

Players have low recent activity, reduced bet amount and bet count versus history, low RTP, negative profit ratio, and the highest 7-day churn proxy. This is the strongest risk state.

### S1: Stable Active State

Players have the highest 7-day activity, almost no betting-day gap, bet amount and count close to historical levels, and the lowest churn proxy. This is closest to a stable/core behavior state.

### S2: Long-Gap Return / Reactivation State

Players have a long gap since the previous betting day, but show a strong spike in bet amount and bet count versus their own historical baseline. This looks more like a return/reactivation state than a lifecycle stage.

### S3: Short-Cycle Positive-Feedback State

Players are active with short gaps and positive RTP/profit signals, but lower bet amount and count versus history. This may represent positive short-term feedback with conservative betting.

## Transition Matrix

```text
From S0 -> S0 49.3%, S1 8.6%,  S2 23.8%, S3 18.3%
From S1 -> S0 18.8%, S1 51.8%, S2 10.8%, S3 18.6%
From S2 -> S0 28.5%, S1 25.2%, S2 25.9%, S3 20.5%
From S3 -> S0 30.0%, S1 15.2%, S2 15.0%, S3 39.7%
```

## Current Conclusion

This HMM output is better interpreted as `behavior_state`, not direct lifecycle stages such as `newbie / early / core`. The median state duration is 1 betting day, which means the states are responsive and jumpy. Short term recommendation:

```text
rule-based lifecycle stage × HMM behavior state × churn probability
```

Use HMM as an additional behavior-state layer and later validate whether it can revise lifecycle rules.

## Next Improvements

- Add exact event-level median betting interval.
- Add exact max consecutive loss streak.
- Add fishing-specific features:
  - high_level_bullet_ratio_today
  - target_selection_entropy_today
  - weighted_kill_rate_today
- Compare K=3, K=4, K=5.
- Validate monthly stability and transition consistency.

