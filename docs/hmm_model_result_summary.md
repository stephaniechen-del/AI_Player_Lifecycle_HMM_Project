# HMM Model Result Summary

## Training Setup

- Model: Gaussian HMM
- States: `K=4`
- Training rows: `204,551`
- Training users: `22,896`
- Training condition: users with at least 3 betting days
- Full labeled rows: `368,333`
- Full users: `167,265`
- Converged: `true`
- Iterations: `125`

## State Interpretation

| State | Manual Label | Row Share | Users | Churn 7D Proxy |
|---|---|---:|---:|---:|
| S0 | Stable regular active state | 20.06% | 23,446 | 19.50% |
| S1 | High-churn low-frequency decline state | 55.83% | 152,324 | 76.57% |
| S2 | High-frequency low-bet active state | 12.04% | 13,967 | 24.18% |
| S3 | Long-gap return / reactivation state | 12.07% | 24,814 | 39.61% |

## Business Reading

### S0: Stable Regular Active State

Players have high recent betting-day activity, almost no no-bet gap, bet amount and count close to historical baseline, healthy RTP, and the lowest churn proxy.

### S1: High-Churn Low-Frequency Decline State

Players have low recent betting-day activity, clear no-bet gaps, reduced bet amount and bet count versus history, lower RTP, and the highest churn proxy. This is the strongest risk state.

### S2: High-Frequency Low-Bet Active State

Players have high recent betting-day activity but current bet amount and count are far below their historical baseline, with lower current-day RTP. This looks like active but conservative/low-stake behavior.

### S3: Long-Gap Return / Reactivation State

Players have a long gap since the previous betting day, but current bet amount and bet count are much higher than their historical baseline. This looks like a return/reactivation state.

## Transition Matrix

```text
From S0 -> S0 57.2%, S1 10.6%, S2 22.5%, S3 9.7%
From S1 -> S0 11.3%, S1 50.2%, S2 11.6%, S3 26.9%
From S2 -> S0 21.6%, S1 22.3%, S2 43.1%, S3 12.9%
From S3 -> S0 27.8%, S1 27.9%, S2 18.4%, S3 25.9%
```

## Current Conclusion

This HMM output is better interpreted as `behavior_state`, not direct lifecycle stages such as `newbie / early / core`. The median state duration is 1 betting day, which means the states are responsive and jumpy. Short term recommendation:

```text
rule-based lifecycle stage × HMM behavior state × churn probability
```

Use HMM as an additional behavior-state layer and later validate whether it can revise lifecycle rules.

## Next Improvements

- Consider exact event-level median betting interval if interval precision becomes important.
- Add fishing-specific features:
  - high_level_bullet_ratio_today
  - target_selection_entropy_today
  - weighted_kill_rate_today
- Compare K=3, K=4, K=5.
- Validate monthly stability and transition consistency.
