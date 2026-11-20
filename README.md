# gta6-trophy-rarity-forecast-01 — Round 4

Forecasts GTA 6 trophy rarities (55 trophies, 5 quantiles each) from
Exophase-style history reads. Stdlib only, deterministic.

Run: `mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`

## What round 4 changes (vs round 3)

1. **Launch-night flow blend.** The cohort earning rate `g_hat` now blends
   the recent-jump estimator 50/50 with the launch-night total-growth
   estimator `g_total = (h_last - h_first) / (n_last - n_first)` (clamped
   `[0, 0.05]`), the direct empirical earning rate of the whole observed
   cohort flow from the first distinct state (launch night) to the last
   read. Recent jumps alone are noisy; the blend anchors the cohort model
   on the full observed history.
2. **Narrower flat-path spread.** `FLAT_SPREAD0` is 0.13 (was 0.20).

Everything else is round 3's skeleton: cohort / power-law / curve blend
(0.30 / 0.35 / 0.35 in logit space), geometric-mean player-count model,
and the 50/50 rate-power / progression-depth mixture for flat histories.

## Backtest (final, this `main.py`)

| Test | r4 | r3 | r2 |
|---|---|---|---|
| A practice p00/p01 mean (range) | **0.1019** (0.0468–0.1477) | 0.1058 | 0.1059 |
| B flat → live Nov-20 | **0.2702** | 0.2680 | 0.3131 |
| C power-law truth, 10 / 30 seeds | 0.2134 / 0.1970 | 0.2021 / 0.1879 | 0.1894 / 0.1801 |
| C2 step/refresh truth, 10 / 30 seeds | 0.1203 / 0.1087 | 0.1200 / 0.1094 | 0.1696 / 0.1587 |
| C3 regime-bend truth, 10 / 30 seeds | 0.1607 / 0.1960 | 0.1746 / 0.2135 | 0.2026 / 0.2428 |

Round 4 improves the practice number slightly (0.1019 vs 0.1058) and C3,
is flat on C2, and is slightly worse on B and C. Net: the launch-night
flow moves the practice number a little, nothing more.

## Honest ceiling

Practice ~0.10 vs the live rank-20 bar of 0.022. This entry does **not**
reach the reward zone — only an exact-split oracle with sigma ~0.07
clears the bar. Round 4 is a test of whether the launch-night flow moves
the practice number at all, not a claim to a paid place.
