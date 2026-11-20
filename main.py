"""GTA 6 trophy-rarity forecast -- round 4.

Reads GTA 6 trophy-rarity forecast cases (``*.in`` JSON) from the directory
in ``MAKEFUN_INPUT_DIR`` (default ``/input``) and writes one forecast file
per case (``*.out`` JSON) to ``MAKEFUN_OUTPUT_DIR`` (default ``/output``).

Each case carries 55 trophies (``id``, ``description``, ``secret``), a
``history`` of reads (``at`` ISO timestamp, ``players_tracked``, ``rates``:
55 percentages in trophy order), a ``target`` timestamp, a horizon in
minutes, and ``levels`` ``[0.05, 0.25, 0.5, 0.75, 0.95]``.  The output is
``{"trophies": [{"id": <id>, "q": [q05, q25, q50, q75, q95]}]}`` with five
quantiles per trophy, each in ``[0.01, 99.99]`` and non-decreasing.

Model skeleton (kept from round 3)
-----------------------------------
Three views of the future rarity of each trophy are blended in logit
space -- cohort 0.30, power-law 0.35, curve 0.35:

* Cohort: the earning rate ``g`` (new holders per new tracked player) is
  estimated from the recent refresh jumps, pooled with a prior regression
  of ``log g`` on ``log rate`` across trophies.
* Power-law: log holders grow linearly in ``log1p`` time; the per-trophy
  slope is shrunk towards the cross-trophy median slope.
* Curve: the logit of the rate itself grows linearly in ``log1p`` time
  over the last day of reads.

The future player count ``N(target)`` is the geometric mean of a power-law
extrapolation and a decaying-refresh-multiplier extrapolation.  When the
history has never refreshed (a single flat state), new holders are
allocated across trophies by a 50/50 mixture of a rate-power term and a
progression-depth term.

Round 4 changes (the only two differences from round 3)
--------------------------------------------------------
1. Launch-night flow blend.  The cohort earning rate now blends the
   recent-jump estimator 50/50 with the launch-night total-growth
   estimator: the direct empirical earning rate of the whole observed
   cohort flow, from the FIRST distinct state (launch night) to the last
   read, ``g_total = (h_last - h_first) / (n_last - n_first)`` clamped to
   ``[0, 0.05]``.  The recent jumps are noisy -- a single refresh can be
   dominated by one game's worth of new players -- while the launch-night
   total averages the entire flow, so the blend anchors the cohort model
   on the full observed history instead of only the last few jumps.
2. Narrower flat-path spread.  ``FLAT_SPREAD0`` is 0.13 instead of 0.20:
   when no refresh has ever been seen, the no-refresh allocation is a
   tighter statement about where the next holders go.

Honest ceiling (read this before trusting any number below)
-----------------------------------------------------------
On the practice reconstruction no honest forecast beats ~0.10 (the Bayes
floor there is 0.082); the live rank-20 bar is 0.022 and only an
exact-split oracle with sigma ~0.07 clears it.  Round 4 is a test of
whether the launch-night flow moves the practice number at all -- it does
not claim to reach the reward zone.

Stdlib only.  Deterministic: same case in, same bytes out.
"""

import json
import math
import os
import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Log time is measured from here (launch night, 2026-11-19 00:00 UTC).
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Blend weights for the three models, in logit space.
COHORT_WEIGHT = 0.30
POWER_WEIGHT = 0.35
CURVE_WEIGHT = 0.35

# Cohort / player-count model.
PRIOR_PLAYERS = 2000.0
FIRST_REFRESH_JUMP = 6.0
LATER_REFRESH_JUMP = 1.65
TYPICAL_GAP_HOURS = 4.5
NEW_PER_PLAYER = 0.032
RATE_POWER = 3.0
DEPTH_POWER = 2.0
MULT_FLOOR = 1.02

# Refresh-path spread: sigma = (SPREAD0 + SPREAD_H * sqrt(horizon_days)
#   + SPREAD_MIX * disagreement) * SPREAD_SCALE, clamped.
SPREAD0 = 0.25
SPREAD_H = 0.05
SPREAD_MIX = 0.30
SPREAD_MIN = 0.25
SPREAD_MAX = 1.1
SPREAD_SCALE = 0.8

# Flat-path spread: sigma = FLAT_SPREAD0 + FLAT_SPREAD_H * sqrt(horizon_days).
# Round 4: FLAT_SPREAD0 is 0.13 (was 0.20 in round 3).
FLAT_SPREAD0 = 0.13
FLAT_SPREAD_H = 0.30

# A trophy nearly every tracked player holds gets a fixed narrow spread.
NEARLY_EVERYONE = 99.9
UNIVERSAL_SPREAD = 0.25

# Cohort pooling: the last five jumps, oldest first, with these weights.
RECENT_JUMPS = 5
COHORT_WEIGHTS = [1.0, 0.85, 0.7, 0.55, 0.4]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_ts(text):
    """ISO string (ending Z) to an aware datetime."""
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def pct_clamp(x):
    """Clamp a percentage to [0.01, 99.99]."""
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def to_logit(pct):
    """Logit of a percentage: log(p / (1 - p)) with p = pct / 100."""
    p = pct_clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(lg):
    """Inverse logit, as a percentage; numerically stable for large |lg|."""
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; integral results come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    if value == int(value):
        return int(value)
    return value


def linfit(xs, ys):
    """Least-squares y ~ a + b*x; (a, b), or None if <2 points or no spread."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx == 0.0:
        return None
    b = sxy / sxx
    a = my - b * mx
    return (a, b)


def middle(vals):
    """Median of a list (0.0 if empty)."""
    if not vals:
        return 0.0
    ordered = sorted(vals)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def make_quantiles(median_pct, sigma):
    """Five quantiles around a median percentage, non-decreasing."""
    centre = to_logit(median_pct)
    out = []
    for z in NORMAL_Z:
        q = round4(from_logit(centre + z * sigma))
        q = pct_clamp(q)
        out.append(q)
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def holders_at(rate, players):
    """Holders implied by a rate and a player count, or None if no rate."""
    if rate is None:
        return None
    return int(math.floor(rate * players / 100.0 + 0.5))


def progression_depth(trophy):
    """How deep into the game a trophy sits, read off its description.

    Rules are checked in order; the first match wins.  The description
    is lowercased before matching.
    """
    text = str(trophy.get("description", "")).lower()
    secret = bool(trophy.get("secret", False))
    if "gone fishing" in text:
        return 0.0
    m = re.search(r"chapter\s+(\d+)", text)
    if m:
        return float(int(m.group(1)))
    if "100% completion" in text:
        return 6.0
    if "final mission" in text and "for " not in text:
        return 5.0
    if "final mission for" in text:
        return 3.5
    if "fishing" in text:
        return 0.5
    if "try each type of activity" in text:
        return 1.0
    if "arcade game" in text:
        return 1.5
    ints = [int(x) for x in re.findall(r"\d+", text)]
    biggest = max(ints) if ints else 0
    if biggest >= 100:
        return 5.0
    if biggest >= 20:
        return 4.0
    if "every location" in text or "every spot" in text:
        return 4.0
    if "all of leonida" in text:
        return 4.0
    if "all " in text and (
        "challenges" in text
        or "sets" in text
        or "properties" in text
        or "locations" in text
    ):
        return 4.0
    if "complete" in text and secret:
        return 2.5
    return 3.0


# ---------------------------------------------------------------------------
# History: distinct states and refresh jumps
# ---------------------------------------------------------------------------

def distinct_reads(history, n_trophies):
    """Collapse runs of identical reads into one state each.

    A read is a step function (the tracker refreshes every few hours), so
    a run of identical reads is one state, stamped at the time it was
    LAST confirmed.  Returns a list of (t_hours, players, rates) states.
    """
    states = []
    prev_key = None
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or players <= 0:
            continue
        if rates is None or len(rates) != n_trophies:
            continue
        key = (players, tuple(rates))
        t_hours = (parse_ts(read["at"]) - EPOCH).total_seconds() / 3600.0
        if key == prev_key:
            # Same state, confirmed again later: move its timestamp.
            states[-1] = (t_hours, players, list(rates))
        else:
            states.append((t_hours, players, list(rates)))
        prev_key = key
    return states


def refresh_jumps(states, n_trophies):
    """One entry per jump in players tracked between consecutive states.

    Each entry is {"dN", "dH", "mult", "t"}: the player increase, the
    per-trophy holder increases (None where a rate is missing), the
    player multiplier, and the time of the later state.  Pairs with no
    player increase are skipped.
    """
    jumps = []
    for j in range(1, len(states)):
        _t_prev, n_prev, rates_prev = states[j - 1]
        t_cur, n_cur, rates_cur = states[j]
        dn = n_cur - n_prev
        if dn <= 0:
            continue
        dh = []
        for i in range(n_trophies):
            h_cur = holders_at(rates_cur[i], n_cur)
            h_prev = holders_at(rates_prev[i], n_prev)
            if h_cur is None or h_prev is None:
                dh.append(None)
            else:
                dh.append(h_cur - h_prev)
        jumps.append({"dN": dn, "dH": dh, "mult": n_cur / n_prev, "t": t_cur})
    return jumps


# ---------------------------------------------------------------------------
# Future player count
# ---------------------------------------------------------------------------

def predict_players(states, jumps, x_last, x_target, horizon_h):
    """N(target) as the geometric mean of two views.

    Power law: log N grows linearly in log1p time, with the slope half
    from all states and half from the last day (when enough exist).
    Multiplier: future refresh multipliers decay geometrically (a linear
    fit of log(mult - 1) against jump index, slope clamped to <= 0),
    one multiplier per typical gap until the horizon is covered.
    """
    xs = [math.log1p(max(t, -0.9)) for (t, _n, _r) in states]
    log_n = [math.log(max(n, 1)) for (_t, n, _r) in states]
    last_t = states[-1][0]
    last_n = states[-1][1]

    fit_all = linfit(xs, log_n)
    day_xs = []
    day_yn = []
    for k, (t, _n, _r) in enumerate(states):
        if t >= last_t - 24.0:
            day_xs.append(xs[k])
            day_yn.append(log_n[k])
    fit_day = linfit(day_xs, day_yn) if len(day_xs) >= 3 else None

    if fit_all is None:
        slope = fit_day[1] if fit_day is not None else 0.0
    elif fit_day is not None:
        slope = 0.5 * fit_all[1] + 0.5 * fit_day[1]
    else:
        slope = fit_all[1]
    slope = min(max(slope, 0.0), 2.5)
    dx = max(x_target - x_last, 0.0)
    n_power = last_n * math.exp(slope * dx)
    if n_power < last_n:
        n_power = float(last_n)

    if len(jumps) < 2:
        return n_power

    mults = [jump["mult"] for jump in jumps]
    decay = linfit(
        list(range(len(mults))),
        [math.log(max(m - 1.0, 1e-3)) for m in mults],
    )
    jump_times = [jump["t"] for jump in jumps]
    gaps = [
        jump_times[k + 1] - jump_times[k]
        for k in range(len(jump_times) - 1)
    ]
    if gaps:
        gap = min(max(middle(gaps), 1.0), 12.0)
    else:
        gap = TYPICAL_GAP_HOURS
    n_future = max(int(horizon_h / gap), 0)

    n_mult = float(last_n)
    med_mult = middle(mults)
    for k in range(len(mults), len(mults) + n_future):
        if decay is None:
            m = med_mult
        else:
            a, b = decay
            b = min(b, 0.0)
            expo = a + b * k
            # The multiplier is clamped to 6 below, so capping the
            # exponent here cannot change the result; it only avoids
            # an overflow in exp for wild fits.
            if expo > 50.0:
                expo = 50.0
            m = 1.0 + math.exp(expo)
        m = min(max(m, MULT_FLOOR), 6.0)
        n_mult *= m
    if n_mult < last_n:
        n_mult = float(last_n)

    return math.exp(0.5 * math.log(n_power) + 0.5 * math.log(n_mult))


# ---------------------------------------------------------------------------
# Forecast with refreshes
# ---------------------------------------------------------------------------

def forecast_with_refreshes(case, states, jumps, n_target, horizon_d):
    """Forecast when the history has at least one refresh jump."""
    trophies = case["trophies"]
    target = parse_ts(case["target"])
    xs = [math.log1p(max(t, -0.9)) for (t, _n, _r) in states]
    x_last = xs[-1]
    target_hours = (target - EPOCH).total_seconds() / 3600.0
    x_target = math.log1p(max(target_hours, -0.9))
    dx = max(x_target - x_last, 0.0)
    t_last = states[-1][0]
    n_last = states[-1][1]
    last_rates = states[-1][2]
    present = [r for r in last_rates if r is not None]
    typical_rate = middle(present) if present else 1.0
    recent = jumps[-RECENT_JUMPS:]

    # Cohort prior regression: over the recent jumps, (log rate, log g)
    # pairs where g = dH / dN > 0, for trophies with rate in (0, 99.9).
    # The rate is each trophy's LAST rate, shared across the jumps.
    pair_x = []
    pair_y = []
    for jump in recent:
        dn = jump["dN"]
        dh = jump["dH"]
        for i, rate in enumerate(last_rates):
            if rate is None:
                continue
            if not (0.0 < rate < NEARLY_EVERYONE):
                continue
            dhi = dh[i]
            if dhi is None:
                continue
            g = dhi / dn
            if g > 0:
                pair_x.append(math.log(rate))
                pair_y.append(math.log(g))
    reg = linfit(pair_x, pair_y) if len(pair_x) >= 5 else None

    # Cross-trophy median holder slope: per trophy, fit log(holders)
    # against log1p time over all states; take the median slope.
    slopes = []
    for i in range(len(trophies)):
        pts_x = []
        pts_y = []
        for j, (_t, n, rates) in enumerate(states):
            rate = rates[i]
            if rate is None:
                continue
            holders = max(rate * n / 100.0, 0.5)
            pts_x.append(xs[j])
            pts_y.append(math.log(holders))
        fit = linfit(pts_x, pts_y)
        if fit is not None:
            slopes.append(fit[1])
    slope_typ = middle(slopes) if slopes else 0.5
    slope_typ = min(max(slope_typ, 0.0), 2.5)

    # x of the first state in the last day (xs[0] if the history is
    # shorter than a day): the day-window starts here.
    x_day = xs[0]
    for j, (t, _n, _r) in enumerate(states):
        if t >= t_last - 24.0:
            x_day = xs[j]
            break

    # Launch state: the first distinct state, for the round-4 flow blend.
    _first_t, first_n, first_rates = states[0]

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = typical_rate
        if r_last >= NEARLY_EVERYONE:
            rows.append(
                {
                    "id": trophy["id"],
                    "q": make_quantiles(min(99.99, r_last), UNIVERSAL_SPREAD),
                }
            )
            continue
        h_last = r_last * n_last / 100.0

        # (1) Cohort model.
        pooled_h = 0.0
        pooled_n = 0.0
        for k, jump in enumerate(recent):
            w = COHORT_WEIGHTS[k] if k < len(COHORT_WEIGHTS) else COHORT_WEIGHTS[-1]
            dhi = jump["dH"][i]
            if dhi is None:
                continue
            pooled_h += w * dhi
            pooled_n += w * jump["dN"]
        if reg is not None:
            g_prior = math.exp(reg[0] + reg[1] * math.log(max(r_last, 0.01)))
        else:
            g_prior = 0.0004 * (max(r_last, 0.01) ** 1.5)
        g_prior = min(max(g_prior, 1e-6), 0.05)
        if pooled_n > 0:
            g_hat = (pooled_h + PRIOR_PLAYERS * g_prior) / (pooled_n + PRIOR_PLAYERS)
        else:
            g_hat = g_prior
        # ROUND 4 -- launch-night flow: blend the recent-jump estimator
        # 50/50 with the total-growth estimator from the launch state to
        # the last read.  This is the direct empirical earning rate of
        # the whole observed cohort flow, anchored at launch night.
        first_rate = first_rates[i]
        if first_rate is None:
            first_rate = 0.0
        h_first = first_rate * first_n / 100.0
        if n_last > first_n:
            g_total = max(h_last - h_first, 0.0) / (n_last - first_n)
            g_total = min(max(g_total, 0.0), 0.05)
            g_hat = 0.5 * g_hat + 0.5 * g_total
        r_cohort = pct_clamp(
            100.0 * (h_last + g_hat * max(n_target - n_last, 0.0)) / n_target
        )

        # (2) Power-law model: log holders linear in log1p time, slope
        # half from all states and half from the last day, shrunk
        # towards the cross-trophy median slope.
        pts_x = []
        pts_y = []
        for j, (_t, n, rates) in enumerate(states):
            rate = rates[i]
            if rate is None:
                continue
            pts_x.append(xs[j])
            pts_y.append(math.log(max(rate * n / 100.0, 0.5)))
        fit_h = linfit(pts_x, pts_y)
        if fit_h is not None:
            day_x = [x for x in pts_x if x >= x_day]
            day_y = [y for x, y in zip(pts_x, pts_y) if x >= x_day]
            fit_h_day = linfit(day_x, day_y) if len(day_x) >= 3 else None
            if fit_h_day is not None:
                b_i = 0.5 * fit_h[1] + 0.5 * fit_h_day[1]
            else:
                b_i = fit_h[1]
            wgt = len(pts_x) / (len(pts_x) + 3.0)
            b_i = wgt * b_i + (1.0 - wgt) * slope_typ
            b_i = min(max(b_i, 0.0), 2.5)
            h_pow = math.exp(math.log(max(h_last, 0.5)) + b_i * dx)
            r_power = pct_clamp(100.0 * h_pow / n_target)
        else:
            r_power = r_cohort

        # (3) Curve model: logit(rate) linear in log1p time over the
        # last day of reads.
        curve_x = []
        curve_y = []
        for j, (_t, _n, rates) in enumerate(states):
            rate = rates[i]
            if rate is None:
                continue
            if xs[j] < x_day:
                continue
            curve_x.append(xs[j])
            curve_y.append(to_logit(rate))
        fit_c = linfit(curve_x, curve_y) if len(curve_x) >= 3 else None
        if fit_c is not None:
            r_curve = pct_clamp(from_logit(fit_c[0] + fit_c[1] * x_target))
        else:
            r_curve = r_power

        # Blend the three views in logit space.
        centre = (
            COHORT_WEIGHT * to_logit(r_cohort)
            + POWER_WEIGHT * to_logit(r_power)
            + CURVE_WEIGHT * to_logit(r_curve)
        )
        med = from_logit(centre)
        lc = to_logit(r_cohort)
        lp = to_logit(r_power)
        lv = to_logit(r_curve)
        disagree = max(abs(lc - lp), abs(lc - lv), abs(lp - lv))
        sigma = (
            SPREAD0 + SPREAD_H * math.sqrt(horizon_d) + SPREAD_MIX * disagree
        ) * SPREAD_SCALE
        sigma = min(max(sigma, SPREAD_MIN * SPREAD_SCALE), SPREAD_MAX)
        rows.append({"id": trophy["id"], "q": make_quantiles(med, sigma)})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Forecast with no refresh ever seen
# ---------------------------------------------------------------------------

def forecast_no_refresh(case, states, horizon_h, horizon_d):
    """Forecast when the history is one flat state (no refresh yet).

    The next refresh is assumed to multiply players by
    FIRST_REFRESH_JUMP, later ones by LATER_REFRESH_JUMP, one per
    typical gap.  The new holders are allocated across trophies by a
    50/50 mixture of a rate-power term and a progression-depth term.
    """
    trophies = case["trophies"]
    _t_last, n_last, last_rates = states[-1]
    present = [r for r in last_rates if r is not None]
    typical_rate = middle(present) if present else 1.0

    if horizon_h >= 1.0:
        n_ref = 1.0 + max(0.0, (horizon_h - TYPICAL_GAP_HOURS) / TYPICAL_GAP_HOURS)
        n_target = (
            n_last
            * (FIRST_REFRESH_JUMP ** min(n_ref, 1.0))
            * (LATER_REFRESH_JUMP ** max(n_ref - 1.0, 0.0))
        )
    else:
        n_target = n_last * (
            1.0 + (FIRST_REFRESH_JUMP - 1.0) * horizon_h / TYPICAL_GAP_HOURS
        )
    d_n = max(n_target - n_last, 0.0)

    rate_parts = []
    depth_parts = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = typical_rate
        if rate >= NEARLY_EVERYONE:
            rate_parts.append(0.0)
            depth_parts.append(0.0)
        else:
            base = max(rate, 0.01)
            rate_parts.append(base ** RATE_POWER)
            depth_parts.append(
                (base ** DEPTH_POWER) * math.exp(-progression_depth(trophy) / 2.0)
            )
    sum_rate = sum(rate_parts)
    sum_depth = sum(depth_parts)
    if sum_rate == 0:
        sum_rate = 1.0
    if sum_depth == 0:
        sum_depth = 1.0

    # Round 4: FLAT_SPREAD0 is 0.13 (was 0.20).
    sigma = FLAT_SPREAD0 + FLAT_SPREAD_H * math.sqrt(horizon_d)

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = typical_rate
        if rate >= NEARLY_EVERYONE:
            rows.append(
                {
                    "id": trophy["id"],
                    "q": make_quantiles(min(99.99, rate), UNIVERSAL_SPREAD),
                }
            )
            continue
        mixed = 0.5 * rate_parts[i] / sum_rate + 0.5 * depth_parts[i] / sum_depth
        h_last = rate * n_last / 100.0
        new_h = NEW_PER_PLAYER * d_n * mixed
        med = pct_clamp(100.0 * (h_last + new_h) / n_target)
        rows.append({"id": trophy["id"], "q": make_quantiles(med, sigma)})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Case dispatch and entry point
# ---------------------------------------------------------------------------

def forecast_case(case):
    """Forecast one case dict, returning {"trophies": [...]}."""
    if case["levels"] != QUANTILE_LEVELS:
        raise ValueError("levels must equal %r" % (QUANTILE_LEVELS,))
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    n_tr = len(trophies)
    states = distinct_reads(history, n_tr)
    if not states:
        raise ValueError("case has no usable states")
    target = parse_ts(case["target"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    x_last = math.log1p(max(states[-1][0], -0.9))
    target_hours = (target - EPOCH).total_seconds() / 3600.0
    x_target = math.log1p(max(target_hours, -0.9))
    jumps = refresh_jumps(states, n_tr)
    if not jumps:
        return forecast_no_refresh(case, states, horizon_h, horizon_d)
    n_target = predict_players(states, jumps, x_last, x_target, horizon_h)
    return forecast_with_refreshes(case, states, jumps, n_target, horizon_d)


def main():
    """Read *.in cases, forecast each, write *.out as compact JSON."""
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        in_path = os.path.join(input_dir, name)
        with open(in_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        stem = name[:-3]
        out_path = os.path.join(output_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
