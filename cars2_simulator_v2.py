"""
CARS2-ST ordinal DCM simulator -- CORRECTED (v2).

Fixes two problems found auditing v1 (`cars2_simulator.py`):

FIX 1 -- CALIBRATION.  v1's delta spacings were far too small relative to the
item slope v, so all 6 ordinal transitions compressed into ~47% of the 3-10y
observation window.  Children rushed floor->ceiling, producing a U-shaped
score distribution with 73% of item ratings at an extreme (exactly 1.0 or
4.0) and a total-score SD of 13.5 -- versus the real published CARS2-ST
distribution (Campbell et al. 2026, N=302: M=36.41, SD=4.40, range 22-50.5).
v2 widens the deltas so transitions spread across the observable age range,
and calibrates to reproduce the published mean and SD.

FIX 2 -- DIRECTION.  v1 had every item's severity monotonically INCREASING
with age, i.e. all children worsen.  That is correct for a neurodegenerative
cohort (the paper's PPMI/Parkinson's application) but contradicts the autism
developmental literature: Fountain et al. (2012, Pediatrics, n=6975) found
most children IMPROVE on social/communication between ages 3-14, with ~7-10%
"bloomers" improving dramatically, while repetitive behaviours stay mostly
stable.  v2 therefore models the population trend as monotone IMPROVEMENT
(CARS2-ST severity decreasing with age) via an explicit per-item `direction`.

REMAINING STRUCTURAL LIMITATION (not fixable by calibration -- see README):
the DCM linear predictor is x(t) = v*e^xi*(t-t0-tau) - v*cum_delta + w, so
dx/dt = v*e^xi has a FIXED SIGN for every individual.  Individuals differ in
timing (tau), rate (xi) and item-profile (w), but never in DIRECTION.  A
cohort where some children improve while others worsen on the same item
cannot be represented.  This is intrinsic to the method, not to this code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# item number -> (factor, published standardized loading from Campbell et al. Fig 1, short label)
ITEM_SPEC = {
    1:  ("SC", 0.79, "Relating to People"),
    2:  ("SC", 0.76, "Imitation"),
    3:  ("ED", 0.23, "Emotional Response"),
    4:  ("RB", 0.45, "Body Use"),
    5:  ("RB", 0.64, "Object Use"),
    6:  ("ED", 0.36, "Adaptation to Change"),
    7:  ("SC", 0.69, "Visual Response"),
    8:  ("SC", 0.57, "Listening Response"),
    9:  ("RB", 0.38, "Taste, Smell, & Touch Response and Use"),
    10: ("ED", 0.51, "Fear or Nervousness"),
    11: ("SC", 0.68, "Verbal Communication"),
    12: ("SC", 0.76, "Nonverbal Communication"),
    13: ("ED", 0.44, "Activity Level"),
    14: ("SC", 0.13, "Level and Consistency of Intellectual Response"),
    15: ("SC", 0.74, "General Impressions"),
}
FACTORS = ["SC", "RB", "ED"]
FACTOR_LABELS = {"SC": "Social Communication",
                 "RB": "Restrictive/Repetitive Behavior/Sensory",
                 "ED": "Emotional and Behavioral Dysregulation"}
FACTOR_CORR = pd.DataFrame(
    [[1.00, 0.87, 0.23],
     [0.87, 1.00, 0.49],
     [0.23, 0.49, 1.00]],
    index=FACTORS, columns=FACTORS,
)

# Per-factor direction of the POPULATION trend over ages 3-10.
#  -1 = severity decreases with age (improvement)   +1 = severity increases
# Set from Fountain et al. (2012): social/communication improve substantially;
# restricted/repetitive behaviours are largely stable (modelled as a weak
# improvement so the item is still estimable); emotional dysregulation
# improves modestly.
FACTOR_DIRECTION = {"SC": -1, "RB": -1, "ED": -1}
FACTOR_SLOPE_SCALE = {"SC": 1.00, "RB": 0.25, "ED": 0.60}   # RB nearly flat

L = 6          # 7 levels (0-6) <-> raw CARS2-ST scores 1.0-4.0 in 0.5 steps
T0 = 10.0   # reference age = END of window: severity is highest early, improving with age
N_SOURCES = 3


def build_item_table(seed: int = 5,
                     v_base: float = 0.055,
                     delta_mean: float = 1.35,
                     p_center: float = 0.50) -> pd.DataFrame:
    """Fixed effects per item.

    delta_mean is now ~1.35 (v1 used 0.55): total cum_delta ~8.1 years, so the
    six transitions spread across (and slightly beyond) the 7-year observation
    window rather than compressing into ~3 years.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for item_num, (factor, loading, label) in ITEM_SPEC.items():
        p = float(np.clip(p_center + rng.normal(0, 0.05), 0.15, 0.85))
        v = float(np.clip((v_base + 0.03 * loading) * FACTOR_SLOPE_SCALE[factor]
                          + rng.normal(0, 0.004), 0.008, 0.30))
        deltas = list(np.clip(rng.normal(delta_mean, 0.20, size=L), 0.5, 3.0))
        rows.append({"item": f"CARS-{item_num:02d}", "item_num": item_num,
                     "factor": factor, "label": label,
                     "published_loading": loading,
                     "direction": FACTOR_DIRECTION[factor],
                     "p": p, "v": v, "deltas": [float(d) for d in deltas]})
    return pd.DataFrame(rows).sort_values("item_num").reset_index(drop=True)


def build_mixing_matrix(item_table: pd.DataFrame, scale: float = 0.055) -> np.ndarray:
    """Space-shift loadings taken from the published standardized loadings."""
    d = len(item_table)
    A = np.zeros((d, N_SOURCES))
    for idx, row in item_table.iterrows():
        A[idx, FACTORS.index(row["factor"])] = scale * row["published_loading"]
    return A


def sigmoid_curve(x, p):
    return 1.0 / (1.0 + (1.0 / p - 1.0) * np.exp(-x / (p * (1 - p))))


def p_ge_level(age, level, tau, xi, w, item):
    """P(y >= level).  `direction` flips the sign of the AGE-VARYING term only;
    the level-offset term keeps a fixed sign so the cumulative-logit
    monotonicity constraint P(y>=l) >= P(y>=l+1) holds either way."""
    cum_delta = sum(item["deltas"][:level])
    raw_time = np.exp(xi) * (age - T0 - tau)
    x = item["direction"] * item["v"] * raw_time - item["v"] * cum_delta + w
    return sigmoid_curve(x, item["p"])


def simulate_item_scores(age, tau, xi, w, item, rng):
    p_ge = [np.ones_like(age)]
    for level in range(1, L + 1):
        p_ge.append(p_ge_level(age, level, tau, xi, w, item))
    p_ge.append(np.zeros_like(age))
    p_ge = np.minimum.accumulate(np.stack(p_ge, axis=0), axis=0)
    probs = np.clip(-np.diff(p_ge, axis=0), 1e-9, None)
    probs = probs / probs.sum(axis=0, keepdims=True)
    return np.array([rng.choice(np.arange(L + 1), p=probs[:, j])
                     for j in range(age.shape[0])], dtype=int)


def simulate_cohort(n_subjects: int = 300, seed: int = 42,
                    age_min: float = 3.0, age_max: float = 10.0,
                    min_visits: int = 2, max_visits: int = 6,
                    two_profiles: bool = True,
                    sigma_tau: float = 2.5, sigma_xi: float = 0.25,
                    profile_shift: float = 0.9,
                    item_table: pd.DataFrame | None = None,
                    A: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    if item_table is None:
        item_table = build_item_table()
    if A is None:
        A = build_mixing_matrix(item_table)
    items = item_table.to_dict("records")
    L_chol = np.linalg.cholesky(FACTOR_CORR.values)

    subject_rows, long_rows = [], []
    for i in range(n_subjects):
        subject_id = f"CARS{i+1:04d}"
        tau_i = rng.normal(0, sigma_tau)
        xi_i = rng.normal(0, sigma_xi)
        s_i = L_chol @ rng.normal(0, 1.0, size=N_SOURCES)

        profile = None
        if two_profiles:
            profile = rng.choice([1, 2])
            s_i = s_i + (-profile_shift if profile == 1 else profile_shift)

        w_i = A @ s_i
        n_visits = rng.integers(min_visits, max_visits + 1)
        ages = np.sort(rng.uniform(age_min, age_max, size=n_visits))

        subject_rows.append({"subject_id": subject_id, "profile": profile,
                             "tau": tau_i, "xi": xi_i,
                             "source_SC": s_i[0], "source_RB": s_i[1],
                             "source_ED": s_i[2], "n_visits": n_visits})

        for k, item in enumerate(items):
            scores = simulate_item_scores(ages, np.full(n_visits, tau_i),
                                          np.full(n_visits, xi_i),
                                          np.full(n_visits, w_i[k]), item, rng)
            for age, score in zip(ages, scores):
                long_rows.append({"subject_id": subject_id,
                                  "age": round(float(age), 3),
                                  "item": item["item"], "factor": item["factor"],
                                  "value": int(score)})

    long_df = (pd.DataFrame(long_rows)
               .sort_values(["subject_id", "age", "item"]).reset_index(drop=True))
    return long_df, pd.DataFrame(subject_rows), item_table, A


def total_score_moments(long_df: pd.DataFrame) -> dict:
    d = long_df.copy()
    d["score"] = 1 + d["value"] * 0.5
    tot = d.groupby(["subject_id", "age"])["score"].sum()
    frac_extreme = ((d.score == 1.0) | (d.score == 4.0)).mean()
    return {"mean": float(tot.mean()), "sd": float(tot.std()),
            "min": float(tot.min()), "max": float(tot.max()),
            "frac_extreme_item_ratings": float(frac_extreme)}


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    long_df, subjects_df, item_table, A = simulate_cohort(n_subjects=300, seed=42)

    long_df.to_csv(out / "cars2_v2_simulated_long_n300.csv", index=False)
    subjects_df.to_csv(out / "cars2_v2_subjects_ground_truth_n300.csv", index=False)
    item_table.to_json(out / "cars2_v2_item_table.json", orient="records", indent=2)
    np.save(out / "cars2_v2_mixing_matrix.npy", A)

    m = total_score_moments(long_df)
    print(f"n={subjects_df.shape[0]} subjects, {len(item_table)} items, {long_df.shape[0]} rows")
    print()
    print("CARS2-ST TOTAL SCORE      simulated (v2)   published (Campbell et al.)")
    print(f"  mean                    {m['mean']:>10.2f}       36.41")
    print(f"  sd                      {m['sd']:>10.2f}        4.40")
    print(f"  min                     {m['min']:>10.1f}       22.0")
    print(f"  max                     {m['max']:>10.1f}       50.5")
    print(f"  % item ratings extreme  {m['frac_extreme_item_ratings']*100:>9.1f}%       (v1 was 73.0%)")
