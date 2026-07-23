"""
CARS2-ST ordinal DCM simulator.

Ground truth is built directly from an externally validated factor
structure -- not an invented grouping:

Campbell, J.M., Feghali, P., Powell, L., Gilchrest, C., Broomell, A., &
Gardner, L. (2026). Evaluating the factor structure of the Childhood Autism
Rating Scale 2 - Standard Version: Evidence for a three-factor model.
Research in Autism, 133, 202908.

Their consensus 3-factor CFA model (their "Model 5", best-fitting of 5
models compared, CFI=.93, RMSEA=.06, N=302):
  - Social Communication (SC):              items 1,2,7,8,11,12,14,15  (8)
  - Restrictive/Repetitive Behavior/Sensory (RB): items 4,5,9           (3)
  - Emotional and Behavioral Dysregulation (ED):  items 3,6,10,13       (4)
Factor correlations (their reported values): SC-RB=.87, RB-ED=.49, SC-ED=.23
-- note SC and RB are nearly collinear, not cleanly separable; ED is more
distinct. Standardized item loadings are taken from their Figure 1.

Item text itself is copyrighted, proprietary test material (Schopler et al.,
2010, WPS) and is NOT reproduced -- items are referred to by number and by
their published short domain label only (e.g. "Item 1 - Relating to
People"), which is standard academic shorthand used throughout the
published literature (including the review paper above), not the
copyrighted behavioral rating criteria itself.

CARS2-ST scoring: each item 1-4 in 0.5 steps (7 ordinal levels), all items
same direction (higher = more autism-related severity). Internally recoded
to integer levels 0-6 for the ordinal DCM (0 <-> score 1.0, 6 <-> score 4.0).
"""

from __future__ import annotations

import json
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
    14: ("SC", 0.13, "Level and Consistency of Intellectual Response"),  # known weak/outlier loading
    15: ("SC", 0.74, "General Impressions"),
}
FACTORS = ["SC", "RB", "ED"]
FACTOR_LABELS = {"SC": "Social Communication", "RB": "Restrictive/Repetitive Behavior/Sensory",
                 "ED": "Emotional and Behavioral Dysregulation"}
# published factor correlation matrix (Campbell et al., Results 4.1)
FACTOR_CORR = pd.DataFrame(
    [[1.00, 0.87, 0.23],
     [0.87, 1.00, 0.49],
     [0.23, 0.49, 1.00]],
    index=FACTORS, columns=FACTORS,
)

L = 6         # 7 levels (0-6), representing raw scores 1.0-4.0 in 0.5 steps
T0 = 3.0
N_SOURCES = 3  # matches the externally validated 3-factor structure


def build_item_table(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item_num, (factor, loading, label) in ITEM_SPEC.items():
        p = float(np.clip(0.35 + rng.normal(0, 0.06), 0.1, 0.75))
        # item slope scaled loosely with its factor loading -- items that load
        # more strongly on their factor are given a bit more discriminating power
        v = float(np.clip(0.10 + 0.12 * loading + rng.normal(0, 0.015), 0.04, 0.3))
        deltas = list(np.clip(rng.normal(0.55, 0.12, size=L), 0.2, 1.3))
        rows.append({"item": f"CARS-{item_num:02d}", "item_num": item_num, "factor": factor,
                      "label": label, "published_loading": loading, "p": p, "v": v,
                      "deltas": [float(d) for d in deltas]})
    return pd.DataFrame(rows).sort_values("item_num").reset_index(drop=True)


def build_mixing_matrix(item_table: pd.DataFrame) -> np.ndarray:
    """Mixing matrix built directly from published loadings: item i's column
    for its OWN factor is set to its published standardized loading (scaled
    down to keep space-shifts from dominating the age signal, same fix as
    the GARS-3 rebuild); small loadings on other factors reflect nothing
    more than baseline noise (the factor *correlation* is handled separately
    at the source-generation step below, not by cross-loading here).
    """
    d = len(item_table)
    A = np.zeros((d, N_SOURCES))
    for idx, row in item_table.iterrows():
        main_col = FACTORS.index(row["factor"])
        A[idx, main_col] = 0.22 * row["published_loading"]  # scaled to item's v-range
    return A


def simulate_cohort(n_subjects: int = 100, seed: int = 42,
                     age_min: float = 3.0, age_max: float = 10.0,
                     min_visits: int = 2, max_visits: int = 6,
                     two_profiles: bool = True):
    rng = np.random.default_rng(seed)
    item_table = build_item_table(seed=5)
    A = build_mixing_matrix(item_table)
    items = item_table.to_dict("records")

    def sigmoid_curve(x, p):
        return 1.0 / (1.0 + (1.0 / p - 1.0) * np.exp(-x / (p * (1 - p))))

    def p_ge_level(age, level, tau, xi, w, item):
        cum_delta = sum(item["deltas"][:level])
        raw_time = np.exp(xi) * (age - T0 - tau)
        x = item["v"] * raw_time - item["v"] * cum_delta + w
        return sigmoid_curve(x, item["p"])

    def simulate_item_scores(age, tau, xi, w, item, rng):
        p_ge = [np.ones_like(age)]
        for level in range(1, L + 1):
            p_ge.append(p_ge_level(age, level, tau, xi, w, item))
        p_ge.append(np.zeros_like(age))
        p_ge = np.stack(p_ge, axis=0)
        p_ge = np.minimum.accumulate(p_ge, axis=0)
        probs = -np.diff(p_ge, axis=0)
        probs = np.clip(probs, 1e-9, None)
        probs = probs / probs.sum(axis=0, keepdims=True)
        scores = np.empty(age.shape, dtype=int)
        for j in range(age.shape[0]):
            scores[j] = rng.choice(np.arange(L + 1), p=probs[:, j])
        return scores

    # published factor correlation matrix -> Cholesky, so ground-truth
    # sources are drawn CORRELATED (matching real SC-RB=.87 etc.), even
    # though the fitting procedure will later assume independent sources
    # (standard ICA identifiability assumption) -- this mismatch is
    # intentional and documented; see README for why it matters.
    L_chol = np.linalg.cholesky(FACTOR_CORR.values)

    subject_rows, long_rows = [], []
    for i in range(n_subjects):
        subject_id = f"CARS{i+1:04d}"
        tau_i = rng.normal(0, 1.2)
        xi_i = rng.normal(0, 0.25)

        z = rng.normal(0, 1.0, size=N_SOURCES)
        s_i = L_chol @ z  # correlated per published factor correlations

        profile = None
        if two_profiles:
            # overall severity subtype: shifts all three factors together
            profile = rng.choice([1, 2])
            shift = -1.1 if profile == 1 else 1.1
            s_i = s_i + shift

        w_i = A @ s_i

        n_visits = rng.integers(min_visits, max_visits + 1)
        ages = np.sort(rng.uniform(age_min, age_max, size=n_visits))

        subject_rows.append({
            "subject_id": subject_id, "profile": profile, "tau": tau_i, "xi": xi_i,
            "source_SC": s_i[0], "source_RB": s_i[1], "source_ED": s_i[2], "n_visits": n_visits,
        })

        for k, item in enumerate(items):
            scores = simulate_item_scores(ages, np.full(n_visits, tau_i), np.full(n_visits, xi_i),
                                           np.full(n_visits, w_i[k]), item, rng)
            for age, score in zip(ages, scores):
                long_rows.append({"subject_id": subject_id, "age": round(float(age), 3),
                                   "item": item["item"], "factor": item["factor"], "value": int(score)})

    long_df = pd.DataFrame(long_rows).sort_values(["subject_id", "age", "item"]).reset_index(drop=True)
    subjects_df = pd.DataFrame(subject_rows)
    return long_df, subjects_df, item_table, A


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    long_df, subjects_df, item_table, A = simulate_cohort(n_subjects=100, seed=42)

    long_df.to_csv(out_dir / "cars2_simulated_long.csv", index=False)
    subjects_df.to_csv(out_dir / "cars2_subjects_ground_truth.csv", index=False)
    item_table.to_json(out_dir / "cars2_item_table.json", orient="records", indent=2)
    np.save(out_dir / "cars2_mixing_matrix.npy", A)
    FACTOR_CORR.to_csv(out_dir / "cars2_published_factor_correlations.csv")

    print(f"Simulated {subjects_df.shape[0]} subjects, {len(item_table)} items, "
          f"{long_df.shape[0]} (subject,visit,item) rows")
    print("\nItems per factor:\n", item_table.groupby("factor").size())
    print("\nValue distribution:\n", long_df["value"].value_counts(normalize=True).sort_index().round(3))
