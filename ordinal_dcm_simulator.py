"""
Ordinal Disease Course Mapping (DCM) synthetic data simulator for autism
progression markers.

Implements the generative process of the ordinal logistic-curves model from:
    Poulet, P.-E. & Durrleman, S. (2023). "Multivariate disease progression
    modelling with longitudinal ordinal data." Statistics in Medicine.
    (Section 2.3, Equation 3; simulation design mirrors Section 3.1.)

Model recap (per item k, level l):
    psi_ik_l(t)  = exp(xi_i) * (t - t0 - tau_i) + t0 - sum_{m=1}^{l} delta_k^m
    P(y_ijk >= l) = sigmoid_curve(v_k * psi_ik_l(t) + w_ik ; p_k)

    where sigmoid_curve(x; p) = [1 + (1/p - 1) * exp(-x / (p*(1-p)))]^-1

Random effects per individual i:
    tau_i  ~ N(0, sigma_tau^2)      time-shift (early/late "onset")
    xi_i   ~ N(0, sigma_xi^2)       log-acceleration (fast/slow progressor)
    s_i    ~ N(0, I_Ns)             independent sources
    w_i = A @ s_i                  space-shift (item-level individual deviation)

This module is deliberately dependency-light (numpy/pandas only) so it can
run standalone before Leaspy is wired in, and so the same population
parameters can later be used to sanity-check a fitted Leaspy ordinal model
against known ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Item definitions
# --------------------------------------------------------------------------
# Each item is rated 0..L (L=4 -> 5-point Likert scale, mirroring MDS-UPDRS
# style items used in the reference paper). p_k, v_k are fixed effects for
# the population-average curve of item k. sign(v_k) sets whether the item
# increases (skill acquisition) or decreases (problem behaviour reduction)
# with disease/developmental time. delta_k are the positive level-transition
# delays (Equation 3) -- larger delta = that level is held longer.

ITEMS = {
    # name                 direction   p_k    v_k    deltas (delta^1..delta^4), years
    "communication_cder":   dict(sign=+1, p=0.20, v=0.55, deltas=[0.8, 1.0, 1.3, 1.6]),
    "social_cder":          dict(sign=+1, p=0.22, v=0.50, deltas=[0.9, 1.1, 1.4, 1.8]),
    "adaptive_vabs":        dict(sign=+1, p=0.18, v=0.45, deltas=[1.0, 1.2, 1.5, 2.0]),
    "hyperactivity_abc":    dict(sign=-1, p=0.75, v=0.40, deltas=[0.9, 1.1, 1.4, 1.7]),
    "irritability_abc":     dict(sign=-1, p=0.70, v=0.42, deltas=[0.8, 1.0, 1.3, 1.6]),
    "insistence_sameness":  dict(sign=+1, p=0.30, v=0.35, deltas=[1.1, 1.3, 1.6, 2.0]),
    "rsm_repetitive":       dict(sign=+1, p=0.40, v=0.08, deltas=[1.5, 1.6, 1.7, 1.8]),  # ~flat: "generally stable"
}
ITEM_NAMES = list(ITEMS.keys())
D = len(ITEM_NAMES)      # number of items
L = 4                    # number of ordinal transitions -> 5 levels (0..4)
T0 = 3.0                 # reference age (years) -- start of observation window

# --------------------------------------------------------------------------
# 2. Population / random-effect hyperparameters
# --------------------------------------------------------------------------

@dataclass
class PopulationParams:
    t0: float = T0
    sigma_tau: float = 1.2     # sd of time-shift (years)
    sigma_xi: float = 0.25     # sd of log-acceleration
    n_sources: int = 2
    noise_free: bool = True    # ordinal outcome is already stochastic (Bernoulli/multinomial draw)

    def item_table(self) -> pd.DataFrame:
        rows = []
        for name, spec in ITEMS.items():
            rows.append({"item": name, **{k: v for k, v in spec.items() if k != "deltas"},
                         "deltas": spec["deltas"]})
        return pd.DataFrame(rows)


def _mixing_matrix(d: int, ns: int, seed: int) -> np.ndarray:
    """Build a random d x ns mixing matrix A (space-shift = A @ sources).
    Columns are not strictly enforced orthogonal to v here (that constraint
    matters for identifiability during *estimation*, not for generating a
    plausible synthetic dataset), but we scale modestly to keep space-shifts
    small relative to the population trajectory spacing.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.6, size=(d, ns))
    return A


# --------------------------------------------------------------------------
# 3. Core ordinal DCM math
# --------------------------------------------------------------------------

def sigmoid_curve(x: np.ndarray, p: float) -> np.ndarray:
    """Logistic-curves-model base sigmoid, eq. (1)/(3) denominator form."""
    return 1.0 / (1.0 + (1.0 / p - 1.0) * np.exp(-x / (p * (1.0 - p))))


def p_ge_level(age: np.ndarray, level: int, t0: float, tau: np.ndarray, xi: np.ndarray,
               p_k: float, v_k: float, sign: int, w_ik: np.ndarray, deltas: list[float]) -> np.ndarray:
    """P(y_ijk >= level) for level in 1..L, vectorised over subjects/visits.

    NOTE on `sign`: the cumulative-logit constraint P(y>=l) must be
    non-increasing in l *regardless* of whether the item trends up or down
    with age. So `sign` is applied only to the age-varying component
    (raw_time), while the level-delay term (-v_k * cum_delta) always uses a
    fixed negative coefficient -- that's what keeps level l+1 strictly
    "further out" than level l on the curve for both increasing and
    decreasing items.
    """
    cum_delta = sum(deltas[:level])
    # centred warped time: 0 at age==t0 when tau=xi=0 (so curves are anchored
    # at the reference age t0, not offset by an extra +t0 as a naive reading
    # of the paper's ψ formula would give -- that offset is an artifact of
    # ψ being calendar-time-valued before being re-centred by (p_k, v_k)).
    raw_time = np.exp(xi) * (age - t0 - tau)
    x = sign * v_k * raw_time - v_k * cum_delta + w_ik
    return sigmoid_curve(x, p_k)


def simulate_item_scores(age: np.ndarray, tau: np.ndarray, xi: np.ndarray,
                          w_ik: np.ndarray, item_spec: dict, t0: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Draw ordinal scores 0..L for one item across all (subject, visit) rows."""
    p_k, v_k, sign, deltas = item_spec["p"], item_spec["v"], item_spec["sign"], item_spec["deltas"]
    # P(y >= l) for l = 1..L, plus P(y>=0)=1 and P(y>=L+1)=0 as boundaries
    p_ge = [np.ones_like(age)]
    for level in range(1, L + 1):
        p_ge.append(p_ge_level(age, level, t0, tau, xi, p_k, v_k, sign, w_ik, deltas))
    p_ge.append(np.zeros_like(age))
    p_ge = np.stack(p_ge, axis=0)  # shape (L+2, N)

    # enforce monotone non-increasing (numerical safety) then take differences
    p_ge = np.minimum.accumulate(p_ge, axis=0)
    probs = -np.diff(p_ge, axis=0)  # shape (L+1, N) = P(y == l) for l=0..L
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=0, keepdims=True)

    scores = np.empty(age.shape, dtype=int)
    for j in range(age.shape[0]):
        scores[j] = rng.choice(np.arange(L + 1), p=probs[:, j])
    return scores


# --------------------------------------------------------------------------
# 4. Cohort simulation
# --------------------------------------------------------------------------

def simulate_cohort(n_subjects: int = 100, seed: int = 42,
                     age_min: float = 3.0, age_max: float = 10.0,
                     min_visits: int = 2, max_visits: int = 6,
                     two_subtypes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    long_df : one row per (subject, visit, item) -- the dataset you'd feed to Leaspy
    subjects_df : one row per subject with ground-truth random effects (for validation)
    """
    rng = np.random.default_rng(seed)
    pop = PopulationParams()
    A = _mixing_matrix(D, pop.n_sources, seed=seed + 1)

    subject_rows = []
    long_rows = []

    for i in range(n_subjects):
        subject_id = f"SIM{i+1:04d}"
        tau_i = rng.normal(0, pop.sigma_tau)
        xi_i = rng.normal(0, pop.sigma_xi)

        # two latent subtypes via source 1, mirroring the paper's cluster design
        cluster = None
        if two_subtypes:
            cluster = rng.choice([1, 2])
            s1 = rng.normal(-2.5, 1.0) if cluster == 1 else rng.normal(2.5, 1.0)
            s_rest = rng.normal(0, 1.0, size=pop.n_sources - 1)
            s_i = np.concatenate([[s1], s_rest])
        else:
            s_i = rng.normal(0, 1.0, size=pop.n_sources)

        w_i = A @ s_i  # shape (D,)

        # visit schedule: irregular, within [age_min, age_max]
        n_visits = rng.integers(min_visits, max_visits + 1)
        ages = np.sort(rng.uniform(age_min, age_max, size=n_visits))

        subject_rows.append({
            "subject_id": subject_id, "cluster": cluster, "tau": tau_i, "xi": xi_i,
            **{f"source_{s+1}": s_i[s] for s in range(pop.n_sources)},
            "n_visits": n_visits,
        })

        for k, item_name in enumerate(ITEM_NAMES):
            spec = ITEMS[item_name]
            scores = simulate_item_scores(ages, np.full(n_visits, tau_i), np.full(n_visits, xi_i),
                                           np.full(n_visits, w_i[k]), spec, pop.t0, rng)
            for age, score in zip(ages, scores):
                long_rows.append({
                    "subject_id": subject_id, "age": round(float(age), 3),
                    "item": item_name, "value": int(score),
                })

    long_df = pd.DataFrame(long_rows).sort_values(["subject_id", "age", "item"]).reset_index(drop=True)
    subjects_df = pd.DataFrame(subject_rows)
    return long_df, subjects_df


def item_metadata_json(path: Path) -> None:
    meta = {
        "items": ITEMS,
        "n_levels_per_item": L + 1,
        "t0": T0,
        "notes": "Scale 0-4 for every item. sign=+1 items increase with age "
                 "(skills), sign=-1 items decrease with age (problem "
                 "behaviours). deltas are the population-level time (years) "
                 "spent progressing from level l-1 to level l on the common "
                 "disease/developmental timeline.",
    }
    path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    long_df, subjects_df = simulate_cohort(n_subjects=100, seed=42)

    long_df.to_csv(out_dir / "simulated_long.csv", index=False)
    subjects_df.to_csv(out_dir / "simulated_subjects_ground_truth.csv", index=False)

    wide_df = long_df.pivot_table(index=["subject_id", "age"], columns="item", values="value").reset_index()
    wide_df.to_csv(out_dir / "simulated_wide.csv", index=False)

    item_metadata_json(out_dir / "item_metadata.json")

    print(f"Simulated {subjects_df.shape[0]} subjects, {long_df.shape[0]} (subject,visit,item) rows")
    print(f"Visits per subject: min={subjects_df.n_visits.min()}, max={subjects_df.n_visits.max()}, "
          f"mean={subjects_df.n_visits.mean():.2f}")
    print("Files written to:", out_dir)
