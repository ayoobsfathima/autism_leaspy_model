"""
Ordinal Disease Course Mapping (DCM) fitter.

Same generative model as `ordinal_dcm_simulator.py` (Poulet & Durrleman 2023,
Eq. 3), fit via joint MAP estimation (gradient descent over fixed effects +
individual random effects with Gaussian priors) using PyTorch autodiff.

WHY NOT LEASPY DIRECTLY: the `leaspy` PyPI package's classic API (1.x, which
has the ordinal model) requires torch<1.12, unavailable for Python>=3.11.
The current 2.x rewrite (which installs fine on modern Python) hasn't ported
ordinal observation models yet. This module is a drop-in stand-in: same
model, same data format (long-format ID/TIME/FEATURE/VALUE), so you can swap
in real leaspy on a Python<=3.10 environment with minimal changes -- see
`fit_with_real_leaspy.py` for that version.

ESTIMATION NOTE: this is joint MAP (point estimate of every random effect,
regularised by its prior), not full MCMC-SAEM. This is a common, much
cheaper approximation (similar in spirit to Laplace/FOCE approaches used
elsewhere in nonlinear mixed-effects modelling) -- fine for validating the
pipeline and getting usable individual trajectories, but it will
underestimate posterior uncertainty compared to the paper's fully Bayesian
approach.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class OrdinalDCM(nn.Module):
    """Ordinal disease course mapping model (joint MAP formulation)."""

    def __init__(self, item_names: list[str], item_signs: dict[str, int],
                 n_subjects: int, n_sources: int = 2, n_levels: int = 4,
                 t0: float = 3.0, sigma_tau: float = 1.2, sigma_xi: float = 0.25):
        super().__init__()
        self.item_names = item_names
        self.d = len(item_names)
        self.n_levels = n_levels           # L: number of transitions -> L+1 levels (0..L)
        self.n_subjects = n_subjects
        self.n_sources = n_sources
        self.t0 = t0
        self.sigma_tau = sigma_tau
        self.sigma_xi = sigma_xi

        signs = torch.tensor([item_signs[name] for name in item_names], dtype=torch.float32)
        self.register_buffer("sign", signs)  # (d,) fixed, assumed known a priori (clinical direction)

        # ---- fixed effects (population parameters) ----
        self.g = nn.Parameter(torch.zeros(self.d))                    # p_k = sigmoid(g_k)
        self.log_v = nn.Parameter(torch.zeros(self.d))                # v_k = softplus(log_v_k)
        self.log_delta = nn.Parameter(torch.zeros(self.d, n_levels))  # delta_k^m = softplus(...)
        self.A = nn.Parameter(torch.randn(self.d, n_sources) * 0.3)   # mixing matrix, w_i = A @ s_i

        # ---- random effects (individual parameters), jointly optimised (MAP) ----
        self.tau = nn.Parameter(torch.zeros(n_subjects))
        self.xi = nn.Parameter(torch.zeros(n_subjects))
        self.sources = nn.Parameter(torch.zeros(n_subjects, n_sources))

    def p_v_delta(self):
        p = torch.sigmoid(self.g)                     # (d,)
        v = nn.functional.softplus(self.log_v) + 1e-3  # (d,)
        delta = nn.functional.softplus(self.log_delta) + 1e-3  # (d, n_levels)
        return p, v, delta

    def w(self):
        return self.sources @ self.A.T  # (n_subjects, d)

    @staticmethod
    def sigmoid_curve(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, 0.03, 0.97)
        denom = p * (1 - p)
        z = torch.clamp(-x / denom, -30.0, 30.0)  # avoid exp overflow
        return 1.0 / (1.0 + (1.0 / p - 1.0) * torch.exp(z))

    def level_probs(self, subj_idx: torch.Tensor, item_idx: torch.Tensor,
                     age: torch.Tensor) -> torch.Tensor:
        """Returns (n_obs, n_levels+1) matrix of P(y == l) for l = 0..n_levels."""
        p, v, delta = self.p_v_delta()
        w_all = self.w()

        tau_i = self.tau[subj_idx]
        xi_i = self.xi[subj_idx]
        w_i = w_all[subj_idx, item_idx]
        p_k = p[item_idx]
        v_k = v[item_idx]
        sign_k = self.sign[item_idx]
        delta_k = delta[item_idx]  # (n_obs, n_levels)

        raw_time = torch.exp(xi_i) * (age - self.t0 - tau_i)  # (n_obs,)

        cum_delta = torch.cumsum(delta_k, dim=1)  # (n_obs, n_levels), cum_delta[:, l-1] = sum_{m<=l} delta^m
        # P(y>=0) = 1 always; P(y>=l) for l=1..n_levels; P(y>=n_levels+1)=0
        ones = torch.ones_like(raw_time).unsqueeze(1)
        zeros = torch.zeros_like(raw_time).unsqueeze(1)

        x_l = sign_k.unsqueeze(1) * v_k.unsqueeze(1) * raw_time.unsqueeze(1) \
              - v_k.unsqueeze(1) * cum_delta + w_i.unsqueeze(1)  # (n_obs, n_levels)
        p_ge_mid = self.sigmoid_curve(x_l, p_k.unsqueeze(1))
        p_ge = torch.cat([ones, p_ge_mid, zeros], dim=1)  # (n_obs, n_levels+2)

        # enforce monotone non-increasing (numerical safety)
        p_ge, _ = torch.cummin(p_ge, dim=1)
        probs = -torch.diff(p_ge, dim=1)  # (n_obs, n_levels+1) = P(y=l)
        probs = torch.clamp(probs, min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def nll(self, subj_idx, item_idx, age, value) -> torch.Tensor:
        probs = self.level_probs(subj_idx, item_idx, age)
        picked = probs.gather(1, value.long().unsqueeze(1)).squeeze(1)
        return -torch.log(picked).sum()

    def prior_penalty(self) -> torch.Tensor:
        tau_pen = 0.5 * (self.tau ** 2).sum() / self.sigma_tau ** 2
        xi_pen = 0.5 * (self.xi ** 2).sum() / self.sigma_xi ** 2
        s_pen = 0.5 * (self.sources ** 2).sum()
        return tau_pen + xi_pen + s_pen


# --------------------------------------------------------------------------
# Fitting driver
# --------------------------------------------------------------------------

def fit(long_df: pd.DataFrame, item_signs: dict[str, int], n_sources: int = 2,
        n_iter: int = 4000, lr: float = 0.005, seed: int = 0,
        verbose_every: int = 500) -> tuple[OrdinalDCM, dict]:
    torch.manual_seed(seed)

    subjects = sorted(long_df["subject_id"].unique())
    items = sorted(item_signs.keys())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    item_to_idx = {k: i for i, k in enumerate(items)}

    subj_idx = torch.tensor(long_df["subject_id"].map(subj_to_idx).values, dtype=torch.long)
    item_idx = torch.tensor(long_df["item"].map(item_to_idx).values, dtype=torch.long)
    age = torch.tensor(long_df["age"].values, dtype=torch.float32)
    value = torch.tensor(long_df["value"].values, dtype=torch.float32)

    model = OrdinalDCM(item_names=items, item_signs=item_signs,
                        n_subjects=len(subjects), n_sources=n_sources)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for it in range(n_iter):
        optimizer.zero_grad()
        loss = model.nll(subj_idx, item_idx, age, value) + model.prior_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(loss.item())
        if verbose_every and it % verbose_every == 0:
            print(f"iter {it:5d}  loss {loss.item():12.1f}")

    meta = {
        "subjects": subjects, "items": items,
        "subj_to_idx": subj_to_idx, "item_to_idx": item_to_idx,
        "history": history,
    }
    return model, meta


def extract_individual_params(model: OrdinalDCM, meta: dict) -> pd.DataFrame:
    tau = model.tau.detach().numpy()
    xi = model.xi.detach().numpy()
    sources = model.sources.detach().numpy()
    rows = []
    for s, i in meta["subj_to_idx"].items():
        row = {"subject_id": s, "tau_hat": tau[i], "xi_hat": xi[i]}
        for k in range(sources.shape[1]):
            row[f"source_{k+1}_hat"] = sources[i, k]
        rows.append(row)
    return pd.DataFrame(rows)


def extract_fixed_effects(model: OrdinalDCM) -> pd.DataFrame:
    p, v, delta = model.p_v_delta()
    p, v, delta = p.detach().numpy(), v.detach().numpy(), delta.detach().numpy()
    rows = []
    for i, item in enumerate(model.item_names):
        rows.append({
            "item": item, "p_hat": p[i], "v_hat": v[i], "sign": model.sign[i].item(),
            **{f"delta_{m+1}_hat": delta[i, m] for m in range(model.n_levels)},
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    base = Path(__file__).parent
    long_df = pd.read_csv(base / "output" / "simulated_long.csv")
    with open(base / "output" / "item_metadata.json") as f:
        meta_json = json.load(f)
    item_signs = {name: spec["sign"] for name, spec in meta_json["items"].items()}

    model, meta = fit(long_df, item_signs, n_sources=2, n_iter=4000, lr=0.005)

    fixed_df = extract_fixed_effects(model)
    indiv_df = extract_individual_params(model, meta)

    out_dir = base / "output"
    fixed_df.to_csv(out_dir / "fitted_fixed_effects.csv", index=False)
    indiv_df.to_csv(out_dir / "fitted_individual_params.csv", index=False)
    torch.save(model.state_dict(), out_dir / "fitted_model_state.pt")

    print("\nFixed effects (fitted):")
    print(fixed_df.round(3))
    print("\nSaved fitted_fixed_effects.csv, fitted_individual_params.csv, fitted_model_state.pt")
