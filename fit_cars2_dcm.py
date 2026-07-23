"""
Ordinal DCM fitter for the CARS2-ST simulator (cars2_simulator.py).

n_levels=6 (7-level 1-4/0.5-step scale), 15 items, 3 ICA sources -- matching
the externally validated 3-factor structure (Campbell et al., 2026). Same
joint-MAP estimation approach as the GARS-3 / earlier fitters (see those
READMEs for why: real leaspy's classic ordinal-model API needs torch<1.12,
incompatible with Python>=3.11).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Cars2OrdinalDCM(nn.Module):
    def __init__(self, items: list[str], n_subjects: int, n_sources: int = 3,
                 n_levels: int = 6, t0: float = 3.0, sigma_tau: float = 1.2, sigma_xi: float = 0.25):
        super().__init__()
        self.items = items
        self.d = len(items)
        self.n_levels = n_levels
        self.n_subjects = n_subjects
        self.n_sources = n_sources
        self.t0 = t0
        self.sigma_tau = sigma_tau
        self.sigma_xi = sigma_xi

        self.g = nn.Parameter(torch.zeros(self.d))
        self.log_v = nn.Parameter(torch.zeros(self.d))
        self.log_delta = nn.Parameter(torch.zeros(self.d, n_levels))
        self.A = nn.Parameter(torch.randn(self.d, n_sources) * 0.15)

        self.tau = nn.Parameter(torch.zeros(n_subjects))
        self.xi = nn.Parameter(torch.zeros(n_subjects))
        self.sources = nn.Parameter(torch.zeros(n_subjects, n_sources))

    def p_v_delta(self):
        p = torch.sigmoid(self.g)
        v = nn.functional.softplus(self.log_v) + 1e-3
        delta = nn.functional.softplus(self.log_delta) + 1e-3
        return p, v, delta

    def w(self):
        return self.sources @ self.A.T

    @staticmethod
    def sigmoid_curve(x, p):
        p = torch.clamp(p, 0.03, 0.97)
        denom = p * (1 - p)
        z = torch.clamp(-x / denom, -30.0, 30.0)
        return 1.0 / (1.0 + (1.0 / p - 1.0) * torch.exp(z))

    def level_probs(self, subj_idx, item_idx, age):
        p, v, delta = self.p_v_delta()
        w_all = self.w()

        tau_i, xi_i = self.tau[subj_idx], self.xi[subj_idx]
        w_i = w_all[subj_idx, item_idx]
        p_k, v_k, delta_k = p[item_idx], v[item_idx], delta[item_idx]

        raw_time = torch.exp(xi_i) * (age - self.t0 - tau_i)
        cum_delta = torch.cumsum(delta_k, dim=1)

        ones = torch.ones_like(raw_time).unsqueeze(1)
        zeros = torch.zeros_like(raw_time).unsqueeze(1)
        x_l = v_k.unsqueeze(1) * raw_time.unsqueeze(1) - v_k.unsqueeze(1) * cum_delta + w_i.unsqueeze(1)
        p_ge_mid = self.sigmoid_curve(x_l, p_k.unsqueeze(1))
        p_ge = torch.cat([ones, p_ge_mid, zeros], dim=1)
        p_ge, _ = torch.cummin(p_ge, dim=1)
        probs = -torch.diff(p_ge, dim=1)
        probs = torch.clamp(probs, min=1e-7)
        return probs / probs.sum(dim=1, keepdim=True)

    def nll(self, subj_idx, item_idx, age, value):
        probs = self.level_probs(subj_idx, item_idx, age)
        picked = probs.gather(1, value.long().unsqueeze(1)).squeeze(1)
        return -torch.log(picked).sum()

    def prior_penalty(self):
        return (0.5 * (self.tau ** 2).sum() / self.sigma_tau ** 2
                + 0.5 * (self.xi ** 2).sum() / self.sigma_xi ** 2
                + 0.5 * (self.sources ** 2).sum())


def fit(long_df: pd.DataFrame, n_sources: int = 3, n_iter: int = 5000, lr: float = 0.006,
        seed: int = 0, verbose_every: int = 1000):
    torch.manual_seed(seed)
    subjects = sorted(long_df["subject_id"].unique())
    items = sorted(long_df["item"].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    item_to_idx = {k: i for i, k in enumerate(items)}

    subj_idx = torch.tensor(long_df["subject_id"].map(subj_to_idx).values, dtype=torch.long)
    item_idx = torch.tensor(long_df["item"].map(item_to_idx).values, dtype=torch.long)
    age = torch.tensor(long_df["age"].values, dtype=torch.float32)
    value = torch.tensor(long_df["value"].values, dtype=torch.float32)

    model = Cars2OrdinalDCM(items=items, n_subjects=len(subjects), n_sources=n_sources)
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

    meta = {"subjects": subjects, "items": items, "subj_to_idx": subj_to_idx,
            "item_to_idx": item_to_idx, "history": history}
    return model, meta


def extract_individual_params(model, meta):
    tau, xi = model.tau.detach().numpy(), model.xi.detach().numpy()
    sources = model.sources.detach().numpy()
    rows = []
    for s, i in meta["subj_to_idx"].items():
        row = {"subject_id": s, "tau_hat": tau[i], "xi_hat": xi[i]}
        for k in range(sources.shape[1]):
            row[f"source_{k+1}_hat"] = sources[i, k]
        rows.append(row)
    return pd.DataFrame(rows)


def extract_fixed_effects(model):
    p, v, delta = model.p_v_delta()
    p, v, delta = p.detach().numpy(), v.detach().numpy(), delta.detach().numpy()
    rows = []
    for i, item in enumerate(model.items):
        rows.append({"item": item, "p_hat": p[i], "v_hat": v[i],
                      **{f"delta_{m+1}_hat": delta[i, m] for m in range(model.n_levels)}})
    return pd.DataFrame(rows)


def item_space_shift_correlation(model):
    w_all = model.w().detach().numpy()
    corr = np.corrcoef(w_all.T)
    return pd.DataFrame(corr, index=model.items, columns=model.items)


if __name__ == "__main__":
    base = Path(__file__).parent
    long_df = pd.read_csv(base / "output" / "cars2_simulated_long.csv")

    model, meta = fit(long_df, n_sources=3, n_iter=5000, lr=0.006)

    fixed_df = extract_fixed_effects(model)
    indiv_df = extract_individual_params(model, meta)
    out_dir = base / "output"
    fixed_df.to_csv(out_dir / "cars2_fitted_fixed_effects.csv", index=False)
    indiv_df.to_csv(out_dir / "cars2_fitted_individual_params.csv", index=False)
    torch.save(model.state_dict(), out_dir / "cars2_fitted_model_state.pt")

    print("\nFitted fixed effects (head):")
    print(fixed_df.head(8).round(3))
