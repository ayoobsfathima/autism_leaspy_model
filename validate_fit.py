"""
End-to-end validation of the ordinal DCM fit against the known ground truth
from the simulator. Mirrors the validation approach in Poulet & Durrleman
(2023) Section 3.1 / Figures 3-4:
  1. Reconstruction error (predicted vs. true ordinal level)
  2. Whether the recovered individual random effects (tau, xi, sources)
     let a simple classifier recover the true latent subtype -- this is
     rotation/sign-invariant, so it's a fair check even though ICA sources
     are only identifiable up to rotation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from fit_ordinal_dcm import fit, extract_individual_params, extract_fixed_effects

BASE = Path(__file__).parent
OUT = BASE / "output"


def main():
    long_df = pd.read_csv(OUT / "simulated_long.csv")
    with open(OUT / "item_metadata.json") as f:
        meta_json = json.load(f)
    item_signs = {name: spec["sign"] for name, spec in meta_json["items"].items()}

    print("Fitting ordinal DCM (joint MAP, torch/Adam)...")
    model, meta = fit(long_df, item_signs, n_sources=2, n_iter=4000, lr=0.005, verbose_every=1000)

    fixed_df = extract_fixed_effects(model)
    indiv_df = extract_individual_params(model, meta)
    fixed_df.to_csv(OUT / "fitted_fixed_effects.csv", index=False)
    indiv_df.to_csv(OUT / "fitted_individual_params.csv", index=False)
    torch.save(model.state_dict(), OUT / "fitted_model_state.pt")

    # ---- reconstruction accuracy ----
    subj_idx = torch.tensor(long_df["subject_id"].map(meta["subj_to_idx"]).values, dtype=torch.long)
    item_idx = torch.tensor(long_df["item"].map(meta["item_to_idx"]).values, dtype=torch.long)
    age = torch.tensor(long_df["age"].values, dtype=torch.float32)
    value = long_df["value"].values

    with torch.no_grad():
        probs = model.level_probs(subj_idx, item_idx, age)
        pred_argmax = probs.argmax(dim=1).numpy()
        pred_expect = (probs * torch.arange(probs.shape[1])).sum(dim=1).numpy()

    mae_argmax = float(np.abs(pred_argmax - value).mean())
    mae_expect = float(np.abs(pred_expect - value).mean())
    mae_baseline = float(np.abs(0 - value).mean())

    # ---- random-effect recovery vs ground truth ----
    gt = pd.read_csv(OUT / "simulated_subjects_ground_truth.csv")
    merged = gt.merge(indiv_df, on="subject_id")
    merged.to_csv(OUT / "merged_validation.csv", index=False)

    corr_tau = float(np.corrcoef(merged.tau, merged.tau_hat)[0, 1])
    corr_xi = float(np.corrcoef(merged.xi, merged.xi_hat)[0, 1])

    valid = merged.dropna(subset=["cluster"])
    X_hat = valid[["tau_hat", "xi_hat", "source_1_hat", "source_2_hat"]].values
    X_true = valid[["tau", "xi", "source_1", "source_2"]].values
    y = valid["cluster"].values
    clf = LogisticRegression(max_iter=1000)
    acc_hat = float(cross_val_score(clf, X_hat, y, cv=5).mean())
    acc_true = float(cross_val_score(clf, X_true, y, cv=5).mean())

    report = {
        "reconstruction_mae_argmax": round(mae_argmax, 4),
        "reconstruction_mae_expectation": round(mae_expect, 4),
        "reconstruction_mae_naive_baseline": round(mae_baseline, 4),
        "correlation_tau_recovered_vs_true": round(corr_tau, 4),
        "correlation_xi_recovered_vs_true": round(corr_xi, 4),
        "cluster_cv_accuracy_from_recovered_random_effects": round(acc_hat, 4),
        "cluster_cv_accuracy_from_true_random_effects_upper_bound": round(acc_true, 4),
    }
    with open(OUT / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nValidation report:")
    print(json.dumps(report, indent=2))

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    tsne = TSNE(n_components=2, random_state=0, perplexity=15)
    emb = tsne.fit_transform(X_hat)
    colors = np.where(y == 1, "tab:blue", "tab:red")
    axes[0].scatter(emb[:, 0], emb[:, 1], c=colors, s=25, alpha=0.8)
    axes[0].set_title(f"t-SNE of recovered random effects\n(colored by true cluster; CV acc={acc_hat:.2f})")
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

    axes[1].scatter(value, pred_expect, s=8, alpha=0.25)
    lims = [-0.3, 4.3]
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)
    axes[1].set_xlabel("true ordinal level"); axes[1].set_ylabel("predicted (expectation)")
    axes[1].set_title(f"Reconstruction: predicted vs true\nMAE={mae_expect:.3f} (naive baseline={mae_baseline:.3f})")

    fig.tight_layout()
    fig.savefig(OUT / "validation_plots.png", dpi=130)
    print("\nSaved validation_plots.png, validation_report.json, merged_validation.csv")


if __name__ == "__main__":
    main()
