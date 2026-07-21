# Autism Ordinal Disease Course Mapping — Synthetic Data Simulator

Generates synthetic longitudinal ordinal marker data for autism progression,
using the same generative model as the ordinal Disease Course Mapping (DCM)
approach in Poulet & Durrleman (2023), *Statistics in Medicine* — the paper
behind the Leaspy library you're planning to use for the Roots app.

## What it simulates

- **7 items**, each on a 0–4 ordinal (Likert-style) scale, standing in for
  the instruments in your protocol:
  - `communication_cder`, `social_cder` — CDER
  - `adaptive_vabs` — VABS-II composite, bucketed to 5 levels
  - `hyperactivity_abc`, `irritability_abc` — ABC
  - `insistence_sameness` — ADI-R "Insistence on Sameness"
  - `rsm_repetitive` — repetitive sensorimotor behaviour (near-flat slope,
    reflecting the "generally stable" pattern noted in your protocol)
- **100 synthetic children**, ages 3–10 (within your 70–116 sample-size
  range), 2–6 irregular visits each.
- **Two latent subtypes**, baked in via one ICA source, so you have a known
  ground truth to check whether a fitted model (Leaspy or otherwise)
  recovers the right cluster structure — same validation trick as the
  paper's Section 3.1 experiment.
- Individual random effects: time-shift (`tau`, early/late developmental
  pace), log-acceleration (`xi`, fast/slow progressor), and item-level
  space-shifts (via a small ICA mixing matrix).

## Files

- `ordinal_dcm_simulator.py` — the model + generation code
- `output/simulated_long.csv` — one row per (subject, visit, item); this is
  the format Leaspy expects (long format)
- `output/simulated_wide.csv` — one row per (subject, visit), items as columns
- `output/simulated_subjects_ground_truth.csv` — true random effects & cluster
  label per subject, for validating a fitted model against ground truth
- `output/item_metadata.json` — item definitions (direction, fixed effects)
- `output/population_curves.png` — sanity-check plot of the population-average
  probability of each ordinal level, by age, for every item

## Run it

```bash
python3 ordinal_dcm_simulator.py
```

Deterministic given `seed=42`; change the seed or `n_subjects` in
`simulate_cohort()` to generate different cohorts.

## Design choices worth revisiting with your clinical team

1. **Item set is a placeholder.** I picked one representative composite per
   instrument family from your protocol rather than every individual item
   (e.g. all CDER sub-items), because with n≈70–116 real patients, item-level
   modelling of 50+ items (like the paper's 59 MDS-UPDRS items on 900
   patients) would almost certainly be under-powered. Worth deciding now
   whether Leaspy will run on subscales or true items once real data exists.
2. **Direction handling.** The ordinal cumulative-logit model (paper eq. 3)
   assumes P(Y≥l) is non-increasing in l, which must hold regardless of
   whether the *item itself* trends up or down with age. I fixed this by
   applying the trend direction (`sign`) only to the age-varying part of the
   warped time, while the level-delay offsets always act in the same
   direction — this is a deviation/clarification from the literal paper
   formula (which only handles increasing items, since Parkinson's markers
   only worsen) and matters for your "improving" items like hyperactivity.
3. **No comorbidity/covariate layer yet.** Your protocol's open questions
   (IQ, puberty onset, sex, seizures, SES, etc.) aren't in this simulator.
   Once the core ordinal DCM is running, those become either (a) covariates
   on the fixed effects, or (b) stratification variables for separate
   population curves — decide based on how large your eventual real sample is.
4. **t0 = 3 years** is used as the shared reference age across items. In the
   original paper t0 is itself a fitted population parameter; here it's
   fixed for simplicity since we're generating (not estimating) data.

## Next step

Fit an ordinal DCM (Leaspy) to `simulated_long.csv`, recover the fixed
effects and individual random effects, and check them against
`simulated_subjects_ground_truth.csv` (in particular: does the 2-cluster
structure separate out via t-SNE on random effects, same as paper Fig. 3?).
That validates the pipeline before you point it at real patient data.

---

## Phase 2: Fitting the ordinal DCM

**Important compatibility note:** the `leaspy` PyPI package's classic API
(1.x line, which has genuine ordinal-model support) pins `torch<1.12`,
which has no wheels for Python ≥3.11. The current 2.x rewrite installs
fine on modern Python but **hasn't ported ordinal observation models yet**
(only Gaussian/Bernoulli/Weibull as of v2.1.0). So there are two paths:

### Path A — real `leaspy`, on your own machine
Requires Python 3.9–3.10 (conda is the easiest way to get an isolated one):
```bash
conda create -n roots-dcm python=3.10 -y
conda activate roots-dcm
pip install "leaspy==1.5.0" pandas numpy matplotlib
python fit_with_real_leaspy.py
```
This runs the actual MCMC-SAEM algorithm from the paper. `fit_with_real_leaspy.py`
is fully written and ready to go — it's just gated on that Python version.

### Path B — `fit_ordinal_dcm.py` (works anywhere, already run and validated here)
Same ordinal DCM model (identical math to the simulator / paper Eq. 3),
fit via **joint MAP estimation** (gradient descent with PyTorch autodiff over
fixed effects + individual random effects, each regularized by its prior)
instead of MCMC-SAEM sampling. This is a standard, much cheaper approximation
(similar in spirit to Laplace/FOCE approaches elsewhere in mixed-effects
modelling) — it will understate posterior uncertainty vs. the paper's fully
Bayesian approach, but recovers the same structure well:

```bash
python3 fit_ordinal_dcm.py     # fits and saves fixed/individual params
python3 validate_fit.py        # fits + validates against ground truth + plots
```

**Validation results on the synthetic cohort** (`output/validation_report.json`):
- Reconstruction MAE (expectation): **0.14** on a 0–4 scale vs. **1.50** for
  a naive "predict 0" baseline
- 5-fold CV accuracy predicting the true latent subtype from the *recovered*
  random effects: **0.98** (vs. 1.00 using the true random effects as an
  upper bound) — this is the same rotation/sign-invariant check the paper
  uses in Section 3.1 / Figure 3, since ICA sources are only identifiable up
  to rotation.
- Correlation of recovered vs. true time-shift (`tau`): 0.79; log-acceleration
  (`xi`): 0.64.

See `output/validation_plots.png` for the t-SNE cluster-recovery plot and
the reconstruction scatter plot (paper-style Figs. 3–4 equivalents).

### Files (Phase 2)
- `fit_ordinal_dcm.py` — the stand-in torch/MAP fitter (model + training loop)
- `fit_with_real_leaspy.py` — reference script for real leaspy (Path A)
- `validate_fit.py` — runs the stand-in fitter + full validation + plots
- `output/fitted_fixed_effects.csv`, `output/fitted_individual_params.csv` —
  fitted parameters
- `output/validation_report.json`, `output/validation_plots.png`,
  `output/merged_validation.csv` — validation outputs

### Note on fixed-effect identifiability
Don't expect `fitted_fixed_effects.csv`'s raw `p_hat`/`v_hat`/`delta_hat`
values to match the simulator's ground-truth item parameters one-to-one —
this model family has known identifiability trade-offs between `p_k`, `v_k`,
and the deltas (multiple parameter combinations can produce near-identical
curves), which the paper itself notes. What matters, and what we validated,
is **reconstruction accuracy** and **recovery of the individual-level
structure** (subtype clusters, relative timing) — not the raw fixed-effect
values in isolation.

## Next step (Phase 3)
Wire this into the app layer: given a new child's visit history, use the
fitted model to personalize (estimate their tau/xi/sources), predict their
next-visit scores, and show where they sit relative to the population curve.
