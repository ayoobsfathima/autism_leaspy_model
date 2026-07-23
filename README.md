# CARS2-ST Ordinal Disease Course Mapping — Final Build (n=300)

Complete rebuild around CARS2-ST at n=300, the sample size that unlocks
reliable item-level factor recovery (see the sample-size sweep from the
earlier n=100 build: ARI goes from -0.13 at n=100 to +0.40 at n=300).

## Why CARS2-ST, why n=300
- **Satisfies "clinician only"**: CARS2-ST is genuine clinician direct
  behavioral observation, not an informant checklist a clinician happens to
  fill out (unlike GARS-3/ABC).
- **Ground truth from a real, published factor analysis**: Campbell et al.
  (2026, *Research in Autism*) — Social Communication (8 items),
  Restrictive/Repetitive Behavior/Sensory (3 items), Emotional/Behavioral
  Dysregulation (4 items), with published factor correlations SC-RB=.87,
  RB-ED=.49, SC-ED=.23.
- **n=300 is where the model actually recovers this structure** — confirmed
  empirically, not assumed (see `output/cars2_sample_size_sweep.csv` from
  the earlier diagnostic run for the full n=100/200/300/500 comparison).

## Pipeline (in order)
1. `cars2_simulator.py` — generates synthetic data. Ground truth sources are
   drawn CORRELATED (matching the published .87/.49/.23), not independent.
2. `fit_cars2_dcm.py` — the working fitter (joint MAP via torch/Adam,
   documented stand-in for real leaspy; see below).
3. `fit_with_real_leaspy.py` — reference script for real `leaspy` (needs
   Python 3.9-3.10, see the script's docstring for exact setup).
4. Validation (reconstruction + random-effect recovery + factor-structure
   recovery against Campbell et al.'s published grouping).
5. `cars2_dcm_demo/index.html` — standalone deployable demo app.

## Results at n=300

```
reconstruction_mae_expectation:                          1.40   (vs 2.51 midpoint baseline, 0-6 internal scale)
correlation_tau_recovered_vs_true:                        0.57
correlation_xi_recovered_vs_true:                         0.64
profile_cv_accuracy_from_recovered_random_effects:        0.63   (vs 0.90 upper bound using true effects)
factor_recovery_adjusted_rand_index_vs_published:         0.40   (vs -0.13 at n=100)
```

Meaningfully better than n=100 across the board, especially the factor
recovery (the thing that was actually broken before). Still imperfect —
SC and RB remain hard to fully separate because they're genuinely
near-collinear in the real instrument (published r=.87), not an artifact of
our modeling. The demo's Population Structure tab shows this directly: SC
and RB items blend somewhat in the correlation heatmap, while ED stands out
more distinctly, matching the published correlation pattern.

## Files

**Simulator & data**
- `cars2_simulator.py`
- `output/cars2_simulated_long_n300.csv` — long-format data (Leaspy-compatible)
- `output/cars2_subjects_ground_truth_n300.csv` — true random effects, for validation
- `output/cars2_item_table.json` — item metadata (factor, published loading, fixed effects)

**Fitting**
- `fit_cars2_dcm.py` — working stand-in fitter
- `fit_with_real_leaspy.py` — real-leaspy reference (Python 3.9-3.10 required)
- `output/cars2_n300_fitted_fixed_effects.csv`
- `output/cars2_n300_fitted_individual_params.csv`
- `output/cars2_n300_fitted_model_state.pt` — raw torch weights
- `output/cars2_n300_fitted_model_export.json` — portable JSON (fixed effects + mixing matrix), used by the demo app

**Validation**
- `output/cars2_n300_validation_report.json`
- `output/cars2_n300_validation_plots.png`
- `output/cars2_n300_merged_validation.csv`

**Demo app**
- `cars2_dcm_demo/index.html` — standalone, no build step, no backend.
  Two tabs: Trajectory Explorer (personalize + predict a child's course) and
  Population Structure (item-clustering heatmap, the Figure-5 analog).

## Known limitations (carried forward honestly, not hidden)
1. Fixed-effect values (`p_hat`, `v_hat`, `delta_hat`) are not uniquely
   identifiable in this model family — trust reconstruction accuracy and
   structure recovery, not raw parameter values.
2. SC/RB separation remains partial even at n=300, because the published
   instrument itself has these factors at r=.87. This is a property of
   CARS2-ST, not a fitting failure.
3. The stand-in estimator (joint MAP) understates posterior uncertainty
   compared to leaspy's full MCMC-SAEM. Point estimates should be fine for
   demonstration purposes; don't treat confidence intervals as available.
4. Simulated data only — no real patient data has been used anywhere in
   this pipeline. Before any real use, refit on real CARS2-ST data and
   re-export the model for the demo app.
