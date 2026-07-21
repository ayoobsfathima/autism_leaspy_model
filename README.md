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
