# Audit of the CARS2-ST n=300 build (against Poulet & Durrleman 2023)

This audits the earlier `cars2_dcm_final/` deliverable: does it correctly
implement the paper's method, and does it support the conclusion that
autism developmental trajectories can be modeled the way the paper models
Parkinson's progression?

## Summary

| Check | Result |
|---|---|
| Simulator implements paper's Eq. 3 | Reparameterized (drops a constant `+v*t0` term), but consistently |
| Simulator, fitter, and HTML demo numerically agree | **Yes** — identical to 1e-7 |
| Previously reported validation numbers reproduce | **Yes** — exact match to 4 decimals |
| Simulated data resembles real CARS2-ST data | **No** — SD 3x too large, 73% of ratings at floor/ceiling |
| Model can represent a child who improves | **No** — structurally impossible (see below) |
| n=300 result (ARI=0.40) holds under realistic data | **No** — collapses toward chance (ARI ~0.05-0.15) |

## 1. Implementation checks (all passed)

`audit1.py`, `audit2.py`: confirmed numerically that the simulator's
generative process and the fitter's likelihood use the identical linear
predictor `x = v*e^xi*(t-t0-tau) - v*cum_delta + w`, agreeing to 1e-7 given
identical parameters. The HTML demo's JavaScript reproduces the same
function to the same precision (see the audit transcript). This differs
from a literal reading of the paper's Eq. 3 by a constant `v*t0` term per
item -- a reparameterization, not an error, and applied consistently
everywhere.

All five previously-reported n=300 validation numbers (reconstruction MAE,
correlation of recovered vs. true tau/xi, profile classification accuracy,
factor-recovery ARI) were re-run from scratch and reproduced exactly.

## 2. Data realism check (failed)

Compared the simulated CARS2-ST total score distribution against the
published real sample (Campbell et al. 2026, N=302):

| | Simulated (original) | Published (real) |
|---|---|---|
| Mean | 40.34 | 36.41 |
| SD | **13.45** | **4.40** |
| Range | 15.0-60.0 (full scale) | 22.0-50.5 |
| % of item ratings at an exact extreme (1.0 or 4.0) | **73.0%** | not reported, but implausible given SD=4.4 |

Root cause: item `delta` (level-transition spacing) parameters were too
small relative to the slope `v`, compressing all 6 ordinal transitions into
~47% of the 3-10y age window. Children rush from floor to ceiling instead
of progressing gradually -- unlike real clinician ratings, which cluster
mid-scale.

## 3. Structural check: can the model represent improvement? (failed)

The DCM's linear predictor has `dx/dt = v*exp(xi)`. Since `v` is a softplus
output (always positive) and `exp(xi)` is always positive, this derivative
has a **fixed sign for every individual, every item, always**. Swept 2,873
combinations of (tau, xi, w): zero produced a decreasing trajectory
(`audit5` in the transcript). This is correct for a neurodegenerative
cohort (monotone worsening) but is a real limitation for autism, where
Fountain et al. (2012, the paper you separately uploaded, n=6975) found
most children **improve** on social/communication measures ages 3-14, with
a ~7-10% "bloomers" subgroup improving dramatically.

Practical implication: the model can represent different *timing* and
*rate* of a single fixed-direction trend, and different *item profiles* via
the space-shift sources -- but not a cohort where some children get better
while others get worse on the same item. If that directional heterogeneity
matters for your use case, this model family needs an extension (e.g. a
per-individual direction term), not just different data.

## 4. Does the n=300 result survive a realistic data regime?

Rebuilt the simulator (`cars2_simulator_v2.py`) with calibrated delta/v/w
parameters that bring the score distribution much closer to the published
moments (extreme-rating fraction down from 73% to 24-38%, SD down from 13.5
to ~5-9 depending on configuration). Refit under this regime, controlling
for two possible confounds (fitter/simulator time-origin mismatch; fixed
t0 across a delta-scale sweep) -- both ruled out as the explanation.

Result: recovery collapses toward chance as the data becomes more
realistic:

| Data regime | corr(tau) | Profile CV accuracy (chance=0.50) | Factor-recovery ARI |
|---|---|---|---|
| Original (73% extreme, SD=13.5) | 0.57 | 0.63 | **0.40** |
| Calibrated (~35% extreme, SD~6-9) | ~0.13-0.26 | ~0.51-0.60 | **~0.02-0.22** |

## Caveat on the calibration itself

The SD=4.40 target is from a **cross-sectional** sample with a narrow mean
age (~3.5y, SD~1.7y in months) -- not a longitudinal 3-10y cohort. A cohort
spanning that full age range should legitimately show more dispersion than
a narrow-age referral sample. My calibration likely over-tightened as a
result. The honest conclusion is that the true answer sits somewhere
between the original build and this calibrated version, and **no published
longitudinal CARS2-ST dataset was found to pin down how much the
instrument actually changes over ages 3-10** -- which is the single number
this entire feasibility question hinges on.

## Files in this folder

- `cars2_simulator_v2.py` -- corrected simulator: adds a `direction` field
  per item/factor (so population trend can be improvement, not just
  decline, following Fountain et al.), and includes calibration utilities
  (`total_score_moments`) to check simulated data against published norms.
- `output/cars2_v2cal_long_n300.csv`, `cars2_v2cal_gt_n300.csv`,
  `cars2_v2cal_item_table.json` -- the calibrated (more realistic) n=300
  dataset used for the final validation table above.
- `output/cars2_signal_sweep.csv` -- the sweep showing recovery vs. amount
  of real score change over time.
- `audit1.py`, `audit2.py` -- the numerical cross-checks of simulator vs.
  fitter math.

## Recommended next step

Before drawing further conclusions from simulation, look for **any**
published longitudinal CARS/CARS2 data (repeated administrations on the
same children over multiple years) to anchor how much the instrument
actually moves over a 3-10y span. Without that anchor, the feasibility
question ("can this method work for autism the way it works for
Parkinson's") cannot be resolved by simulation alone -- the simulation's
conclusion flips depending on an assumption we don't have real data for.
