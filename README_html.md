# Roots — developmental progress tracker (prototype)

A single-file, no-build-step web app that personalizes a child's developmental
trajectory using the ordinal disease-course-mapping model from
`fit_ordinal_dcm.py`, fit on the synthetic cohort.

## What it does
- Add a child, log visits (age + 0–4 scores on 7 items)
- Runs the same ordinal DCM math client-side (in JS) to personalize that
  child's individual parameters (time-shift, pace, profile) given their visits
- Plots each item's population-average curve vs. this child's personalized
  curve, with observed visits marked
- Predicts scores at any future age

## Try it now
Open `index.html` directly in a browser — no server, no build step. Click
"Load demo children" to see it working on two of the synthetic cohort's kids.

## Deploying it for real
Because it's a single static file, any of these work:
- **GitHub Pages**: push this folder to a repo, enable Pages on it
- **Netlify / Vercel**: drag the folder onto their web UI
- Any static file host / `python3 -m http.server` for local testing

## Important limitations before this touches real patient data

1. **Population parameters are fit on synthetic data.** Before any real use,
   refit `fit_ordinal_dcm.py` (or real leaspy — see the sibling
   `autism_dcm_sim/` folder) on real patient data, then re-export and swap
   the `MODEL` constant at the top of `index.html`'s script with the new
   fixed effects.
2. **Storage is not clinical-grade.** The app persists data via this preview's
   built-in storage, or `localStorage` when deployed standalone. Neither has
   access controls, encryption at rest, or audit logging. A real deployment
   handling identifiable child health data needs a proper backend (auth,
   encrypted database, audit trail) — likely a hard requirement for your ICMR
   ethics approval, not just a nice-to-have.
3. **Fixed-effect identifiability caveat still applies** (see the sibling
   README) — trust trends and reconstruction, not raw parameter values.
4. **"Profile coordinates" (the two ICA sources) are shown raw, deliberately
   uninterpreted.** Turning them into a clinical subtype label needs the same
   validation the paper does in Section 3.2 (correlating sources against
   known clinical patterns) — don't let the app assert a subtype without that.

## Next steps
- Swap in the real fitted model once you have real cohort data
- Add authentication + a real backend before any real child's data goes in
- Consider a "confidence" indicator on predictions when a child has very few
  visits (personalization with 1-2 visits is barely constrained by data and
  leans heavily on the population prior)
