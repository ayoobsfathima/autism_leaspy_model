"""
Reference script for fitting the ordinal DCM with the REAL `leaspy` package
(classic 1.x API, which has genuine ordinal support and full MCMC-SAEM
estimation -- this is the actual method from Poulet & Durrleman 2023).

WHY THIS ISN'T RUN IN THE SANDBOX: leaspy's classic API (needed for ordinal
models) pins `torch<1.12`, and there are no torch<1.12 wheels for Python
>=3.11. `fit_ordinal_dcm.py` in this folder is a from-scratch stand-in
(same model, joint MAP estimation) that runs anywhere modern, and was
validated against this simulator's ground truth. Once you have real leaspy
running, you can point THIS script at the same `simulated_long.csv` (or
real patient data) and compare -- the model is identical, only the
estimation algorithm differs (MCMC-SAEM here vs. gradient-based MAP there).

--------------------------------------------------------------------------
SETUP (run once, on your own machine -- NOT in this sandbox):

    conda create -n roots-dcm python=3.10 -y
    conda activate roots-dcm
    pip install "leaspy==1.5.0"
    pip install pandas numpy matplotlib

Then run:  python fit_with_real_leaspy.py
--------------------------------------------------------------------------
"""

import pandas as pd
from leaspy import Leaspy, Data, AlgorithmSettings


def main():
    # --- 1. Load data in leaspy's expected long format: ID, TIME, <features...> ---
    long_df = pd.read_csv("output/simulated_long.csv")
    wide_df = long_df.pivot_table(index=["subject_id", "age"], columns="item", values="value").reset_index()
    wide_df = wide_df.rename(columns={"subject_id": "ID", "age": "TIME"})

    data = Data.from_dataframe(wide_df)

    # --- 2. Instantiate the ordinal logistic DCM model ---
    # source_dimension mirrors n_sources in our simulator (2)
    leaspy_model = Leaspy("logistic", noise_model="ordinal", source_dimension=2)

    # --- 3. Calibrate with MCMC-SAEM (the actual algorithm from the paper) ---
    settings = AlgorithmSettings(
        "mcmc_saem",
        n_iter=8000,          # paper used ~20000 for the full 59-item PPMI model;
                               # this synthetic 7-item cohort needs far fewer
        seed=0,
        progress_bar=True,
    )
    leaspy_model.fit(data, settings)

    leaspy_model.save("output/leaspy_model_real.json")
    print("Saved fitted model to output/leaspy_model_real.json")

    # --- 4. Personalize: estimate each subject's individual random effects ---
    personalize_settings = AlgorithmSettings("scipy_minimize", seed=0, progress_bar=True)
    individual_parameters = leaspy_model.personalize(data, personalize_settings)
    individual_parameters.save("output/leaspy_individual_params_real.json")
    print("Saved individual parameters to output/leaspy_individual_params_real.json")

    # --- 5. From here: compare tau/xi/sources against
    #        output/simulated_subjects_ground_truth.csv, same as validate_fit.py does
    #        for the stand-in estimator, to sanity-check the real fit too.


if __name__ == "__main__":
    main()
