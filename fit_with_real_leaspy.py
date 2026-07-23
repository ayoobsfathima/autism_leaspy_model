"""
Reference script for fitting the CARS2-ST ordinal DCM with the REAL `leaspy`
package (classic 1.x API, genuine ordinal support, full MCMC-SAEM
estimation) -- the actual method from Poulet & Durrleman (2023).

WHY THIS ISN'T RUN DIRECTLY HERE: leaspy's classic API (needed for ordinal
models) pins `torch<1.12`, and there are no torch<1.12 wheels for Python
>=3.11. `fit_cars2_dcm.py` in this folder is a from-scratch stand-in (same
model, joint MAP estimation via PyTorch autodiff) that runs anywhere modern,
and has been validated against this simulator's ground truth (see
cars2_n300_validation_report.json). Once you have real leaspy running, this
script fits the SAME data with the SAME model -- only the estimation
algorithm differs (MCMC-SAEM here vs. gradient-based MAP in the stand-in).

--------------------------------------------------------------------------
SETUP (on your own machine -- NOT in this sandbox):

    conda create -n roots-dcm python=3.10 -y
    conda activate roots-dcm
    pip install "leaspy==1.5.0" pandas numpy matplotlib

Then run:  python fit_with_real_leaspy.py
--------------------------------------------------------------------------
"""

import pandas as pd
from leaspy import Leaspy, Data, AlgorithmSettings


def main():
    # --- 1. Load data in leaspy's expected long format: ID, TIME, <features...> ---
    long_df = pd.read_csv("output/cars2_simulated_long_n300.csv")
    wide_df = long_df.pivot_table(index=["subject_id", "age"], columns="item", values="value").reset_index()
    wide_df = wide_df.rename(columns={"subject_id": "ID", "age": "TIME"})

    data = Data.from_dataframe(wide_df)

    # --- 2. Instantiate the ordinal logistic DCM model ---
    # source_dimension=3, matching the externally validated Campbell et al.
    # (2026) 3-factor structure (Social Communication / Restrictive-
    # Repetitive-Behavior-Sensory / Emotional-Behavioral-Dysregulation)
    leaspy_model = Leaspy("logistic", noise_model="ordinal", source_dimension=3)

    # --- 3. Calibrate with MCMC-SAEM (the actual algorithm from the paper) ---
    settings = AlgorithmSettings(
        "mcmc_saem",
        n_iter=8000,
        seed=0,
        progress_bar=True,
    )
    leaspy_model.fit(data, settings)

    leaspy_model.save("output/cars2_leaspy_model_real.json")
    print("Saved fitted model to output/cars2_leaspy_model_real.json")

    # --- 4. Personalize: estimate each subject's individual random effects ---
    personalize_settings = AlgorithmSettings("scipy_minimize", seed=0, progress_bar=True)
    individual_parameters = leaspy_model.personalize(data, personalize_settings)
    individual_parameters.save("output/cars2_leaspy_individual_params_real.json")
    print("Saved individual parameters to output/cars2_leaspy_individual_params_real.json")

    # --- 5. From here: compare tau/xi/sources against
    #        output/cars2_subjects_ground_truth_n300.csv, and compare the
    #        item space-shift correlation structure against Campbell et
    #        al.'s published SC/RB/ED factor structure, same checks as
    #        validate_cars2_fit.py does for the stand-in estimator.


if __name__ == "__main__":
    main()
