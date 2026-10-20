"""Fit Collins WMH model 26 separately to early and late blocks.

Place beside general_fitting_script.py in parameter_sweep/.
The original model definitions and likelihood come directly from that script.

Split: per participant, sort complete blocks by block number; first floor(N/2)
blocks are early; remaining blocks are late (middle block goes to late).

Run from anywhere: python parameter_sweep/fit_early_late.py
Outputs two .mat files in parameter_sweep/pooled_data/.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from general_fitting_script import define_models, RLWM_llh

Demo = False
Dataset = "pooled_expe1_3_5"


def main(Demo=Demo, Dataset=Dataset, niter=None, seed=None):
    root = Path(__file__).resolve().parent
    pooled_folder = root / "pooled_data"
    expe_data = loadmat(pooled_folder / f"{Dataset}.mat")["expe_data"]
    all_subjects = np.unique(expe_data[:, 0])

    # Preserve original model and optimization definitions.
    Ms = define_models()
    m = 26
    fit_model = Ms[m - 1]
    if not isinstance(fit_model, dict) or fit_model["name"] != "RLWM":
        raise ValueError("Expected RLWM-based WMH model 26")
    pmin = np.asarray(fit_model["pMin"], dtype=float)
    pmax = np.asarray(fit_model["pMax"], dtype=float)
    niter = (2 if Demo else 10) if niter is None else niter
    if niter < 1:
        raise ValueError("niter must be at least 1")
    subjects_list = all_subjects[:2] if Demo else all_subjects
    if seed is not None:
        np.random.seed(seed)

    for phase in ["early", "late"]:
        All_Params = [np.array([]) for _ in range(len(Ms))]
        All_fits = np.full((len(all_subjects), 6, len(Ms)), np.nan)
        fitmeasures, fitparams = [], []
        block_counts = np.zeros((len(all_subjects), 2), dtype=int)

        for k, s in enumerate(subjects_list):
            participant = expe_data[expe_data[:, 0] == s]
            # Column 1 is block number; sort unique block IDs chronologically.
            blocks = np.sort(np.unique(participant[:, 1]))
            if len(blocks) < 2:
                raise ValueError(f"Subject {s} has fewer than two complete blocks")
            midpoint = len(blocks) // 2
            selected_blocks = blocks[:midpoint] if phase == "early" else blocks[midpoint:]
            selected = participant[np.isin(participant[:, 1], selected_blocks)]
            if len(selected) == 0:
                raise ValueError(f"No {phase} trials for subject {s}")
            # Exact columns from general_fitting_script.py: stim/action/reward/set size/block/RT.
            data = selected[:, [4, 9, 11, 2, 1, 13]]
            subject_idx = np.flatnonzero(all_subjects == s)[0]
            block_counts[subject_idx] = [len(selected_blocks), len(data)]
            print(f"{phase}: Model {m}, subject {k+1}/{len(subjects_list)} "
                  f"(blocks={len(selected_blocks)}, trials={len(data)})", flush=True)

            # Same random-start initialization, K grid, SLSQP optimizer, and
            # model likelihood as the original fitting script.
            sofar = []
            pars = pmin + np.random.random((niter, len(pmin))) * (pmax - pmin)
            for K in range(2, 6):
                myfitfun = lambda p: RLWM_llh(p, K, data, fit_model)
                for it in range(niter):
                    par = pars[it, :]
                    output = minimize(myfitfun, par, method="SLSQP",
                                      bounds=list(zip(pmin, pmax)),
                                      options={"disp": False, "maxiter": 1000, "ftol": 1e-6})
                    sofar.append(np.r_[output.x, K, output.fun])
            sofar = np.asarray(sofar)
            i = np.argmin(sofar[:, -1])
            llh = sofar[i, -1]
            param = sofar[i, :-1]
            ntrials = len(data)
            AIC = 2 * llh + 2 * (len(param) + 1)
            BIC = 2 * llh + np.log(ntrials) * (len(param) + 1)
            AIC0 = -2 * np.log(1/3) * ntrials
            psr2 = (AIC0 - AIC) / AIC0
            fitmeasures.append([s, -llh, AIC, BIC, psr2, AIC0])
            fitparams.append(param)

        fitparams = np.asarray(fitparams)
        fitmeasures = np.asarray(fitmeasures)
        full_params = np.full((len(all_subjects), fitparams.shape[1]), np.nan)
        for row, s in enumerate(subjects_list):
            pos = np.flatnonzero(all_subjects == s)
            full_params[pos[0], :] = fitparams[row]
            All_fits[pos[0], :, m - 1] = fitmeasures[row]
        All_Params[m - 1] = full_params

        cells = np.empty((1, len(Ms)), dtype=object)
        params_cells = np.empty((1, len(All_Params)), dtype=object)
        for j, model in enumerate(Ms):
            cells[0, j] = model
        for j, params in enumerate(All_Params):
            params_cells[0, j] = params
        output_path = pooled_folder / f"FitRLWM_{phase}_dataset{Dataset}.mat"
        if Demo:
            output_path = pooled_folder / f"FitRLWM_{phase}_demo_dataset{Dataset}.mat"
        savemat(output_path, {
            "Ms": cells,
            "All_Params": params_cells,
            "All_fits": All_fits,
            "subjects_list": np.asarray(all_subjects)[None, :],
            "block_trial_counts": block_counts,
            "split_rule": "first floor(n_blocks/2) early; middle block late",
        }, oned_as="row")
        print(f"Saved {phase} fit: {output_path}", flush=True)


if __name__ == "__main__":
    main()
