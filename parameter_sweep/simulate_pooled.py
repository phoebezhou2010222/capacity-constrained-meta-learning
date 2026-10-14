from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from Collins_translated_scripts.Models.RLWM import RLWM

root = Path(__file__).resolve().parent
pooled_folder = root / "pooled_data"
output_folder = root / "simulated_pooled_data"

# Position of each parameter in the fitted parameter row.
parameter_positions = {
    "alphaRL": 0,
    "stick": 1, 
    "rho": 2, 
    "forget": 3, 
    "epsilon": 4, 
    "bias": 5, 
    "K": 6
}

sweep_values = {
    "alphaRL": np.round(np.linspace(0, 1, 11), 1),
    "stick": np.round(np.linspace(-1, 1, 21), 1),
    "rho": np.round(np.linspace(0, 1, 11), 1),
    "forget": np.round(np.linspace(0, 1, 11), 1),
    "epsilon": np.round(np.linspace(0, 1, 11), 1),
    "bias": np.round(np.linspace(0, 1, 11), 1),
    "K": [2, 3, 4, 5]
}

# Load pooled experimental data.
experiment = loadmat(pooled_folder / "pooled_expe1_3_5.mat")
pooled_data = experiment["expe_data"]

# Load model 26 and each participant's fitted parameters.
fit = loadmat(
    pooled_folder / "FitRLWM_datasetpooled_expe1_3_5.mat",
    simplify_cells=True
)

model = fit["Ms"][25]
fitted_parameters = fit["All_Params"][25]
subjects = np.asarray(fit["subjects_list"]).reshape(-1)

# Simulate baseline using all fitted parameters unchanged
baseline_folder = output_folder / "baseline"
baseline_folder.mkdir(parents=True, exist_ok=True)
np.random.seed(123)

for repeat in range(1, 21):
    simulated_data = pooled_data.copy()

    for participant_index, subject in enumerate(subjects):
        participant_rows = pooled_data[:, 0] == subject
        participant_data = pooled_data[participant_rows].copy()
        parameters = fitted_parameters[participant_index].copy()
        simulated_data[participant_rows] = RLWM(parameters, participant_data, model)
    result = {
        "expe_data": simulated_data, 
        "repeat": repeat
    }
    if "columns" in experiment:
        result["columns"] = experiment["columns"]

    savemat(
        str(baseline_folder / f"repeat_{repeat:02d}.mat"),
        result,
        do_compression = True
    )


for parameter_name, position in parameter_positions.items():
    for value in sweep_values[parameter_name]:
        np.random.seed(123)
        value_label = str(int(value)) if parameter_name == "K" else f"{value:.1f}"

        value_folder = (
            output_folder
            / parameter_name
            / f"{parameter_name}_{value_label}"
        )
        value_folder.mkdir(parents=True, exist_ok=True)

        for repeat in range(1, 21):
            simulated_data = pooled_data.copy()

            for participant_index, subject in enumerate(subjects):
                participant_rows = pooled_data[:, 0] == subject
                participant_data = pooled_data[participant_rows].copy()

                # Keep this participant's other fitted parameters.
                parameters = fitted_parameters[participant_index].copy()
                parameters[position] = value

                simulated_data[participant_rows] = RLWM(
                    parameters,
                    participant_data,
                    model
                )

            result = {
                "expe_data": simulated_data,
                "parameter_name": parameter_name,
                "parameter_value": value,
                "repeat": repeat
            }

            if "columns" in experiment:
                result["columns"] = experiment["columns"]

            savemat(
                str(value_folder / f"repeat_{repeat:02d}.mat"),
                result,
                do_compression=True
            )

            print(
                f"{parameter_name} = {value_label}: "
                f"saved repeat {repeat}/20",
                flush=True
            )
