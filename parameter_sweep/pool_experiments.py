from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd

data_folder = Path("/Users/phoebezhou/AnneCollinsWMH/RLWM/DataSets")

expe1 = loadmat(data_folder / "Expe1.mat")["expe_data"]
expe3 = loadmat(data_folder / "Expe3.mat")["expe_data"]
expe5 = loadmat(data_folder / "Expe5.mat")["expe_data"]

for name, data in [("Expe1", expe1), ("Expe3", expe3), ("Expe5", expe5)]:
    print(name, "shape:", data.shape)
    print("Participants:", len(np.unique(data[:, 0])))

datasets = []
participant_records = []
next_subject = 1

for experiment, data in [("Expe1", expe1), ("Expe3", expe3), ("Expe5", expe5)]:
    data = data.copy()
    original_subjects = data[:, 0].copy()

    for original_subject in pd.unique(original_subjects):
        participant_rows = original_subjects == original_subject
        data[participant_rows, 0] = next_subject
        participant_records.append({
            "subject": next_subject,
            "experiment": experiment,
            "original_subject": int(original_subject)
        })
        next_subject += 1
    datasets.append(data)

pooled_data = np.concatenate(datasets, axis = 0)
participant_lookup = pd.DataFrame(participant_records)

output_folder = Path(__file__).resolve().parent / "pooled_data"
output_folder.mkdir(exist_ok=True)

columns = [
    "subject", "block", "set_size", "time", "stimulus", "image", "folder",
    "iteration", "correct_action", "choice", "key", "correct", "reward",
    "rt", "condition", "previous_correct", "delay"
]

savemat(str(output_folder / "pooled_expe1_3_5.mat"), {
    "expe_data": pooled_data,
    "columns": np.array(columns, dtype=object)
})

participant_lookup.to_csv(
    output_folder / "participant_lookup.csv",
    index=False
)

print("Pooled data shape:", pooled_data.shape)
print("Total participants:", len(participant_lookup))
print(participant_lookup.head())
