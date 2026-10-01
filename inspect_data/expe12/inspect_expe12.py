from pathlib import Path
import pandas as pd
from scipy.io import loadmat

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path.home() / "AnneCollinsWMH" / "RLWM" / "DataSets" / "Expe12.mat"
PROJECT_DIR = Path.home() / "Desktop" / "capacity_constrained_meta_learning_1"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "expe12" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

mat = loadmat(DATA_PATH)

print("MATLAB keys:")
print(mat.keys())

print("\nStored column labels:")
print(mat["columns"])

print("\nNumber of column labels:")
print(len(mat["columns"][0]))
# Stored column labels:
# subno, block, ns, time, stimseq, imageseq, folderseq, iterseq,
# corAseq, choice, key, cor, rew, rt, condition (HC=0,SZ=1), pcor,
# delay

X = mat["expe_data"]

print("\nFirst 10 rows of raw expe_data:")
print(X[:10])

column_names = [
    "subject", "block", "set_size", "time", "stimulus", "image", "folder",
    "iteration", "correct_action", "choice", "key", "correct", "reward",
    "rt", "condition", "previous_correct", "delay"
]

df = pd.DataFrame(X, columns=column_names)

int_cols = ["subject", "block", "set_size", "time", "stimulus", "iteration", "correct_action", "choice", "correct"]
for col in int_cols:
    df[col] = df[col].astype("Int64")

group_labels = {
    1.0: "Kids",
    2.0: "Teens",
    3.0: "RPP",
    4.0: "Age>25"
}

subject_condition = df[["subject", "condition"]].drop_duplicates().sort_values(["subject", "condition"])

print("\nParticipants per condition:")
print(subject_condition["condition"].value_counts(dropna=False).sort_index())
# condition 1: 101 participants
# condition 2: 89 participants
# condition 3: 67 participants
# condition 4: 43 participants

condition_per_subject = df.groupby("subject")["condition"].nunique(dropna=False)

print("\nDistinct condition values within participant:")
print(condition_per_subject.value_counts().sort_index())
# all 300 participants have exactly one condition value

# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n======================================")
print("EXPE12 BASIC STRUCTURE")
print("======================================")

print("\nShape:")
print(df.shape)
# (140124, 17)

print("\nNumber of participants:")
print(df["subject"].nunique())
# 300

print("\nCondition values:")
print(df["condition"].value_counts(dropna=False).sort_index())
# condition 1: 47045 trials
# condition 2: 41625 trials
# condition 3: 31346 trials
# condition 4: 20108 trials

# ============================================================
# BLOCK METADATA
# ============================================================

block_summary = (
    df.dropna(subset=["subject", "block", "set_size"])
    .groupby(["subject", "block"], as_index=False)
    .agg(
        condition=("condition", "first"),
        set_size=("set_size", "first"),
        n_trials=("time", "size"),
        n_stimuli=("stimulus", "nunique"),
        min_iteration=("iteration", "min"),
        max_iteration=("iteration", "max"),
        accuracy=("correct", "mean")
    )
)

block_summary["stimuli_match_setsize"] = block_summary["n_stimuli"] == block_summary["set_size"]
block_summary = block_summary.sort_values(["subject", "block"])
block_summary["block_order"] = block_summary.groupby("subject").cumcount() + 1
block_summary["set_size_exposure"] = block_summary.groupby(["subject", "set_size"]).cumcount() + 1
block_summary["group"] = block_summary["condition"].map(group_labels)

# ============================================================
# BLOCK / SET-SIZE STRUCTURE
# ============================================================

print("\n======================================")
print("BLOCK / SET-SIZE STRUCTURE")
print("======================================")

print("\nBlocks per participant:")
print(block_summary.groupby("subject").size().value_counts().sort_index())
# all 300 participants have 10 blocks

setsize_counts = block_summary.groupby(["subject", "set_size"]).size().unstack(fill_value=0)

print("\nSet-size counts per participant:")
print(setsize_counts.head(10))

print("\nUnique set-size-count patterns:")
print(setsize_counts.value_counts())
# for all participants:
# set size 2: 2
# set size 3: 3
# set size 4: 2
# set size 5: 3

print("\nCheck n_stimuli == set_size:")
print(block_summary["stimuli_match_setsize"].value_counts())
# all 3000 blocks have n_stimuli == set_size

print("\nObserved trial counts by set size:")
for ns in sorted(block_summary["set_size"].dropna().unique()):
    vals = sorted(block_summary.loc[block_summary["set_size"] == ns, "n_trials"].unique())
    print(f"ns={int(ns)}: {vals}")
# block lengths vary slightly within each set size

# ============================================================
# CHRONOLOGICAL SET-SIZE ORDER
# ============================================================

print("\n======================================")
print("CHRONOLOGICAL SET-SIZE ORDER")
print("======================================")

for subject in block_summary["subject"].drop_duplicates().head(10):
    temp = block_summary[block_summary["subject"] == subject].sort_values("block_order")
    print(f"Subject {subject}: {temp['set_size'].astype(int).tolist()}")

# ============================================================
# WHERE EACH SET-SIZE EXPOSURE USUALLY OCCURS
# ============================================================

exposure_positions = block_summary.groupby(
    ["set_size", "set_size_exposure"], as_index=False
).agg(
    mean_block_order=("block_order", "mean"),
    median_block_order=("block_order", "median"),
    min_block_order=("block_order", "min"),
    max_block_order=("block_order", "max"),
    n_participants=("subject", "nunique")
)

print("\n======================================")
print("WHERE EACH SET-SIZE EXPOSURE USUALLY OCCURS")
print("======================================")

print("\nThe overall estimated chronological position of each set-size exposure:")
print(exposure_positions.to_string(index=False))
# set size 2:
# exposure 1 always occurs at block 1
# exposure 2 always occurs at block 7
#
# set size 3:
# exposure 1 occurs at blocks 2-3
# exposure 2 always occurs at block 5
# exposure 3 occurs at blocks 8-9
#
# set size 4:
# exposure 1 occurs at blocks 2-4
# exposure 2 occurs at blocks 8-10
#
# set size 5:
# exposure 1 occurs at blocks 3-4
# exposure 2 always occurs at block 6
# exposure 3 occurs at blocks 9-10

# ============================================================
# SAVE CSVs
# ============================================================

block_summary.to_csv(OUTPUT_DIR / "expe12_block_summary.csv", index=False)
exposure_positions.to_csv(OUTPUT_DIR / "expe12_exposure_positions.csv", index=False)

print(f"\nSaved inspection outputs to: {OUTPUT_DIR}")