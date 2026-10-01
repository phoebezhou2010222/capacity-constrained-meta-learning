from pathlib import Path
import pandas as pd
from scipy.io import loadmat

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path.home() / "AnneCollinsWMH" / "RLWM" / "DataSets" / "Expe5.mat"
PROJECT_DIR = Path.home() / "Desktop" / "capacity_constrained_meta_learning_1"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "expe5" / "inspection"
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

# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n======================================")
print("EXPE5 BASIC STRUCTURE")
print("======================================")

print("\nShape:")
print(df.shape)
# (19656, 17)

print("\nNumber of participants:")
print(df["subject"].nunique())
# 26

print("\nCondition values:")
print(df["condition"].value_counts(dropna=False).sort_index())
# condition is NaN for all trials

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

def get_third(block_order):
    if block_order <= 6:
        return "First"
    elif block_order <= 12:
        return "Middle"
    return "Final"

block_summary["experiment_third"] = block_summary["block_order"].apply(get_third)

# ============================================================
# BLOCK / SET-SIZE STRUCTURE
# ============================================================

print("\n======================================")
print("BLOCK / SET-SIZE STRUCTURE")
print("======================================")

print("\nBlocks per participant:")
print(block_summary.groupby("subject").size().value_counts().sort_index())
# all 26 participants have 18 blocks

setsize_counts = block_summary.groupby(["subject", "set_size"]).size().unstack(fill_value=0)

print("\nSet-size counts per participant:")
print(setsize_counts.head(10))

print("\nUnique set-size-count patterns:")
print(setsize_counts.value_counts())
# for all participants:
# set size 1: 3
# set size 2: 3
# set size 3: 3
# set size 4: 3
# set size 5: 3
# set size 6: 3

print("\nCheck n_stimuli == set_size:")
print(block_summary["stimuli_match_setsize"].value_counts())
# all 468 blocks have n_stimuli == set_size

print("\nObserved trial counts by set size:")
for ns in sorted(block_summary["set_size"].dropna().unique()):
    vals = sorted(block_summary.loc[block_summary["set_size"] == ns, "n_trials"].unique())
    print(f"ns={int(ns)}: {vals}")
# fixed block length within each set size:
# set size 1: 12 trials
# set size 2: 24 trials
# set size 3: 36 trials
# set size 4: 48 trials
# set size 5: 60 trials
# set size 6: 72 trials

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
# exposure 1 always occurs in blocks 1-6
# exposure 2 always occurs in blocks 7-12
# exposure 3 always occurs in blocks 13-18

# ============================================================
# EXPOSURE NUMBER / EXPERIMENT THIRD ALIGNMENT
# ============================================================

exposure_third = (
    block_summary.groupby(["set_size", "set_size_exposure", "experiment_third"], as_index=False)
    .size()
    .rename(columns={"size": "n_blocks"})
)

print("\n======================================")
print("EXPOSURE NUMBER / EXPERIMENT THIRD ALIGNMENT")
print("======================================")

print("\nExposure number x experiment third:")
print(exposure_third.to_string(index=False))
# for every set size:
# exposure 1 = First third
# exposure 2 = Middle third
# exposure 3 = Final third
# each combination contains all 26 participants

# ============================================================
# SAVE CSVs
# ============================================================

block_summary.to_csv(OUTPUT_DIR / "expe5_block_summary.csv", index=False)
exposure_positions.to_csv(OUTPUT_DIR / "expe5_exposure_positions.csv", index=False)

print(f"\nSaved inspection outputs to: {OUTPUT_DIR}")