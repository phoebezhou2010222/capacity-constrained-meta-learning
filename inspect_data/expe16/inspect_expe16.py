from pathlib import Path
import pandas as pd
from scipy.io import loadmat

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path.home() / "AnneCollinsWMH" / "RLWM" / "DataSets" / "Expe16.mat"
PROJECT_DIR = Path.home() / "Desktop" / "capacity_constrained_meta_learning_1"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "expe16" / "inspection"
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
# corAseq, choice, key, cor, rew, rt, condition (WL), pcor, delay

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
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

subject_condition = df[["subject", "condition"]].drop_duplicates()

print("\nParticipants by condition:")
print(subject_condition["condition"].value_counts(dropna=False).sort_index())
# condition 0: 30 participant-condition entries
# condition 1: 30 participant-condition entries

print("\nDistinct condition values within participant:")
print(df.groupby("subject")["condition"].nunique(dropna=False).value_counts().sort_index())
# all 30 participants have 2 condition values

# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n======================================")
print("EXPE16 BASIC STRUCTURE")
print("======================================")

print("\nShape:")
print(df.shape)
# (27090, 17)

print("\nNumber of participants:")
print(df["subject"].nunique())
# 30

print("\nCondition values:")
print(df["condition"].value_counts(dropna=False).sort_index())
# condition 0: 12870 trials
# condition 1: 14220 trials

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

# ============================================================
# BLOCK / SET-SIZE STRUCTURE
# ============================================================

print("\n======================================")
print("BLOCK / SET-SIZE STRUCTURE")
print("======================================")

print("\nBlocks per participant:")
print(block_summary.groupby("subject").size().value_counts().sort_index())
# all 30 participants have 18 blocks

setsize_counts = block_summary.groupby(["subject", "set_size"]).size().unstack(fill_value=0)

print("\nSet-size counts per participant:")
print(setsize_counts.head(10))

print("\nUnique set-size-count patterns:")
print(setsize_counts.value_counts())
# for all participants:
# set size 2: 4
# set size 3: 4
# set size 4: 4
# set size 5: 3
# set size 6: 3

print("\nCheck n_stimuli == set_size:")
print(block_summary["stimuli_match_setsize"].value_counts())
# all 540 blocks have n_stimuli == set_size

print("\nObserved trial counts by set size:")
for ns in sorted(block_summary["set_size"].dropna().unique()):
    vals = sorted(block_summary.loc[block_summary["set_size"] == ns, "n_trials"].unique())
    print(f"ns={int(ns)}: {vals}")
# two block lengths occur within each set size:
# set size 2: 26 or 28 trials
# set size 3: 39 or 42 trials
# set size 4: 52 or 56 trials
# set size 5: 65 or 70 trials
# set size 6: 78 or 84 trials

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
# exposure 4 always occurs at block 18
#
# set size 3:
# exposures 1-2 occur early
# exposures 3-4 occur late
#
# set size 4:
# exposures are distributed across early, middle, and late blocks
#
# set size 5:
# exposures occur approximately in early, middle, and later blocks
#
# set size 6:
# exposure 1 can occur relatively late compared with other set sizes
# exposures 2-3 occur mostly in the middle-to-late part of the experiment

# ============================================================
# SAVE CSVs
# ============================================================

block_summary.to_csv(OUTPUT_DIR / "expe16_block_summary.csv", index=False)
exposure_positions.to_csv(OUTPUT_DIR / "expe16_exposure_positions.csv", index=False)

print(f"\nSaved inspection outputs to: {OUTPUT_DIR}")