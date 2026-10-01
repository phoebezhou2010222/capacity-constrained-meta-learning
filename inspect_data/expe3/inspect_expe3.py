from pathlib import Path
import pandas as pd
from scipy.io import loadmat

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path.home() / "AnneCollinsWMH" / "RLWM" / "DataSets" / "Expe3.mat"
PROJECT_DIR = Path.home() / "Desktop" / "capacity_constrained_meta_learning_1"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "expe3" / "inspection"
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
# subno, block, ns, time, stimseq, imageseq, folderseq, interseq,
# corAseq, choice, key, cor, rew, rt, condition (HC=0, SZ=1), pcor,
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
print("EXPE3 BASIC STRUCTURE")
print("======================================")

print("\nShape:")
print(df.shape)
# (30403, 17)

print("\nNumber of participants:")
print(df["subject"].nunique())
# 40

print("\nCondition values:")
print(df["condition"].value_counts(dropna=False).sort_index())
# all 40 participants are condition 1

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
# all 40 participants have 22 blocks

setsize_counts = block_summary.groupby(["subject", "set_size"]).size().unstack(fill_value=0)

print("\nSet-size counts per participant:")
print(setsize_counts.head(10))

print("\nUnique set-size-count patterns:")
print(setsize_counts.value_counts())
# for all participants:
# set size 1: 3
# set size 2: 6
# set size 3: 4
# set size 4: 3
# set size 5: 3
# set size 6: 3

print("\nCheck n_stimuli == set_size:")
print(block_summary["stimuli_match_setsize"].value_counts())

print("\nObserved trial counts by set size:")
for ns in sorted(block_summary["set_size"].dropna().unique()):
    vals = sorted(block_summary.loc[block_summary["set_size"] == ns, "n_trials"].unique())
    print(f"ns={int(ns)}: {vals}")
# blocks in the same set size, depending on the participant and
# the stopping rule, have different trial counts

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

# ============================================================
# SAVE CSVs
# ============================================================

block_summary.to_csv(OUTPUT_DIR / "expe3_block_summary.csv", index=False)
exposure_positions.to_csv(OUTPUT_DIR / "expe3_exposure_positions.csv", index=False)

print(f"\nSaved inspection outputs to: {OUTPUT_DIR}")