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

X = mat["expe_data"]

print("\nFirst 10 rows of raw expe_data:")
print(X[:10])

column_names = [
    "subject", "block", "set_size", "time", "stimulus", "image", "folder",
    "iteration", "correct_action", "choice", "key", "correct", "reward",
    "rt", "condition", "previous_correct", "delay"
]

df = pd.DataFrame(X, columns=column_names)

int_cols = [
    "subject", "block", "set_size", "time", "stimulus",
    "iteration", "correct_action", "choice", "correct"
]

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

print("\nNumber of participants:")
print(df["subject"].nunique())

print("\nCondition values:")
print(df["condition"].value_counts(dropna=False).sort_index())

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

block_summary["stimuli_match_setsize"] = (
    block_summary["n_stimuli"] == block_summary["set_size"]
)

block_summary = block_summary.sort_values(["subject", "block"])

block_summary["block_order"] = (
    block_summary.groupby("subject").cumcount() + 1
)

block_summary["set_size_exposure"] = (
    block_summary.groupby(["subject", "set_size"]).cumcount() + 1
)

block_summary["n_blocks_total"] = (
    block_summary.groupby("subject")["block_order"].transform("max")
)

block_summary["normalized_block_position"] = (
    (block_summary["block_order"] - 1) /
    (block_summary["n_blocks_total"] - 1)
)

# ============================================================
# BLOCK / SET-SIZE STRUCTURE
# ============================================================

print("\n======================================")
print("BLOCK / SET-SIZE STRUCTURE")
print("======================================")

print("\nBlocks per participant:")
print(block_summary.groupby("subject").size().value_counts().sort_index())

setsize_counts = (
    block_summary
    .groupby(["subject", "set_size"])
    .size()
    .unstack(fill_value=0)
)

print("\nSet-size counts per participant:")
print(setsize_counts.head(10))

print("\nUnique set-size-count patterns:")
print(setsize_counts.value_counts())

print("\nTotal number of blocks of each set size:")
print(block_summary["set_size"].value_counts().sort_index())

print("\nCheck n_stimuli == set_size:")
print(block_summary["stimuli_match_setsize"].value_counts())

print("\nObserved trial counts by set size:")
for ns in sorted(block_summary["set_size"].dropna().unique()):
    vals = sorted(
        block_summary.loc[
            block_summary["set_size"] == ns,
            "n_trials"
        ].unique()
    )
    print(f"ns={int(ns)}: {vals}")

# ============================================================
# CHRONOLOGICAL SET-SIZE ORDER
# ============================================================

print("\n======================================")
print("CHRONOLOGICAL SET-SIZE ORDER")
print("======================================")

for subject in block_summary["subject"].drop_duplicates().head(10):
    temp = block_summary[
        block_summary["subject"] == subject
    ].sort_values("block_order")

    print(
        f"Subject {subject}: "
        f"{temp['set_size'].astype(int).tolist()}"
    )

# ============================================================
# CHRONOLOGY OF SET-SIZE EXPOSURES
# ============================================================

exposure_positions = (
    block_summary
    .groupby(["set_size", "set_size_exposure"], as_index=False)
    .agg(
        mean_block_order=("block_order", "mean"),
        median_block_order=("block_order", "median"),
        min_block_order=("block_order", "min"),
        max_block_order=("block_order", "max"),
        mean_normalized_position=("normalized_block_position", "mean"),
        sd_normalized_position=("normalized_block_position", "std"),
        n_participants=("subject", "nunique")
    )
)

print("\n======================================")
print("CHRONOLOGY OF SET-SIZE EXPOSURES")
print("======================================")

print(exposure_positions.to_string(index=False))

chronology_mean_matrix = exposure_positions.pivot(
    index="set_size",
    columns="set_size_exposure",
    values="mean_normalized_position"
)

chronology_sd_matrix = exposure_positions.pivot(
    index="set_size",
    columns="set_size_exposure",
    values="sd_normalized_position"
)

print("\nMean normalized position matrix:")
print(chronology_mean_matrix)

print("\nSD normalized position matrix:")
print(chronology_sd_matrix)

# ============================================================
# MAPPING COMPOSITION
# ============================================================

stimulus_mapping = (
    df.dropna(
        subset=["subject", "block", "stimulus", "correct_action"]
    )
    .groupby(["subject", "block", "stimulus"], as_index=False)
    .agg(correct_action=("correct_action", "first"))
)

action_counts = (
    stimulus_mapping
    .groupby(["subject", "block", "correct_action"])
    .size()
    .unstack(fill_value=0)
)

for action in [1, 2, 3]:
    if action not in action_counts.columns:
        action_counts[action] = 0

action_counts = action_counts[[1, 2, 3]]

action_counts["mapping_composition"] = action_counts.apply(
    lambda row: "-".join(
        map(str, sorted(row.astype(int).tolist(), reverse=True))
    ),
    axis=1
)

mapping_blocks = (
    block_summary
    .merge(
        action_counts["mapping_composition"].reset_index(),
        on=["subject", "block"],
        how="left"
    )
)

mapping_counts = (
    mapping_blocks
    .groupby(
        ["set_size", "set_size_exposure", "mapping_composition"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "n_blocks"})
)

mapping_counts["proportion"] = (
    mapping_counts["n_blocks"] /
    mapping_counts.groupby(
        ["set_size", "set_size_exposure"]
    )["n_blocks"].transform("sum")
)

print("\n======================================")
print("MAPPING COMPOSITION BY EXPOSURE")
print("======================================")

print(mapping_counts.to_string(index=False))

# ============================================================
# SAVE CSVs
# ============================================================

block_summary.to_csv(
    OUTPUT_DIR / "expe3_block_summary.csv",
    index=False
)

exposure_positions.to_csv(
    OUTPUT_DIR / "expe3_exposure_positions.csv",
    index=False
)

chronology_mean_matrix.to_csv(
    OUTPUT_DIR / "expe3_chronology_mean_matrix.csv"
)

chronology_sd_matrix.to_csv(
    OUTPUT_DIR / "expe3_chronology_sd_matrix.csv"
)

mapping_counts.to_csv(
    OUTPUT_DIR / "expe3_mapping_composition_long.csv",
    index=False
)

print(f"\nSaved inspection outputs to: {OUTPUT_DIR}")