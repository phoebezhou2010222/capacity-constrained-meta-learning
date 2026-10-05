from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ============================================================
# 0. PATHS
# ============================================================

DATA_DIR = Path.home() / "AnneCollinsWMH" / "RLWM" / "DataSets"

DATA_PATHS = {
    "Expe1": DATA_DIR / "Expe1.mat",
    "Expe3": DATA_DIR / "Expe3.mat",
    "Expe5": DATA_DIR / "Expe5.mat"
}

PROJECT_DIR = Path.home() / "Desktop" / "capacity_constrained_meta_learning_1"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "pooled_expe1_3_5" / "meta_learning"
CSV_DIR = OUTPUT_DIR / "csv"
FIG_DIR = OUTPUT_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD + POOL EXPE1, EXPE3, EXPE5
# ============================================================

column_names = [
    "subject", "block", "set_size", "time", "stimulus", "image", "folder",
    "iteration", "correct_action", "choice", "key", "correct", "reward",
    "rt", "condition", "previous_correct", "delay"
]

int_cols = [
    "subject", "block", "set_size", "time", "stimulus",
    "iteration", "correct_action", "choice", "correct"
]

datasets = []

for experiment, path in DATA_PATHS.items():

    mat = loadmat(path)
    X = mat["expe_data"]

    temp = pd.DataFrame(X, columns=column_names)

    for col in int_cols:
        temp[col] = pd.to_numeric(
            temp[col],
            errors="coerce"
        ).astype("Int64")

    # Keep original participant number for traceability.
    temp["original_subject"] = temp["subject"]

    # Keep experiment identity, but do NOT use it as an analysis group.
    temp["experiment"] = experiment

    datasets.append(temp)

df = pd.concat(
    datasets,
    ignore_index=True
)

# ============================================================
# UNIQUE PARTICIPANT IDS ACROSS EXPERIMENTS
# ============================================================

participant_key = (
    df["experiment"].astype(str)
    + "_"
    + df["original_subject"].astype(str)
)

df["subject"] = pd.factorize(participant_key)[0] + 1

# Everyone is treated as belonging to one pooled analysis group.
df["group"] = "All"
GROUP_ORDER = ["All"]

print("\n======================================")
print("POOLED DATASET: EXPE1 + EXPE3 + EXPE5")
print("======================================")

print("\nRows by experiment:")
print(df["experiment"].value_counts().sort_index())

print("\nParticipants by experiment:")
print(
    df[
        ["experiment", "subject"]
    ]
    .drop_duplicates()
    .groupby("experiment")
    .size()
)

print("\nTotal pooled participants:")
print(df["subject"].nunique())

# Expected:
# Expe1 = 79
# Expe3 = 40
# Expe5 = 26
# Total = 145

# Save participant lookup so pooled IDs can always be traced back.
participant_lookup = (
    df[
        ["subject", "experiment", "original_subject"]
    ]
    .drop_duplicates()
    .sort_values("subject")
)

participant_lookup.to_csv(
    CSV_DIR / "participant_lookup.csv",
    index=False
)

# ============================================================
# 2. BLOCK METADATA + SET-SIZE EXPOSURE
# ============================================================

block_meta = (
    df.dropna(subset=["subject", "block", "set_size"])
    .groupby(["subject", "block"], as_index=False)
    .agg(
        experiment=("experiment", "first"),
        original_subject=("original_subject", "first"),
        set_size=("set_size", "first"),
        group=("group", "first")
    )
    .sort_values(["subject", "block"])
)

block_meta["block_order"] = (
    block_meta.groupby("subject").cumcount() + 1
)

block_meta["set_size_exposure"] = (
    block_meta
    .groupby(["subject", "set_size"])
    .cumcount() + 1
)

block_meta.to_csv(
    CSV_DIR / "block_metadata.csv",
    index=False
)

merge_back_cols = [
    "subject",
    "block",
    "block_order",
    "set_size_exposure",
    "group"
]

df = df.merge(
    block_meta[merge_back_cols],
    on=["subject", "block"],
    how="left",
    suffixes=("", "_block")
)

if "group_block" in df.columns:
    df = df.drop(columns=["group_block"])

# ============================================================
# 3. COLLINS RT FILTER
# ============================================================

# Collins-style accuracy analyses exclude RT <= 150 ms.
# Error-history analysis later uses the original unfiltered df.
accuracy_df = df[
    df["rt"] > 0.15
].copy()

# ============================================================
# 4. LEARNING CURVES — EXACT ITERATIONS 1-9
# ============================================================

# Fixed 1-9 window across all three datasets.
#
# Accuracy at iteration i = accuracy on exactly that presentation,
# not cumulative accuracy through iteration i.

learning_trials = accuracy_df[
    accuracy_df["iteration"].between(1, 9)
].copy()

learning_curve_block = (
    learning_trials
    .groupby(
        [
            "subject",
            "group",
            "block",
            "set_size",
            "set_size_exposure",
            "iteration"
        ],
        as_index=False
    )
    .agg(
        accuracy=("correct", "mean"),
        n_stimuli_contributing=("stimulus", "nunique")
    )
)

learning_curve_block.to_csv(
    CSV_DIR / "learning_curve_block_level.csv",
    index=False
)

learning_curve_summary = (
    learning_curve_block
    .groupby(
        [
            "group",
            "set_size",
            "set_size_exposure",
            "iteration"
        ],
        as_index=False
    )
    .agg(
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
        n_participants=("subject", "nunique")
    )
)

learning_curve_summary["sem_accuracy"] = (
    learning_curve_summary["sd_accuracy"]
    / np.sqrt(
        learning_curve_summary["n_participants"]
    )
)

learning_curve_summary.to_csv(
    CSV_DIR / "learning_curve_group_summary.csv",
    index=False
)

# One learning-curve figure per set size.
for group in GROUP_ORDER:

    group_sub = learning_curve_summary[
        learning_curve_summary["group"] == group
    ]

    for ns in sorted(
        group_sub["set_size"].dropna().unique()
    ):

        sub = group_sub[
            group_sub["set_size"] == ns
        ]

        if sub.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        for exposure in sorted(
            sub["set_size_exposure"].unique()
        ):

            temp = sub[
                sub["set_size_exposure"] == exposure
            ].sort_values("iteration")

            ax.errorbar(
                temp["iteration"],
                temp["mean_accuracy"],
                yerr=temp["sem_accuracy"],
                marker="o",
                capsize=3,
                label=f"Exposure {int(exposure)}"
            )

        ax.axhline(
            1 / 3,
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )

        ax.set_xlabel(
            "Stimulus iteration"
        )

        ax.set_ylabel(
            "P(correct)"
        )

        ax.set_title(
            f"Pooled Expe1 + Expe3 + Expe5: ns={int(ns)}"
        )

        ax.set_xticks(
            range(1, 10)
        )

        ax.set_ylim(
            0.2,
            1.02
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIG_DIR /
            f"learning_curve_ns{int(ns)}_by_exposure.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

# ============================================================
# 5. OVERALL ACCURACY — FIXED ITERATIONS 1-9
# ============================================================

overall_accuracy = (
    learning_trials
    .groupby(
        [
            "subject",
            "group",
            "block",
            "set_size",
            "set_size_exposure"
        ],
        as_index=False
    )
    .agg(
        accuracy_iter1_9=("correct", "mean"),
        n_trials_iter1_9=("correct", "size")
    )
)

# ============================================================
# 6. ITERATION-2 ACCURACY
# ============================================================

iteration2_accuracy = (
    accuracy_df[
        accuracy_df["iteration"] == 2
    ]
    .groupby(
        [
            "subject",
            "group",
            "block",
            "set_size",
            "set_size_exposure"
        ],
        as_index=False
    )
    .agg(
        iteration2_accuracy=("correct", "mean"),
        n_iteration2=("correct", "size")
    )
)

# ============================================================
# 7. MATCH ITERATION 1 -> ITERATION 2 FOR EACH STIMULUS
# ============================================================

keys = [
    "subject",
    "group",
    "block",
    "set_size",
    "set_size_exposure",
    "stimulus"
]

first = (
    accuracy_df[
        accuracy_df["iteration"] == 1
    ][
        keys
        + [
            "choice",
            "correct",
            "correct_action"
        ]
    ]
    .rename(
        columns={
            "choice": "choice1",
            "correct": "correct1",
            "correct_action": "correct_action1"
        }
    )
)

second = (
    accuracy_df[
        accuracy_df["iteration"] == 2
    ][
        keys
        + [
            "choice",
            "correct",
            "correct_action"
        ]
    ]
    .rename(
        columns={
            "choice": "choice2",
            "correct": "correct2",
            "correct_action": "correct_action2"
        }
    )
)

iteration2_stimulus = first.merge(
    second,
    on=keys,
    how="inner"
)

# ============================================================
# 8. FIRST-FEEDBACK MEASURES
# ============================================================

iteration2_stimulus[
    "positive_retention"
] = np.where(
    iteration2_stimulus["correct1"] == 1,
    (
        iteration2_stimulus["choice2"]
        ==
        iteration2_stimulus["correct_action1"]
    ).astype(float),
    np.nan
)

iteration2_stimulus[
    "negative_avoidance"
] = np.where(
    iteration2_stimulus["correct1"] == 0,
    (
        iteration2_stimulus["choice2"]
        !=
        iteration2_stimulus["choice1"]
    ).astype(float),
    np.nan
)

iteration2_stimulus[
    "correct_after_error"
] = np.where(
    iteration2_stimulus["correct1"] == 0,
    iteration2_stimulus["correct2"].astype(float),
    np.nan
)

iteration2_stimulus.to_csv(
    CSV_DIR / "iteration2_stimulus_level.csv",
    index=False
)

def count_non_nan(series):
    return series.notna().sum()

feedback_block = (
    iteration2_stimulus
    .groupby(
        [
            "subject",
            "group",
            "block",
            "set_size",
            "set_size_exposure"
        ],
        as_index=False
    )
    .agg(
        positive_retention=(
            "positive_retention",
            "mean"
        ),
        n_positive=(
            "positive_retention",
            count_non_nan
        ),
        negative_avoidance=(
            "negative_avoidance",
            "mean"
        ),
        n_negative=(
            "negative_avoidance",
            count_non_nan
        ),
        correct_after_error=(
            "correct_after_error",
            "mean"
        ),
        n_correct_after_error=(
            "correct_after_error",
            count_non_nan
        )
    )
)

# ============================================================
# 9. CHOSEN / UNCHOSEN ERROR HISTORY
# ============================================================

# No RT filter here, matching Collins-style error-history logic.

error_rows = []

for (
    subject,
    block,
    stimulus
), stim_df in df.groupby(
    ["subject", "block", "stimulus"]
):

    stim_df = stim_df.sort_values(
        "iteration"
    )

    previous_choices = []

    for _, row in stim_df.iterrows():

        if (
            pd.isna(row["choice"])
            or pd.isna(row["correct_action"])
            or pd.isna(row["correct"])
        ):
            continue

        choice = int(
            row["choice"]
        )

        correct_action = int(
            row["correct_action"]
        )

        correct = int(
            row["correct"]
        )

        iteration = int(
            row["iteration"]
        )

        if (
            correct == 0
            and iteration > 1
        ):

            wrong_actions = [
                a
                for a in [1, 2, 3]
                if a != correct_action
            ]

            if choice in wrong_actions:

                other_wrong = [
                    a
                    for a in wrong_actions
                    if a != choice
                ][0]

                n_chosen = sum(
                    c == choice
                    for c in previous_choices
                )

                n_unchosen = sum(
                    c == other_wrong
                    for c in previous_choices
                )

                error_rows.append(
                    {
                        "subject": int(subject),
                        "group": row["group"],
                        "experiment": row["experiment"],
                        "block": int(block),
                        "set_size": int(
                            row["set_size"]
                        ),
                        "set_size_exposure": int(
                            row[
                                "set_size_exposure"
                            ]
                        ),
                        "stimulus": int(
                            stimulus
                        ),
                        "iteration": iteration,
                        "chosen_error_count": n_chosen,
                        "unchosen_error_count": n_unchosen,
                        "error_avoidance_D":
                            n_unchosen - n_chosen
                    }
                )

        previous_choices.append(
            choice
        )

error_trials = pd.DataFrame(
    error_rows
)

error_trials.to_csv(
    CSV_DIR / "error_trials_with_history.csv",
    index=False
)

error_block = (
    error_trials
    .groupby(
        [
            "subject",
            "group",
            "block",
            "set_size",
            "set_size_exposure"
        ],
        as_index=False
    )
    .agg(
        mean_chosen_error=(
            "chosen_error_count",
            "mean"
        ),
        mean_unchosen_error=(
            "unchosen_error_count",
            "mean"
        ),
        error_avoidance_D=(
            "error_avoidance_D",
            "mean"
        ),
        n_error_trials=(
            "error_avoidance_D",
            "size"
        )
    )
)

error_block.to_csv(
    CSV_DIR / "error_block_metrics.csv",
    index=False
)

# ============================================================
# 10. MASTER BLOCK-LEVEL DATASET
# ============================================================

merge_keys = [
    "subject",
    "group",
    "block",
    "set_size",
    "set_size_exposure"
]

block_metrics = (
    block_meta
    .merge(
        overall_accuracy,
        on=merge_keys,
        how="left"
    )
    .merge(
        iteration2_accuracy,
        on=merge_keys,
        how="left"
    )
    .merge(
        feedback_block,
        on=merge_keys,
        how="left"
    )
    .merge(
        error_block,
        on=merge_keys,
        how="left"
    )
)

block_metrics.to_csv(
    CSV_DIR / "block_metrics.csv",
    index=False
)

# ============================================================
# 11. GENERIC EXPOSURE SUMMARIES + PLOTS
# ============================================================

def summarize_metric(
    data,
    metric
):

    temp = data.dropna(
        subset=[metric]
    )

    summary = (
        temp
        .groupby(
            [
                "group",
                "set_size",
                "set_size_exposure"
            ],
            as_index=False
        )
        .agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            n_participants=(
                "subject",
                "nunique"
            )
        )
    )

    summary["sem"] = (
        summary["sd"]
        /
        np.sqrt(
            summary["n_participants"]
        )
    )

    return summary


def plot_metric(
    metric,
    ylabel,
    filename,
    reference_line=None
):

    summary = summarize_metric(
        block_metrics,
        metric
    )

    summary.to_csv(
        CSV_DIR /
        f"{filename}_summary.csv",
        index=False
    )

    for group in GROUP_ORDER:

        sub = summary[
            summary["group"] == group
        ]

        if sub.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 5.5)
        )

        for ns in sorted(
            sub["set_size"].unique()
        ):

            temp = sub[
                sub["set_size"] == ns
            ].sort_values(
                "set_size_exposure"
            )

            ax.errorbar(
                temp["set_size_exposure"],
                temp["mean"],
                yerr=temp["sem"],
                marker="o",
                capsize=3,
                label=f"ns={int(ns)}"
            )

        if reference_line is not None:

            ax.axhline(
                reference_line,
                linestyle="--",
                linewidth=1,
                alpha=0.5
            )

        ax.set_xlabel(
            "Exposure number to same set size"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"Pooled Expe1 + Expe3 + Expe5: {ylabel}"
        )

        ax.legend(
            title="Set size"
        )

        fig.tight_layout()

        fig.savefig(
            FIG_DIR /
            f"{filename}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


plot_metric(
    "accuracy_iter1_9",
    "P(correct), iterations 1-9",
    "overall_accuracy_by_exposure",
    1 / 3
)

plot_metric(
    "iteration2_accuracy",
    "P(correct) at iteration 2",
    "iteration2_accuracy_by_exposure",
    1 / 3
)

plot_metric(
    "positive_retention",
    "Positive-feedback retention",
    "positive_retention_by_exposure",
    1 / 3
)

plot_metric(
    "negative_avoidance",
    "Negative-feedback avoidance",
    "negative_avoidance_by_exposure",
    2 / 3
)

plot_metric(
    "correct_after_error",
    "P(correct at iteration 2 | iteration 1 incorrect)",
    "correct_after_error_by_exposure",
    1 / 3
)

plot_metric(
    "error_avoidance_D",
    "Unchosen errors - chosen errors",
    "error_avoidance_D_by_exposure",
    0
)

# ============================================================
# 12. CHOSEN VS UNCHOSEN ERROR-HISTORY PLOTS
# ============================================================

for group in GROUP_ORDER:

    group_errors = error_block[
        error_block["group"] == group
    ]

    for ns in sorted(
        group_errors["set_size"].unique()
    ):

        sub = group_errors[
            group_errors["set_size"] == ns
        ]

        if sub.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        for metric, label in [
            (
                "mean_chosen_error",
                "Chosen error"
            ),
            (
                "mean_unchosen_error",
                "Unchosen error"
            )
        ]:

            summary = (
                sub
                .groupby(
                    "set_size_exposure",
                    as_index=False
                )
                .agg(
                    mean=(metric, "mean"),
                    sd=(metric, "std"),
                    n=(
                        "subject",
                        "nunique"
                    )
                )
            )

            summary["sem"] = (
                summary["sd"]
                /
                np.sqrt(
                    summary["n"]
                )
            )

            ax.errorbar(
                summary["set_size_exposure"],
                summary["mean"],
                yerr=summary["sem"],
                marker="o",
                capsize=3,
                label=label
            )

        ax.set_xlabel(
            "Exposure number to same set size"
        )

        ax.set_ylabel(
            "Mean # previous errors"
        )

        ax.set_title(
            f"Pooled Expe1 + Expe3 + Expe5: ns={int(ns)}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIG_DIR /
            f"chosen_unchosen_ns{int(ns)}_by_exposure.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

# ============================================================
# 13. PRINT MAIN NUMBERS
# ============================================================

print("\n======================================")
print("POOLED EXPE1 + EXPE3 + EXPE5 ANALYSIS COMPLETE")
print("======================================")

print("\nParticipants by experiment:")
print(
    block_meta[
        ["experiment", "subject"]
    ]
    .drop_duplicates()
    .groupby("experiment")
    .size()
)

print("\nTotal participants:")
print(
    block_meta["subject"].nunique()
)

print("\nBlocks by experiment:")
print(
    block_meta.groupby(
        "experiment"
    ).size()
)

print("\nExposure counts by set size:")
print(
    block_meta
    .groupby("set_size")[
        "set_size_exposure"
    ]
    .max()
)

print("\nParticipants contributing to each set size x exposure:")
print(
    block_meta
    .groupby(
        [
            "set_size",
            "set_size_exposure"
        ]
    )["subject"]
    .nunique()
    .to_string()
)

print("\nMean iterations 1-9 accuracy:")
print(
    block_metrics
    .groupby(
        [
            "group",
            "set_size",
            "set_size_exposure"
        ]
    )["accuracy_iter1_9"]
    .mean()
    .to_string()
)

print("\nSaved outputs to:")
print(
    OUTPUT_DIR
)

print("\nMain block-level CSV:")
print(
    CSV_DIR / "block_metrics.csv"
)