from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

script_folder = Path(__file__).resolve().parent
root = script_folder.parent  # parameter_sweep
simulation_folder = root / "simulated_pooled_data"
output_folder = script_folder / "output"
N_REPEATS = 20
GRID_COLUMNS = 4

# Same parameter values and folder names as simulate_pooled.py.
sweep_values = {
    "alphaRL": np.round(np.linspace(0, 1, 11), 1),
    "stick": np.round(np.linspace(-1, 1, 21), 1),
    "rho": np.round(np.linspace(0, 1, 11), 1),
    "forget": np.round(np.linspace(0, 1, 11), 1),
    "epsilon": np.round(np.linspace(0, 1, 11), 1),
    "bias": np.round(np.linspace(0, 1, 11), 1),
    "K": [2, 3, 4, 5]
}

column_names = [
    "subject", "block", "set_size", "time", "stimulus", "image", "folder",
    "iteration", "correct_action", "choice", "key", "correct", "reward",
    "rt", "condition", "previous_correct", "delay"
]

# Same measures as your pooled analysis, plus chosen/unchosen error means.
metric_labels = {
    "accuracy_iter1_9": "P(correct), iterations 1-9",
    "iteration2_accuracy": "P(correct) at iteration 2",
    "positive_retention": "Positive-feedback retention",
    "negative_avoidance": "Negative-feedback avoidance",
    "correct_after_error": "P(correct after first error)",
    "error_avoidance_D": "Unchosen errors - chosen errors",
    "mean_chosen_error": "Mean previous chosen errors",
    "mean_unchosen_error": "Mean previous unchosen errors"
}


def calculate_measurements(X, participant_lookup):
    """Analyze one complete pooled experiment; no files or plots are created."""
    df = pd.DataFrame(X, columns=column_names)
    df = df.merge(participant_lookup, on="subject", how="left", validate="many_to_one")
    df["group"] = "All"
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
    # Error-history analysis uses this same filtered dataset.
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

    # RT-excluded responses never enter the error averages or history counts.

    error_rows = []

    for (
        subject,
        block,
        stimulus
    ), stim_df in accuracy_df.groupby(
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

    error_trials = pd.DataFrame(error_rows, columns=["subject","group","experiment","block","set_size","set_size_exposure","stimulus","iteration","chosen_error_count","unchosen_error_count","error_avoidance_D"])


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


    # ============================================================
    tables = {
        "block_metadata.csv": block_meta,
        "block_metrics.csv": block_metrics,
        "learning_curve_block_level.csv": learning_curve_block,
        "error_block_metrics.csv": error_block,
        "iteration2_stimulus_level.csv": iteration2_stimulus,
        "error_trials_with_history.csv": error_trials
    }
    return block_metrics, learning_curve_block, tables


def measurements_as_table(block_metrics, learning_curve):
    """Put all measurements in one table so averaging uses the same steps."""
    keys = ["subject", "group", "block", "set_size", "set_size_exposure"]
    count_columns = ["n_trials_iter1_9", "n_iteration2", "n_positive", "n_negative",
                     "n_correct_after_error", "n_error_trials"]
    # No error trials means count zero, but the error means remain undefined.
    block_metrics = block_metrics.copy()
    block_metrics["n_error_trials"] = block_metrics["n_error_trials"].fillna(0)
    block_table = block_metrics.melt(
        id_vars=keys, value_vars=list(metric_labels) + count_columns,
        var_name="metric", value_name="measurement"
    )
    block_table["iteration"] = np.nan  # These are block-level measures.

    learning_table = learning_curve.melt(
        id_vars=keys + ["iteration"], value_vars=["accuracy", "n_stimuli_contributing"],
        var_name="metric", value_name="measurement"
    )
    learning_table["metric"] = learning_table["metric"].replace({"accuracy": "learning_accuracy"})
    return pd.concat([block_table, learning_table], ignore_index=True)


def average_and_summarize(repeat_tables):
    """Average repeats within participant/block, then summarize participants."""
    all_repeats = pd.concat(repeat_tables, ignore_index=True)
    participant_keys = ["subject", "group", "block", "set_size", "set_size_exposure",
                        "iteration", "metric"]
    averaged = all_repeats.groupby(participant_keys, as_index=False, dropna=False).agg(
        measurement=("measurement", "mean"),
        n_repeats_with_measurement=("measurement", "count")
    )

    # Repeat simulations are not additional participants for SEM.
    valid = averaged.dropna(subset=["measurement"])
    group_keys = ["group", "set_size", "set_size_exposure", "iteration", "metric"]
    summary = valid.groupby(group_keys, as_index=False, dropna=False).agg(
        mean=("measurement", "mean"),
        sd=("measurement", "std"),
        n_participants=("subject", "nunique"),
        min_valid_repeats=("n_repeats_with_measurement", "min"),
        max_valid_repeats=("n_repeats_with_measurement", "max")
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    return summary


def average_block_tables(tables, keys):
    """Average each block's calculated measurements across repeats."""
    combined = pd.concat(tables, ignore_index=True)
    numeric_columns = [c for c in combined.select_dtypes(include="number").columns if c not in keys]
    other_columns = [c for c in combined.columns if c not in keys + numeric_columns]
    averaged = combined.groupby(keys, as_index=False, dropna=False).agg(
        {**{c: "mean" for c in numeric_columns}, **{c: "first" for c in other_columns}}
    )
    return averaged[tables[0].columns]


def analyze_repeats(folder, participant_lookup, expected_shape, csv_folder):
    csv_folder.mkdir(parents=True, exist_ok=True)
    repeat_tables = []
    block_tables, learning_tables, error_tables = [], [], []
    block_keys = ["subject", "group", "block", "set_size", "set_size_exposure"]

    for repeat in range(1, N_REPEATS + 1):
        X = loadmat(folder / f"repeat_{repeat:02d}.mat")["expe_data"]
        if X.shape != expected_shape:
            raise ValueError(f"Unexpected trial matrix shape in {folder}, repeat {repeat}")
        block_metrics, learning_curve, tables = calculate_measurements(X, participant_lookup)
        block_metrics = block_metrics.copy()
        block_metrics["n_error_trials"] = block_metrics["n_error_trials"].fillna(0)
        repeat_tables.append(measurements_as_table(block_metrics, learning_curve))
        block_tables.append(block_metrics)
        learning_tables.append(learning_curve)

        # Error means remain undefined in repeats with no eligible errors.
        # Error-trial counts in those repeats are zero.
        error_block = tables["block_metadata.csv"][block_keys].merge(
            tables["error_block_metrics.csv"], on=block_keys, how="left"
        )
        error_block["n_error_trials"] = error_block["n_error_trials"].fillna(0)
        error_tables.append(error_block)

        if repeat == 1:
            tables["block_metadata.csv"].to_csv(csv_folder / "block_metadata.csv", index=False)
            participant_lookup.to_csv(csv_folder / "participant_lookup.csv", index=False)

        # Raw trials differ across repeats, so retain a repeat column instead
        # of averaging action IDs or mixing their histories together.
        for name in ["iteration2_stimulus_level.csv", "error_trials_with_history.csv"]:
            detailed = tables[name].copy()
            detailed["repeat"] = repeat
            detailed.to_csv(csv_folder / name, index=False,
                            mode="w" if repeat == 1 else "a", header=(repeat == 1))
        print(f"{folder.name}: analyzed repeat {repeat}/{N_REPEATS}", flush=True)

    # Original block-level CSV layouts, averaged over the 20 repeats.
    average_block_tables(block_tables, block_keys).to_csv(csv_folder / "block_metrics.csv", index=False)
    average_block_tables(learning_tables, block_keys + ["iteration"]).to_csv(
        csv_folder / "learning_curve_block_level.csv", index=False
    )
    average_block_tables(error_tables, block_keys).to_csv(csv_folder / "error_block_metrics.csv", index=False)

    summary = average_and_summarize(repeat_tables)
    group_columns = ["group", "set_size", "set_size_exposure", "mean", "sd", "n_participants", "sem"]
    filenames = {
        "accuracy_iter1_9": "overall_accuracy_by_exposure_summary.csv",
        "iteration2_accuracy": "iteration2_accuracy_by_exposure_summary.csv",
        "positive_retention": "positive_retention_by_exposure_summary.csv",
        "negative_avoidance": "negative_avoidance_by_exposure_summary.csv",
        "correct_after_error": "correct_after_error_by_exposure_summary.csv",
        "error_avoidance_D": "error_avoidance_D_by_exposure_summary.csv"
    }
    for metric, filename in filenames.items():
        summary.loc[summary.metric == metric, group_columns].to_csv(csv_folder / filename, index=False)

    learning_summary = summary.loc[summary.metric == "learning_accuracy",
        ["group", "set_size", "set_size_exposure", "iteration", "mean", "sd", "n_participants", "sem"]]
    learning_summary = learning_summary.rename(columns={
        "mean": "mean_accuracy", "sd": "sd_accuracy", "sem": "sem_accuracy"
    })
    learning_summary.to_csv(csv_folder / "learning_curve_group_summary.csv", index=False)
    return summary


def new_grid(panel_count):
    columns = min(GRID_COLUMNS, panel_count)
    rows = int(np.ceil(panel_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[panel_count:]:
        ax.set_visible(False)
    return fig, axes


def finish_grid(fig, axes, title, path):
    # One shared legend, collected from whichever panels have data.
    legend_items = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend_items.update(zip(labels, handles))
    fig.suptitle(title)
    if legend_items:
        fig.legend(list(legend_items.values()), list(legend_items),
                   loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.90, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_curve(ax, table, x_column, label, color):
    table = table.sort_values(x_column)
    ax.errorbar(table[x_column], table["mean"], yerr=table["sem"],
                marker="o", markersize=3, capsize=2, linewidth=1,
                label=label, color=color)


def plot_grids(results, destination, title):
    """Called only after every value of one parameter has been analyzed."""
    destination.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(list(results.values()), ignore_index=True)
    set_sizes = sorted(combined["set_size"].unique())
    colors = plt.get_cmap("tab10")

    # 1. Learning curves: one grid per set size, exposure curves in each panel.
    for ns in set_sizes:
        fig, axes = new_grid(len(results))
        for ax, (label, summary) in zip(axes, results.items()):
            sub = summary[(summary.metric == "learning_accuracy") & (summary.set_size == ns)]
            for exposure in sorted(sub["set_size_exposure"].unique()):
                temp = sub[sub.set_size_exposure == exposure]
                draw_curve(ax, temp, "iteration", f"Exposure {int(exposure)}",
                           colors((int(exposure) - 1) % 10))
            ax.axhline(1/3, color="gray", linestyle="--", linewidth=0.7)
            ax.set(title=label, xlabel="Stimulus iteration", ylabel="P(correct)", ylim=(0, 1.02))
            ax.set_xticks(range(1, 10))
        finish_grid(fig, axes, f"{title}: learning curves, set size {int(ns)}",
                    destination / f"learning_curve_ns{int(ns)}_grid.png")

    # 2. The six original exposure-based measurement plots.
    references = {"accuracy_iter1_9": 1/3, "iteration2_accuracy": 1/3,
                  "positive_retention": 1/3, "negative_avoidance": 2/3,
                  "correct_after_error": 1/3, "error_avoidance_D": 0}
    for metric, reference in references.items():
        fig, axes = new_grid(len(results))
        for ax, (label, summary) in zip(axes, results.items()):
            sub = summary[summary.metric == metric]
            for ns in sorted(sub["set_size"].unique()):
                temp = sub[sub.set_size == ns]
                draw_curve(ax, temp, "set_size_exposure", f"ns={int(ns)}",
                           colors((int(ns) - 1) % 10))
            if sub.empty:
                ax.text(.5, .5, "No eligible measurements", transform=ax.transAxes, ha="center")
            ax.axhline(reference, color="gray", linestyle="--", linewidth=0.7)
            ax.set(title=label, xlabel="Exposure to same set size", ylabel=metric_labels[metric])
            if metric != "error_avoidance_D":
                ax.set_ylim(0, 1.02)
        finish_grid(fig, axes, f"{title}: {metric_labels[metric]}",
                    destination / f"{metric}_grid.png")

    # 3. Chosen and unchosen previous errors: one grid per set size.
    for ns in set_sizes:
        fig, axes = new_grid(len(results))
        for ax, (label, summary) in zip(axes, results.items()):
            sub = summary[summary.set_size == ns]
            for metric, curve_label, color in [
                ("mean_chosen_error", "Chosen error", colors(0)),
                ("mean_unchosen_error", "Unchosen error", colors(1))
            ]:
                temp = sub[sub.metric == metric]
                draw_curve(ax, temp, "set_size_exposure", curve_label, color)
            if sub[sub.metric.isin(["mean_chosen_error", "mean_unchosen_error"])].empty:
                ax.text(.5, .5, "No eligible errors", transform=ax.transAxes, ha="center")
            ax.set(title=label, xlabel="Exposure to same set size", ylabel="Mean previous errors")
        # Set the shared range after all panels, so later values are not clipped.
        all_errors = combined[(combined.set_size == ns) & combined.metric.isin(
            ["mean_chosen_error", "mean_unchosen_error"])]
        upper = (all_errors["mean"] + all_errors["sem"].fillna(0)).max()
        if not np.isfinite(upper) or upper <= 0:
            upper = 1
        axes[0].set_ylim(0, upper * 1.15)
        finish_grid(fig, axes, f"{title}: chosen/unchosen errors, set size {int(ns)}",
                    destination / f"chosen_unchosen_ns{int(ns)}_grid.png")


def main():
    pooled_folder = root / "pooled_data"
    experiment = loadmat(pooled_folder / "pooled_expe1_3_5.mat")["expe_data"]
    participant_lookup = pd.read_csv(pooled_folder / "participant_lookup.csv")
    participant_lookup = participant_lookup[["subject", "experiment", "original_subject"]]

    # First calculate the fitted baseline, averaging its 20 repeats.
    baseline_output = output_folder / "baseline"
    baseline = analyze_repeats(simulation_folder / "baseline", participant_lookup,
                               experiment.shape, baseline_output / "csv")
    plot_grids({"Fitted baseline": baseline}, baseline_output / "figures", "Fitted baseline")

    # Then calculate each parameter/value, using the same measurements.
    for parameter_name, values in sweep_values.items():
        parameter_output = output_folder / parameter_name
        csv_folder = parameter_output / "csv"
        csv_folder.mkdir(parents=True, exist_ok=True)
        results_for_all_values = {"Fitted baseline": baseline}

        for value in values:
            value_label = str(int(value)) if parameter_name == "K" else f"{value:.1f}"
            value_folder = simulation_folder / parameter_name / f"{parameter_name}_{value_label}"

            # analyze_repeats contains the loop over repeat_01.mat ... repeat_20.mat.
            value_csv_folder = csv_folder / f"{parameter_name}_{value_label}"
            summary = analyze_repeats(value_folder, participant_lookup,
                                      experiment.shape, value_csv_folder)
            results_for_all_values[f"{parameter_name} = {value_label}"] = summary

        # Plot only after the loop over all values has finished.
        plot_grids(results_for_all_values, parameter_output / "figures", parameter_name)
        print(f"Finished all values and grids for {parameter_name}", flush=True)


if __name__ == "__main__":
    main()
