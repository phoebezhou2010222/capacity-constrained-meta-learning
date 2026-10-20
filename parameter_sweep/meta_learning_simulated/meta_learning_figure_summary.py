"""Plot exposure-averaged parameter-sweep summaries from existing block CSVs.

Place alongside meta_learning_simulated.py and run:
    python meta_learning_figure_summary.py

Only creates PNGs in output/<parameter>/figure_summary/; never changes CSVs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / 'output'
FITS = ROOT / 'pooled_data' / 'FitRLWM_datasetpooled_expe1_3_5.mat'

PARAMETER_POSITIONS = {'alphaRL': 0, 'stick': 1, 'rho': 2,
                       'forget': 3, 'epsilon': 4, 'bias': 5, 'K': 6}
METRICS = {
    'accuracy_iter1_9': ('Accuracy, iterations 1–9', (0, 1)),
    'iteration2_accuracy': ('Accuracy, iteration 2', (0, 1)),
    'positive_retention': ('Positive-feedback retention', (0, 1)),
    'negative_avoidance': ('Negative-feedback avoidance', (0, 1)),
    'correct_after_error': ('P(correct after first error)', (0, 1)),
    'error_avoidance_D': ('Unchosen errors − chosen errors', None),
    'mean_chosen_error': ('Mean previous chosen errors', None),
    'mean_unchosen_error': ('Mean previous unchosen errors', None),
}


def pooled_by_subject_and_set_size(csv_path, metric):
    """Within subject+set size: mean across eligible blocks/exposures.

    Across subjects: mean and between-participant SEM. A participant with more
    exposures does not get more weight. Measurements retain their original
    eligibility/RT filtering and repeat-averaging from block_metrics.csv.
    """
    frame = pd.read_csv(csv_path, usecols=['subject', 'set_size', metric])
    frame[metric] = pd.to_numeric(frame[metric], errors='coerce')
    frame = frame.dropna(subset=['subject', 'set_size', metric])
    subject_means = (frame.groupby(['subject', 'set_size'], as_index=False)
                     [metric].mean())
    result = (subject_means.groupby('set_size')[metric]
              .agg(['mean', 'std', 'count']).reset_index())
    result['sem'] = result['std'] / np.sqrt(result['count'])
    return result


def fitted_reference_positions():
    fit = loadmat(FITS, simplify_cells=True)
    fitted = np.asarray(fit['All_Params'][25], dtype=float)
    if fitted.ndim != 2 or fitted.shape[1] < len(PARAMETER_POSITIONS):
        raise ValueError(f'Unexpected fitted parameter array shape: {fitted.shape}')
    # Each row is an individual participant, so the reference is median across
    # participant-specific fitted values, NOT a globally fitted parameter.
    return {name: (float(np.nanmedian(fitted[:, position])),
                   float(np.nanpercentile(fitted[:, position], 25)),
                   float(np.nanpercentile(fitted[:, position], 75)))
            for name, position in PARAMETER_POSITIONS.items()}


def parameter_values(parameter_dir, parameter):
    folders = []
    for folder in (parameter_dir / 'csv').glob(f'{parameter}_*'):
        if not folder.is_dir():
            continue
        try:
            value = float(folder.name[len(parameter) + 1:])
        except ValueError:
            continue
        if (folder / 'block_metrics.csv').is_file():
            folders.append((value, folder / 'block_metrics.csv'))
    return sorted(folders)


def plot_metric(parameter, values, fitted_reference, metric, ylabel, ylim, destination):
    fitted_x, q1, q3 = fitted_reference
    evaluated = [(value, pooled_by_subject_and_set_size(csv, metric).set_index('set_size'))
                 for value, csv in values]
    sizes = sorted(set().union(*(set(data.index) for _, data in evaluated)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap('tab10')
    for i, ns in enumerate(sizes):
        xs, ys, sems = [], [], []
        for value, table in evaluated:
            if ns in table.index and np.isfinite(table.loc[ns, 'mean']):
                xs.append(value)
                ys.append(float(table.loc[ns, 'mean']))
                sems.append(float(table.loc[ns, 'sem']))
        color = cmap(i % 10)
        if xs:
            ax.plot(xs, ys, '-o', markersize=4, linewidth=1.6,
                    color=color, label=f'Set size {int(ns)}')
            sem_array = np.nan_to_num(np.asarray(sems), nan=0.0)
            ax.fill_between(xs, np.asarray(ys) - sem_array,
                            np.asarray(ys) + sem_array, alpha=0.13, color=color)
    ax.axvspan(q1, q3, color='gray', alpha=0.15, zorder=0,
               label=f'Fitted IQR (Q1={q1:.3g}, Q3={q3:.3g})')
    ax.axvline(fitted_x, linestyle='--', linewidth=1.3, color='black', alpha=0.75,
               label=f'Median fitted {parameter} = {fitted_x:.3g}')
    ax.set_xlabel(f'{parameter} (swept value)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs. {parameter}\nAveraged across exposures within each participant and set size')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    ax.legend(loc='best', fontsize=8)
    fig.text(0.5, 0.005,
             'Gray band: fitted-parameter IQR across participants; curve shading: between-participant SEM',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(destination / f'{metric}_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    reference = fitted_reference_positions()
    for parameter in PARAMETER_POSITIONS:
        parameter_dir = OUTPUT / parameter
        values = parameter_values(parameter_dir, parameter)
        if not values:
            print(f'Skipping {parameter}: no sweep block_metrics.csv found')
            continue
        destination = parameter_dir / 'figure_summary'
        destination.mkdir(parents=True, exist_ok=True)
        for metric, (ylabel, ylim) in METRICS.items():
            plot_metric(parameter, values, reference[parameter],
                        metric, ylabel, ylim, destination)
        print(f'{parameter}: saved {len(METRICS)} plots to {destination}; '
              f'median fitted {parameter} = {reference[parameter][0]:.4g}; '
              f'IQR = [{reference[parameter][1]:.4g}, {reference[parameter][2]:.4g}]')


if __name__ == '__main__':
    main()
