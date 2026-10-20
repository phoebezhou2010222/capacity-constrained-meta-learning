"""Compare paired early/late Model 26 fits. Run from anywhere.

Outputs (in pooled_data/fitted_parameter_change_early_late/):
  01_early_vs_late.png   paired values + medians/IQR
  02_paired_changes.png  participant changes + median/IQR + mean/95% CI
  statistics.csv         paired change summaries and two-sided paired tests

Sign test: H0 = equally likely increases/decreases among nonzero changes.
Paired t-test: H0 = population mean late-minus-early change is zero.
Each test family receives its own Benjamini-Hochberg FDR adjustment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import binomtest, ttest_1samp, t
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / 'pooled_data'
OUT = DATA / 'fitted_parameter_change_early_late'
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ['alphaRL', 'stick', 'rho', 'forget', 'epsilon', 'biasWM/biasRL', 'K']
MODEL_INDEX = 25  # MATLAB Model 26; Python is zero-indexed


def read_fit(period):
    file = DATA / f'FitRLWM_{period}_datasetpooled_expe1_3_5.mat'
    mat = loadmat(file)
    values = np.asarray(mat['All_Params'][0, MODEL_INDEX], dtype=float)
    ids = np.asarray(mat['subjects_list']).ravel()
    if values.ndim != 2 or values.shape != (len(ids), 7):
        raise ValueError(f'Unexpected Model 26 parameter array: {values.shape}')
    if len(set(ids)) != len(ids):
        raise ValueError('Subject IDs are not unique; cannot safely pair.')
    return pd.DataFrame(values, index=ids, columns=NAMES)


early = read_fit('early')
late = read_fit('late')
if set(early.index) != set(late.index):
    raise ValueError('Early and late subject IDs do not match')
late = late.loc[early.index]
changes = late - early

# Paired statistical tests and descriptive statistics.
records = []
for name in NAMES:
    d = changes[name].to_numpy()
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        raise ValueError(f'Not enough valid paired observations for {name}')
    pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    sign_p = binomtest(pos, pos + neg, p=0.5).pvalue if pos + neg else 1.0
    mean, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    if sd == 0:
        # If all changes are exactly zero, the null is trivially compatible.
        # A nonzero constant vector is incompatible with the t-test null.
        t_p = 1.0 if mean == 0 else 0.0
        ci_low = ci_high = mean
    else:
        t_p = float(ttest_1samp(d, popmean=0).pvalue)  # paired t-test on differences
        margin = t.ppf(0.975, df=n - 1) * sd / np.sqrt(n)
        ci_low, ci_high = mean - margin, mean + margin
    records.append({
        'parameter': name,
        'mean_change': mean,
        'sd_change': sd,
        'median_change': float(np.median(d)),
        'sign_test_p': sign_p,
        'paired_t_p': t_p,
        'paired_t_95CI_lower': ci_low,
        'paired_t_95CI_upper': ci_high,
    })
statistics = pd.DataFrame(records)


def fdr_bh(p_values):
    """Benjamini-Hochberg adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(ranked, 1)
    return adjusted


statistics['sign_test_FDR_BH_q'] = fdr_bh(statistics['sign_test_p'])
statistics['paired_t_FDR_BH_q'] = fdr_bh(statistics['paired_t_p'])
statistics.to_csv(OUT / 'statistics.csv', index=False)

# Consistent, deterministic horizontal positions for participants.
rng = np.random.default_rng(2026)
fig, axes = plt.subplots(2, 4, figsize=(15, 7.3), constrained_layout=True)
for j, name in enumerate(NAMES):
    ax = axes.flat[j]
    a, b = early[name].to_numpy(), late[name].to_numpy()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    for av, bv in zip(a, b):
        ax.plot([0, 1], [av, bv], color='#9babbc', lw=0.55, alpha=0.23, zorder=1)
    for ix, vals, color in [(0, a, '#2873a8'), (1, b, '#cc7c3d')]:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([ix, ix], [q1, q3], lw=9, color=color, alpha=.36, solid_capstyle='butt', zorder=3)
        ax.plot([ix-.14, ix+.14], [med, med], lw=3, color=color, zorder=4)
    ax.set(xlim=(-.35, 1.35), xticks=[0,1], xticklabels=['Early','Late'], title=name)
    ax.grid(axis='y', alpha=.17)
axes.flat[-1].axis('off')
fig.suptitle('A. Early vs late fitted parameters (paired by participant)\nThick bars: IQR (Q1–Q3); short lines: medians', fontsize=13)
fig.savefig(OUT / '01_early_vs_late.png', dpi=210)
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(15, 7.3), constrained_layout=True)
for j, name in enumerate(NAMES):
    ax = axes.flat[j]
    d = changes[name].to_numpy()
    d = d[np.isfinite(d)]
    jitter = rng.uniform(-.19, .19, len(d))
    ax.scatter(jitter, d, s=12, alpha=.28, color='#5189ad', edgecolor='none', zorder=2)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    ax.plot([0, 0], [q1, q3], color='#2376aa', lw=15, alpha=.40, solid_capstyle='butt', zorder=3)
    ax.plot([-.14, .14], [med, med], color='#10496e', lw=3, zorder=4)
    # Mean and t-based 95% CI, shifted right from the median/IQR.
    mean = np.mean(d)
    sd = np.std(d, ddof=1)
    margin = t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
    ax.errorbar(.23, mean, yerr=margin, fmt='o', color='#cc7c3d',
                markersize=5.5, capsize=4, elinewidth=2, zorder=5)
    ax.axhline(0, color='#bd5b4a', ls='--', lw=1.1, zorder=1)
    ax.set(xlim=(-.35, .48), xticks=[0, .23], xticklabels=['Median', 'Mean'], title=name)
    ax.grid(axis='y', alpha=.17)
axes.flat[-1].axis('off')
fig.suptitle('B. Participant-level parameter changes (Late − Early)\nBlue: participant dots, median + IQR; orange: mean + 95% CI; dashed: zero', fontsize=13)
fig.savefig(OUT / '02_paired_changes.png', dpi=210)
plt.close(fig)
print(f'Saved 2 PNGs and statistics.csv in {OUT}')
print(statistics.to_string(index=False))
