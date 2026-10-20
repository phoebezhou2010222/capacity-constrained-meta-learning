"""Behavioral inference for pooled Expe1/3/5 (no uneligible trials).

Run from anywhere: python statistics.py
Input:  output/csv/block_metrics.csv (one row per subject/set size/exposure)
Output: output/statistics/ (five CSV files; no additional figures)

Paired comparisons: two-sided participant-matched t tests; all exposure
comparisons are within set size, with no duplicate comparisons counted twice.
Exposure regression: participant fixed-effects OLS with standard errors
clustered by participant, fitted separately per metric and set size.
Participant effects implicitly control experiment-specific baselines.
This describes linear exposure trends, not a binomial trial-level accuracy model.
BH FDR is applied separately to each metric within each comparison family;
D-vs-zero is one prespecified family. NaN values are omitted pairwise.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t as t_distribution
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'output' / 'csv' / 'block_metrics.csv'
OUT = BASE / 'output' / 'statistics'
OUT.mkdir(parents=True, exist_ok=True)
METRICS = {'accuracy': 'accuracy_iter1_9', 'D': 'error_avoidance_D'}


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (n, float(np.mean(x)) if n else np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    half = float(t_distribution.ppf(.975, n-1) * se)
    p = float(ttest_1samp(x, 0).pvalue) if sd > 0 else (1.0 if mean == 0 else 0.0)
    return n, mean, sd, mean-half, mean+half, p


def fdr(table, groups=('metric',)):
    if table.empty:
        table['FDR_BH_q'] = pd.Series(dtype=float)
        return table
    table['FDR_BH_q'] = np.nan
    for _, indices in table.groupby(list(groups), dropna=False).groups.items():
        pvals = table.loc[indices, 'p_two_sided']
        good = pvals.notna()
        if good.any():
            table.loc[pvals.index[good], 'FDR_BH_q'] = multipletests(pvals[good], method='fdr_bh')[1]
    return table


def paired_test(part, col, exp_a, exp_b, metric, ns):
    w = part.pivot(index='subject', columns='set_size_exposure', values=col)
    if exp_a not in w.columns or exp_b not in w.columns:
        return None
    w = w[[exp_a, exp_b]].dropna()
    n, avg, sd, lo, hi, p = mean_ci(w[exp_b].to_numpy() - w[exp_a].to_numpy())
    return dict(metric=metric, set_size=ns, exposure_from=int(exp_a), exposure_to=int(exp_b),
                n_pairs=n, mean_change=avg, sd_change=sd, CI95_lower=lo, CI95_upper=hi,
                p_two_sided=p)


def compare(data):
    first_last, vs_first, consecutive = [], [], []
    for ns, part in data.groupby('set_size'):
        exposures = sorted(part['set_size_exposure'].dropna().unique())
        if len(exposures) < 2:
            continue
        for metric, col in METRICS.items():
            row = paired_test(part, col, exposures[0], exposures[-1], metric, ns)
            if row: first_last.append(row)
            for end in exposures[1:]:
                row = paired_test(part, col, exposures[0], end, metric, ns)
                if row: vs_first.append(row)
            for start, end in zip(exposures[:-1], exposures[1:]):
                row = paired_test(part, col, start, end, metric, ns)
                if row: consecutive.append(row)
    for filename, rows in [('first_vs_last.csv', first_last),
                           ('vs_first.csv', vs_first), ('consecutive.csv', consecutive)]:
        result = fdr(pd.DataFrame(rows))
        result.to_csv(OUT / filename, index=False)
        print(f'{filename}: {len(result)} comparisons')


def vs_zero(data):
    rows = []
    for ns in (5, 6):
        subset = data[data['set_size'] == ns]
        exposures = sorted(subset['set_size_exposure'].dropna().unique())
        for exposure in exposures:
            x = subset.loc[subset['set_size_exposure'] == exposure, 'error_avoidance_D'].to_numpy()
            n, avg, sd, lo, hi, p = mean_ci(x)
            rows.append(dict(set_size=ns, exposure=int(exposure), n_participants=n,
                             mean_D=avg, sd_D=sd, CI95_lower=lo, CI95_upper=hi,
                             direction=('positive' if avg > 0 else 'negative' if avg < 0 else 'zero'),
                             p_two_sided=p))
    result = pd.DataFrame(rows)
    # One adjustment family across all prespecified ns5/ns6 D-versus-zero tests.
    result['FDR_BH_q'] = np.nan
    valid = result['p_two_sided'].notna()
    result.loc[valid, 'FDR_BH_q'] = multipletests(result.loc[valid, 'p_two_sided'], method='fdr_bh')[1]
    result.to_csv(OUT / 'D_vs_zero.csv', index=False)
    print(f'D_vs_zero.csv: {len(result)} tests')


def trajectory(data):
    rows = []
    for ns, subset in data.groupby('set_size'):
        for metric, col in METRICS.items():
            clean = subset[['subject', 'experiment', 'set_size_exposure', col]].dropna().copy()
            clean = clean.rename(columns={col: 'measurement', 'set_size_exposure': 'exposure'})
            if clean['subject'].nunique() < 3 or clean['exposure'].nunique() < 2:
                continue
            # Subject fixed effects prevent differences between participants (and
            # experiment baselines) from confounding the exposure coefficient.
            # Cluster-robust SE accounts for repeated observations per participant.
            fit = smf.ols('measurement ~ exposure + C(subject)', clean).fit(
                cov_type='cluster', cov_kwds={'groups': clean['subject']})
            method = 'subject_fixed_effects_cluster_robust'
            ci = fit.conf_int().loc['exposure']
            rows.append(dict(metric=metric, set_size=ns, n_participants=clean['subject'].nunique(),
                             n_observations=len(clean), min_exposure=int(clean['exposure'].min()),
                             max_exposure=int(clean['exposure'].max()), method=method,
                             slope_per_exposure=float(fit.params['exposure']),
                             CI95_lower=float(ci.iloc[0]), CI95_upper=float(ci.iloc[1]),
                             p_two_sided=float(fit.pvalues['exposure'])))
    result = fdr(pd.DataFrame(rows))
    result.to_csv(OUT / 'exposure_regression.csv', index=False)
    print(f'exposure_regression.csv: {len(result)} models')


def main():
    data = pd.read_csv(INPUT)
    needed = {'subject', 'experiment', 'set_size', 'set_size_exposure', *METRICS.values()}
    if not needed.issubset(data.columns):
        raise ValueError(f'Missing CSV columns: {sorted(needed - set(data.columns))}')
    if data.duplicated(['subject', 'set_size', 'set_size_exposure']).any():
        raise ValueError('Duplicate participant × set size × exposure rows; aggregate explicitly before testing')
    compare(data)
    vs_zero(data)
    trajectory(data)
    print(f'All results saved to {OUT}')


if __name__ == '__main__':
    main()
