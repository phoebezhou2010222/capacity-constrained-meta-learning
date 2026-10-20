"""RT by set-size exposure from pooled Expe1/3/5.

Put this file in parameter_sweep/ beside pooled_data/pooled_expe1_3_5.mat.
Outputs are written to pooled_data/reaction_time/:
  rt_by_exposure.png, rt_by_exposure.csv, block_rt.csv
  rt_by_log_set_size.png, rt_by_log_set_size.csv

Each point: mean across participants of participant-level mean block RTs.
95% CI: Student t interval across participant-level values, not across trials.
Eligibility: RT > 0.15 seconds, matching meta_learning_expe1_3_5.py.
Responses with invalid choices also have ineligible RTs in this pooled dataset.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import t
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'pooled_data'
OUT = DATA / 'reaction_time'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = DATA / 'pooled_expe1_3_5.mat'
MIN_RT = 0.15  # seconds: exclude RT <= 150 ms, matching behavioral analysis

mat = loadmat(INPUT)
columns = [str(x[0]) for x in mat['columns'].ravel()]
df = pd.DataFrame(mat['expe_data'], columns=columns)

# An exposure is the participant's first, second, ... block of that set size.
blocks = df[['subject', 'block', 'set_size']].drop_duplicates().sort_values(['subject', 'block'])
blocks['exposure'] = blocks.groupby(['subject', 'set_size']).cumcount() + 1
df = df.merge(blocks, on=['subject', 'block', 'set_size'], how='left', validate='many_to_one')

eligible = df.loc[
    df['rt'] > MIN_RT,
    ['subject', 'block', 'set_size', 'exposure', 'rt']
].copy()

# First average all eligible RTs inside each participant's block.
block_rt = (eligible.groupby(['subject', 'block', 'set_size', 'exposure'], as_index=False)
            .agg(mean_rt_s=('rt', 'mean'), n_eligible_trials=('rt', 'size')))
block_rt.to_csv(OUT / 'block_rt.csv', index=False)

# Each participant contributes at most one observation per set size and exposure.
records = []
for (ns, exp), grp in block_rt.groupby(['set_size', 'exposure'], sort=True):
    values = grp['mean_rt_s'].to_numpy()
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    ci = float(t.ppf(.975, n-1) * sem) if n > 1 else np.nan
    records.append({'set_size': int(ns), 'exposure': int(exp), 'n_participants': n,
                    'mean_rt_s': mean, 'sd_participants_s': sd,
                    'ci95_lower_s': mean-ci, 'ci95_upper_s': mean+ci,
                    'n_eligible_trials': int(grp['n_eligible_trials'].sum())})
summary = pd.DataFrame(records).sort_values(['set_size','exposure'])
summary.to_csv(OUT / 'rt_by_exposure.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
for ns, grp in summary.groupby('set_size'):
    grp = grp.sort_values('exposure')
    x = grp['exposure'].to_numpy()
    y = grp['mean_rt_s'].to_numpy()
    lower = grp['ci95_lower_s'].to_numpy()
    upper = grp['ci95_upper_s'].to_numpy()
    (line,) = ax.plot(x, y, marker='o', linewidth=2, markersize=5, label=f'ns{int(ns)}')
    ax.fill_between(x, lower, upper, alpha=.17, color=line.get_color())
ax.set(xlabel='Exposure (within set size)', ylabel='Mean reaction time (seconds)',
       title='Reaction time across repeated set-size exposures')
ax.set_xticks(sorted(summary['exposure'].unique()))
ax.grid(alpha=.2)
ax.legend(title='Set size', ncol=2)
fig.savefig(OUT / 'rt_by_exposure.png', dpi=220)
plt.close(fig)
# Second view: RT versus log(set size), pooling exposures within each participant.
# Average block means (not individual trials) so every eligible block has equal
# influence on its participant's summary; then average across participants.
participant_set_size = (block_rt.groupby(['subject', 'set_size'], as_index=False)
                        .agg(mean_rt_s=('mean_rt_s', 'mean'),
                             n_blocks=('block', 'size'),
                             n_eligible_trials=('n_eligible_trials', 'sum')))

records = []
for ns, grp in participant_set_size.groupby('set_size', sort=True):
    values = grp['mean_rt_s'].to_numpy()
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    margin = float(t.ppf(.975, n - 1) * se) if n > 1 else np.nan
    records.append({
        'set_size': int(ns), 'log_set_size': float(np.log(ns)),
        'n_participants': n, 'mean_rt_s': mean,
        'sd_participants_s': sd, 'ci95_lower_s': mean - margin,
        'ci95_upper_s': mean + margin,
        'n_blocks': int(grp['n_blocks'].sum()),
        'n_eligible_trials': int(grp['n_eligible_trials'].sum()),
    })
log_summary = pd.DataFrame(records).sort_values('set_size')
log_summary.to_csv(OUT / 'rt_by_log_set_size.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
x = log_summary['log_set_size'].to_numpy()
y = log_summary['mean_rt_s'].to_numpy()
yerr = np.vstack([y - log_summary['ci95_lower_s'].to_numpy(),
                  log_summary['ci95_upper_s'].to_numpy() - y])
ax.errorbar(x, y, yerr=yerr, fmt='o-', color='black', capsize=4,
            linewidth=2, markersize=6, label='Mean RT ± 95% CI')
ax.set_xticks(x, [f'ln({int(ns)})' for ns in log_summary['set_size']])
ax.set(xlabel='Natural log of set size', ylabel='Mean reaction time (seconds)',
       title='Reaction time versus log(set size)\n(All eligible blocks and exposures)')
ax.grid(alpha=.2)
fig.savefig(OUT / 'rt_by_log_set_size.png', dpi=220)
plt.close(fig)

print('Saved:', OUT)
print('Eligible RTs:', len(eligible), '/', len(df))
print(summary[['set_size', 'exposure', 'n_participants', 'mean_rt_s']].to_string(index=False))
