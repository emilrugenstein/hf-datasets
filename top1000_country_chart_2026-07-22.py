"""Reproduction of alex-repo/country_charts.py (top-1000 config) on the 2026-07-22 snapshot.

Owner classification reused from alex-repo/owner_classification.csv — all 585 owners of the
new top-1,000 were already researched there (no new owners entered vs the 2026-07-15 slice).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED = '#4064d9', '#eda100', '#9a9a9a', '#1a1a1a', '#666666'


def fmt_bytes(v):
    for cut, suf in [(1e12, 'TB'), (1e9, 'GB'), (1e6, 'MB'), (1e3, 'kB')]:
        if v >= cut:
            return f'{v/cut:,.1f} {suf}'.replace('.0 ', ' ')
    return f'{v:,.0f} B'


df = pd.read_parquet('data/top1000_classified_2026-07-22.parquet')
n_owners = df['author'].nunique()
n_evid = (df['country'] != 'Unknown').sum()

title = (f'Top 1,000 datasets by all-time downloads — by owner country  '
         f'(own classification of all {n_owners} owners)')
foot = (f'All {n_owners} owners individually researched (HF profiles, org pages, web). '
        f'{n_evid}/1,000 datasets have an evidenced country; Unknown = no public evidence '
        '(never inferred from names/language). Single-dataset countries show no whisker. '
        'Whiskers = IQR in log₁₀ space. Snapshot 2026-07-22.')

rows = []
for c, g in df.groupby('country'):
    ls = np.log10(g.loc[g['mainSize'] > 0, 'mainSize'])
    rows.append({'country': c, 'downloads': g['downloadsAllTime'].sum(), 'n': len(g), 'n_size': len(ls),
                 'med': ls.median(), 'q1': ls.quantile(.25), 'q3': ls.quantile(.75)})
st = pd.DataFrame(rows).sort_values('downloads', ascending=False).reset_index(drop=True)

h = max(6.5, 0.42 * len(st) + 2.6)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, h), dpi=140)
fig.suptitle(title, fontsize=15, fontweight='bold', color=INK, y=0.985)
y = np.arange(len(st))[::-1]
grayish = st['country'].str.startswith(('Unknown', 'Other'))

axL.barh(y, st['downloads'] / 1e6, color=np.where(grayish, GRAY, BLUE), height=0.62)
for yi, v, cnt in zip(y, st['downloads'] / 1e6, st['n']):
    axL.text(v + st['downloads'].max() / 1e6 * 0.012, yi, f'{v:,.0f}M ({cnt})', va='center', fontsize=10, color=INK)
axL.set_yticks(y, st['country'], fontsize=11)
axL.set_xlim(0, st['downloads'].max() / 1e6 * 1.22)
axL.set_xlabel('downloads (millions)', fontsize=11, color=MUTED)
axL.set_title('downloads by country  (n datasets in parens)', fontsize=13, color=INK, pad=14)

axR.barh(y, st['med'], color=np.where(grayish, GRAY, ORANGE), height=0.62)
multi = st['n_size'] > 1
axR.errorbar(st.loc[multi, 'med'], y[multi.values],
             xerr=[(st['med'] - st['q1'])[multi], (st['q3'] - st['med'])[multi]],
             fmt='none', ecolor='#333333', elinewidth=1.1, capsize=3)
for yi, med, q3, m in zip(y, st['med'], st['q3'], multi):
    axR.text((q3 if m else med) + 0.25, yi, fmt_bytes(10 ** med), va='center', fontsize=10, color=INK)
axR.set_yticks(y, [''] * len(st))
axR.set_xlim(0, max(st['q3'].max(), st['med'].max()) + 2.6)
axR.set_xlabel('log₁₀ size (bytes)', fontsize=11, color=MUTED)
axR.set_title('dataset size — median log₁₀(bytes), whiskers = IQR, label = median', fontsize=13, color=INK, pad=14)

for ax in (axL, axR):
    ax.grid(axis='x', color='#e3e3e3', lw=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.tick_params(axis='y', colors=MUTED, length=0)
fig.text(0.5, 0.008, foot, ha='center', fontsize=8.5, color=MUTED)
plt.tight_layout(rect=[0, 0.025, 1, 0.97])
plt.savefig('top1000_by_country_2026-07-22.png', bbox_inches='tight', facecolor='white')
plt.close()
print('saved top1000_by_country_2026-07-22.png')
