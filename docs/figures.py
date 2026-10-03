from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {2: '#2a78d6', 5: '#eb6834'}

metrics = pd.read_csv(ROOT / 'results' / 'metrics.csv')
words = sorted(metrics.word.unique())

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, model in zip(axes, ['SVM', 'Naive Bayes']):
    for k, window in enumerate([2, 5]):
        f1 = metrics[(metrics.model == model) & (metrics.window == window)].set_index('word').f1
        x = [i + (k - 0.5) * 0.36 for i in range(len(words))]
        ax.bar(x, f1[words], width=0.34, color=COLORS[window], label=f'W = ±{window}')
    ax.set_title(model)
    ax.set_xticks(range(len(words)), words, rotation=35, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('Macro F1')
axes[1].legend(frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(ROOT / 'docs' / 'figures' / 'window_f1.pdf')
