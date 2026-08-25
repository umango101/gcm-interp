"""Plot per-layer steering effect against layer index.

Consumes the scorer's per-cell output. ``eval_pipeline_bias.py`` writes
``{plots_dir}/{metric}_dataset.csv`` with rows indexed by N and one column per
value of the sweep's third field — which in ``--sweep_mode per_layer`` is the
layer index, so that file is already effect-by-layer and needs no new scoring
stage.

    python -m layers.plot_layer_sweep \\
        --dataset_csv results_pipeline/.../plots/judge_0.5_dataset.csv \\
        --layer_effects results_layers/Qwen1.5-14B-Chat/from_female-long_to_male-long/atp/eval/layer_effects.csv \\
        --out per_layer_effect.png

The overlay is the point of the exercise. The per-layer sweep is the measured
ground truth: what actually happens when you steer layer L. The attribution map
is a prediction: what ATP said would happen. Plotting them on shared axes, and
reporting the rank correlation between them, tests the localization claim
directly — far more informative than a top-k curve, which collapses forty
measurements into seven and confounds "ATP ranked layers well" with "steering
more layers does more".

Read the correlation, not the peak agreement. A method that ranks the single
best layer correctly but is otherwise uninformative is a different finding from
one that gets the whole profile right, and only the correlation distinguishes them.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _spearman(a, b):
    """Rank correlation without a scipy dependency.

    Average ranks for ties, which matters here: a genuinely uninformative
    attribution map produces many near-equal scores, and ordinal ranking would
    manufacture a spurious ordering among them.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float('nan'), 0

    def rank(x):
        order = np.argsort(x, kind='mergesort')
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (float((ra * rb).sum() / denom) if denom else float('nan')), int(a.size)


def load_dataset_csv(path):
    """-> DataFrame indexed by N, columns = integer layer index, sorted."""
    df = pd.read_csv(path, index_col=0)
    df.columns = [int(float(c)) for c in df.columns]
    return df[sorted(df.columns)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset_csv', required=True,
                    help="Scorer output: {plots_dir}/{metric}_dataset.csv")
    ap.add_argument('--layer_effects', default=None,
                    help="Optional layer_effects.csv from the localization pass, "
                         "for the ATP overlay and rank correlation.")
    ap.add_argument('--attribution_col', default='cumulative',
                    choices=['cumulative', 'marginal'])
    ap.add_argument('--baseline', type=float, default=None,
                    help="Optional unsteered score, drawn as a horizontal reference. "
                         "Without it, a flat profile is hard to read: it could mean "
                         "no layer does anything, or that every layer saturates.")
    ap.add_argument('--out', default='per_layer_effect.png')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    df = load_dataset_csv(args.dataset_csv)
    layers = list(df.columns)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for n in df.index:
        ax.plot(layers, df.loc[n].values, marker='o', ms=3, lw=1.4, label=f"N={n}")
    if args.baseline is not None:
        ax.axhline(args.baseline, color='0.4', ls=':', lw=1.2, label='unsteered')

    ax.set_xlabel('layer index')
    ax.set_ylabel(os.path.basename(args.dataset_csv).replace('_dataset.csv', ''))
    ax.set_title(args.title or 'Steering effect by layer')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='best')

    rho = None
    if args.layer_effects and os.path.exists(args.layer_effects):
        eff = pd.read_csv(args.layer_effects).set_index('layer')[args.attribution_col]
        aligned = np.array([eff.get(l, np.nan) for l in layers], float)

        # Strongest measured row, so the correlation is not read off an N where
        # steering was too weak to move anything.
        best_n = df.mean(axis=1).abs().idxmax()
        measured = df.loc[best_n].values
        rho, n_pts = _spearman(aligned, measured)

        ax2 = ax.twinx()
        ax2.plot(layers, aligned, color='crimson', ls='--', lw=1.6, alpha=0.75,
                 label=f'ATP ({args.attribution_col})')
        ax2.set_ylabel(f'ATP {args.attribution_col} effect', color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')
        ax2.legend(fontsize=8, loc='lower right')
        ax.set_title((args.title or 'Steering effect by layer') +
                     f"\nSpearman(ATP, measured @ N={best_n}) = {rho:.3f}  (n={n_pts})")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    out_csv = args.out.rsplit('.', 1)[0] + '_by_layer.csv'
    tidy = df.T
    tidy.index.name = 'layer'
    tidy.columns = [f"N={c}" for c in tidy.columns]
    if args.layer_effects and os.path.exists(args.layer_effects):
        eff = pd.read_csv(args.layer_effects).set_index('layer')
        tidy['atp_cumulative'] = [eff['cumulative'].get(l, np.nan) for l in tidy.index]
        tidy['atp_marginal'] = [eff['marginal'].get(l, np.nan) for l in tidy.index]
    tidy.to_csv(out_csv)

    print(f"wrote {args.out} and {out_csv}")
    if rho is not None:
        print(f"Spearman(ATP, measured) = {rho:.3f}")
        print("Near zero is the head-agnostic result extended to layers: ATP's "
              "layer ranking would carry no information about what steering "
              "actually does. Strongly positive means the localization is real.")


if __name__ == "__main__":
    main()
