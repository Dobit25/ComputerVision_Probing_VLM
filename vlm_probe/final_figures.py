"""WS3 (1124.md, Chang 2): four IEEE-size figures from audited artifacts (no inference).

    python -m vlm_probe.final_figures

Writes reports/final_diagnostics_v1/figures/{fig_baseline_prompts, fig_counting_negation,
fig_spatial_overview, fig_vietnamese_hybrid}.{pdf,png}. Fonts: Times New Roman; minus signs are
rendered as hyphens. One column = 3.5 in, two columns = 7.1 in.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .depth.common import ROOT, read_json
from .io_utils import sha256, write_json

FIGS = ROOT / 'reports/final_diagnostics_v1/figures'
DIAG = ROOT / 'results/final_diagnostics_v1/metrics.json'
VI_TABLE = ROOT / 'results/final_diagnostics_v1/vietnamese_per_class.csv'
MULTIDOMAIN = ROOT / 'reports/research_review_v2/multidomain_summary.csv'
T2 = ROOT / 'results/full_spatial_ablation_v1/metrics.json'
SET_C = ROOT / 'results/full_natural_unseen_holdout_v1/metrics.json'
T1 = ROOT / 'results/full_natural_recognition_v1/metrics.json'
BASELINES = [('cifar100', 'CIFAR-100', 'full_cifar100_clip_b32_v1'), ('flowers102', 'Flowers102', 'full_flowers102_clip_b32_v2'),
             ('oxford_pet', 'Pet', 'full_oxford_pet_clip_b32_v3'), ('dtd', 'DTD', 'full_dtd_clip_b32_v3'),
             ('food101', 'Food-101', 'full_food101_clip_b32_v3')]
ONE_COL, TWO_COL = 3.5, 7.1
GREY, BLUE, ORANGE, GREEN, RED = '#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 8,
                         'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
    return plt


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGS / f'{name}.png', bbox_inches='tight', dpi=300)


def worst_class(run):
    with (ROOT / 'results' / run / 'predictions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    stats = {}
    for r in rows:
        s = stats.setdefault(r['target_name'], [0, 0])
        s[0] += int(r['target_id']) == int(r['prediction_id'])
        s[1] += 1
    name, (k, n) = min(stats.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1], kv[0]))
    return name.replace('_', ' '), k, n


def fig_baseline_prompts(plt):
    with MULTIDOMAIN.open(encoding='utf-8', newline='') as f:
        md = {r['dataset']: r for r in csv.DictReader(f)}
    labels, base, ens, lo, hi = [], [], [], [], []
    for key, label, run in BASELINES:
        r = md[key]
        name, k, n = worst_class(run)
        labels.append(f'{label}\nworst: {name}\n{k}/{n}')
        base.append(float(r['baseline_top1'])); ens.append(float(r['ensemble_top1']))
        lo.append(float(r['delta_ci_low'])); hi.append(float(r['delta_ci_high']))
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.3))
    ax.bar(x - w / 2, base, w, color=GREY, label='Baseline prompt')
    bars = ax.bar(x + w / 2, ens, w, color=BLUE, label='Embedding ensemble (7 templates)')
    for b, d0, d1, e, b0 in zip(bars, lo, hi, ens, base):
        ax.text(b.get_x() + b.get_width() / 2, e + 1.5, f'{e - b0:+.2f}\n[{d0:.2f}, {d1:.2f}]', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x, labels); ax.set_ylabel('Top-1 accuracy (%)'); ax.set_ylim(0, 105)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, ncol=2)
    save(fig, 'fig_baseline_prompts'); plt.close(fig)


def fig_counting_negation(plt, diag):
    c, n = diag['counting'], diag['negation']
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.3), gridspec_kw={'width_ratios': [1.1, 1.0, 1.1]})
    ax = axes[0]
    counts = np.arange(1, 7)
    for scope, color, label in (('all', GREY, 'All images (n=1,080)'), ('control_correct', BLUE, 'Control correct (n=804)')):
        acc = [c['by_true_count'][str(k)][scope]['accuracy']['percent'] for k in counts]
        ax.plot(counts, acc, marker='o', ms=3, color=color, label=label)
    ax.axhline(100 / 6, color='black', lw=0.6, ls=':', label='Chance (16.67%)')
    ax.set_xticks(counts); ax.set_xlabel('True number of objects'); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 80)
    ax.set_title('(a) Counting by true count'); ax.legend(frameon=False, loc='upper right', fontsize=6)
    ax = axes[1]
    conf = np.array(c['confusion_rows_true_cols_pred'])
    im = ax.imshow(conf / conf.sum(1, keepdims=True) * 100, cmap='Blues', vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(conf[i, j]), ha='center', va='center', fontsize=6,
                    color='white' if conf[i, j] / conf[i].sum() > 0.45 else 'black')
    ax.set_xticks(range(6), counts); ax.set_yticks(range(6), counts)
    ax.set_xlabel('Predicted count'); ax.set_ylabel('True count'); ax.set_title('(b) Counting confusion (row %)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(True)
    ax = axes[2]
    templates = [('with_without', 'with / without'), ('contains_does_not', 'contains /\ndoes not contain')]
    x = np.arange(len(templates)); w = 0.36
    pres = [n[t]['accuracy_present']['percent'] for t, _ in templates]
    absn = [n[t]['accuracy_absent']['percent'] for t, _ in templates]
    ax.bar(x - w / 2, pres, w, color=GREEN, label='Object present (affirmative)')
    ax.bar(x + w / 2, absn, w, color=ORANGE, label='Object absent (negative)')
    for i, (t, _) in enumerate(templates):
        rate = n[t]['affirmative_prediction_rate']['percent']
        ax.text(x[i], 102, f'affirmative chosen\n{rate:.2f}%', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x, [lab for _, lab in templates]); ax.set_ylim(0, 120); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Accuracy (%)'); ax.set_title('(c) Negation by true label')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_counting_negation'); plt.close(fig)


def fig_spatial_overview(plt):
    t2, sc, t1 = read_json(T2), read_json(SET_C), read_json(T1)
    groups = [('Set A (200 sources, 4-way)', t2['sets']['A']['conditions'], 25.0),
              ('Set B reviewed (98 sources, 2-way)', t2['sets']['B']['conditions'], 50.0)]
    conds = [('holistic_center_crop', 'Holistic (center crop)', GREY), ('holistic_full_padded', 'Holistic (full image, padded)', '#bcbcbc'),
             ('native_half', 'Native half', BLUE), ('padded_half', 'Padded half', '#9ecae1'),
             ('bbox_ideal_half', 'Oracle bbox, ideal half', GREEN)]
    set_c = {'holistic_center_crop': sc['holistic_t2_o0'], 'native_half': sc['native_half'], 'padded_half': sc['padded_half'],
             'bbox_ideal_half': {'primary_accuracy_percent': sc['bbox_geometric_baseline']['ideal_half_canvas_2way_accuracy_percent']}}
    groups.append(('Set C unseen (100 sources, 2-way)', set_c, 50.0))
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.5))
    w = 0.16
    for gi, (label, data, chance) in enumerate(groups):
        for ci, (key, clabel, color) in enumerate(conds):
            if key not in data:
                continue
            v = data[key]['primary_accuracy_percent']
            ci95 = data[key].get('primary_ci95_percent')
            xpos = gi + (ci - 2) * w
            err = None if not ci95 else [[v - ci95[0]], [ci95[1] - v]]
            ax.bar(xpos, v, w, color=color, label=clabel if gi == 0 else None, yerr=err, capsize=1.5,
                   error_kw={'elinewidth': 0.6})
            ax.text(xpos, v + (ci95[1] - v if ci95 else 0) + 1.5, f'{v:.1f}', ha='center', va='bottom', fontsize=5.5)
        ax.hlines(chance, gi - 2.6 * w, gi + 2.6 * w, colors='black', linestyles=':', lw=0.7)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups]); ax.set_ylim(0, 115); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Top-1 relation accuracy (%)')
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color='black', ls=':', lw=0.7)); labels.append('Chance')
    ax.legend(handles, labels, frameon=False, ncol=6, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=6)
    save(fig, 'fig_spatial_overview'); plt.close(fig)


def fig_vietnamese_hybrid(plt, diag):
    v = diag['vietnamese']
    order = [('en_baseline', 'English'), ('vi_diacritics', 'Vietnamese'), ('vi_no_diacritics', 'VI, no\ndiacritics'),
             ('en_frame_vi_label', 'EN frame,\nVI name'), ('vi_frame_en_label', 'VI frame,\nEN name'),
             ('vi_en_bilingual', 'Bilingual')]
    with VI_TABLE.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.3), gridspec_kw={'width_ratios': [1.35, 1.0]})
    ax = axes[0]
    vals = [v['overall_top1_percent'][k] for k, _ in order]
    colors = [GREY, ORANGE, ORANGE, ORANGE, BLUE, BLUE]
    bars = ax.bar(range(len(order)), vals, color=colors)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f'{val:.2f}', ha='center', va='bottom', fontsize=6)
    ax.axhline(1.0, color='black', lw=0.6, ls=':')
    ax.text(2.0, 30, 'Dotted line: chance (1%)', ha='center', fontsize=6)
    ax.set_xticks(range(len(order)), [lab for _, lab in order]); ax.set_ylim(0, 75)
    ax.set_ylabel('CIFAR-100 top-1 (%)'); ax.set_title('(a) Prompt language on the same 10,000 images')
    ax = axes[1]
    tok = np.array([float(r['vi_label_tokens']) for r in rows]); drop = np.array([float(r['drop_vi_diacritics_pp']) for r in rows])
    cog = np.array([r['cognate_like'] == 'True' for r in rows])
    jitter = (np.arange(len(tok)) % 5 - 2) * 0.06
    ax.scatter(tok[~cog] + jitter[~cog], drop[~cog], s=7, color=ORANGE, alpha=0.7, label='Other classes')
    ax.scatter(tok[cog] + jitter[cog], drop[cog], s=12, color=BLUE, marker='s', label='Name close to English (heuristic)')
    s = v['spearman']['vi_label_tokens_vs_drop_vi']
    ax.text(0.98, 0.04, f"Spearman rho = {s['spearman_rho']:.2f}\n95% CI [{s['ci95_class_bootstrap'][0]:.2f}, {s['ci95_class_bootstrap'][1]:.2f}]",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
    ax.set_xlabel('BPE tokens in Vietnamese class name'); ax.set_ylabel('Accuracy drop vs English (points)')
    ax.set_ylim(-50, 128)
    ax.set_title('(b) Per-class drop vs tokenization')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig_vietnamese_hybrid'); plt.close(fig)


def main():
    plt = style()
    diag = read_json(DIAG)
    fig_baseline_prompts(plt)
    fig_counting_negation(plt, diag)
    fig_spatial_overview(plt)
    fig_vietnamese_hybrid(plt, diag)
    inputs = [DIAG, VI_TABLE, MULTIDOMAIN, T2, SET_C, T1] + [ROOT / 'results' / run / 'predictions.csv' for _, _, run in BASELINES]
    write_json(FIGS / 'figures_manifest.json', {
        'code_sha256': sha256(Path(__file__)),
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
        'figures': sorted(p.name for p in FIGS.iterdir() if p.suffix in ('.pdf', '.png'))})
    print(json.dumps(sorted(p.name for p in FIGS.iterdir()), indent=1), flush=True)


if __name__ == '__main__':
    main()
