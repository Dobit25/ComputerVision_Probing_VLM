"""Build tables and figures from audited frozen runs; never rerun inference."""
from pathlib import Path
import numpy as np

from ..io_utils import sha256, write_json
from .common import ROOT, read_json, write_csv
from .audit import verify_hashes
from .pairs import analyze


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ROOT / 'reports/probing_depth_v1'
    out.mkdir(parents=True, exist_ok=True)
    metrics, inputs, pair_metrics = {}, {}, {}
    for stage in ['historical', 'spatial', 'category', 'natural']:
        folder = ROOT / f'results/full_depth_{stage}_v1'
        verify_hashes(folder)
        audit_path = folder.parent / (folder.name + '_audit.json')
        audit = read_json(audit_path)
        if not audit.get('verified') or not audit.get('metrics_recomputed'):
            raise ValueError('Run must pass independent recomputation before reporting.')
        metrics[stage] = read_json(folder / 'metrics.json')
        for path in [folder/'manifest.json', folder/'metrics.json', audit_path]:
            inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
        if stage in ('spatial', 'natural'):
            summary, records = analyze(folder)
            pair_metrics[stage] = summary
            write_csv(out / f'{stage}_exploratory_pairs.csv', records)
    write_json(out / 'exploratory_pair_metrics.json', pair_metrics)
    for name in ['README.md', 'bao_cao_probing_chuyen_sau.tex']:
        path = out / name
        if path.exists():
            inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name in ['vlm_probe/depth/report.py', 'vlm_probe/depth/pairs.py',
                 'docs/depth_exploratory_pair_analysis.md', 'docs/natural_depth_review.md',
                 'results/depth_natural_selection_v1/review.json',
                 'results/depth_natural_selection_v1/selected.json']:
        inputs[name] = sha256(ROOT / name)

    s, n, c = (metrics[k] for k in ['spatial', 'natural', 'category'])
    spatial_rows = []
    for domain, result in [('synthetic', s), ('natural', n)]:
        for condition, values in result['conditions'].items():
            spatial_rows.append({'domain': domain, 'condition': condition, 'n': values['n'],
                'accuracy_percent': values['accuracy_percent'],
                'ci95_low': values['accuracy_ci95_percent'][0], 'ci95_high': values['accuracy_ci95_percent'][1],
                'mean_signed_margin': values['mean_signed_margin'],
                **{f'predicted_{name}': count for name, count in zip(['left','right','above','below'], values['prediction_counts'])}})
    write_csv(out / 'spatial_conditions.csv', spatial_rows)
    write_csv(out / 'category_conditions.csv', [{'condition': k, **v} for k,v in c['conditions'].items()])
    write_csv(out / 'category_planned_comparisons.csv', [{'comparison': k, 'delta_pp': v['delta_pp'],
              'ci95_low': v['ci95_pp'][0], 'ci95_high': v['ci95_pp'][1]} for k,v in c['planned_comparisons'].items()])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.savefig(out / (name+'.png'), dpi=200, bbox_inches='tight')
        fig.savefig(out / (name+'.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for ax, result, variant, title in [(axes[0], s, 'all', 'Hình học: 1.440 ảnh'),
                                      (axes[1], n, 'original', 'Tự nhiên: 200 ảnh gốc')]:
        keys = [f't{t}_o{o}_{variant}' for t in range(3) for o in range(2)]
        vals = np.array([result['conditions'][k]['accuracy_percent'] for k in keys])
        cis = np.array([result['conditions'][k]['accuracy_ci95_percent'] for k in keys]).T
        ax.bar(range(6), vals, color=['#2878a6','#8fc2dc']*3)
        ax.errorbar(range(6), vals, yerr=np.vstack([vals-cis[0],cis[1]-vals]), fmt='none', color='black', capsize=3)
        ax.axhline(25, color='#ad403c', linestyle='--', label='Ngẫu nhiên: 25%')
        ax.set_xticks(range(6), ['T0→','T0←','T1→','T1←','T2→','T2←'])
        ax.set(ylabel='Accuracy (%) và CI 95%', title=title, ylim=(0, 50))
        ax.legend(fontsize=8, loc='upper left')
    save(fig, 'spatial_accuracy')

    fig, axes = plt.subplots(1,2,figsize=(9,3.5), constrained_layout=True)
    for ax, result, key, title in [(axes[0],s,'t0_o0_all','Hình học, T0 thuận'),
                                  (axes[1],n,'t2_o0_original','Tự nhiên, T2 thuận')]:
        counts = np.array(result['conditions'][key]['confusion'])
        rates = 100*counts/counts.sum(1,keepdims=True)
        ax.imshow(rates, vmin=0, vmax=100, cmap='Blues')
        for (i,j),v in np.ndenumerate(rates):
            ax.text(j,i,f'{v:.1f}%\n({counts[i,j]})',ha='center',va='center',fontsize=9,color='white' if v>55 else 'black')
        ax.set_xticks(range(4), ['Trái','Phải','Trên','Dưới'])
        ax.set_yticks(range(4), ['Trái','Phải','Trên','Dưới'])
        ax.set(xlabel='Dự đoán',ylabel='Nhãn thật',title=title)
    save(fig, 'spatial_confusion')

    fig, ax = plt.subplots(figsize=(9,3.5), constrained_layout=True)
    names = list(c['conditions'])
    values = [c['conditions'][k]['accuracy_percent'] for k in names]
    bars = ax.bar(range(8),values,color=['#2878a6','#d98b32','#58a27c']+['#a7a9ae']*5)
    ax.bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=3,fontsize=9)
    ax.set_xticks(range(8),['Baseline','Category\nđúng','Trung tính']+[f'Sai {i}' for i in range(1,6)])
    ax.set(ylabel='Top-1 accuracy (%)',ylim=(0,75),title='CIFAR-100: giữ đủ 100 lớp, cùng 10.000 ảnh')
    save(fig, 'category_controls')

    def number(value):
        return f'{value:.2f}'.replace('.', ',')
    def ci(value):
        return '['+'; '.join(number(x) for x in value)+']'
    lines = []
    for t in range(3):
        for order in range(2):
            a,b = s['conditions'][f't{t}_o{order}_all'], n['conditions'][f't{t}_o{order}_original']
            lines.append(f"T{t} / {'thuận' if order == 0 else 'đảo tương đương'} & {number(a['accuracy_percent'])} & {ci(a['accuracy_ci95_percent'])} & {number(b['accuracy_percent'])} & {ci(b['accuracy_ci95_percent'])} " + r'\\')
    (out/'spatial_table.tex').write_text(
        '\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n'
        + 'Mẫu câu/thứ tự & Hình học & CI 95\\% & Tự nhiên & CI 95\\% \\\\\n\\midrule\n'
        + '\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    labels = ['Baseline','Category đúng','Nội dung trung tính']+[f'Category sai, seed {i}' for i in range(271,276)]
    lines = [f"{label} & {number(c['conditions'][key]['accuracy_percent'])} & {number(c['conditions'][key]['delta_vs_baseline_pp'])} " + r'\\' for label,key in zip(labels,names)]
    (out/'category_table.tex').write_text(
        '\\begin{tabular}{@{}lrr@{}}\n\\toprule\n'
        + 'Điều kiện & Accuracy (\\%) & Chênh lệch \\\\\n\\midrule\n'
        + '\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    macros = {'SyntheticAcc': number(s['conditions']['t0_o0_all']['accuracy_percent']),
              'NaturalAcc': number(n['conditions']['t2_o0_original']['accuracy_percent']),
              'NaturalCI': ci(n['conditions']['t2_o0_original']['accuracy_ci95_percent']),
              'NaturalFlipBoth': number(n['flips']['t2_o0']['both_correct_percent']),
              'NaturalHorizontalBoth': number(n['flips']['t2_o0']['horizontal_both_correct_percent']),
              'NaturalOrderAgreement': number(n['equivalent_orders']['t2']['prediction_agreement_percent']),
              'NaturalPairAssignment': number(pair_metrics['natural']['t2_o0_horizontal']['assignment_accuracy_percent']),
              'NaturalPairCI': ci(pair_metrics['natural']['t2_o0_horizontal']['assignment_ci95_percent'])}
    (out/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    write_json(out/'sources.json', {'description': 'Full runs audited before reporting. Pair assignment is explicitly post-hoc.',
               'input_sha256': inputs, 'derived_sha256': {p.name: sha256(p) for p in sorted(out.iterdir())
                    if p.is_file() and p.suffix in ('.csv','.png','.pdf','.tex','.json') and p.name not in ['sources.json','bao_cao_probing_chuyen_sau.pdf','bao_cao_probing_chuyen_sau.tex']}})
    print('Report tables, plots and LaTeX numbers:', out, flush=True)


if __name__ == '__main__':
    run()
