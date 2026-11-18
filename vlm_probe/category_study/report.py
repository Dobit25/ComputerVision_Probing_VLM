"""Generate the Vietnamese report after auditing a completed full study."""

import argparse
import json
from pathlib import Path

from ..io_utils import inside_repo, sha256, write_json
from ..report import plotting
from .audit import verify
from .runner import csv_rows


def report(root, directory):
    audit = verify(root, directory)
    if audit['mode'] != 'full':
        raise ValueError('Report requires the full test split.')
    output = root / 'reports/cifar100_category_v1'
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((directory / 'metrics.json').read_text())
    paired = metrics['paired']
    classes = csv_rows(directory / 'per_class.csv')
    groups = csv_rows(directory / 'per_category.csv')
    run = directory.relative_to(root).as_posix()
    low, high = paired['delta_ci95_percentage_points']
    lines = ['# CIFAR-100: thêm category vào prompt', '',
             'Một checkpoint OpenAI CLIP ViT-B/32, cùng 10.000 ảnh test và cùng vector ảnh cho hai điều kiện. '
             'Mỗi ảnh luôn chọn giữa đủ 100 lớp. Ánh xạ nhóm cha lấy từ nhãn fine/coarse của **train**, '
             'không dùng nhãn nhóm thật của ảnh test để thu hẹp ứng viên.', '',
             '| Điều kiện | Prompt | Đúng / ảnh test | Top-1 | Mean per-class |',
             '| --- | --- | ---: | ---: | ---: |']
    for name in ('baseline', 'category'):
        m = metrics[name]
        template = 'a photo of a {class}.' if name == 'baseline' else 'a photo of a {class}, a type of {category}.'
        lines.append(f"| {name} | `{template}` | {m['correct']} / {m['num_samples']} | "
                     f"{m['top1_accuracy_percent']:.2f}% | {m['mean_per_class_accuracy_percent']:.2f}% |")
    improved = sum(float(r['delta_percentage_points']) > 0 for r in classes)
    declined = sum(float(r['delta_percentage_points']) < 0 for r in classes)
    lines += ['', f"**Chênh lệch category − baseline: {paired['delta_percentage_points']:+.2f} điểm phần trăm.** "
              f'CI 95% bootstrap ghép cặp, phân tầng theo fine class: [{low:+.2f}; {high:+.2f}] điểm phần trăm '
              '(1.000 lần, seed 0).', '',
              '| Chuyển trạng thái | Số ảnh |', '| --- | ---: |',
              f"| Sai → đúng | {paired['wrong_to_correct']} |",
              f"| Đúng → sai | {paired['correct_to_wrong']} |",
              f"| Cả hai đúng | {paired['both_correct']} |",
              f"| Cả hai sai | {paired['both_wrong']} |", '',
              f"Tổng ảnh đổi nhãn dự đoán: {paired['prediction_changed']}; bao gồm cả trường hợp đổi từ một nhãn sai sang nhãn sai khác.", '',
              '## Thay đổi theo lớp', '',
              f'Trong 100 lớp: **{improved} lớp tăng, {declined} lớp giảm, {100-improved-declined} lớp không đổi accuracy**. '
              'Các lớp tăng/giảm nhiều nhất dưới đây được chọn sau khi tổng hợp để minh họa; '
              'kết quả đầy đủ của cả 100 lớp được giữ trong CSV.', '']
    for title, descending in [('Tăng nhiều nhất', True), ('Giảm nhiều nhất', False)]:
        lines += [f'### {title}', '', '| Lớp | Nhóm | Baseline | Category | Chênh lệch (điểm %) |',
                  '| --- | --- | ---: | ---: | ---: |']
        selected = sorted(classes, key=lambda r: ((-1 if descending else 1) * float(r['delta_percentage_points']), r['class_name']))[:10]
        for row in selected:
            lines.append(f"| {row['class_name']} | {row['coarse_name']} | {float(row['baseline_accuracy_percent']):.2f}% | "
                         f"{float(row['category_accuracy_percent']):.2f}% | {float(row['delta_percentage_points']):+.2f} |")
        lines.append('')
    lines += ['## Lỗi trong và ngoài nhóm lớn', '',
              '| Điều kiện | Sai nhưng cùng coarse group | Sai sang coarse group khác |', '| --- | ---: | ---: |']
    for name in ('baseline', 'category'):
        within = sum(int(r[name + '_errors_within_group']) for r in groups)
        across = sum(int(r[name + '_errors_across_groups']) for r in groups)
        lines.append(f'| {name} | {within} | {across} |')
    lines += ['', 'Các nhóm dùng để phân tích sau dự đoán; không lọc ứng viên khi suy luận.', '',
              '![Chênh lệch theo nhóm](category_delta.png)', '', '## Kiểm chứng và giới hạn', '',
              'Baseline chạy lại khớp đủ 10.000 dự đoán với baseline core. Cosine của lớp đúng/lớp được chọn '
              f"và margin có sai khác tuyệt đối tối đa {audit['baseline_parity']['max_absolute_cosine_or_margin_difference']:.9g} "
              '(ngưỡng 1e-6). Audit tính lại dự đoán từ cả hai ma trận cosine 10.000 × 100 và tính lại metrics/CI.', '',
              'Phép thử đo tác động của **một cách thêm category cố định**. Chưa tách hoàn toàn tác động của '
              'thông tin ngữ nghĩa khỏi độ dài/cấu trúc câu; chưa phải nghiên cứu tám template hoặc ensembling. '
              'Không dùng kết quả để chọn lại cách viết category. Hai nhóm vehicles cùng dùng từ vehicle. '
              'Kết quả không so trực tiếp với fine-grained oracle 5-way.', '',
              f'[Toàn bộ 100 lớp và prompt](../../{run}/classes.json) · '
              f'[Accuracy từng lớp](../../{run}/per_class.csv) · [20 nhóm](../../{run}/per_category.csv) · '
              f'[Dự đoán từng ảnh](../../{run}/predictions.csv) · [Cosine scores](../../{run}/scores.npz) · '
              '[Giao thức](../../docs/cifar100_category_protocol.md) · [Audit](audit.json)', '',
              '```powershell', f'python -m vlm_probe.category_study.audit {run}',
              f'python -m vlm_probe.category_study.report {run}', '```', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    plt = plotting(root)
    ordered = sorted(groups, key=lambda r: float(r['delta_percentage_points']))
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    values = [float(r['delta_percentage_points']) for r in ordered]
    ax.barh([r['coarse_name'].replace('_', ' ') for r in ordered], values,
            color=['#3a7ca5' if v >= 0 else '#bf4b45' for v in values])
    ax.axvline(0, color='black', linewidth=.8)
    ax.set(xlabel='Category minus baseline (percentage points)', title='CIFAR-100 — all 100 candidates for every image')
    fig.savefig(output / 'category_delta.png', dpi=150)
    plt.close(fig)
    write_json(output / 'audit.json', {'audit': audit, 'run': run,
        'manifest_sha256': sha256(directory / 'manifest.json'), 'report_source_sha256': sha256(Path(__file__))})
    print(f'Report: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report(root, inside_repo(root, args.run_directory))


if __name__ == '__main__':
    main()
