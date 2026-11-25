"""Independent second-reviewer (R2) package export, validation, freeze, and IAA analysis for Set B.

Never modifies R1's frozen review (reviews/human_review_final_v1/) or original v2 blanks
(reviews/human_review_v2/). Replays sensitivity on the R1 ∩ R2 consensus subset using frozen
scores without any new CLIP inference.
"""
from __future__ import annotations

import argparse
import csv
import html
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .depth.common import ROOT, cluster_ci, read_json, write_csv
from .depth.confirmation import paired_metrics
from .extension_artifacts import close_enough, verify_manifest
from .human_review_followup import normalize_sheet, read_sheet, subset_indices
from .io_utils import sha256, source_snapshot, write_json
from .spatial_diagnostics.natural_regions import audit as region_audit
from .spatial_diagnostics.natural_regions import baseline_canvas, summarize

META = ROOT / 'reviews/human_review_iaa_set_b_v1'
PACKAGE = ROOT / 'data/human_review_iaa_set_b_v1'
OUTPUT = ROOT / 'results/human_review_iaa_set_b_v1'

R1_FINAL = ROOT / 'reviews/human_review_final_v1/confirmation_100_R1.csv'
R1_FREEZE = ROOT / 'reviews/human_review_final_v1/freeze.json'
BLANK_TEMPLATE = ROOT / 'reviews/human_review_v2/confirmation_100_blank.csv'
MAPPING_FILE = ROOT / 'results/human_review_followup_v1/blind_source_mapping.json'
SELECTED_FILE = ROOT / 'results/natural_confirmation_selection_v1/selected.json'
REGION = ROOT / 'results/full_natural_regions_confirmation_review_v2'
HOLISTIC = ROOT / 'results/full_natural_confirmation_v1'

IAA_FIELDS = (
    'overall_accept',
    'both_objects_visible',
    'class_names_correct',
    'relation_clear',
    'perceived_relation',
)

R2_GUIDE_VI = """# Hướng dẫn chấm độc lập lần 2 (R2) — Set B (100 nguồn COCO)

1. **Nguyên tắc độc lập & mù hoàn toàn:** Người chấm **R2 bắt buộc phải là thành viên khác với R1** (người đã chấm vòng trước), chưa từng xem quyết định của R1 và chưa xem bất kỳ điểm số/dự đoán nào của CLIP. Ghi mã người chấm vào cột `annotator_id` là `R2` (hoặc tên/mã riêng khác `R1`).
2. **Cách xem ảnh:** Giải nén gói `reviewer_r2_package.zip` vào thư mục riêng và mở `index.html`. Mỗi mục `CONFIRMATION_100_xxx` hiển thị:
   - Bên trái: ảnh nguồn với khung `A` (xanh dương) và `B` (cam).
   - Bên phải: **đúng vùng crop 224×224** mà CLIP nhìn thấy (phóng to 1,5 lần để dễ quan sát).
   - **Quyết định phải dựa trên vùng crop 224×224**; ảnh nguồn chỉ giúp nhận diện vật bị cắt biên.
3. **Điền phiếu `confirmation_100_R2_blank.csv` (UTF-8):**
   - Giữ nguyên toàn bộ 100 dòng và các cột nguồn (`blind_id`, `object_a`, `object_b`, `expected_relation`).
   - Điền `yes` / `no` cho `both_objects_visible`, `class_names_correct`, `relation_clear`.
   - Điền `perceived_relation` bằng đúng một trong các giá trị: `left`, `right`, `above`, `below`, `unclear` (hướng của A so với B trên mặt phẳng ảnh).
   - `overall_accept` chỉ bằng `yes` khi cả 3 câu hỏi đầu là `yes` **và** `perceived_relation == expected_relation`. Nếu `no`, bắt buộc ghi `exclusion_reason` (`OCCLUDED_OR_TINY`, `MULTIPLE_INSTANCES`, `AMBIGUOUS_LABEL`, `AMBIGUOUS_RELATION`, hoặc `OTHER`) và mô tả ngắn ở `notes`.
4. Lưu file hoàn chỉnh dạng CSV UTF-8 (ví dụ `confirmation_100_R2.csv`) và gửi lại điều phối viên để chạy `validate`, `freeze`, `run` và `audit`.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def multiclass_cohens_kappa(seq_a, seq_b):
    """Compute raw agreement (%) and Cohen's kappa for any discrete label sequences."""
    if len(seq_a) != len(seq_b) or not seq_a:
        raise ValueError('Label sequences must be non-empty and have equal length.')
    n = len(seq_a)
    a = [str(x) for x in seq_a]
    b = [str(x) for x in seq_b]
    agree = sum(x == y for x, y in zip(a, b))
    p_o = agree / n
    ca, cb = Counter(a), Counter(b)
    labels = sorted(set(ca) | set(cb))
    p_e = sum((ca[k] / n) * (cb[k] / n) for k in labels)
    kappa = None if abs(1.0 - p_e) <= 1e-12 else (p_o - p_e) / (1.0 - p_e)
    return {
        'n': n,
        'raw_agreement_count': agree,
        'raw_agreement_percent': 100.0 * p_o,
        'expected_agreement_percent': 100.0 * p_e,
        'cohens_kappa': float(kappa) if kappa is not None else None,
        'kappa_defined': kappa is not None,
        'marginal_a': dict(sorted(ca.items())),
        'marginal_b': dict(sorted(cb.items())),
    }


def validate_r2_rows(rows, blanks):
    """Validate a filled Set B R2 sheet against the frozen blank template."""
    if len(rows) != len(blanks):
        raise ValueError(f'Expected exactly {len(blanks)} rows, got {len(rows)}.')
    expected_fields = list(blanks[0])
    if list(rows[0]) != expected_fields:
        raise ValueError(f'CSV columns differ from template: {list(rows[0])}')
    blank_by_id = {r['blind_id']: r for r in blanks}
    ids = [r.get('blind_id', '') for r in rows]
    if len(set(ids)) != len(blanks):
        raise ValueError('Duplicate blind_id values found.')
    if set(ids) != set(blank_by_id):
        raise ValueError('Missing or unknown blind_id values.')
    annotators = set()
    for row in rows:
        bid = row['blind_id']
        ann = (row.get('annotator_id') or '').strip()
        if not ann:
            raise ValueError(f'{bid}: annotator_id is required.')
        if ann.lower() == 'r1':
            raise ValueError(f'{bid}: R2 reviewer must be independent from R1 (annotator_id cannot be R1).')
        annotators.add(ann)
        for col in ('object_a', 'object_b', 'expected_relation'):
            if row.get(col) != blank_by_id[bid][col]:
                raise ValueError(f'{bid}: frozen column {col!r} was modified.')
        yn_cols = ('both_objects_visible', 'class_names_correct', 'relation_clear', 'overall_accept')
        for col in yn_cols:
            if row.get(col) not in ('yes', 'no'):
                raise ValueError(f'{bid}: {col} must be yes or no, got {row.get(col)!r}.')
        if row.get('perceived_relation') not in ('left', 'right', 'above', 'below', 'unclear'):
            raise ValueError(f'{bid}: invalid perceived_relation {row.get("perceived_relation")!r}.')
        if row['overall_accept'] == 'yes' and (
            any(row[c] != 'yes' for c in yn_cols[:3])
            or row['perceived_relation'] != row['expected_relation']
        ):
            raise ValueError(f'{bid}: overall_accept=yes contradicts sub-questions or perceived_relation.')
        if row['overall_accept'] == 'no' and not (row.get('exclusion_reason') or '').strip():
            raise ValueError(f'{bid}: overall_accept=no requires a non-empty exclusion_reason.')
    if len(annotators) != 1:
        raise ValueError(f'Expected a single consistent R2 annotator_id, got {sorted(annotators)}.')
    return {'valid': True, 'n': len(rows), 'annotator_id': next(iter(annotators))}


def export_r2():
    """Export a completely score-blind, R1-blind review packet for independent reviewer R2."""
    from PIL import Image, ImageDraw

    META.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    images_dir = PACKAGE / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    blanks = read_sheet(BLANK_TEMPLATE)
    mapping = read_json(MAPPING_FILE)
    selected = {r['source_image_id']: r for r in read_json(SELECTED_FILE)}
    by_blind = {item['blind_id']: item for item in mapping['items']}
    if len(blanks) != 100 or set(by_blind) != {r['blind_id'] for r in blanks}:
        raise ValueError('Mismatch between confirmation_100_blank.csv and blind_source_mapping.json.')

    blank_out = []
    licenses = []
    sections = []
    for row in blanks:
        bid = row['blind_id']
        item = by_blind[bid]
        src = selected[item['source_image_id']]
        source_path = ROOT / src['image_path']
        if sha256(source_path) != src['image_sha256']:
            raise ValueError(f'Source image checksum mismatch: {source_path}')
        with Image.open(source_path) as im:
            rgb = im.convert('RGB')
        crop = baseline_canvas(rgb, size=224)
        shown = rgb.copy()
        draw = ImageDraw.Draw(shown)
        for side, color in (('a', '#0072B2'), ('b', '#E69F00')):
            x, y, w, h = src[f'bbox_{side}']
            draw.rectangle((x, y, x + w, y + h), outline=color, width=4)
            draw.text((x, max(0, y - 12)), side.upper(), fill=color)
        shown.thumbnail((640, 480))
        shown.save(images_dir / f'{bid}_source.jpg', quality=92)
        crop.save(images_dir / f'{bid}_crop.png')

        rel_name = ('left', 'right', 'above', 'below')[int(src['target'])]
        caption = f"A: {src['category_a']} — B: {src['category_b']} — {rel_name}"
        sections.append(
            f'<section id="{bid}"><h2>{bid}</h2><p>{html.escape(caption)}</p>'
            f'<img src="images/{bid}_source.jpg" alt="original with A/B boxes"> '
            f'<img width="336" src="images/{bid}_crop.png" alt="actual CLIP 224x224 crop">'
            '<p>Trái: ảnh nguồn có box; phải: vùng nhìn 224×224 của CLIP (phóng to 1,5 lần). '
            'Chỉ đánh giá dựa trên vùng crop bên phải.</p></section>'
        )
        blank_out.append({
            'blind_id': bid,
            'object_a': src['category_a'],
            'object_b': src['category_b'],
            'expected_relation': rel_name,
            'annotator_id': '',
            'both_objects_visible': '',
            'class_names_correct': '',
            'relation_clear': '',
            'perceived_relation': '',
            'overall_accept': '',
            'exclusion_reason': '',
            'notes': '',
        })
        licenses.append({
            'blind_id': bid,
            'source_url': src['download_url'],
            'license': src['license'],
            'original_sha256': src['image_sha256'],
        })

    blank_meta_path = META / 'confirmation_100_R2_blank.csv'
    write_csv(blank_meta_path, blank_out)
    (META / 'README.md').write_text(R2_GUIDE_VI, encoding='utf-8')

    (PACKAGE / 'confirmation_100_R2_blank.csv').write_bytes(blank_meta_path.read_bytes())
    (PACKAGE / 'HUONG_DAN_R2.md').write_text(R2_GUIDE_VI, encoding='utf-8')
    write_json(PACKAGE / 'sources_and_licenses.json', licenses)
    (PACKAGE / 'index.html').write_text(
        '<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<title>Review độc lập lần 2 (R2) — 100 nguồn Set B</title>'
        '<style>body{font:16px Arial;margin:24px}section{border-bottom:1px solid #bbb;padding:12px}'
        'img{vertical-align:top;max-width:60%}</style>'
        '<h1>Rà soát độc lập lần 2 (R2) — 100 nguồn Set B</h1>'
        '<p>Gói hoàn toàn mù: không chứa quyết định của R1, không chứa scores/dự đoán CLIP. '
        'Đọc HUONG_DAN_R2.md trước khi chấm.</p>'
        + ''.join(sections)
        + '</html>',
        encoding='utf-8',
    )

    archive = PACKAGE / 'reviewer_r2_package.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            rel = p.relative_to(PACKAGE)
            allowed = rel.parts[0] == 'images' or rel.as_posix() in {
                'index.html',
                'confirmation_100_R2_blank.csv',
                'HUONG_DAN_R2.md',
                'sources_and_licenses.json',
            }
            if p.is_file() and allowed:
                z.write(p, rel.as_posix())

    manifest = {
        'study': 'human_review_iaa_set_b_v1',
        'exported_at_utc': now(),
        'n_items': len(blank_out),
        'r2_review_completed': False,
        'zero_r1_decisions_included': True,
        'zero_scores_included': True,
        'reviewer_zip_local_path': archive.relative_to(ROOT).as_posix(),
        'zip_sha256': sha256(archive),
        'html_sha256': sha256(PACKAGE / 'index.html'),
        'blank_csv_sha256': sha256(blank_meta_path),
        'source_mapping_sha256': sha256(MAPPING_FILE),
    }
    write_json(META / 'package_manifest.json', manifest)
    print(f'Exported R2 blind package: {archive} (n={len(blank_out)}, zip_sha256={manifest["zip_sha256"][:12]})', flush=True)
    return manifest


def validate_r2_file(review_path):
    blanks = read_sheet(META / 'confirmation_100_R2_blank.csv')
    rows = read_sheet(Path(review_path))
    res = validate_r2_rows(rows, blanks)
    res['sha256'] = sha256(Path(review_path))
    print(res, flush=True)
    return res


def freeze_r2(review_path):
    freeze_path = META / 'freeze.json'
    if freeze_path.exists():
        raise FileExistsError(f'R2 review already frozen: {freeze_path}')
    src = Path(review_path)
    blanks = read_sheet(META / 'confirmation_100_R2_blank.csv')
    history = META / 'history'
    history.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, history / 'received_confirmation_100_R2.csv')
    dest = META / 'confirmation_100_R2.csv'
    rows = normalize_sheet(src, dest, 'images')
    # Re-run strict R2 checks (especially distinct reviewer_id != R1).
    info = validate_r2_rows(rows, blanks)
    record = {
        'status': 'frozen',
        'frozen_at_utc': now(),
        'reviewer_id': info['annotator_id'],
        'n_items': len(rows),
        'accepted_count': sum(r['overall_accept'] == 'yes' for r in rows),
        'sha256': {
            p.relative_to(META).as_posix(): sha256(p)
            for p in sorted(META.rglob('*'))
            if p.is_file() and p.name != 'freeze.json'
        },
    }
    write_json(freeze_path, record)
    print(f'Frozen R2 review ({info["annotator_id"]}): {record["accepted_count"]}/100 accepted.', flush=True)
    return record


def _check_freezes():
    r1_manifest = read_json(R1_FREEZE)
    for name, digest in r1_manifest['sha256'].items():
        if sha256(R1_FREEZE.parent / name) != digest:
            raise ValueError(f'Frozen R1 file changed: {name}')
    r2_manifest = read_json(META / 'freeze.json')
    for name, digest in r2_manifest['sha256'].items():
        if sha256(META / name) != digest:
            raise ValueError(f'Frozen R2 file changed: {name}')
    return r1_manifest, r2_manifest


def calculate_iaa():
    r1_manifest, r2_manifest = _check_freezes()
    for run_dir in (REGION, HOLISTIC):
        verify_manifest(run_dir)
    region_audit(REGION)

    r1_rows = {r['blind_id']: r for r in read_sheet(R1_FINAL)}
    r2_rows = {r['blind_id']: r for r in read_sheet(META / 'confirmation_100_R2.csv')}
    if set(r1_rows) != set(r2_rows) or len(r1_rows) != 100:
        raise ValueError('R1 and R2 row IDs do not match 100 Set B items.')

    mapping = read_json(MAPPING_FILE)
    lookup = {item['blind_id']: item for item in mapping['items']}
    ordered_ids = [item['blind_id'] for item in mapping['items']]

    iaa_metrics = {
        field: multiclass_cohens_kappa([r1_rows[b][field] for b in ordered_ids], [r2_rows[b][field] for b in ordered_ids])
        for field in IAA_FIELDS
    }

    disagreements = []
    membership = []
    for bid in ordered_ids:
        a, b = r1_rows[bid], r2_rows[bid]
        item = lookup[bid]
        r1_acc = a['overall_accept'] == 'yes'
        r2_acc = b['overall_accept'] == 'yes'
        membership.append({
            'blind_id': bid,
            'source_image_id': item['source_image_id'],
            'group_id': item['group_id'],
            'r1_accept': r1_acc,
            'r2_accept': r2_acc,
            'both_accept': r1_acc and r2_acc,
            'r1_reason': a['exclusion_reason'],
            'r2_reason': b['exclusion_reason'],
        })
        diff_fields = [f for f in IAA_FIELDS if a[f] != b[f]]
        if diff_fields:
            disagreements.append({
                'blind_id': bid,
                'source_image_id': item['source_image_id'],
                'object_a': a['object_a'],
                'object_b': a['object_b'],
                'expected_relation': a['expected_relation'],
                'differing_fields': ';'.join(diff_fields),
                'r1_accept': a['overall_accept'],
                'r2_accept': b['overall_accept'],
                'r1_perceived': a['perceived_relation'],
                'r2_perceived': b['perceived_relation'],
                'r1_reason': a['exclusion_reason'],
                'r2_reason': b['exclusion_reason'],
                'r1_notes': a['notes'],
                'r2_notes': b['notes'],
            })

    consensus_keep = {m['group_id'] for m in membership if m['both_accept']}
    rows = read_json(REGION / 'samples.json')
    cfg = read_json(REGION / 'study_config.json')
    with np.load(REGION / 'scores.npz', allow_pickle=False) as z:
        scores = {k: z[k] for k in ('relational', 'padded_relational', 'holistic')}
    trials = read_json(HOLISTIC / 'trials.json')
    old_cfg = read_json(HOLISTIC / 'config.json')
    with np.load(HOLISTIC / 'scores.npz', allow_pickle=False) as z:
        old_scores = z['scores']

    ix = subset_indices(rows, consensus_keep)
    subset = [rows[i] for i in ix]
    consensus_bundle = {}
    for method, field in (('holistic', 'holistic'), ('native_half', 'relational'), ('padded_half', 'padded_relational')):
        result, predictions, pairs = summarize(subset, scores[field][ix], scores['holistic'][ix], 2, cfg)
        result['both_images_correct_ci95_percent'] = cluster_ci(
            [p['both_images_correct'] for p in pairs],
            [p['group_id'] for p in pairs],
            cfg['bootstrap_draws'],
            cfg['bootstrap_seed'],
        )
        result['correct_images'] = sum(p['correct'] for p in predictions)
        result['assignment_correct_sources'] = sum(p['correct'] for p in pairs)
        consensus_bundle[method] = result
    paired_consensus, _ = paired_metrics(trials, old_scores, old_cfg, keep_ids=consensus_keep)

    metrics = {
        'study': 'human_review_iaa_set_b_v1',
        'r1_reviewer': r1_manifest['reviewer'],
        'r2_reviewer': r2_manifest['reviewer_id'],
        'r1_accepted': sum(m['r1_accept'] for m in membership),
        'r2_accepted': sum(m['r2_accept'] for m in membership),
        'both_accepted': len(consensus_keep),
        'either_accepted': sum(m['r1_accept'] or m['r2_accept'] for m in membership),
        'both_rejected': sum((not m['r1_accept']) and (not m['r2_accept']) for m in membership),
        'iaa': iaa_metrics,
        'disagreement_count_any_field': len(disagreements),
        'disagreement_count_overall_accept': sum(m['r1_accept'] != m['r2_accept'] for m in membership),
        'consensus_sensitivity_no_new_inference': {
            'n_sources': len(consensus_keep),
            'region_comparison': consensus_bundle,
            'holistic_t2_o0': paired_consensus['t2_o0'],
        },
    }
    return metrics, disagreements, membership


def run():
    if OUTPUT.exists():
        raise FileExistsError(f'IAA output already exists; use audit: {OUTPUT}')
    metrics, disagreements, membership = calculate_iaa()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / 'metrics.json', metrics)
    write_csv(OUTPUT / 'disagreements.csv', disagreements)
    write_csv(OUTPUT / 'r1_r2_membership.csv', membership)
    inputs = [
        R1_FREEZE,
        R1_FINAL,
        META / 'freeze.json',
        META / 'confirmation_100_R2.csv',
        MAPPING_FILE,
        *(p for p in REGION.iterdir() if p.is_file()),
        *(p for p in HOLISTIC.iterdir() if p.is_file()),
    ]
    hashes, snapshot = source_snapshot(ROOT, [R1_FREEZE, R1_FINAL, META / 'freeze.json', META / 'confirmation_100_R2.csv'])
    manifest = {
        'status': 'completed',
        'created_at_utc': now(),
        'new_inference': False,
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in set(inputs)},
        'artifact_sha256': {
            name: sha256(OUTPUT / name)
            for name in ('metrics.json', 'disagreements.csv', 'r1_r2_membership.csv')
        },
        'source_sha256': hashes,
        'source_snapshot': snapshot,
    }
    write_json(OUTPUT / 'manifest.json', manifest)
    print(f'IAA computed: kappa(overall_accept)={metrics["iaa"]["overall_accept"]["cohens_kappa"]}, both_accepted={metrics["both_accepted"]}/100', flush=True)


def audit():
    manifest = read_json(OUTPUT / 'manifest.json')
    for name, digest in manifest['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed IAA input: {name}')
    for name, digest in manifest['artifact_sha256'].items():
        if sha256(OUTPUT / name) != digest:
            raise ValueError(f'Changed IAA artifact: {name}')
    metrics, disagreements, membership = calculate_iaa()
    close_enough(metrics, read_json(OUTPUT / 'metrics.json'))
    for fname, recs in (('disagreements.csv', disagreements), ('r1_r2_membership.csv', membership)):
        if read_sheet(OUTPUT / fname) != [{k: str(v) for k, v in r.items()} for r in recs]:
            raise ValueError(f'Recomputed IAA trace differs: {fname}')
    result = {
        'verified': True,
        'new_inference': False,
        'overall_accept_kappa': metrics['iaa']['overall_accept']['cohens_kappa'],
        'overall_accept_agreement_percent': metrics['iaa']['overall_accept']['raw_agreement_percent'],
        'both_accepted': metrics['both_accepted'],
    }
    print(result, flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['export-r2', 'validate', 'freeze', 'run', 'audit'])
    p.add_argument('--review', help='Path to filled R2 CSV for validate/freeze.')
    a = p.parse_args(argv)
    if a.action == 'export-r2':
        export_r2()
    elif a.action == 'validate':
        if not a.review:
            p.error('--review is required for validate')
        validate_r2_file(a.review)
    elif a.action == 'freeze':
        if not a.review:
            p.error('--review is required for freeze')
        freeze_r2(a.review)
    elif a.action == 'run':
        run()
    else:
        audit()


if __name__ == '__main__':
    main()
