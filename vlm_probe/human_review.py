"""Export a score-blind human review package; never supplies human decisions."""
import argparse
import csv
import html
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from .depth.common import ROOT, read_json, write_csv
from .depth.blinded_review import export_blinded_sheets
from .io_utils import sha256, write_json
from .spatial_diagnostics.natural_regions import baseline_canvas

META = ROOT / 'reviews/human_review_v2'
PACKAGE = ROOT / 'data/human_review_v2'


def export():
    META.mkdir(parents=True, exist_ok=True); PACKAGE.mkdir(parents=True, exist_ok=True)
    # Private key stays local; only blank sheets are committed.
    export_blinded_sheets(output_dir=PACKAGE / 'coordinator_private')
    key = read_json(PACKAGE / 'coordinator_private/answer_key_confirmation_100.json')
    selected = {r['source_image_id']: r for r in read_json(ROOT / 'results/natural_confirmation_selection_v1/selected.json')}
    images = PACKAGE / 'images'; images.mkdir(exist_ok=True)
    rows, licenses, sections = [], [], []
    for item in key['items']:
        bid = item['blind_id']; row = selected[item['source_image_id']]
        source = ROOT / row['image_path']
        if sha256(source) != row['image_sha256']: raise ValueError('Image source changed')
        with Image.open(source) as im: image = im.convert('RGB')
        crop = baseline_canvas(image)
        shown = image.copy(); draw = ImageDraw.Draw(shown)
        for side, color in [('a', '#0072B2'), ('b', '#E69F00')]:
            x, y, w, h = row[f'bbox_{side}']; draw.rectangle((x, y, x+w, y+h), outline=color, width=4)
            draw.text((x, max(0, y-12)), side.upper(), fill=color)
        shown.thumbnail((640, 480)); shown.save(images / f'{bid}_source.jpg', quality=92)
        crop.save(images / f'{bid}_crop.png')
        caption = f"A: {row['category_a']} — B: {row['category_b']} — {('left','right','above','below')[row['target']]}"
        sections.append(f'<section id="{bid}"><h2>{bid}</h2><p>{html.escape(caption)}</p>'
            f'<img src="images/{bid}_source.jpg" alt="original with A/B boxes"> '
            f'<img width="336" src="images/{bid}_crop.png" alt="actual CLIP center crop">'
            '<p>Trái: ảnh nguồn có box; phải: vùng nhìn 224×224 của CLIP (phóng to 1,5 lần). Điền cùng ID trong CSV.</p></section>')
        rows.append({'blind_id': bid, 'object_a': row['category_a'], 'object_b': row['category_b'],
                     'expected_relation': ('left','right','above','below')[row['target']],
                     'annotator_id': '', 'both_objects_visible': '', 'class_names_correct': '',
                     'relation_clear': '', 'perceived_relation': '', 'overall_accept': '', 'exclusion_reason': '', 'notes': ''})
        licenses.append({'blind_id': bid, 'source_url': row['download_url'], 'license': row['license'],
                         'original_sha256': row['image_sha256']})
    write_csv(META / 'confirmation_100_blank.csv', rows)
    classes = read_json(ROOT / 'results/full_vietnamese_cifar100_v1/multilingual_classes.json')
    translations = [{'class_id': r['class_id'], 'en_label': r['en_label'], 'vi_label_current': r['vi_label'],
        'annotator_id': '', 'status': '', 'proposed_vi_label': '', 'reason': '', 'reference': ''} for r in classes]
    write_csv(META / 'vietnamese_100_blank.csv', translations)
    write_json(META / 'status.json', {'image_review': 'pending_human', 'translation_review': 'pending_human',
        'image_items': 100, 'translation_items': 100,
        'source_sha256': {'natural_selection': sha256(ROOT/'results/natural_confirmation_selection_v1/selected.json'),
                          'translations': sha256(ROOT/'results/full_vietnamese_cifar100_v1/multilingual_classes.json')}})
    write_json(PACKAGE / 'sources_and_licenses.json', licenses)
    (PACKAGE / 'index.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<title>Review 100 ảnh</title><style>body{font:16px Arial;margin:24px}section{border-bottom:1px solid #bbb;padding:12px}img{vertical-align:top;max-width:60%}</style>'
        '<h1>Rà soát nhãn — 100 nguồn ảnh đã cố định</h1><p>Không có scores hoặc quyết định AI. Đọc HUONG_DAN.md trước khi chấm.</p>'
        + ''.join(sections) + '</html>', encoding='utf-8')
    for name in ['confirmation_100_blank.csv', 'vietnamese_100_blank.csv']:
        (PACKAGE / name).write_bytes((META / name).read_bytes())
    guide = ROOT / 'docs/human_review_assignment_v2.md'
    if guide.exists(): (PACKAGE / 'HUONG_DAN.md').write_bytes(guide.read_bytes())
    archive = PACKAGE / 'reviewer_package.zip'
    # The private key and filled decisions are excluded from the reviewer archive.
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            rel = p.relative_to(PACKAGE)
            allowed = (rel.parts[0] == 'images' or rel.as_posix() in {
                'index.html','confirmation_100_blank.csv','vietnamese_100_blank.csv',
                'HUONG_DAN.md','sources_and_licenses.json'})
            if p.is_file() and allowed:
                z.write(p, p.relative_to(PACKAGE).as_posix())
    write_json(META / 'package_manifest.json', {'reviewer_zip_local_path': archive.relative_to(ROOT).as_posix(),
        'sha256': sha256(archive), 'html_sha256': sha256(PACKAGE / 'index.html'),
        'contains': '100 annotated source previews, 100 actual center crops, blank sheets, instructions, licenses; no scores or AI review decisions',
        'human_review_completed': False})
    print(archive, flush=True)


def validate(path, kind):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    expected = read_json(ROOT / 'results/full_vietnamese_cifar100_v1/multilingual_classes.json')
    if len(rows) != 100: raise ValueError('Exactly 100 reviewed rows required')
    key = 'blind_id' if kind == 'images' else 'class_id'
    ids = [r[key] for r in rows]
    if len(set(ids)) != 100: raise ValueError('Duplicate IDs')
    with (META / ('confirmation_100_blank.csv' if kind == 'images' else 'vietnamese_100_blank.csv')).open(encoding='utf-8', newline='') as f:
        blanks = {r[key]: r for r in csv.DictReader(f)}
    if set(ids) != set(blanks): raise ValueError('Missing or unknown IDs')
    for row in rows:
        if not row['annotator_id'].strip(): raise ValueError('Human annotator ID required')
        fixed = ['object_a','object_b','expected_relation'] if kind == 'images' else ['en_label','vi_label_current']
        if any(row[k] != blanks[row[key]][k] for k in fixed): raise ValueError('Frozen source columns changed')
        if kind == 'translations':
            if row['status'] not in ['accepted', 'needs_revision', 'uncertain']: raise ValueError('Invalid translation status')
            if row['status'] != 'accepted' and not row['reason'].strip(): raise ValueError('Revision/uncertain reason required')
            if row['status'] == 'needs_revision' and not row['proposed_vi_label'].strip(): raise ValueError('Proposed label required')
        else:
            cols = ['both_objects_visible','class_names_correct','relation_clear','overall_accept']
            if any(row[k] not in ['yes','no'] for k in cols): raise ValueError('Use yes/no for all review questions')
            if row['perceived_relation'] not in ['left','right','above','below','unclear']: raise ValueError('Relation response required')
            if row['overall_accept'] == 'yes' and (any(row[k] != 'yes' for k in cols[:3]) or row['perceived_relation'] != row['expected_relation']):
                raise ValueError('Contradictory accepted annotation')
            if row['overall_accept'] == 'no' and not row['exclusion_reason'].strip(): raise ValueError('Rejection reason required')
    print({'valid_completed_review': True, 'kind': kind, 'n': len(rows), 'sha256': sha256(Path(path))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('export'); q = sub.add_parser('validate'); q.add_argument('path'); q.add_argument('--kind', choices=['images','translations'], required=True)
    a = p.parse_args(); export() if a.action == 'export' else validate(a.path, a.kind)
