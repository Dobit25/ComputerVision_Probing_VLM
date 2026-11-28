"""Pre-registered unseen held-out COCO Set C (T3): disjoint selection, 2-reviewer blind review,
single-shot evaluation (Holistic vs Native/Padded Half vs Oracle Bbox Geometric Baseline), and offline audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import random
import shutil
import struct
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..depth import natural
from ..depth.common import ROOT, Engine, ImageFiles, cluster_ci, finish, read_json, write_csv
from ..depth.confirmation import paired_metrics, wilson
from ..depth.design import HFLIP, spatial_trials
from ..extension_artifacts import begin_extension, close_enough
from ..human_review_iaa import multiclass_cohens_kappa, validate_r2_rows
from ..io_utils import sha256, write_json
from .natural_regions import baseline_canvas, pad_half, summarize
from .quadrant_mitigation import _box_half_overlap, crop_half_canvases, relational_scores_from_regions

CONFIG = ROOT / 'configs/natural_unseen_holdout_v1.json'
REVIEWS = ROOT / 'reviews/natural_unseen_holdout_v1'
PACKAGE = ROOT / 'data/natural_unseen_holdout_v1'
NATURAL_DATA = ROOT / 'data/probing_depth_v1/natural'

REVIEW_FIELDS = [
    'blind_id', 'object_a', 'object_b', 'expected_relation', 'annotator_id',
    'both_objects_visible', 'class_names_correct', 'relation_clear',
    'perceived_relation', 'overall_accept', 'exclusion_reason', 'notes',
]

ADJ_FIELDS = [
    'blind_id', 'object_a', 'object_b', 'expected_relation',
    'ra_annotator', 'ra_accept', 'ra_perceived', 'ra_reason', 'ra_notes',
    'rb_annotator', 'rb_accept', 'rb_perceived', 'rb_reason', 'rb_notes',
    'adjudicator_id', 'final_accept', 'adjudication_reason', 'adjudication_notes',
]

SET_C_GUIDE_VI = """# Hướng dẫn chấm mù kép (Dual Blind Review) — Set C (160 nguồn COCO mới)

1. **Mục đích & Nguyên tắc mù:** Hai thành viên độc lập (**Slot RA** và **Slot RB**) tự chấm riêng rẽ trên cùng 160 nguồn ảnh COCO mới hoàn toàn (`UNSEEN_C_001` .. `UNSEEN_C_160`).
   - Hai người chấm **không trao đổi kết quả với nhau** trước khi điều phối viên chạy phân xử (`adjudicate`).
   - Không có bất kỳ điểm số CLIP nào được sinh ra trước khi phiếu được khóa (`freeze`).
2. **Cách xem ảnh:** Giải nén `reviewer_package_set_c.zip` và mở `index.html`. Mỗi mục hiển thị ảnh nguồn có box `A` (xanh dương)/`B` (cam) bên trái và **đúng crop 224×224 của CLIP** bên phải. Phải quyết định dựa trên vùng crop bên phải.
3. **Điền đúng file của mình (`unseen_c_blank_RA.csv` hoặc `unseen_c_blank_RB.csv`):**
   - Ghi mã người chấm vào `annotator_id` (hai người phải dùng hai mã khác nhau, ví dụ tên riêng không dấu).
   - Giữ nguyên 160 dòng và 4 cột đầu (`blind_id`, `object_a`, `object_b`, `expected_relation`).
   - Điền `yes`/`no` cho `both_objects_visible`, `class_names_correct`, `relation_clear`; điền `left`/`right`/`above`/`below`/`unclear` cho `perceived_relation`.
   - `overall_accept = yes` chỉ khi cả 3 câu đầu đều `yes` và `perceived_relation == expected_relation`. Nếu `no`, ghi rõ `exclusion_reason`.
4. Lưu file CSV UTF-8 và gửi lại để tính IAA, phân xử bất đồng (nếu có) và đóng băng trước khi chạy GPU 1 lượt duy nhất.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv_sig(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_seen_exclusions(cfg):
    """Load all previously screened source_image_ids, image_sha256s, and paths from Set A & Set B."""
    seen_ids = set()
    seen_hashes = set()
    seen_rows = []
    for rel_path in cfg['exclude_files']:
        rows = read_json(ROOT / rel_path)
        for r in rows:
            sid = int(r['source_image_id'])
            if sid not in seen_ids:
                seen_ids.add(sid)
                seen_rows.append(r)
            if r.get('image_sha256'):
                seen_hashes.add(r['image_sha256'])
    return seen_ids, seen_hashes, seen_rows


def load_train2014_annotations(cfg):
    """Stream annotations/instances_train2014.json safely without touching val2014 provenance."""
    NATURAL_DATA.mkdir(parents=True, exist_ok=True)
    member = cfg['annotation_member']
    member_name = member.split('/')[-1]
    path = NATURAL_DATA / member_name
    prov = NATURAL_DATA / (member_name + '.source.json')
    if path.exists() and prov.exists():
        if sha256(path) != read_json(prov)['sha256']:
            raise ValueError('COCO train2014 annotations checksum changed.')
        return read_json(path), read_json(prov)

    url = cfg['annotation_url']
    import urllib.request
    req = urllib.request.Request(url, headers={'Range': 'bytes=-65536', 'User-Agent': 'VLM-Probe-Research/1.0'})
    with urllib.request.urlopen(req, timeout=45) as resp:
        tail = resp.read()
        total = int(resp.headers['Content-Range'].split('/')[-1])
    with zipfile.ZipFile(io.BytesIO(tail)) as z:
        info = z.getinfo(member)
    offset = info.header_offset + total - len(tail)
    header, _ = natural.get_bytes(url, offset, offset + 29)
    parts = struct.unpack('<4s5H3I2H', header)
    if parts[0] != b'PK\x03\x04' or info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError('Unsupported ZIP member.')
    start = offset + 30 + parts[-2] + parts[-1]
    blocks = [
        (s, min(start + info.compress_size - 1, s + 2 * 1024 * 1024 - 1))
        for s in range(start, start + info.compress_size, 2 * 1024 * 1024)
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        chunks = list(pool.map(lambda pair: natural.get_bytes(url, *pair)[0], blocks))
    payload = zlib.decompress(b''.join(chunks), -15)
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise ValueError('COCO train2014 ZIP member CRC/size mismatch.')
    path.write_bytes(payload)
    prov_obj = {
        'url': url,
        'member': member,
        'archive_bytes': total,
        'member_crc32': info.CRC,
        'sha256': sha256(path),
        'terms': 'https://cocodataset.org/#termsofuse',
        'annotation_license': 'CC BY 4.0',
    }
    write_json(prov, prov_obj)
    print(f'COCO train2014 annotations verified: {len(payload)} bytes', flush=True)
    return json.loads(payload), prov_obj


def dhash_from_path(path, mirror=False):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = im.convert('RGB')
        side = min(im.size)
        x, y = (im.width - side) / 2, (im.height - side) / 2
        im = im.crop((x, y, x + side, y + side))
        if mirror:
            im = ImageOps.mirror(im)
        v = np.asarray(im.convert('L').resize((9, 8)))
    return v[:, 1:] > v[:, :-1]


def download_set_c_image(row):
    from PIL import Image
    path = PACKAGE / 'raw_images' / row['file_name']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload, _ = natural.get_bytes(row['download_url'])
        path.write_bytes(payload)
    with Image.open(path) as im:
        if im.size != (row['width'], row['height']):
            raise ValueError(f'COCO image dimensions differ for {row["file_name"]}.')
        im.verify()
    digest = sha256(path)
    if row.get('image_sha256') and digest != row['image_sha256']:
        raise ValueError(f'Downloaded Set C image differs from recorded hash: {path}')
    return {**row, 'image_path': path.relative_to(ROOT).as_posix(), 'image_sha256': digest}


def select_disjoint_candidates(possible, seen_ids, seen_hashes, seen_dhash_orig, seen_dhash_flip,
                               download_fn, dhash_fn, max_dist, per_relation):
    """Select balanced left/right candidates strictly disjoint by ID, SHA-256, and perceptual dHash."""
    kept = []
    duplicates = []
    quotas = Counter()
    kept_dhashes = []

    for cand in sorted(possible, key=lambda r: r['rank']):
        if quotas[0] >= per_relation and quotas[1] >= per_relation:
            break
        if quotas[cand['target']] >= per_relation:
            continue
        if cand['source_image_id'] in seen_ids:
            duplicates.append({'source_image_id': cand['source_image_id'], 'reason': 'excluded_id'})
            continue
        row = download_fn(cand)
        if row['image_sha256'] in seen_hashes:
            duplicates.append({'source_image_id': row['source_image_id'], 'reason': 'sha256_match_seen'})
            continue
        h = dhash_fn(ROOT / row['image_path'], mirror=False)
        if len(seen_dhash_orig):
            d_orig = np.count_nonzero(seen_dhash_orig != h, axis=(1, 2))
            d_flip = np.count_nonzero(seen_dhash_flip != h, axis=(1, 2))
            dists = np.minimum(d_orig, d_flip)
            best_idx = int(dists.argmin())
            if int(dists[best_idx]) <= max_dist:
                duplicates.append({
                    'source_image_id': row['source_image_id'],
                    'reason': 'near_duplicate_seen',
                    'distance': int(dists[best_idx]),
                })
                continue
        match_c = next(
            ((sid, int(min(np.count_nonzero(h != ho), np.count_nonzero(h != hf))))
             for sid, ho, hf in kept_dhashes
             if min(np.count_nonzero(h != ho), np.count_nonzero(h != hf)) <= max_dist),
            None,
        )
        if match_c is not None:
            duplicates.append({
                'source_image_id': row['source_image_id'],
                'reason': 'near_duplicate_within_set_c',
                'matches_source_id': match_c[0],
                'distance': match_c[1],
            })
            continue
        kept_dhashes.append((row['source_image_id'], h, dhash_fn(ROOT / row['image_path'], mirror=True)))
        seen_ids.add(row['source_image_id'])
        seen_hashes.add(row['image_sha256'])
        quotas[row['target']] += 1
        kept.append(row)
    return kept, duplicates, dict(quotas)


def prepare(cfg):
    """Prepare 160 disjoint Set C candidates and dual-reviewer blind review packets (no CLIP model)."""
    if (REVIEWS / 'candidates.json').exists():
        raise FileExistsError(f'Set C candidates already prepared: {REVIEWS / "candidates.json"}')
    from PIL import Image, ImageDraw

    base = read_json(ROOT / cfg['base_config'])
    seen_ids, seen_hashes, seen_rows = load_seen_exclusions(cfg)
    annotation, prov_obj = load_train2014_annotations(cfg)

    filtered_ann = {
        **annotation,
        'images': [im for im in annotation['images'] if im['id'] not in seen_ids],
    }
    nc = {
        **base['natural'],
        'image_split': cfg['image_split'],
        'allowed_relations': cfg['allowed_relations'],
        'selection_seed': cfg['selection_seed'],
        'candidate_reserve_per_relation': 100000,
    }
    possible, counts = natural.candidates(filtered_ann, nc)

    # Precompute dHash on all 578 seen images (Set A + Set B).
    for r in seen_rows:
        p = ROOT / r['image_path']
        if not p.exists():
            raise FileNotFoundError(f'Seen Set A/B image missing for duplicate screening: {p}')
        seen_hashes.add(sha256(p))
    seen_dhash_orig = np.stack([dhash_from_path(ROOT / r['image_path'], mirror=False) for r in seen_rows])
    seen_dhash_flip = np.stack([dhash_from_path(ROOT / r['image_path'], mirror=True) for r in seen_rows])

    # Pre-fetch top candidates in small waves using ThreadPoolExecutor until 80+80 clean sources collected.
    per_rel = cfg['review_pool_per_relation']
    by_rel = {0: [], 1: []}
    for row in sorted(possible, key=lambda r: r['rank']):
        if len(by_rel[row['target']]) < per_rel + 40:
            by_rel[row['target']].append(row)
    prefetch = sorted(by_rel[0] + by_rel[1], key=lambda r: r['rank'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download_set_c_image, prefetch))

    kept, duplicates, quotas = select_disjoint_candidates(
        possible, set(seen_ids), set(seen_hashes),
        seen_dhash_orig, seen_dhash_flip,
        download_set_c_image, dhash_from_path,
        cfg['near_duplicate_dhash_max_distance'], per_rel,
    )
    if quotas.get(0, 0) < per_rel or quotas.get(1, 0) < per_rel:
        raise RuntimeError(f'Insufficient clean disjoint Set C candidates: {quotas}')

    # Interleave deterministically for blind reviewers.
    shuffled = list(kept)
    random.Random(cfg['blind_shuffle_seed']).shuffle(shuffled)

    REVIEWS.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    rev_img_dir = PACKAGE / 'review_images'
    rev_img_dir.mkdir(parents=True, exist_ok=True)
    (PACKAGE / 'coordinator_private').mkdir(parents=True, exist_ok=True)

    blank_rows, mapping_items, licenses, sections = [], [], [], []
    for idx, row in enumerate(shuffled):
        bid = f'UNSEEN_C_{idx + 1:03d}'
        row['blind_id'] = bid
        row['review_index'] = idx
        rel_name = ('left', 'right')[int(row['target'])]

        with Image.open(ROOT / row['image_path']) as im:
            rgb = im.convert('RGB')
        crop = baseline_canvas(rgb, size=cfg['canvas_size'])
        shown = rgb.copy()
        draw = ImageDraw.Draw(shown)
        for side, color in (('a', '#0072B2'), ('b', '#E69F00')):
            x, y, w, h = row[f'bbox_{side}']
            draw.rectangle((x, y, x + w, y + h), outline=color, width=4)
            draw.text((x, max(0, y - 12)), side.upper(), fill=color)
        shown.thumbnail((640, 480))
        shown.save(rev_img_dir / f'{bid}_source.jpg', quality=92)
        crop.save(rev_img_dir / f'{bid}_crop.png')

        caption = f"A: {row['category_a']} — B: {row['category_b']} — {rel_name}"
        sections.append(
            f'<section id="{bid}"><h2>{bid}</h2><p>{html.escape(caption)}</p>'
            f'<img src="review_images/{bid}_source.jpg" alt="original with A/B boxes"> '
            f'<img width="336" src="review_images/{bid}_crop.png" alt="actual CLIP 224x224 crop">'
            '<p>Trái: ảnh nguồn có box; phải: vùng nhìn 224×224 của CLIP (phóng to 1,5 lần). '
            'Quyết định dựa trên crop bên phải.</p></section>'
        )
        blank_rows.append({
            'blind_id': bid,
            'object_a': row['category_a'],
            'object_b': row['category_b'],
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
        mapping_items.append({
            'blind_id': bid,
            'source_image_id': row['source_image_id'],
            'rank': row['rank'],
            'target': row['target'],
            'image_path': row['image_path'],
            'image_sha256': row['image_sha256'],
        })
        licenses.append({
            'blind_id': bid,
            'source_url': row['download_url'],
            'license': row['license'],
            'original_sha256': row['image_sha256'],
        })

    cand_meta = {
        'study': cfg['study'],
        'prepared_at_utc': now(),
        'config_sha256': sha256(CONFIG),
        'protocol_sha256': sha256(ROOT / cfg['protocol']),
        'annotation_source': prov_obj,
        'excluded_seen_id_count': len(seen_ids),
        'zero_overlap_verified': True,
        'scores_used': False,
        'quotas': quotas,
        'candidate_pool_counts': counts,
        'candidates': shuffled,
    }
    write_json(REVIEWS / 'candidates.json', cand_meta)
    write_json(REVIEWS / 'duplicate_screen.json', {'screened_duplicates': duplicates})
    write_csv(REVIEWS / 'unseen_c_blank_RA.csv', blank_rows)
    write_csv(REVIEWS / 'unseen_c_blank_RB.csv', blank_rows)
    (REVIEWS / 'README.md').write_text(SET_C_GUIDE_VI, encoding='utf-8')

    write_json(PACKAGE / 'coordinator_private/blind_mapping.json', {'items': mapping_items})
    write_json(PACKAGE / 'sources_and_licenses.json', licenses)
    (PACKAGE / 'HUONG_DAN_SET_C.md').write_text(SET_C_GUIDE_VI, encoding='utf-8')
    (PACKAGE / 'unseen_c_blank_RA.csv').write_bytes((REVIEWS / 'unseen_c_blank_RA.csv').read_bytes())
    (PACKAGE / 'unseen_c_blank_RB.csv').write_bytes((REVIEWS / 'unseen_c_blank_RB.csv').read_bytes())
    (PACKAGE / 'index.html').write_text(
        '<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<title>Review mù kép Set C — 160 nguồn COCO mới</title>'
        '<style>body{font:16px Arial;margin:24px}section{border-bottom:1px solid #bbb;padding:12px}'
        'img{vertical-align:top;max-width:60%}</style>'
        '<h1>Rà soát mù kép Set C — 160 nguồn COCO train2014 mới hoàn toàn</h1>'
        '<p>Không có bất kỳ điểm số CLIP nào tồn tại trước khi phiếu được khóa. '
        'Đọc HUONG_DAN_SET_C.md trước khi chấm.</p>'
        + ''.join(sections)
        + '</html>',
        encoding='utf-8',
    )

    archive = PACKAGE / 'reviewer_package_set_c.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            rel = p.relative_to(PACKAGE)
            allowed = rel.parts[0] == 'review_images' or rel.as_posix() in {
                'index.html',
                'unseen_c_blank_RA.csv',
                'unseen_c_blank_RB.csv',
                'HUONG_DAN_SET_C.md',
                'sources_and_licenses.json',
            }
            if p.is_file() and allowed:
                z.write(p, rel.as_posix())

    pkg_manifest = {
        'study': cfg['study'],
        'prepared_at_utc': cand_meta['prepared_at_utc'],
        'n_candidates': len(shuffled),
        'quotas': quotas,
        'review_completed': False,
        'scores_used': False,
        'zero_overlap_verified': True,
        'reviewer_zip_local_path': archive.relative_to(ROOT).as_posix(),
        'zip_sha256': sha256(archive),
        'html_sha256': sha256(PACKAGE / 'index.html'),
        'ra_blank_sha256': sha256(REVIEWS / 'unseen_c_blank_RA.csv'),
        'rb_blank_sha256': sha256(REVIEWS / 'unseen_c_blank_RB.csv'),
    }
    write_json(REVIEWS / 'package_manifest.json', pkg_manifest)
    print(f'Prepared Set C: {len(shuffled)} candidates {quotas}, zip={archive}', flush=True)
    return pkg_manifest


def validate_slot(review_path, slot):
    if slot not in ('RA', 'RB'):
        raise ValueError('Slot must be RA or RB.')
    blanks = read_csv_sig(REVIEWS / f'unseen_c_blank_{slot}.csv')
    rows = read_csv_sig(review_path)
    # Reuse strict per-row validation (allows any non-empty annotator_id != R1; check pair distinctness in adjudicate/freeze).
    info = validate_r2_rows(rows, blanks)
    info['slot'] = slot
    info['sha256'] = sha256(Path(review_path))
    print(info, flush=True)
    return info


def _load_and_check_pair(ra_path, rb_path):
    blanks_a = read_csv_sig(REVIEWS / 'unseen_c_blank_RA.csv')
    blanks_b = read_csv_sig(REVIEWS / 'unseen_c_blank_RB.csv')
    ra_rows = read_csv_sig(ra_path)
    rb_rows = read_csv_sig(rb_path)
    info_a = validate_r2_rows(ra_rows, blanks_a)
    info_b = validate_r2_rows(rb_rows, blanks_b)
    if info_a['annotator_id'].strip().lower() == info_b['annotator_id'].strip().lower():
        raise ValueError(f'RA and RB must have distinct annotator_ids, both got {info_a["annotator_id"]!r}.')
    return {r['blind_id']: r for r in ra_rows}, {r['blind_id']: r for r in rb_rows}, info_a['annotator_id'], info_b['annotator_id']


def adjudicate(ra_path, rb_path):
    ra_map, rb_map, ann_a, ann_b = _load_and_check_pair(ra_path, rb_path)
    ordered = [r['blind_id'] for r in read_csv_sig(REVIEWS / 'unseen_c_blank_RA.csv')]
    iaa = {
        field: multiclass_cohens_kappa([ra_map[b][field] for b in ordered], [rb_map[b][field] for b in ordered])
        for field in ('overall_accept', 'both_objects_visible', 'class_names_correct', 'relation_clear', 'perceived_relation')
    }
    adj_rows = []
    for bid in ordered:
        a, b = ra_map[bid], rb_map[bid]
        if a['overall_accept'] != b['overall_accept']:
            adj_rows.append({
                'blind_id': bid,
                'object_a': a['object_a'],
                'object_b': a['object_b'],
                'expected_relation': a['expected_relation'],
                'ra_annotator': ann_a,
                'ra_accept': a['overall_accept'],
                'ra_perceived': a['perceived_relation'],
                'ra_reason': a['exclusion_reason'],
                'ra_notes': a['notes'],
                'rb_annotator': ann_b,
                'rb_accept': b['overall_accept'],
                'rb_perceived': b['perceived_relation'],
                'rb_reason': b['exclusion_reason'],
                'rb_notes': b['notes'],
                'adjudicator_id': '',
                'final_accept': '',
                'adjudication_reason': '',
                'adjudication_notes': '',
            })
    if adj_rows:
        write_csv(REVIEWS / 'adjudication_blank.csv', adj_rows)
    summary = {
        'n_candidates': len(ordered),
        'annotator_ra': ann_a,
        'annotator_rb': ann_b,
        'both_accept': sum(ra_map[b]['overall_accept'] == 'yes' and rb_map[b]['overall_accept'] == 'yes' for b in ordered),
        'both_reject': sum(ra_map[b]['overall_accept'] == 'no' and rb_map[b]['overall_accept'] == 'no' for b in ordered),
        'disagreements': len(adj_rows),
        'overall_accept_iaa': iaa['overall_accept'],
    }
    write_json(REVIEWS / 'pre_adjudication_iaa.json', {'iaa': iaa, 'summary': summary})
    print(summary, flush=True)
    return summary


def select_balanced_cohort(accepted_rows, target_sources, target_per_relation):
    """Select up to target_sources deterministically by rank, targeting 50/50 left/right balance."""
    ranked = sorted(accepted_rows, key=lambda r: r['rank'])
    left = [r for r in ranked if int(r['target']) == 0]
    right = [r for r in ranked if int(r['target']) == 1]
    chosen = left[:target_per_relation] + right[:target_per_relation]
    if len(chosen) < target_sources:
        chosen_ids = {r['source_image_id'] for r in chosen}
        rest = [r for r in ranked if r['source_image_id'] not in chosen_ids]
        chosen.extend(rest[: target_sources - len(chosen)])
    return sorted(chosen, key=lambda r: r['rank'])


def freeze(ra_path, rb_path, adjudication_path=None, cfg=None):
    cfg = cfg or read_json(CONFIG)
    if (REVIEWS / 'freeze.json').exists():
        raise FileExistsError('Set C review already frozen.')
    ra_map, rb_map, ann_a, ann_b = _load_and_check_pair(ra_path, rb_path)
    ordered = [r['blind_id'] for r in read_csv_sig(REVIEWS / 'unseen_c_blank_RA.csv')]
    disagreements = [b for b in ordered if ra_map[b]['overall_accept'] != rb_map[b]['overall_accept']]
    adj_map = {}
    if disagreements:
        if not adjudication_path:
            raise ValueError(f'{len(disagreements)} disagreements exist; provide --adjudication CSV.')
        adj_rows = read_csv_sig(adjudication_path)
        adj_map = {r['blind_id']: r for r in adj_rows}
        if set(adj_map) != set(disagreements):
            raise ValueError('Adjudication CSV IDs do not match exact disagreement set.')
        for bid, r in adj_map.items():
            if not (r.get('adjudicator_id') or '').strip():
                raise ValueError(f'{bid}: adjudicator_id required.')
            if r.get('final_accept') not in ('yes', 'no'):
                raise ValueError(f'{bid}: final_accept must be yes or no.')
            if not (r.get('adjudication_reason') or '').strip():
                raise ValueError(f'{bid}: adjudication_reason required.')

    cand_meta = read_json(REVIEWS / 'candidates.json')
    by_blind = {c['blind_id']: c for c in cand_meta['candidates']}

    accepted_candidates = []
    final_decisions = []
    for bid in ordered:
        a, b = ra_map[bid], rb_map[bid]
        if bid in adj_map:
            final_yes = adj_map[bid]['final_accept'] == 'yes'
            source_of_decision = 'adjudicated'
        else:
            final_yes = a['overall_accept'] == 'yes'
            source_of_decision = 'consensus'
        final_decisions.append({
            'blind_id': bid,
            'source_image_id': by_blind[bid]['source_image_id'],
            'ra_accept': a['overall_accept'],
            'rb_accept': b['overall_accept'],
            'decision_type': source_of_decision,
            'final_accept': final_yes,
        })
        if final_yes:
            accepted_candidates.append(by_blind[bid])

    if len(accepted_candidates) < cfg['minimum_sources']:
        raise RuntimeError(
            f'Only {len(accepted_candidates)} accepted sources (< minimum {cfg["minimum_sources"]}); '
            'stop Set C without inference or commit a pre-inference amendment.'
        )

    selected = select_balanced_cohort(accepted_candidates, cfg['target_sources'], cfg['target_per_relation'])
    shutil.copyfile(ra_path, REVIEWS / 'unseen_c_filled_RA.csv')
    shutil.copyfile(rb_path, REVIEWS / 'unseen_c_filled_RB.csv')
    if adjudication_path:
        shutil.copyfile(adjudication_path, REVIEWS / 'adjudication_filled.csv')

    iaa = {
        field: multiclass_cohens_kappa([ra_map[b][field] for b in ordered], [rb_map[b][field] for b in ordered])
        for field in ('overall_accept', 'both_objects_visible', 'class_names_correct', 'relation_clear', 'perceived_relation')
    }
    write_json(REVIEWS / 'selected.json', selected)
    write_json(REVIEWS / 'final_decisions.json', final_decisions)
    record = {
        'status': 'frozen',
        'frozen_at_utc': now(),
        'config_sha256': sha256(CONFIG),
        'protocol_sha256': sha256(ROOT / cfg['protocol']),
        'annotators': {'RA': ann_a, 'RB': ann_b},
        'n_candidates': len(ordered),
        'n_accepted_pool': len(accepted_candidates),
        'n_selected': len(selected),
        'selected_by_target': dict(Counter(int(r['target']) for r in selected)),
        'iaa_before_adjudication': iaa,
        'sha256': {
            p.relative_to(REVIEWS).as_posix(): sha256(p)
            for p in sorted(REVIEWS.rglob('*'))
            if p.is_file() and p.name != 'freeze.json'
        },
    }
    write_json(REVIEWS / 'freeze.json', record)
    print(f'Frozen Set C: {len(selected)} sources selected (from {len(accepted_candidates)} accepted).', flush=True)
    return record


def frozen_set_c(cfg):
    record = read_json(REVIEWS / 'freeze.json')
    if record['status'] != 'frozen':
        raise RuntimeError('Set C review is not frozen.')
    if record['config_sha256'] != sha256(CONFIG) or record['protocol_sha256'] != sha256(ROOT / cfg['protocol']):
        raise ValueError('Config or protocol changed after Set C review freeze.')
    for name, digest in record['sha256'].items():
        if sha256(REVIEWS / name) != digest:
            raise ValueError(f'Frozen Set C review file changed: {name}')
    selected = read_json(REVIEWS / 'selected.json')
    seen_ids, seen_hashes, _ = load_seen_exclusions(cfg)
    sel_ids = {r['source_image_id'] for r in selected}
    if seen_ids & sel_ids:
        raise ValueError('Set C selection overlaps Set A/B source_image_ids!')
    if any(r['image_sha256'] in seen_hashes for r in selected):
        raise ValueError('Set C selection overlaps Set A/B image_sha256!')
    return record, selected


def bbox_geometric_baseline(rows, canvas_size=224):
    """Compute Oracle COCO Bbox centroid and ideal half-canvas overlap accuracy on the 224x224 crops."""
    from .natural_recognition import canvas_box
    centroid_correct = []
    half_correct = []
    same_half_flags = []
    groups = []
    for r in rows:
        ba = canvas_box(r['bbox_a'], r['width'], r['height'], canvas_size)
        bb = canvas_box(r['bbox_b'], r['width'], r['height'], canvas_size)
        # canvas_box returns (x1, y1, x2, y2); mirror horizontally if horizontal_flip
        if r['transform'] == 'horizontal_flip':
            ba = (canvas_size - ba[2], ba[1], canvas_size - ba[0], ba[3])
            bb = (canvas_size - bb[2], bb[1], canvas_size - bb[0], bb[3])
        cx_a = 0.5 * (ba[0] + ba[2])
        cx_b = 0.5 * (bb[0] + bb[2])
        # target 0 = A left of B (cx_a < cx_b), target 1 = A right of B (cx_a > cx_b)
        cent_pred = 0 if cx_a < cx_b else (1 if cx_a > cx_b else -1)
        centroid_correct.append(cent_pred == int(r['target']))

        box_a = [ba[0], ba[1], max(1e-6, ba[2] - ba[0]), max(1e-6, ba[3] - ba[1])]
        box_b = [bb[0], bb[1], max(1e-6, bb[2] - bb[0]), max(1e-6, bb[3] - bb[1])]
        sa = _box_half_overlap(box_a, canvas_size, canvas_size)
        sb = _box_half_overlap(box_b, canvas_size, canvas_size)
        rel = relational_scores_from_regions(sa, sb)[:2]
        half_pred = int(rel.argmax()) if rel[0] != rel[1] else -1
        half_correct.append(half_pred == int(r['target']))
        same_half_flags.append(bool((sa[0] > 0.99 and sb[0] > 0.99) or (sa[1] > 0.99 and sb[1] > 0.99)))
        groups.append(r['group_id'])
    return {
        'n_images': len(rows),
        'n_sources': len(set(groups)),
        'bbox_centroid_2way_accuracy_percent': float(100.0 * np.mean(centroid_correct)),
        'ideal_half_canvas_2way_accuracy_percent': float(100.0 * np.mean(half_correct)),
        'both_objects_same_half_percent': float(100.0 * np.mean(same_half_flags)),
    }


def compute_set_c_metrics(rows, holistic_scores, native_scores, padded_scores, cfg):
    """Recompute all Set C metrics directly from saved score arrays and geometric boxes."""
    base_cfg = read_json(ROOT / cfg['base_config'])
    trials = spatial_trials(rows, base_cfg, 'photo')
    t, o = cfg['holistic_template'], cfg['holistic_order']
    sel_trials = [tr for tr in trials if tr['template'] == t and tr['order'] == o]
    if len(sel_trials) != len(rows):
        raise ValueError('Holistic trial alignment mismatch.')
    paired_holistic, _ = paired_metrics(sel_trials, holistic_scores, cfg) if False else (None, None)

    hol_res, hol_pred, hol_pairs = summarize(rows, holistic_scores, holistic_scores, 2, cfg)
    nat_res, nat_pred, nat_pairs = summarize(rows, native_scores, holistic_scores, 2, cfg)
    pad_res, pad_pred, pad_pairs = summarize(rows, padded_scores, holistic_scores, 2, cfg)

    for res, pairs in ((hol_res, hol_pairs), (nat_res, nat_pairs), (pad_res, pad_pairs)):
        k = sum(p['correct'] for p in pairs)
        n = len(pairs)
        res['horizontal_assignment']['wilson_ci95_percent'] = wilson(k, n)
        res['horizontal_assignment']['binomial_p_greater_0_5'] = float(sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n))
        res['both_images_correct_ci95_percent'] = cluster_ci(
            [p['both_images_correct'] for p in pairs],
            [p['group_id'] for p in pairs],
            cfg['bootstrap_draws'],
            cfg['bootstrap_seed'],
        )

    metrics = {
        'study': cfg['study'],
        'separate_from_set_a_and_b': True,
        'n_sources': len({r['group_id'] for r in rows}),
        'n_images': len(rows),
        'holistic_t2_o0': hol_res,
        'native_half': nat_res,
        'padded_half': pad_res,
        'bbox_geometric_baseline': bbox_geometric_baseline(rows, cfg['canvas_size']),
    }
    return metrics, nat_pred, nat_pairs


def run(cfg, smoke=False, run_name=None):
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset

    record, selected = frozen_set_c(cfg)
    if len(selected) < cfg['minimum_sources'] and not smoke:
        raise RuntimeError(f'Only {len(selected)} Set C sources (< {cfg["minimum_sources"]}).')
    if smoke:
        selected = selected[:8]
    for r in selected:
        download_set_c_image(r)
    rows = natural.make_images(selected)

    base_cfg = read_json(ROOT / cfg['base_config'])
    name = run_name or f'{"smoke" if smoke else "full"}_natural_unseen_holdout_v1'
    inputs = [REVIEWS / 'freeze.json', REVIEWS / 'selected.json', ROOT / cfg['protocol']]
    out, _, manifest = begin_extension('natural_unseen_holdout', CONFIG, inputs=inputs, smoke=smoke, run_name=name)
    if record['frozen_at_utc'] >= manifest['started_at_utc']:
        raise ValueError('Set C review must be frozen before inference starts.')
    engine = Engine(out, base_cfg, manifest)

    # 1. Holistic T2/o0
    trials = [tr for tr in spatial_trials(rows, base_cfg, 'photo')
              if tr['template'] == cfg['holistic_template'] and tr['order'] == cfg['holistic_order']]
    hol_pool = list(dict.fromkeys(c for tr in trials for c in tr['captions']))
    hol_ix = {c: i for i, c in enumerate(hol_pool)}
    hol_feats = engine.images(ImageFiles(rows, engine.adapter.preprocess), {'set_c_holistic': [r['image_id'] for r in rows]})
    hol_all = engine.scores(hol_feats, engine.texts(hol_pool))
    hol_scores = np.array([[hol_all[tr['image_index'], hol_ix[c]] for c in tr['captions']] for tr in trials], dtype=np.float32)

    # 2. Native & Padded Half-Canvas
    obj_prompts = list(dict.fromkeys(cfg['object_template'].format(object=r[s]) for r in rows for s in ('a', 'b')))
    obj_ix = {p: i for i, p in enumerate(obj_prompts)}
    obj_feats = engine.texts(obj_prompts)
    REGIONS = ('left', 'right', 'top', 'bottom')

    class SetCCrops(Dataset):
        def __len__(self):
            return 8 * len(rows)

        def __getitem__(self, i):
            row = rows[i // 8]
            path = ROOT / row['image_path']
            if sha256(path) != row['image_sha256']:
                raise ValueError(f'Changed source image: {path}')
            with Image.open(path) as im:
                image = im.convert('RGB')
            if row['transform'] == 'horizontal_flip':
                image = ImageOps.mirror(image)
            image = baseline_canvas(image, cfg['canvas_size'])
            crop = crop_half_canvases(image)[REGIONS[i % 4]]
            if i % 8 >= 4:
                crop = pad_half(crop, cfg['canvas_size'], cfg['pad_rgb'])
            return engine.adapter.preprocess(crop)

    import torch
    loader = DataLoader(SetCCrops(), batch_size=32, shuffle=False, num_workers=0)
    crop_feats = []
    with torch.inference_mode():
        for batch in loader:
            crop_feats.append(engine.norm(engine.adapter.encode_images(batch.to('cuda'))).cpu())
    crop_scores = engine.scores(torch.cat(crop_feats), obj_feats).reshape(len(rows), 2, 4, len(obj_prompts))

    native_scores = np.zeros((len(rows), 4), dtype=np.float32)
    padded_scores = np.zeros((len(rows), 4), dtype=np.float32)
    for idx, r in enumerate(rows):
        ia, ib = obj_ix[cfg['object_template'].format(object=r['a'])], obj_ix[cfg['object_template'].format(object=r['b'])]
        native_scores[idx] = relational_scores_from_regions(crop_scores[idx, 0, :, ia], crop_scores[idx, 0, :, ib])
        padded_scores[idx] = relational_scores_from_regions(crop_scores[idx, 1, :, ia], crop_scores[idx, 1, :, ib])

    metrics, predictions, pairs = compute_set_c_metrics(rows, hol_scores, native_scores, padded_scores, cfg)
    write_json(out / 'images.json', rows)
    np.savez_compressed(out / 'scores.npz', holistic=hol_scores, native=native_scores, padded=padded_scores, raw_crops=crop_scores)
    write_json(out / 'metrics.json', metrics)
    write_csv(out / 'predictions.csv', predictions)
    write_csv(out / 'pairs.csv', pairs)
    manifest.update(num_sources=metrics['n_sources'], num_images=metrics['n_images'], review_frozen_at_utc=record['frozen_at_utc'])
    finish(out, manifest)
    print(json.dumps(metrics['native_half']['horizontal_assignment'], indent=1), flush=True)


def audit(run_name=None, directory=None):
    out = Path(directory) if directory else ROOT / 'results' / (run_name or 'full_natural_unseen_holdout_v1')
    m = read_json(out / 'manifest.json')
    if m['status'] != 'completed':
        raise ValueError('Set C run incomplete.')
    if {p.name for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'} != set(m['artifact_sha256']):
        raise ValueError('Unexpected or missing Set C artifacts.')
    for name, digest in m['artifact_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError(f'Changed Set C artifact: {name}')
    for name, digest in m['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed Set C input: {name}')

    cfg = read_json(out / 'study_config.json')
    record, selected = frozen_set_c(cfg)
    if record['frozen_at_utc'] >= m['started_at_utc']:
        raise ValueError('Set C review not frozen before inference.')
    if m['mode'] == 'smoke':
        selected = selected[:8]
    rows = natural.make_images(selected)
    if rows != read_json(out / 'images.json'):
        raise ValueError('Set C images differ from frozen selection.')
    with np.load(out / 'scores.npz', allow_pickle=False) as z:
        hol_scores, nat_scores, pad_scores = z['holistic'], z['native'], z['padded']
    metrics, predictions, pairs = compute_set_c_metrics(rows, hol_scores, nat_scores, pad_scores, cfg)
    close_enough(metrics, read_json(out / 'metrics.json'))
    as_str = lambda rs: [{k: str(v) for k, v in r.items()} for r in rs]
    if as_str(predictions) != read_csv_sig(out / 'predictions.csv') or as_str(pairs) != read_csv_sig(out / 'pairs.csv'):
        raise ValueError('Set C predictions or pairs differ from recomputation.')
    result = {
        'verified': True,
        'recomputed_from_scores': True,
        'review_frozen_before_inference': True,
        'zero_overlap_with_set_a_b': True,
        'n_sources': metrics['n_sources'],
        'holistic_2x2': metrics['holistic_t2_o0']['horizontal_assignment']['accuracy_percent'],
        'native_2x2': metrics['native_half']['horizontal_assignment']['accuracy_percent'],
    }
    write_json(out.parent / (out.name + '_audit.json'), result)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['prepare', 'validate', 'adjudicate', 'freeze', 'run', 'audit'])
    p.add_argument('--review')
    p.add_argument('--slot', choices=['RA', 'RB'])
    p.add_argument('--ra')
    p.add_argument('--rb')
    p.add_argument('--adjudication')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--run-name')
    a = p.parse_args(argv)
    cfg = read_json(CONFIG)
    if a.action == 'prepare':
        prepare(cfg)
    elif a.action == 'validate':
        if not a.review or not a.slot:
            p.error('--review and --slot are required for validate')
        validate_slot(a.review, a.slot)
    elif a.action == 'adjudicate':
        if not a.ra or not a.rb:
            p.error('--ra and --rb are required for adjudicate')
        adjudicate(a.ra, a.rb)
    elif a.action == 'freeze':
        if not a.ra or not a.rb:
            p.error('--ra and --rb are required for freeze')
        freeze(a.ra, a.rb, a.adjudication, cfg)
    elif a.action == 'run':
        run(cfg, a.smoke, a.run_name)
    else:
        audit(run_name=a.run_name)


if __name__ == '__main__':
    main()
