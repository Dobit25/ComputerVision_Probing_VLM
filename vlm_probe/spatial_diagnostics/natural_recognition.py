"""T1 (1118.md): two-object recognition control on the reviewed COCO confirmation crops.

Stages are kept separate so that no CLIP score exists before foils are frozen:

    prepare   rank score-blind foil candidates, write a review round (CSV + HTML packet)
    validate  check a filled review CSV against the round it answers (schema, IDs, logic)
    freeze    freeze all filled rounds once every source/role is resolved
    run       CLIP inference (requires config status 'frozen' and a frozen review)
    audit     recompute trials, predictions and metrics from saved scores (no pixels/model)

See docs/natural_recognition_protocol_v1.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..depth.common import ROOT, cluster_ci, csv_rows, finish, read_json, write_csv
from ..io_utils import sha256, write_json

CONFIG = ROOT / 'configs/natural_recognition_v1.json'
REVIEWS = ROOT / 'reviews/natural_recognition_v1'
PACKETS = ROOT / 'data/natural_recognition_v1'
ROLES = ('replace_a', 'replace_b')
DECISIONS = ('accepted', 'revise', 'uncertain')
YES_NO = ('yes', 'no', 'unsure')
REVIEW_FIELDS = ['review_id', 'source_image_id', 'canvas_rgb_sha256', 'role', 'replaced_object', 'kept_object',
                 'foil', 'foil_tier', 'candidate_version', 'reviewer_id', 'foil_absent', 'unambiguous',
                 'decision', 'reason', 'notes']
FILLED_FIELDS = ('reviewer_id', 'foil_absent', 'unambiguous', 'decision', 'reason', 'notes')
NEAR_TIE = 1e-6


def now():
    return datetime.now(timezone.utc).isoformat()


def article(name):
    from ..depth.natural import article as natural_article
    return natural_article(name)


def caption(template, a_name, b_name):
    return template.format(a=article(a_name), b=article(b_name))


def _rank(seed, source_id, role, name):
    return hashlib.sha256(f'{seed}/{source_id}/{role}/{name}'.encode()).hexdigest()


# ---------------------------------------------------------------- foils (score-blind)

def foil_candidates(source, role, categories, present, cfg):
    """Ranked foil candidates for one source/role; never uses model outputs.

    categories: [{'name', 'supercategory'}]; present: category names annotated in the image.
    """
    replaced = source['category_a'] if role == 'replace_a' else source['category_b']
    supercat = {c['name']: c['supercategory'] for c in categories}
    if replaced not in supercat:
        raise ValueError(f'Unknown COCO category: {replaced}')
    contested = {frozenset(p) for p in cfg['contested_pairs']}
    blocked = {source['category_a'], source['category_b'], *present}
    eligible = [c['name'] for c in categories if c['name'] not in blocked
                and frozenset((c['name'], replaced)) not in contested]
    tier1 = [n for n in eligible if supercat[n] == supercat[replaced]]
    tier, pool = (1, tier1) if tier1 else (2, [n for n in eligible if supercat[n] != supercat[replaced]])
    ranked = sorted(pool, key=lambda n: _rank(cfg['selection_seed'], source['source_image_id'], role, n))
    return [{'name': n, 'tier': tier} for n in ranked]


def review_id(source_id, role, version):
    return f'NR1_{source_id}_{role}_v{version}'


def review_row(source, role, version, candidate, canvas_hash):
    kept = source['category_b'] if role == 'replace_a' else source['category_a']
    replaced = source['category_a'] if role == 'replace_a' else source['category_b']
    return {'review_id': review_id(source['source_image_id'], role, version),
            'source_image_id': str(source['source_image_id']), 'canvas_rgb_sha256': canvas_hash, 'role': role,
            'replaced_object': replaced, 'kept_object': kept, 'foil': candidate['name'],
            'foil_tier': str(candidate['tier']), 'candidate_version': str(version),
            **{k: '' for k in FILLED_FIELDS}}


# ---------------------------------------------------------------- review validation and resolution

def validate_review(filled, presented, reviewer_id=None):
    """Return a list of problems (empty if valid). `presented` is the blank round as written."""
    problems = []
    if not filled:
        return ['empty review']
    if list(filled[0]) != REVIEW_FIELDS:
        problems.append(f'columns differ from schema: {list(filled[0])}')
    want = {r['review_id']: r for r in presented}
    got = Counter(r.get('review_id') for r in filled)
    problems += [f'duplicate review_id {k}' for k, n in got.items() if n > 1]
    problems += [f'missing review_id {k}' for k in want if k not in got]
    problems += [f'unknown review_id {k}' for k in got if k not in want]
    for r in filled:
        rid = r.get('review_id')
        if rid not in want:
            continue
        fixed = {k: v for k, v in want[rid].items() if k not in FILLED_FIELDS}
        changed = [k for k, v in fixed.items() if r.get(k) != v]
        if changed:
            problems.append(f'{rid}: presented fields changed {changed}')
        if not r['reviewer_id'].strip():
            problems.append(f'{rid}: reviewer_id empty')
        elif reviewer_id and r['reviewer_id'] != reviewer_id:
            problems.append(f'{rid}: reviewer_id {r["reviewer_id"]} != assigned {reviewer_id}')
        if r['foil_absent'] not in YES_NO or r['unambiguous'] not in YES_NO:
            problems.append(f'{rid}: foil_absent/unambiguous must be yes/no/unsure')
        if r['decision'] not in DECISIONS:
            problems.append(f'{rid}: decision must be one of {DECISIONS}')
        elif r['decision'] == 'accepted' and (r['foil_absent'], r['unambiguous']) != ('yes', 'yes'):
            problems.append(f'{rid}: accepted requires foil_absent=yes and unambiguous=yes')
        elif r['decision'] != 'accepted' and not r['reason'].strip():
            problems.append(f'{rid}: {r["decision"]} requires a reason')
    return problems


def resolve(rounds, max_versions):
    """Final state per (source_id, role) from ordered filled rounds (later rounds win)."""
    state = {}
    for rows in rounds:
        for r in rows:
            state[(r['source_image_id'], r['role'])] = r
    out = {}
    for key, r in state.items():
        if r['decision'] == 'accepted':
            out[key] = {'status': 'accepted', 'foil': r['foil'], 'foil_tier': int(r['foil_tier']),
                        'candidate_version': int(r['candidate_version']), 'review_id': r['review_id']}
        elif r['decision'] == 'revise' and int(r['candidate_version']) >= max_versions:
            out[key] = {'status': 'no_valid_foil', 'reason': r['reason'], 'review_id': r['review_id']}
        else:
            out[key] = {'status': 'unresolved', 'decision': r['decision'], 'review_id': r['review_id']}
    return out


def next_round_rows(previous_filled, candidates_by_key, canvas_by_source, sources_by_id, max_versions):
    """Rows for the next round: next candidate for 'revise', same candidate for 'uncertain'."""
    rows = []
    for r in previous_filled:
        key = (r['source_image_id'], r['role'])
        version = int(r['candidate_version'])
        if r['decision'] == 'accepted' or (r['decision'] == 'revise' and version >= max_versions):
            continue
        if r['decision'] == 'revise':
            version += 1
            if version > len(candidates_by_key[key]):
                continue
        source = sources_by_id[r['source_image_id']]
        row = review_row(source, r['role'], version, candidates_by_key[key][version - 1], canvas_by_source[r['source_image_id']])
        if r['decision'] == 'uncertain':
            row['review_id'] += '_adj'
        rows.append(row)
    return rows


def analysed_sources(final, cohort_ids):
    """Primary rule: a source is analysed only if both roles have an accepted foil."""
    keep, excluded = [], []
    for sid in cohort_ids:
        states = {role: final.get((str(sid), role), {'status': 'missing'}) for role in ROLES}
        if all(s['status'] == 'accepted' for s in states.values()):
            keep.append(sid)
        else:
            excluded.append({'source_image_id': sid, **{role: states[role]['status'] for role in ROLES}})
    return keep, excluded


# ---------------------------------------------------------------- trials and metrics

def make_trials(images, foils, cfg):
    """images: natural.make_images rows; foils: {(source_id str, role): foil name}."""
    trials = []
    for i, im in enumerate(images):
        sid = str(im['source_image_id'])
        a, b = im['category_a'], im['category_b']
        for role in ROLES:
            foil = foils[(sid, role)]['foil']
            true = caption(cfg['caption_template'], a, b)
            false = caption(cfg['caption_template'], foil, b) if role == 'replace_a' else caption(cfg['caption_template'], a, foil)
            if true == false:
                raise ValueError('Foil caption equals true caption.')
            sample_id = f"{im['image_id']}/{role}"
            # Displayed order is balanced by a fixed hash; scoring is order-independent.
            true_first = int(hashlib.sha256(f"{cfg['selection_seed']}/{sample_id}".encode()).hexdigest(), 16) % 2 == 0
            trials.append({'sample_id': sample_id, 'image_index': i, 'image_id': im['image_id'],
                           'group_id': im['group_id'], 'source_image_id': im['source_image_id'],
                           'variant': im['variant'], 'role': role, 'foil': foil,
                           'foil_tier': foils[(sid, role)]['foil_tier'],
                           'true_caption': true, 'foil_caption': false,
                           'captions': [true, false] if true_first else [false, true],
                           'true_index': 0 if true_first else 1})
    return trials


def _rate(flags, groups, cfg):
    flags = [bool(f) for f in flags]
    return {'n': len(flags), 'correct': sum(flags), 'accuracy_percent': 100 * sum(flags) / len(flags),
            'ci95_percent': cluster_ci(flags, groups, cfg['metrics']['bootstrap_draws'], cfg['metrics']['bootstrap_seed'])}


def summarize(trials, scores, relation_correct, cfg):
    """scores: (n_trials, 2) as [s(true), s(foil)]; relation_correct: image_id -> holistic T2/o0 2-way correct."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(trials), 2) or not np.isfinite(scores).all():
        raise ValueError('Invalid recognition score matrix.')
    gaps = scores[:, 0] - scores[:, 1]
    correct = gaps > 0
    groups = [t['group_id'] for t in trials]
    m = {'primary': {**_rate(correct, groups, cfg), 'n_sources': len(set(groups)),
                     'n_images': len({t['image_id'] for t in trials}),
                     'exact_ties': int((gaps == 0).sum()), 'near_ties': int((np.abs(gaps) <= NEAR_TIE).sum()),
                     'mean_gap': float(gaps.mean())}}
    for key in ('role', 'variant', 'foil_tier'):
        m[f'by_{key}'] = {}
        for value in sorted({t[key] for t in trials}, key=str):
            ix = [i for i, t in enumerate(trials) if t[key] == value]
            m[f'by_{key}'][str(value)] = _rate(correct[ix], [groups[i] for i in ix], cfg)
    by_image, by_source = defaultdict(list), defaultdict(list)
    for i, t in enumerate(trials):
        by_image[t['image_id']].append(i)
        by_source[t['group_id']].append(i)
    if any(len(v) != 2 for v in by_image.values()) or any(len(v) != 4 for v in by_source.values()):
        raise ValueError('Expected 2 roles per image and 4 trials per source.')
    image_ids = list(by_image)
    image_groups = [trials[by_image[k][0]]['group_id'] for k in image_ids]
    both = [bool(correct[by_image[k]].all()) for k in image_ids]
    m['both_roles_correct_per_image'] = _rate(both, image_groups, cfg)
    m['all_four_correct_per_source'] = _rate([correct[v].all() for v in by_source.values()], list(by_source), cfg)
    missing = [k for k in image_ids if k not in relation_correct]
    if missing:
        raise ValueError(f'No holistic relation result for {missing[:3]}')
    rel = [bool(relation_correct[k]) for k in image_ids]
    table = Counter((b, r) for b, r in zip(both, rel))
    m['relation_crosstab_descriptive'] = {
        'note': 'Holistic T2/o0 two-way relation vs recognition on the same images. Subgroups are selected by model behaviour: descriptive only.',
        'relation_accuracy_same_images': _rate(rel, image_groups, cfg),
        'recognition_both_correct_and_relation_correct': table[(True, True)],
        'recognition_both_correct_and_relation_wrong': table[(True, False)],
        'recognition_not_both_and_relation_correct': table[(False, True)],
        'recognition_not_both_and_relation_wrong': table[(False, False)],
        'relation_accuracy_given_recognition_both_correct_percent':
            100 * table[(True, True)] / sum(both) if any(both) else None,
        'images_with_recognition_both_correct': sum(both),
    }
    predictions = [{'sample_id': t['sample_id'], 'image_id': t['image_id'], 'group_id': t['group_id'],
                    'variant': t['variant'], 'role': t['role'], 'foil': t['foil'], 'foil_tier': t['foil_tier'],
                    'true_score': float(scores[i, 0]), 'foil_score': float(scores[i, 1]), 'gap': float(gaps[i]),
                    'correct': bool(correct[i]), 'exact_tie': bool(gaps[i] == 0),
                    'near_tie': bool(abs(gaps[i]) <= NEAR_TIE)} for i, t in enumerate(trials)]
    images = [{'image_id': k, 'group_id': g, 'recognition_both_roles_correct': b, 'holistic_relation_correct': r}
              for k, g, b, r in zip(image_ids, image_groups, both, rel)]
    return m, predictions, images


# ---------------------------------------------------------------- data helpers (need COCO files)

def cohort(cfg):
    accepted = {int(r['source_image_id']) for r in csv_rows(ROOT / cfg['cohort_membership']) if r['accepted'] == 'True'}
    return [r for r in read_json(ROOT / cfg['source_selection']) if r['source_image_id'] in accepted]


def relation_correct_by_image(cfg):
    return {r['image_id']: r['correct'] == 'True'
            for r in csv_rows(ROOT / cfg['holistic_reference']['followup_predictions'])}


def canvas_for(row, size=224):
    from PIL import Image
    from .natural_regions import baseline_canvas
    path = ROOT / row['image_path']
    if sha256(path) != row['image_sha256']:
        raise ValueError(f'Source image changed: {path}')
    with Image.open(path) as im:
        return baseline_canvas(im, size)


def canvas_hash(canvas):
    return hashlib.sha256(f'{canvas.size}/{canvas.mode}/'.encode() + canvas.tobytes()).hexdigest()


def canvas_box(box, width, height, size=224):
    scale = size / min(width, height)
    rw, rh = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    x0, y0 = round((rw - size) / 2), round((rh - size) / 2)
    x, y, w, h = box
    return (x * scale - x0, y * scale - y0, (x + w) * scale - x0, (y + h) * scale - y0)


def read_review_csv(path):
    """Tolerates a UTF-8 BOM (spreadsheet exports); content is otherwise read verbatim."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def round_path(n, kind):
    return REVIEWS / f'round_{n:02d}_{kind}.csv'


def write_review_csv(path, rows):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite review file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader(); w.writerows(rows)


def write_packet(n, rows, sources_by_id, canvases):
    """HTML review packet with the exact model canvas; pixels stay outside Git (data/)."""
    from PIL import ImageDraw
    import html
    folder = PACKETS / f'review_round_{n:02d}'
    (folder / 'img').mkdir(parents=True, exist_ok=True)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r['source_image_id']].append(r)
    cards = []
    for sid, items in by_source.items():
        src = sources_by_id[sid]
        canvas = canvases[sid].copy(); draw = ImageDraw.Draw(canvas)
        for box, color, label in ((src['bbox_a'], '#0072B2', 'A'), (src['bbox_b'], '#E69F00', 'B')):
            rect = canvas_box(box, src['width'], src['height'])
            draw.rectangle(rect, outline=color, width=2); draw.text((rect[0] + 3, rect[1] + 2), label, fill=color)
        canvas.resize((448, 448)).save(folder / 'img' / f'{sid}.png')
        lis = ''.join(
            f"<li><code>{html.escape(x['review_id'])}</code>: thay <b>{html.escape(x['replaced_object'])}</b> "
            f"→ foil <b>{html.escape(x['foil'])}</b> (tầng {x['foil_tier']}, v{x['candidate_version']}); "
            f"giữ {html.escape(x['kept_object'])}</li>" for x in items)
        cards.append(f"<section id='card-{sid}'><img src='img/{sid}.png' width='448' height='448' alt='COCO {sid}'>"
                     f"<p>COCO {sid} — A: {html.escape(src['category_a'])}, B: {html.escape(src['category_b'])}</p>"
                     f"<ul>{lis}</ul></section>")
    page = ("<!doctype html><meta charset='utf-8'><title>T1 foil review round %02d</title>"
            "<style>body{font-family:sans-serif;max-width:960px;margin:auto;padding:16px}"
            "section{border-top:1px solid #ccc;padding:12px 0}img{image-rendering:auto;max-width:100%%;height:auto}</style>"
            "<h1>T1 — review foil, vòng %02d</h1><p>Ảnh là đúng vùng 224×224 mô hình thấy (phóng 2×). "
            "Với mỗi dòng: foil có <b>vắng mặt</b> trong vùng ảnh không (foil_absent), và có chắc người xem "
            "<b>không thể gọi</b> vật bị thay bằng tên foil không (unambiguous). Điền vào CSV đi kèm. "
            "Không có điểm mô hình nào trong gói này.</p>%s") % (n, n, ''.join(cards))
    (folder / 'index.html').write_text(page, encoding='utf-8')
    return folder


def load_coco(cfg):
    from ..depth import natural
    base = read_json(ROOT / cfg['base_config'])
    annotation = natural.annotations(base)
    frozen = read_json(ROOT / 'results/natural_confirmation_selection_v1/preparation.json')['annotation_source']
    path = natural.DATA / base['natural']['annotation_member'].split('/')[-1]
    if sha256(path) != frozen['sha256']:
        raise ValueError('COCO annotations differ from the frozen confirmation source.')
    return annotation


def prepare(cfg):
    """Round 1 (or the next round after the latest filled one). No model is loaded."""
    from ..depth import natural
    sources = cohort(cfg)
    annotation = load_coco(cfg)
    categories = [{'name': c['name'], 'supercategory': c['supercategory']} for c in annotation['categories']]
    names = {c['id']: c['name'] for c in annotation['categories']}
    present = defaultdict(set)
    for a in annotation['annotations']:
        present[a['image_id']].add(names[a['category_id']])
    for s in sources:
        natural.download_image(s)
    by_id = {str(s['source_image_id']): s for s in sources}
    canvases = {sid: canvas_for(s) for sid, s in by_id.items()}
    hashes = {sid: canvas_hash(c) for sid, c in canvases.items()}
    cands = {(sid, role): foil_candidates(s, role, categories, present[s['source_image_id']], cfg)
             for sid, s in by_id.items() for role in ROLES}
    REVIEWS.mkdir(parents=True, exist_ok=True)
    meta_path = REVIEWS / 'candidates.json'
    meta = {'config_sha256': sha256(CONFIG), 'protocol_sha256': sha256(ROOT / cfg['protocol']),
            'annotation_sha256': read_json(ROOT / 'results/natural_confirmation_selection_v1/preparation.json')['annotation_source']['sha256'],
            'scores_used': False,
            'sources': [{'source_image_id': int(sid), 'image_sha256': by_id[sid]['image_sha256'],
                         'canvas_rgb_sha256': hashes[sid], 'annotated_categories': sorted(present[int(sid)]),
                         'candidates': {role: cands[(sid, role)][:cfg['foil_rule']['max_candidate_versions']] for role in ROLES}}
                        for sid in by_id]}
    if meta_path.exists():
        if read_json(meta_path)['sources'] != meta['sources']:
            raise ValueError('Foil candidates differ from the recorded candidate list.')
    else:
        write_json(meta_path, meta)
    filled = sorted(REVIEWS.glob('round_*_filled.csv'))
    if not filled:
        n, rows = 1, [review_row(by_id[sid], role, 1, cands[(sid, role)][0], hashes[sid])
                      for sid in by_id for role in ROLES if cands[(sid, role)]]
        empty = [f'{sid}/{role}' for sid in by_id for role in ROLES if not cands[(sid, role)]]
        if empty:
            print(f'No eligible foil for: {empty}', flush=True)
    else:
        last = filled[-1]; n = int(last.name.split('_')[1]) + 1
        rows = next_round_rows(read_review_csv(last), cands, hashes, by_id, cfg['foil_rule']['max_candidate_versions'])
        if not rows:
            print('Nothing left to review; run freeze.', flush=True); return
    write_review_csv(round_path(n, 'blank'), rows)
    folder = write_packet(n, rows, by_id, canvases)
    print(f'Round {n}: {len(rows)} rows for {len({r["source_image_id"] for r in rows})} sources. '
          f'Blank CSV: {round_path(n, "blank").relative_to(ROOT)}; packet: {folder.relative_to(ROOT)}/index.html '
          '(status: pending_review)', flush=True)


def _round_of(path):
    m = re.fullmatch(r'round_(\d{2})_filled\.csv', Path(path).name)
    if not m:
        raise ValueError('Filled review must be named round_NN_filled.csv inside reviews/natural_recognition_v1/.')
    return int(m.group(1))


def validate(path, cfg):
    path = Path(path).resolve()
    n = _round_of(path)
    reviewer = cfg['team_decisions'].get('reviewer')
    problems = validate_review(read_review_csv(path), read_review_csv(round_path(n, "blank")), reviewer)
    if problems:
        raise ValueError('Review invalid:\n  ' + '\n  '.join(problems[:50]))
    counts = Counter(r['decision'] for r in read_review_csv(path))
    print(f'Round {n} valid: {dict(counts)}', flush=True)
    return counts


def freeze(cfg):
    if (REVIEWS / 'freeze.json').exists():
        raise FileExistsError('Review already frozen; a change needs a new review version.')
    if cfg['status'] != 'frozen':
        raise RuntimeError('Freeze the config (team sign-off) before freezing the review.')
    filled = sorted(REVIEWS.glob('round_*_filled.csv'))
    if not filled:
        raise ValueError('No filled review rounds.')
    for p in filled:
        validate(p, cfg)
    final = resolve([read_review_csv(p) for p in filled], cfg['foil_rule']['max_candidate_versions'])
    unresolved = {f'{k[0]}/{k[1]}': v for k, v in final.items() if v['status'] == 'unresolved'}
    if unresolved:
        raise ValueError(f'Unresolved rows; prepare another round first: {list(unresolved)[:10]}')
    ids = [s['source_image_id'] for s in cohort(cfg)]
    keep, excluded = analysed_sources(final, ids)
    floor = cfg['team_decisions'].get('minimum_sources_for_inference')
    record = {'status': 'frozen', 'frozen_at_utc': now(), 'config_sha256': sha256(CONFIG),
              'reviewer_ids': sorted({r['reviewer_id'] for p in filled for r in read_review_csv(p)}),
              'final': [{'source_image_id': k[0], 'role': k[1], **v} for k, v in sorted(final.items())],
              'analysed_sources': keep, 'excluded_sources': excluded, 'cohort_sources': len(ids),
              'minimum_sources_for_inference': floor,
              'sha256': {p.relative_to(REVIEWS).as_posix(): sha256(p) for p in sorted(REVIEWS.rglob('*'))
                         if p.is_file() and p.name != 'freeze.json'}}
    write_json(REVIEWS / 'freeze.json', record)
    print(f'Frozen: {len(keep)}/{len(ids)} sources analysable; {len(excluded)} excluded.', flush=True)


def frozen_review():
    record = read_json(REVIEWS / 'freeze.json')
    for name, digest in record['sha256'].items():
        if sha256(REVIEWS / name) != digest:
            raise ValueError(f'Frozen review file changed: {name}')
    foils = {(r['source_image_id'], r['role']): r for r in record['final'] if r['status'] == 'accepted'}
    return record, foils


def study_rows(cfg, record, smoke):
    from ..depth.natural import make_images
    keep = set(record['analysed_sources'])
    sources = [s for s in cohort(cfg) if s['source_image_id'] in keep]
    if smoke:
        sources = sources[:8]
    return make_images(sources)


# ---------------------------------------------------------------- inference and audit

def run(cfg, smoke=False, run_name=None):
    if cfg['status'] != 'frozen':
        raise RuntimeError('Config is not frozen by the team; inference refused (see protocol §7).')
    record, foils = frozen_review()
    if record['config_sha256'] != sha256(CONFIG):
        raise ValueError('Config changed after the review freeze.')
    floor = cfg['team_decisions'].get('minimum_sources_for_inference')
    if isinstance(floor, int) and len(record['analysed_sources']) < floor and not smoke:
        raise RuntimeError(f'Only {len(record["analysed_sources"])} sources have accepted foils (< {floor}); T1 not feasible.')
    from ..depth import natural
    from ..depth.common import Engine, ImageFiles
    from ..extension_artifacts import begin_extension
    ref = ROOT / cfg['holistic_reference']['run']
    all_images = natural.make_images(read_json(ROOT / cfg['source_selection']))
    if all_images != read_json(ref / 'images.json'):
        raise ValueError('Holistic reference images differ from selection.')
    for s in read_json(ROOT / cfg['source_selection']):
        natural.download_image(s)
    name = run_name or f'{"smoke" if smoke else "full"}_natural_recognition_v1'
    inputs = [REVIEWS / 'freeze.json', ROOT / cfg['source_selection'], ROOT / cfg['cohort_membership'],
              ROOT / cfg['holistic_reference']['followup_predictions'], ref / 'manifest.json', ROOT / cfg['protocol']]
    out, _, manifest = begin_extension('natural_recognition', CONFIG, inputs=inputs, smoke=smoke, run_name=name)
    if record['frozen_at_utc'] >= manifest['started_at_utc']:
        raise ValueError('Review must be frozen before inference starts.')
    engine = Engine(out, read_json(ROOT / cfg['base_config']), manifest)
    # Identical cache identity to full_natural_confirmation_v1 -> identical image features.
    identity = {'images': [{k: r.get(k) for k in ['image_id', 'image_sha256', 'transform']} for r in all_images]}
    features = engine.images(ImageFiles(all_images, engine.adapter.preprocess), identity)
    # Parity: rebuild frozen holistic T2/o0 scores from these features.
    ref_trials = read_json(ref / 'trials.json')
    with np.load(ref / 'scores.npz', allow_pickle=False) as z:
        ref_scores = z['scores']
    t, o = cfg['holistic_reference']['template'], cfg['holistic_reference']['order']
    sel = [i for i, r in enumerate(ref_trials) if r['template'] == t and r['order'] == o]
    rel_pool = list(dict.fromkeys(c for i in sel for c in ref_trials[i]['captions']))
    rel_all = engine.scores(features, engine.texts(rel_pool)); rel_ix = {c: j for j, c in enumerate(rel_pool)}
    rebuilt = np.array([[rel_all[ref_trials[i]['image_index'], rel_ix[c]] for c in ref_trials[i]['captions']] for i in sel])
    parity = {'trials': len(sel), 'max_abs_diff': float(np.abs(rebuilt - ref_scores[sel]).max()), 'tolerance': 1e-6}
    if parity['max_abs_diff'] > parity['tolerance']:
        raise ValueError(f'Holistic parity failed: {parity}')
    rows = study_rows(cfg, record, smoke)
    index = {r['image_id']: i for i, r in enumerate(all_images)}
    trials = make_trials(rows, foils, cfg)
    pool = list(dict.fromkeys(c for tr in trials for c in tr['captions'])); pix = {c: j for j, c in enumerate(pool)}
    all_scores = engine.scores(features, engine.texts(pool))
    scores = np.array([[all_scores[index[tr['image_id']], pix[tr['true_caption']]],
                        all_scores[index[tr['image_id']], pix[tr['foil_caption']]]] for tr in trials], dtype=np.float32)
    metrics, predictions, images = summarize(trials, scores, relation_correct_by_image(cfg), cfg)
    metrics['coverage'] = {'cohort_sources': record['cohort_sources'], 'analysed_sources': len(record['analysed_sources']),
                           'run_sources': len({r['group_id'] for r in rows}), 'excluded_sources': record['excluded_sources']}
    metrics['holistic_parity'] = parity
    write_json(out / 'images.json', rows); write_json(out / 'trials.json', trials)
    write_json(out / 'text_pool.json', pool); write_json(out / 'parity.json', parity)
    np.savez_compressed(out / 'scores.npz', scores=scores)
    write_json(out / 'metrics.json', metrics)
    write_csv(out / 'predictions.csv', predictions); write_csv(out / 'image_records.csv', images)
    manifest.update(num_sources=len({r['group_id'] for r in rows}), num_images=len(rows), num_trials=len(trials),
                    review_frozen_at_utc=record['frozen_at_utc'])
    finish(out, manifest)
    print(json.dumps(metrics['primary'], indent=1), flush=True)


def audit(cfg_unused=None, run_name=None, directory=None):
    out = Path(directory) if directory else ROOT / 'results' / (run_name or 'full_natural_recognition_v1')
    m = read_json(out / 'manifest.json')
    if m['status'] != 'completed':
        raise ValueError('Run incomplete.')
    if {p.name for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'} != set(m['artifact_sha256']):
        raise ValueError('Unexpected or missing artifacts.')
    for name, digest in m['artifact_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError(f'Changed result artifact: {name}')
    for name, digest in m['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed input: {name}')
    cfg = read_json(out / 'study_config.json')
    record, foils = frozen_review()
    if record['frozen_at_utc'] >= m['started_at_utc']:
        raise ValueError('Review not frozen before inference.')
    rows = study_rows(cfg, record, m['mode'] == 'smoke')
    if rows != read_json(out / 'images.json'):
        raise ValueError('Images differ from frozen review cohort.')
    trials = make_trials(rows, foils, cfg)
    if trials != read_json(out / 'trials.json'):
        raise ValueError('Trials differ from frozen foils/config.')
    with np.load(out / 'scores.npz', allow_pickle=False) as z:
        scores = z['scores']
    if scores.shape != (len(trials), 2) or not np.isfinite(scores).all() or (np.abs(scores) > 1.00001).any():
        raise ValueError('Invalid scores.')
    metrics, predictions, images = summarize(trials, scores, relation_correct_by_image(cfg), cfg)
    saved = read_json(out / 'metrics.json')
    for key in ('coverage', 'holistic_parity'):
        metrics[key] = saved[key]
    if json.loads(json.dumps(metrics)) != saved:
        raise ValueError('Metrics differ from recomputation.')
    as_str = lambda rs: [{k: str(v) for k, v in r.items()} for r in rs]
    if as_str(predictions) != csv_rows(out / 'predictions.csv') or as_str(images) != csv_rows(out / 'image_records.csv'):
        raise ValueError('Predictions or image records differ from recomputation.')
    if saved['holistic_parity']['max_abs_diff'] > saved['holistic_parity']['tolerance']:
        raise ValueError('Run recorded a failed parity check.')
    result = {'verified': True, 'recomputed_from_scores': True, 'n_trials': len(trials),
              'review_frozen_before_inference': True, 'primary': metrics['primary']}
    write_json(out.parent / (out.name + '_audit.json'), result)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['prepare', 'validate', 'freeze', 'run', 'audit'])
    p.add_argument('--review'); p.add_argument('--smoke', action='store_true'); p.add_argument('--run-name')
    a = p.parse_args(argv)
    cfg = read_json(CONFIG)
    if a.action == 'prepare': prepare(cfg)
    elif a.action == 'validate': validate(a.review, cfg)
    elif a.action == 'freeze': freeze(cfg)
    elif a.action == 'run': run(cfg, a.smoke, a.run_name)
    else: audit(run_name=a.run_name)


if __name__ == '__main__':
    main()
