"""COCO subset with geometric filtering and score-blind visual review records."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib

from ..io_utils import sha256, write_json
from .common import ROOT, CONFIG, read_json, begin
from .design import HFLIP

DATA = ROOT / 'data/probing_depth_v1/natural'
SELECTION = ROOT / 'results/depth_natural_selection_v1'
BASE = 'https://s3.amazonaws.com/images.cocodataset.org/'


def get_bytes(url, start=None, end=None):
    headers = {'User-Agent': 'VLM-Probe-Research/1.0'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
                if start is not None and (response.status != 206 or len(payload) != end-start+1):
                    raise ValueError('Range response mismatch.')
                return payload, dict(response.headers)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def annotations(config):
    DATA.mkdir(parents=True, exist_ok=True)
    member_name = config['natural']['annotation_member'].split('/')[-1]
    path = DATA / member_name
    provenance = DATA / (member_name + '.source.json')
    if path.exists() and provenance.exists():
        if sha256(path) != read_json(provenance)['sha256']:
            raise ValueError('COCO annotations checksum changed.')
        if (SELECTION / 'source.json').exists():
            frozen = read_json(SELECTION / 'source.json')
            if frozen['url'] == config['natural']['annotation_url'] and sha256(path) != frozen['sha256']:
                raise ValueError('Annotations differ from frozen selection source.')
        return read_json(path)
    cfg = config['natural']
    request = urllib.request.Request(cfg['annotation_url'], headers={'Range': 'bytes=-65536'})
    with urllib.request.urlopen(request, timeout=45) as response:
        tail = response.read()
        total = int(response.headers['Content-Range'].split('/')[-1])
    with zipfile.ZipFile(io.BytesIO(tail)) as z:
        info = z.getinfo(cfg['annotation_member'])
    offset = info.header_offset + total - len(tail)
    header, _ = get_bytes(cfg['annotation_url'], offset, offset+29)
    parts = struct.unpack('<4s5H3I2H', header)
    if parts[0] != b'PK\x03\x04' or info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError('Unsupported ZIP member.')
    start = offset + 30 + parts[-2] + parts[-1]
    blocks = [(s, min(start+info.compress_size-1, s+1024*1024-1))
              for s in range(start, start+info.compress_size, 1024*1024)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(lambda pair: get_bytes(cfg['annotation_url'], *pair)[0], blocks))
    payload = zlib.decompress(b''.join(chunks), -15)
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise ValueError('COCO ZIP member CRC/size mismatch.')
    path.write_bytes(payload)
    if (SELECTION / 'source.json').exists():
        frozen = read_json(SELECTION / 'source.json')
        if frozen['url'] == cfg['annotation_url'] and sha256(path) != frozen['sha256']:
            raise ValueError('Downloaded annotations differ from frozen selection source.')
    write_json(provenance, {'url': cfg['annotation_url'], 'member': cfg['annotation_member'],
                           'archive_bytes': total, 'member_crc32': info.CRC, 'sha256': sha256(path)})
    print(f'COCO annotations verified: {len(payload)} bytes', flush=True)
    return json.loads(payload)


def relation_from_boxes(a, b, width, height, cfg):
    side = min(width, height)
    left, top = (width-side)/2, (height-side)/2
    boxes = []
    for x, y, w, h in (a, b):
        border = cfg['crop_border_fraction'] * side
        if w*h < cfg['min_box_area_fraction']*width*height or min(w/width, h/height) < cfg['min_box_side_fraction']:
            return None
        if x < left+border or y < top+border or x+w > left+side-border or y+h > top+side-border:
            return None
        boxes.append((x, y, x+w, y+h, w, h))
    a, b = boxes
    overlap_x = max(0, min(a[2], b[2])-max(a[0], b[0])) / min(a[4], b[4])
    overlap_y = max(0, min(a[3], b[3])-max(a[1], b[1])) / min(a[5], b[5])
    gap = cfg['min_separation_fraction'] * side
    candidates = [a[2]+gap <= b[0] and overlap_y >= cfg['min_orthogonal_overlap_fraction'],
                  b[2]+gap <= a[0] and overlap_y >= cfg['min_orthogonal_overlap_fraction'],
                  a[3]+gap <= b[1] and overlap_x >= cfg['min_orthogonal_overlap_fraction'],
                  b[3]+gap <= a[1] and overlap_x >= cfg['min_orthogonal_overlap_fraction']]
    return candidates.index(True) if sum(candidates) == 1 else None


def article(name):
    return ('an ' if name[0].lower() in 'aeiou' else 'a ') + name


def candidates(annotation, cfg):
    names = {c['id']: c['name'] for c in annotation['categories']}
    licenses = {c['id']: c for c in annotation['licenses']}
    by_image = defaultdict(list)
    for ann in annotation['annotations']:
        by_image[ann['image_id']].append(ann)
    possible = []
    for im in annotation['images']:
        instances = by_image[im['id']]
        counts = Counter(a['category_id'] for a in instances)
        eligible = [a for a in instances if not a['iscrowd'] and counts[a['category_id']] == 1]
        for a in eligible:
            for b in eligible:
                if a['id'] == b['id']:
                    continue
                relation = relation_from_boxes(a['bbox'], b['bbox'], im['width'], im['height'], cfg)
                if relation is None or relation not in cfg.get('allowed_relations', range(4)):
                    continue
                rank = hashlib.sha256(f"{cfg['selection_seed']}/{im['id']}/{a['id']}/{b['id']}".encode()).hexdigest()
                possible.append({'rank': rank, 'source_image_id': im['id'], 'file_name': im['file_name'],
                    'width': im['width'], 'height': im['height'], 'annotation_a': a['id'], 'annotation_b': b['id'],
                    'bbox_a': a['bbox'], 'bbox_b': b['bbox'], 'category_a': names[a['category_id']],
                    'category_b': names[b['category_id']], 'category_id_a': a['category_id'], 'category_id_b': b['category_id'],
                    'target': relation, 'coco_url': im['coco_url'], 'flickr_url': im.get('flickr_url'),
                    'download_url': BASE + cfg.get('image_split', 'val2017') + '/' + im['file_name'], 'license': licenses[im['license']]})
    selected, used, quotas = [], set(), Counter()
    for row in sorted(possible, key=lambda r: r['rank']):
        if row['source_image_id'] in used or quotas[row['target']] >= cfg['candidate_reserve_per_relation']:
            continue
        selected.append(row); used.add(row['source_image_id']); quotas[row['target']] += 1
    return selected, {'eligible_directed_pairs': len(possible), 'eligible_source_images': len({r['source_image_id'] for r in possible}),
                      'reserve_by_relation': dict(quotas)}


def download_image(row):
    from PIL import Image
    path = DATA / 'images' / row['file_name']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload, _ = get_bytes(row['download_url'])
        path.write_bytes(payload)
    with Image.open(path) as image:
        if image.size != (row['width'], row['height']):
            raise ValueError('COCO image dimensions differ.')
        image.verify()
    if row.get('image_sha256') and sha256(path) != row['image_sha256']:
        raise ValueError('Downloaded image differs from reviewed bytes.')
    return {**row, 'image_path': path.relative_to(ROOT).as_posix(), 'image_sha256': sha256(path)}


def contact_sheets(rows, prefix='review'):
    from PIL import Image, ImageDraw, ImageFont
    folder = DATA / 'review_sheets'
    folder.mkdir(exist_ok=True)
    # Pin the engine used for the original review; Pillow's implicit default can
    # switch when optional RAQM libraries become available in a later process.
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15, layout_engine=ImageFont.Layout.BASIC)
    legend = ['LEFT', 'RIGHT', 'ABOVE', 'BELOW']
    for start in range(0, len(rows), 16):
        sheet = Image.new('RGB', (1280, 1504), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(rows[start:start+16]):
            x, y = i % 4 * 320, i // 4 * 376
            with Image.open(ROOT / r['image_path']) as im:
                side = min(im.size); left = (im.width-side)/2; top = (im.height-side)/2
                crop = im.convert('RGB').crop((left, top, left+side, top+side)).resize((312, 312))
            cd = ImageDraw.Draw(crop)
            for box, color, label in [(r['bbox_a'], '#0072B2', 'A'), (r['bbox_b'], '#E69F00', 'B')]:
                bx, by, bw, bh = box
                rect = ((bx-left)*312/side, (by-top)*312/side, (bx+bw-left)*312/side, (by+bh-top)*312/side)
                cd.rectangle(rect, outline=color, width=3)
                cd.text((rect[0]+2, rect[1]+2), label, fill=color, font=font, stroke_width=1, stroke_fill='white')
            sheet.paste(crop, (x+4, y))
            draw.text((x+4, y+314), f"{r['review_index']:03d} | COCO {r['source_image_id']} | {legend[r['target']]}", fill='black', font=font)
            draw.text((x+4, y+333), f"A: {r['category_a']}  B: {r['category_b']}", fill='black', font=font)
        sheet.save(folder / f'{prefix}_{start//16:02d}.png')


def prepare():
    cfg = read_json(CONFIG)
    annotation = annotations(cfg)
    SELECTION.mkdir(exist_ok=True)
    if (SELECTION / 'candidates.json').exists():
        existing = read_json(SELECTION / 'candidates.json')
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(download_image, existing))
        # Preserve the original first-pass/reserve layout recorded in review.json.
        initial_count = min(60 * 4, len(existing))
        contact_sheets(existing[:initial_count])
        if len(existing) > initial_count:
            contact_sheets(existing[initial_count:], prefix='reserve')
        if (SELECTION / 'review.json').exists():
            for name, digest in read_json(SELECTION / 'review.json')['review_sheet_sha256'].items():
                if sha256(ROOT / name) != digest:
                    raise ValueError('Restored review sheet differs; check Pillow/font versions.')
        print('Fixed candidates restored and checked; use existing review records.', flush=True)
        return
    rows, counts = candidates(annotation, cfg['natural'])
    write_json(SELECTION / 'candidate_pool.json', rows)
    write_json(SELECTION / 'selection_counts.json', counts)
    print(counts, flush=True)
    # Review up to 60 candidates per relation first; remaining candidates are a fixed reserve.
    review, quotas = [], Counter()
    for r in rows:
        if quotas[r['target']] < 60:
            review.append(r); quotas[r['target']] += 1
    if min(quotas.values(), default=0) < cfg['natural']['target_per_relation']:
        raise ValueError('Insufficient eligible images: record pre-inference protocol amendment.')
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(download_image, review))
    for i, row in enumerate(downloaded):
        row['review_index'] = i
    write_json(SELECTION / 'candidates.json', downloaded)
    annotation_source = DATA / (cfg['natural']['annotation_member'].split('/')[-1] + '.source.json')
    write_json(SELECTION / 'source.json', {**read_json(annotation_source),
        'config_sha256': sha256(CONFIG), 'protocol_sha256': sha256(ROOT / 'docs/probing_depth_protocol.md'),
        'terms': 'https://cocodataset.org/#termsofuse', 'annotation_license': 'CC BY 4.0',
        'image_licenses': 'per image metadata; original photos not redistributed in Git'})
    contact_sheets(downloaded)
    print(f'Prepared {len(downloaded)} images for score-blind visual review.', flush=True)


def finalize_selection(config):
    candidates_list = read_json(SELECTION / 'candidates.json')
    reviews = read_json(SELECTION / 'review.json')
    decisions = {r['source_image_id']: r for r in reviews['decisions']}
    if reviews['candidate_sha256'] != sha256(SELECTION / 'candidates.json') or not reviews['blinded_to_clip_outputs']:
        raise ValueError('Review identity/blinding record mismatch.')
    if len(decisions) != len(candidates_list):
        raise ValueError('Every presented candidate must have a visual review decision.')
    selected, quota = [], Counter()
    for row in candidates_list:
        decision = decisions[row['source_image_id']]
        if decision['image_sha256'] != row['image_sha256'] or decision['status'] not in ['accepted', 'rejected']:
            raise ValueError('Invalid review record.')
        if decision['status'] == 'rejected':
            if not decision.get('reason'):
                raise ValueError('Rejection requires a reason.')
            continue
        if quota[row['target']] < config['natural']['target_per_relation']:
            selected.append(row); quota[row['target']] += 1
    if quota != Counter({i: config['natural']['target_per_relation'] for i in range(4)}):
        raise ValueError(f'Insufficient accepted images; review fixed reserves: {dict(quota)}')
    if len({r['source_image_id'] for r in selected}) != len(selected):
        raise ValueError('Repeated source image.')
    if (SELECTION / 'selected.json').exists():
        if read_json(SELECTION / 'selected.json') != selected:
            raise ValueError('Frozen natural selection changed.')
    else:
        write_json(SELECTION / 'selected.json', selected)
    return selected


def extend_review():
    """Present the rest of the already ranked reserve, without any model scores."""
    current = read_json(SELECTION / 'candidates.json')
    used = {r['source_image_id'] for r in current}
    extra = [r for r in read_json(SELECTION / 'candidate_pool.json') if r['source_image_id'] not in used]
    with ThreadPoolExecutor(max_workers=6) as pool:
        extra = list(pool.map(download_image, extra))
    for i, row in enumerate(extra, start=len(current)):
        row['review_index'] = i
    write_json(SELECTION / 'candidates.json', current + extra)
    contact_sheets(extra, prefix='reserve')
    print(f'Added {len(extra)} fixed reserve candidates for review.', flush=True)


def make_images(selected):
    rows = []
    for row in selected:
        for transform in ['original', 'horizontal_flip']:
            rows.append({**row, 'image_id': f"coco/{row['source_image_id']}/{transform}",
                         'group_id': f"coco/{row['source_image_id']}", 'variant': transform, 'transform': transform,
                         'target': HFLIP[row['target']] if transform == 'horizontal_flip' else row['target'],
                         'a': article(row['category_a']), 'b': article(row['category_b'])})
    return rows


def validate_selected(images, config):
    """Independent check against original annotation and center ordering."""
    annotation = annotations(config)
    anns = {a['id']: a for a in annotation['annotations']}
    names = {c['id']: c['name'] for c in annotation['categories']}
    counts = Counter((a['image_id'], a['category_id']) for a in annotation['annotations'])
    selected = finalize_selection(config)
    expected = make_images(selected)[:len(images)]
    if len(images) not in (24, 400):
        raise ValueError('Unexpected natural image count.')
    if images != expected:
        raise ValueError('Natural inputs differ from reviewed selection.')
    for row in images:
        a, b = anns[row['annotation_a']], anns[row['annotation_b']]
        if a['image_id'] != row['source_image_id'] or b['image_id'] != row['source_image_id']:
            raise ValueError('Annotation image mismatch.')
        if a['bbox'] != row['bbox_a'] or b['bbox'] != row['bbox_b']:
            raise ValueError('Annotation boxes changed.')
        if a['iscrowd'] or b['iscrowd'] or a['category_id'] == b['category_id']:
            raise ValueError('Invalid natural object identities.')
        if counts[(a['image_id'], a['category_id'])] != 1 or counts[(b['image_id'], b['category_id'])] != 1:
            raise ValueError('Target category has multiple annotations.')
        if row['category_a'] != names[a['category_id']] or row['category_b'] != names[b['category_id']]:
            raise ValueError('Caption category differs from source annotation.')
        ax, ay, aw, ah = a['bbox']; bx, by, bw, bh = b['bbox']
        dx, dy = (bx+bw/2)-(ax+aw/2), (by+bh/2)-(ay+ah/2)
        if row['transform'] == 'horizontal_flip':
            dx = -dx
        valid = (dx > 0, dx < 0, dy > 0, dy < 0)[row['target']]
        if not valid:
            raise ValueError('Independent center-based label check failed.')
        original_relation = relation_from_boxes(a['bbox'], b['bbox'], row['width'], row['height'], config['natural'])
        target = HFLIP[original_relation] if row['transform'] == 'horizontal_flip' else original_relation
        if target != row['target']:
            raise ValueError('Natural geometry filter mismatch.')


def run(smoke=False, run_name=None):
    cfg = read_json(CONFIG)
    selected = finalize_selection(cfg)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download_image, selected))
    inputs = [SELECTION / name for name in ['selected.json', 'review.json', 'source.json']]
    out, cfg, manifest = begin('natural', smoke, inputs, run_name=run_name)
    images = make_images(selected)
    if smoke:
        images = images[:24]
    validate_selected(images, cfg)
    manifest['annotation_source'] = read_json(SELECTION / 'source.json')
    manifest['review'] = {'method': 'assistant visual review before CLIP inference; independent geometric validator',
                          'independent_human_annotators': 0, 'selected_original_images': len(images)//2}
    from .spatial import evaluate
    evaluate(out, cfg, manifest, images, 'photo')
