"""Outcome-blind stimulus and caption construction."""
import hashlib
import json
from itertools import permutations, product

import numpy as np

from ..probing.stimuli import COLORS, SHAPES, RELATIONS, obj, render, rng_for
from ..io_utils import sha256
from .common import ROOT

INVERSE = (1, 0, 3, 2)
HFLIP = (1, 0, 2, 3)


def captions(a, b, template, reversed_order, medium, config):
    result = []
    for r in range(4):
        first, second, relation = (b, a, INVERSE[r]) if reversed_order else (a, b, r)
        result.append(config['spatial']['templates'][template].format(
            medium=medium, a=first, b=second, relation=RELATIONS[relation]))
    return result


def spatial_images(config, materialize=True):
    cfg = config['spatial']
    rows = []
    pairs = list(product(permutations(range(3), 2), permutations(range(3), 2)))
    folder = ROOT / 'data/probing_depth_v1/spatial'
    if materialize:
        folder.mkdir(parents=True, exist_ok=True)
    for group in range(cfg['base_groups']):
        rng = rng_for(cfg['renderer_seed'], f'depth/{group}')
        (ca, cb), (sa, sb) = pairs[rng.randrange(len(pairs))]
        cx, cy = rng.uniform(100, 124), rng.uniform(100, 124)
        half, radius = rng.uniform(42, 62), rng.uniform(14, 23)
        positions = [((cx-half, cy), (cx+half, cy)), ((cx+half, cy), (cx-half, cy)),
                     ((cx, cy-half), (cx, cy+half)), ((cx, cy+half), (cx, cy-half))]
        for variant in cfg['appearance_variants']:
            ci = [(ca + (variant == 'color_cycle')) % 3, (cb + (variant == 'color_cycle')) % 3]
            si = [(sa + (variant == 'shape_cycle')) % 3, (sb + (variant == 'shape_cycle')) % 3]
            for relation, (pos_a, pos_b) in enumerate(positions):
                objects = [obj(list(COLORS)[c], SHAPES[s], *pos, radius)
                           for c, s, pos in zip(ci, si, [pos_a, pos_b])]
                sid = f'spatial/{group:03d}/{variant}/{relation}'
                row = {'image_id': sid, 'group_id': f'spatial/{group:03d}', 'variant': variant,
                       'target': relation, 'objects': objects, 'background': 245 if group % 2 else 255,
                       'a': f'a {list(COLORS)[ci[0]]} {SHAPES[si[0]]}',
                       'b': f'a {list(COLORS)[ci[1]]} {SHAPES[si[1]]}'}
                other_color = next(c for c in COLORS if c not in [o['color'] for o in objects])
                control = [f"a drawing of {row['a']} and {row['b']}.",
                           f"a drawing of a {other_color} {objects[0]['shape']} and {row['b']}."]
                reverse = (group + relation + cfg['appearance_variants'].index(variant)) % 2
                row.update(control_captions=control[::-1] if reverse else control, control_target=reverse)
                if materialize:
                    identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    path = folder / f'{identity}.png'
                    render(row).save(path)
                    row.update(image_path=path.relative_to(ROOT).as_posix(), image_sha256=sha256(path))
                rows.append(row)
    return rows


def spatial_trials(images, config, medium):
    rows = []
    for i, im in enumerate(images):
        for t in range(3):
            for reverse in (False, True):
                rows.append({'sample_id': f"{im['image_id']}/t{t}/o{int(reverse)}", 'image_index': i,
                             'group_id': im['group_id'], 'target': im['target'], 'template': t,
                             'order': int(reverse), 'variant': im['variant'],
                             'captions': captions(im['a'], im['b'], t, reverse, medium, config)})
    return rows


def shuffled_categories(classes, seed):
    texts = {c['coarse_id']: c['category_text'] for c in classes}
    rng = np.random.default_rng(seed)
    for _ in range(100000):
        perm = rng.permutation(20)
        if all(texts[c] != texts[int(perm[c])] for c in range(20)):
            return [texts[int(perm[c['coarse_id']])] for c in classes], perm.tolist()
    raise RuntimeError('Unable to generate a category derangement.')


def category_prompts(classes, config, tokenizer):
    cfg = config['category']
    prompts = {'baseline': [r['baseline_prompt'] for r in classes],
               'category': [r['category_prompt'] for r in classes]}
    metadata, neutral = [], []
    for row in classes:
        candidates = [cfg['neutral_template'].format(**{'class': row['class_name'].replace('_', ' '), 'suffix': suffix})
                      for suffix in cfg['neutral_suffix_pool']]
        lengths = (tokenizer([row['category_prompt'], *candidates]) != 0).sum(1).tolist()
        pick = min(range(len(candidates)), key=lambda i: (abs(lengths[i+1]-lengths[0]), i))
        neutral.append(candidates[pick])
        metadata.append({'class_id': row['class_id'], 'category_tokens': lengths[0],
                         'neutral_tokens': lengths[pick+1], 'difference': lengths[pick+1]-lengths[0],
                         'neutral_suffix': cfg['neutral_suffix_pool'][pick]})
    prompts['neutral'] = neutral
    permutations_log = {}
    for seed in cfg['shuffle_seeds']:
        wrong, perm = shuffled_categories(classes, seed)
        name = f'shuffle_{seed}'
        prompts[name] = [cfg['category_template'].format(**{'class': r['class_name'].replace('_', ' '), 'category': w})
                         for r, w in zip(classes, wrong)]
        permutations_log[name] = perm
    return prompts, metadata, permutations_log
