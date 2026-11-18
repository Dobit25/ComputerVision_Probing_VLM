"""Fixed English category wording; mapping is derived from official train labels."""

import pickle
from collections import Counter

from ..assets import digest_file
from ..io_utils import sha256

CATEGORY_TEXT = {
    'aquatic_mammals': 'aquatic mammal', 'fish': 'fish', 'flowers': 'flower',
    'food_containers': 'food container', 'fruit_and_vegetables': 'fruit or vegetable',
    'household_electrical_devices': 'household electrical device',
    'household_furniture': 'household furniture', 'insects': 'insect',
    'large_carnivores': 'large carnivore',
    'large_man-made_outdoor_things': 'man-made outdoor structure',
    'large_natural_outdoor_scenes': 'natural outdoor feature',
    'large_omnivores_and_herbivores': 'large omnivore or herbivore',
    'medium_mammals': 'medium-sized mammal',
    'non-insect_invertebrates': 'non-insect invertebrate', 'people': 'person',
    'reptiles': 'reptile', 'small_mammals': 'small mammal', 'trees': 'tree',
    'vehicles_1': 'vehicle', 'vehicles_2': 'vehicle',
}
SOURCE_MD5 = {'train': '16019d7e3df5f24257cddd939b257f8d',
              'meta': '7973b15100ade9c7d40fb424638fde48',
              'test': 'f0ef6b0ae62326f3e7ffdfab6717acfc'}
STUDY = {'study_version': 1, 'base_config': 'configs/clip_b32_cifar100.json',
         'baseline_run': 'results/full_cifar100_clip_b32_v1',
         'baseline_template': 'a photo of a {class}.',
         'category_template': 'a photo of a {class}, a type of {category}.',
         'taxonomy_source': 'official_train_fine_coarse_labels', 'candidate_count': 100,
         'bootstrap_draws': 1000, 'bootstrap_seed': 0}


def read_source(root, name):
    path = root / 'data/cifar-100-python' / name
    if digest_file(path, 'md5') != SOURCE_MD5[name]:
        raise ValueError(f'Official CIFAR checksum mismatch: {name}')
    with path.open('rb') as stream:
        return pickle.load(stream, encoding='latin1')


def build_classes(meta, train, config):
    names, coarse_names = meta['fine_label_names'], meta['coarse_label_names']
    if len(names) != 100 or set(coarse_names) != set(CATEGORY_TEXT):
        raise ValueError('Unexpected CIFAR taxonomy names.')
    fine_labels, coarse_labels = train['fine_labels'], train['coarse_labels']
    if len(fine_labels) != 50000 or len(coarse_labels) != 50000:
        raise ValueError('Unexpected train split size.')
    mapping = {}
    for fine, coarse in zip(fine_labels, coarse_labels):
        if type(fine) is not int or type(coarse) is not int or not 0 <= fine < 100 or not 0 <= coarse < 20:
            raise ValueError('Invalid taxonomy label.')
        if fine in mapping and mapping[fine] != coarse:
            raise ValueError('Conflicting fine/coarse mapping.')
        mapping[fine] = coarse
    if set(mapping) != set(range(100)) or Counter(mapping.values()) != Counter({i: 5 for i in range(20)}):
        raise ValueError('Expected exactly five fine classes in each of twenty groups.')
    rows = []
    for i, name in enumerate(names):
        coarse = coarse_names[mapping[i]]
        values = {'class': name.replace('_', ' '), 'category': CATEGORY_TEXT[coarse]}
        rows.append({'class_id': i, 'class_name': name, 'coarse_id': mapping[i], 'coarse_name': coarse,
                     'category_text': CATEGORY_TEXT[coarse],
                     'baseline_prompt': config['baseline_template'].format(**values),
                     'category_prompt': config['category_template'].format(**values)})
    return rows


def load_classes(root, config):
    classes = build_classes(read_source(root, 'meta'), read_source(root, 'train'), config)
    provenance = {name: {'md5': SOURCE_MD5[name], 'sha256': sha256(root / 'data/cifar-100-python' / name)}
                  for name in ('train', 'meta')}
    return classes, provenance
