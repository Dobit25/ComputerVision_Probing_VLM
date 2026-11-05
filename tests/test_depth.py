import copy
import unittest
from collections import Counter, defaultdict

import numpy as np

from vlm_probe.depth.common import CONFIG, read_json, cluster_ci, paired_class_ci
from vlm_probe.depth.design import captions, INVERSE, spatial_images, shuffled_categories
from vlm_probe.depth.spatial import compute_metrics


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(CONFIG)

    def test_spatial_labels_bounds_and_counterfactual_positions(self):
        rows = spatial_images(self.cfg, materialize=False)
        self.assertEqual(len(rows), 1440)
        self.assertEqual(Counter(r['target'] for r in rows), Counter({r: 360 for r in range(4)}))
        groups = defaultdict(list)
        for r in rows:
            a, b = r['objects']
            for o in (a, b):
                self.assertLessEqual(o['radius'], min(o['x'], o['y'], 224-o['x'], 224-o['y']))
            dx, dy = b['x']-a['x'], b['y']-a['y']
            actual = (0 if dx > 0 else 1) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)
            self.assertEqual(actual, r['target'])
            self.assertNotEqual(a['color'], b['color'])
            self.assertNotEqual(a['shape'], b['shape'])
            self.assertGreater(max(abs(dx), abs(dy)), a['radius']+b['radius'])
            groups[(r['group_id'], r['variant'])].append(r)
        for group in groups.values():
            for first, second in [(group[0], group[1]), (group[2], group[3])]:
                self.assertEqual([(o['x'], o['y']) for o in first['objects']],
                                 [(o['x'], o['y']) for o in second['objects']][::-1])

    def test_appearance_changes_preserve_geometry(self):
        rows = spatial_images(self.cfg, materialize=False)
        groups = defaultdict(list)
        for row in rows:
            groups[(row['group_id'], row['target'])].append(row)
        for group in groups.values():
            geometry = [[(o['x'], o['y'], o['radius']) for o in r['objects']] for r in group]
            self.assertEqual(geometry[0], geometry[1])
            self.assertEqual(geometry[0], geometry[2])
            self.assertNotEqual(group[0]['a'], group[1]['a'])
            self.assertNotEqual(group[0]['a'], group[2]['a'])

    def test_reverse_order_keeps_canonical_semantics(self):
        result = captions('a red circle', 'a blue square', 0, True, 'drawing', self.cfg)
        self.assertEqual(result[0], 'a drawing of a blue square to the right of a red circle.')
        self.assertEqual(result[2], 'a drawing of a blue square below a red circle.')
        self.assertEqual(tuple(INVERSE[i] for i in INVERSE), tuple(range(4)))

    def test_derangement_preserves_category_multiset_including_vehicles(self):
        classes = [{'coarse_id': c, 'category_text': f'group {c}' if c < 18 else 'vehicle'}
                   for c in range(20) for _ in range(5)]
        for seed in self.cfg['category']['shuffle_seeds']:
            wrong, perm = shuffled_categories(classes, seed)
            self.assertEqual(sorted(perm), list(range(20)))
            self.assertEqual(Counter(wrong), Counter(c['category_text'] for c in classes))
            self.assertTrue(all(w != c['category_text'] for w, c in zip(wrong, classes)))

    def test_cluster_bootstrap_does_not_treat_repeated_captions_as_independent(self):
        original = cluster_ci([0, 1, 0, 1], ['a', 'b', 'c', 'd'], draws=100)
        repeated = cluster_ci(np.repeat([0, 1, 0, 1], 6), np.repeat(['a', 'b', 'c', 'd'], 6), draws=100)
        self.assertEqual(original, repeated)
        self.assertEqual(paired_class_ci(np.zeros(20), np.repeat([0, 1], 10), draws=100), [0, 0])

    def test_constant_predictions_are_consistent_but_not_counterfactually_correct(self):
        rows = []
        for t in range(3):
            for order in range(2):
                for relation in range(4):
                    rows.append({'sample_id': f'{t}/{order}/{relation}', 'image_index': relation,
                                 'group_id': 'group', 'target': relation, 'template': t,
                                 'order': order, 'variant': 'original'})
        scores = np.tile([.4, .1, .2, .3], (len(rows), 1)).astype(np.float32)
        result, _ = compute_metrics(rows, scores, config={'bootstrap_draws': 20, 'bootstrap_seed': 0})
        self.assertEqual(result['conditions']['t0_o0_all']['accuracy_percent'], 25)
        self.assertEqual(result['conditions']['t0_o0_all']['all_four_relations_correct_percent'], 0)
        self.assertEqual(result['conditions']['t0_o0_all']['horizontal_pair_both_correct_percent'], 0)
        self.assertEqual(result['equivalent_orders']['t0']['prediction_agreement_percent'], 100)
        broken = copy.deepcopy(rows)
        broken[1]['sample_id'] = broken[0]['sample_id']
        with self.assertRaises(ValueError):
            compute_metrics(broken, scores)

    def test_natural_geometry_rejects_diagonal_and_cropped_targets(self):
        from vlm_probe.depth.natural import relation_from_boxes
        from vlm_probe.depth.design import HFLIP
        a, b = [20, 90, 50, 50], [155, 90, 50, 50]
        self.assertEqual(relation_from_boxes(a, b, 224, 224, self.cfg['natural']), 0)
        af = [224-a[0]-a[2], a[1], a[2], a[3]]
        bf = [224-b[0]-b[2], b[1], b[2], b[3]]
        self.assertEqual(relation_from_boxes(af, bf, 224, 224, self.cfg['natural']), HFLIP[0])
        self.assertIsNone(relation_from_boxes([20, 20, 50, 50], [155, 155, 50, 50], 224, 224, self.cfg['natural']))
        self.assertIsNone(relation_from_boxes(a, b, 400, 224, self.cfg['natural']))

    def test_neutral_selection_is_token_based_and_preserves_all_classes(self):
        import open_clip
        from vlm_probe.depth.common import ROOT
        from vlm_probe.depth.design import category_prompts
        classes = read_json(ROOT / 'results/full_cifar100_category_v1/classes.json')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        prompts, metadata, perms = category_prompts(classes, self.cfg, tokenizer)
        self.assertEqual(len(prompts), 8)
        self.assertTrue(all(len(texts) == 100 for texts in prompts.values()))
        self.assertLessEqual(max(abs(r['difference']) for r in metadata), 1)
        self.assertEqual(prompts['baseline'], [c['baseline_prompt'] for c in classes])
        self.assertEqual(prompts['category'], [c['category_prompt'] for c in classes])

    def test_assignment_gap_removes_additive_bias_without_erasing_interaction(self):
        from vlm_probe.depth.pairs import assignment_gap
        scores = np.array([[.3, .1], [.2, .4]])
        image_bias = np.array([[.07], [-.11]])
        text_bias = np.array([[.6, -.2]])
        self.assertAlmostEqual(assignment_gap(scores), .4)
        self.assertAlmostEqual(assignment_gap(scores + image_bias + text_bias), .4)
        self.assertAlmostEqual(assignment_gap(scores[:, ::-1]), -.4)
        with self.assertRaises(ValueError):
            assignment_gap(np.ones((2, 3)))

    def test_audit_rejects_modified_scores_and_labels(self):
        from vlm_probe.depth.audit import equal
        with self.assertRaises(ValueError):
            equal({'scores': [.1, .2]}, {'scores': [.1, .3]})
        with self.assertRaises(ValueError):
            equal({'target': 0}, {'target': 1})


if __name__ == '__main__':
    unittest.main()
