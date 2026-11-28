import unittest

import numpy as np

from vlm_probe.depth.common import read_json
from vlm_probe.depth.natural import make_images
from vlm_probe.spatial_diagnostics import ablation_v1 as t2
from vlm_probe.spatial_diagnostics.quadrant_mitigation import relational_scores_from_regions


def fake_source(target, bbox_a, bbox_b, width=640, height=480, sid=1):
    return {'source_image_id': sid, 'category_a': 'cup', 'category_b': 'laptop', 'target': target,
            'width': width, 'height': height, 'bbox_a': bbox_a, 'bbox_b': bbox_b}


class SpatialAblationTests(unittest.TestCase):
    def test_config_is_frozen_with_preregistered_statistics(self):
        cfg = read_json(t2.CONFIG)
        self.assertEqual(cfg['status'], 'frozen')
        self.assertEqual((cfg['bootstrap_draws'], cfg['bootstrap_seed']), (2000, 1402))
        self.assertEqual(cfg['parity_tolerance'], 1e-6)
        self.assertEqual(cfg['sets']['B']['primary_choices'], 2)
        self.assertEqual(cfg['sets']['A']['primary_choices'], 4)

    def test_swapped_regions_flip_every_two_way_prediction(self):
        rng = np.random.default_rng(0)
        regions = rng.normal(size=(50, 4, 2))
        native = relational_scores_from_regions(regions[:, :, 0], regions[:, :, 1])[:, :2].argmax(1)
        swapped = t2.swapped_scores(regions)[:, :2].argmax(1)
        self.assertTrue(np.all(native != swapped))

    def test_bbox_rules_follow_the_horizontal_flip(self):
        # A in the left part, B in the right part of the raw image: "A left of B" (target 0).
        rows = make_images([fake_source(0, [40, 200, 80, 60], [500, 210, 90, 70])])
        centroid, half, same = t2.bbox_scores(rows, 224, 0.99)
        self.assertEqual([int(r['target']) for r in rows], [0, 1])
        self.assertEqual(centroid[:, :2].argmax(1).tolist(), [0, 1])
        self.assertEqual(half[:, :2].argmax(1).tolist(), [0, 1])
        self.assertFalse(same.any())

    def test_same_half_flag_when_both_objects_share_a_half(self):
        rows = make_images([fake_source(0, [100, 200, 40, 40], [200, 210, 40, 40])])
        _, _, same = t2.bbox_scores(rows, 224, 0.99)
        self.assertTrue(same.all())

    def test_vertical_relation_uses_vertical_axis(self):
        rows = make_images([fake_source(2, [300, 20, 60, 60], [300, 380, 60, 60])])
        centroid, half, _ = t2.bbox_scores(rows, 224, 0.99)
        self.assertEqual(int(centroid[0].argmax()), 2)
        self.assertEqual(int(half[0].argmax()), 2)

    def test_pad_full_keeps_the_whole_image(self):
        from PIL import Image
        image = Image.new('RGB', (60, 20), (255, 0, 0))
        canvas = t2.pad_full(image, (1, 2, 3))
        self.assertEqual(canvas.size, (60, 60))
        self.assertEqual(canvas.getpixel((30, 30)), (255, 0, 0))
        self.assertEqual(canvas.getpixel((0, 0)), (1, 2, 3))
        pixels = np.asarray(canvas).reshape(-1, 3)
        self.assertEqual(int((pixels == (255, 0, 0)).all(1).sum()), 60 * 20)

    def test_set_b_cohort_keeps_98_complete_pairs(self):
        cfg = read_json(t2.CONFIG)
        _, rows = t2.set_rows(cfg, 'B')
        mask = t2.cohort_mask(cfg, 'B', rows)
        self.assertEqual(int(mask.sum()), 196)
        kept = [r['group_id'] for r, m in zip(rows, mask) if m]
        self.assertTrue(all(kept.count(g) == 2 for g in set(kept)))
        _, rows_a = t2.set_rows(cfg, 'A')
        self.assertTrue(t2.cohort_mask(cfg, 'A', rows_a).all())

    def test_interpretation_rule_matches_protocol(self):
        cfg = read_json(t2.CONFIG)
        self.assertEqual(t2.interpret(cfg, {'paired_delta_ci95_pp': [-2.0, 8.0]}), 'center_crop_not_main_cause')
        self.assertEqual(t2.interpret(cfg, {'paired_delta_ci95_pp': [12.0, 30.0]}), 'center_crop_contributes')
        self.assertEqual(t2.interpret(cfg, {'paired_delta_ci95_pp': [4.0, 15.0]}), 'unclear')


if __name__ == '__main__':
    unittest.main()
