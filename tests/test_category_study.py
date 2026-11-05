import copy
import unittest

from vlm_probe.category_study.metrics import summarize
from vlm_probe.category_study.runner import get_predictions
from vlm_probe.category_study.taxonomy import CATEGORY_TEXT, STUDY, build_classes


class CategoryStudyTests(unittest.TestCase):
    def test_complete_train_taxonomy_and_shared_vehicle_text(self):
        names = list(CATEGORY_TEXT)
        meta = {'fine_label_names': [f'class_{i}' for i in range(100)], 'coarse_label_names': names}
        train = {'fine_labels': list(range(100)) * 500, 'coarse_labels': [i // 5 for i in range(100)] * 500}
        classes = build_classes(meta, train, STUDY)
        self.assertEqual(len(classes), 100)
        self.assertEqual(len({r['category_text'] for r in classes}), 19)
        self.assertEqual(classes[0]['category_prompt'], 'a photo of a class 0, a type of aquatic mammal.')
        wrong = copy.deepcopy(train)
        wrong['coarse_labels'][100] = 1
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            build_classes(meta, wrong, STUDY)
        missing = copy.deepcopy(train)
        missing['fine_labels'] = [0 if x == 99 else x for x in missing['fine_labels']]
        missing['coarse_labels'] = [x // 5 for x in missing['fine_labels']]
        with self.assertRaisesRegex(ValueError, 'five fine classes'):
            build_classes(meta, missing, STUDY)

    def test_all_100_candidates_and_tie_rule(self):
        import numpy as np
        scores = {k: np.zeros((2, 100), dtype=np.float32) for k in ('baseline', 'category')}
        scores['category'][0, 99] = .5
        predictions = get_predictions(scores, 2)
        self.assertEqual(predictions['category'], [99, 0])
        bad = {k: v[:, :5] for k, v in scores.items()}
        with self.assertRaisesRegex(ValueError, '100'):
            get_predictions(bad, 2)
        scores['category'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid cosine'):
            get_predictions(scores, 2)

    def test_paired_counts_and_error_destinations(self):
        classes = [{'class_name': f'c{i}', 'coarse_id': i // 2, 'coarse_name': f'g{i // 2}'} for i in range(4)]
        # One fix, one regression, one both-correct, one both-wrong.
        y, b, c = [0, 0, 1, 2], [1, 0, 1, 3], [0, 2, 1, 3]
        metrics, rows, groups, _ = summarize(y, b, c, classes, draws=100, seed=0)
        self.assertEqual(metrics['paired']['wrong_to_correct'], 1)
        self.assertEqual(metrics['paired']['correct_to_wrong'], 1)
        self.assertEqual(metrics['paired']['both_correct'], 1)
        self.assertEqual(metrics['paired']['both_wrong'], 1)
        self.assertEqual(metrics['paired']['delta_percentage_points'], 0)
        self.assertEqual(groups[0]['baseline_errors_within_group'], 1)
        self.assertEqual(groups[0]['category_errors_across_groups'], 1)
        self.assertIsNone(rows[3]['category_accuracy_percent'])
        same = summarize(y, b, b, classes, draws=100, seed=0)[0]
        self.assertEqual(same['paired']['delta_ci95_percentage_points'], [0, 0])
