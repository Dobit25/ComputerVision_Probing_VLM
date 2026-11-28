import unittest

import numpy as np

from vlm_probe import final_diagnostics as fd


def negation_row(group, stratum, template, prediction):
    target = 0 if stratum == 'present' else 1
    return {'task': 'negation', 'group_id': group, 'stratum': stratum, 'condition': template,
            'target': target, 'prediction': prediction, 'correct': prediction == target}


class FinalDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fd.load_probing()

    def test_denominators(self):
        tasks = [r['task'] for r in self.rows]
        self.assertEqual(tasks.count('counting'), 1080)
        self.assertEqual(tasks.count('negation'), 1440)
        with np.load(fd.VI / 'scores.npz', allow_pickle=False) as z:
            self.assertEqual(z['vi_diacritics'].shape, (10000, 100))

    def test_negation_matches_known_counts(self):
        neg, _ = fd.negation_metrics(self.rows)
        self.assertEqual(neg['n_trials'], 1440)
        self.assertEqual(neg['all']['accuracy']['k'], 1240)
        self.assertEqual(neg['four_trial_consistency']['all_correct'], 211)
        self.assertAlmostEqual(neg['all']['affirmative_prediction_rate']['percent'], 100 * 692 / 1440)

    def test_affirmative_bias_formula(self):
        rows = []
        for g in range(4):
            for template in ('with_without', 'contains_does_not'):
                rows.append(negation_row(f'n/{g}', 'present', template, 0))
                rows.append(negation_row(f'n/{g}', 'absent', template, 0 if g == 0 else 1))
        neg, _ = fd.negation_metrics(rows)
        # 8 present rows all answered affirmative, plus 2 absent rows of group 0: 10 / 16.
        self.assertAlmostEqual(neg['all']['affirmative_prediction_rate']['percent'], 62.5)
        self.assertEqual(neg['all']['confusion_true_x_pred']['absent'], {'pred_present': 2, 'pred_absent': 6})
        self.assertEqual(neg['four_trial_consistency']['all_correct'], 3)

    def test_counting_groups_partition_the_images(self):
        cnt, table = fd.counting_metrics(self.rows)
        self.assertEqual(cnt['n_images'], 1080)
        self.assertEqual(cnt['control_correct']['accuracy']['n'] + cnt['control_wrong']['accuracy']['n'], 1080)
        self.assertEqual(cnt['control_correct']['accuracy']['n'], 804)
        self.assertEqual(sum(map(sum, cnt['confusion_rows_true_cols_pred'])), 1080)
        self.assertEqual(len(table), 12)

    def test_spearman(self):
        x = np.arange(10)
        self.assertAlmostEqual(fd.spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(fd.spearman(x, -x), -1.0)
        self.assertEqual(fd.rankdata([3, 1, 1, 2]).tolist(), [4.0, 1.5, 1.5, 3.0])

    def test_cognate_heuristic_is_string_based(self):
        self.assertEqual(fd.cognate_ratio('hamster', 'chuot hamster'), 1.0)
        self.assertLess(fd.cognate_ratio('apple', 'qua tao'), fd.COGNATE_MIN_RATIO)


if __name__ == '__main__':
    unittest.main()
