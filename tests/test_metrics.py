import unittest

from vlm_probe.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_imbalanced_classes_and_absent_class(self):
        metrics, per_class, matrix = classification_metrics([0, 0, 0, 1], [0, 1, 1, 1], ["a", "b", "c"])
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["top1_accuracy"], 0.5)
        self.assertEqual(metrics["top1_accuracy_percent"], 50.0)
        self.assertEqual(matrix, [[1, 2, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(per_class[0]["accuracy"], 1 / 3)
        self.assertEqual(per_class[1]["accuracy"], 1.0)
        self.assertEqual(per_class[2]["support"], 0)
        self.assertIsNone(per_class[2]["accuracy"])

    def test_invalid_predictions_are_rejected(self):
        for targets, predictions in [([], []), ([0], []), ([0], [-1]), ([0], [2]), ([0], [0.0])]:
            with self.subTest(targets=targets, predictions=predictions), self.assertRaises(ValueError):
                classification_metrics(targets, predictions, ["a", "b"])

    def test_perfect_predictions(self):
        metrics, _, matrix = classification_metrics([1, 0, 1], [1, 0, 1], ["a", "b"])
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(matrix, [[1, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()
