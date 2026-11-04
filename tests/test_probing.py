import copy
import unittest
from collections import Counter, defaultdict

from vlm_probe.probing.stimuli import scenes, validate_scenes, render, ordered_samples
from vlm_probe.probing.metrics import score_row, aggregate, bootstrap_ci


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = scenes()

    def test_counts_and_balanced_targets(self):
        self.assertEqual(Counter(r["task"] for r in self.rows), {"counting": 1080, "spatial": 720, "negation": 1440})
        for task, expected in [("counting", {i: 180 for i in range(6)}), ("spatial", {i: 180 for i in range(4)}),
                               ("negation", {0: 720, 1: 720})]:
            self.assertEqual(Counter(r["target"] for r in self.rows if r["task"] == task), expected)

    def test_full_and_smoke_input_order_matches_evaluation_task_order(self):
        tasks = ["counting", "spatial", "negation"]
        full = ordered_samples(self.rows, tasks)
        self.assertEqual([r["task"] for r in full], ["counting"] * 1080 + ["spatial"] * 720 + ["negation"] * 1440)
        self.assertEqual({r["sample_id"] for r in full}, {r["sample_id"] for r in self.rows})
        self.assertEqual([r["task"] for r in ordered_samples(self.rows, tasks, 64)],
                         [task for task in tasks for _ in range(64)])

    def test_geometry_and_area_controls(self):
        validate_scenes(self.rows)
        for row in self.rows:
            if row["task"] == "counting" and row["condition"] == "fixed_total_area":
                self.assertAlmostEqual(sum(o["radius"] ** 2 for o in row["objects"]), 1024)
        broken = copy.deepcopy(self.rows[0])
        broken["objects"][0]["x"] = 0
        with self.assertRaisesRegex(ValueError, "clipped"):
            validate_scenes([broken])

    def test_counterfactual_negation_changes_only_target_color(self):
        groups = defaultdict(list)
        for row in self.rows:
            if row["task"] == "negation" and row["condition"] == "with_without":
                groups[row["group_id"]].append(row)
        self.assertEqual(len(groups), 360)
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            a, b = pair
            self.assertEqual(a["objects"][1], b["objects"][1])
            self.assertEqual({k: v for k, v in a["objects"][0].items() if k != "color"},
                             {k: v for k, v in b["objects"][0].items() if k != "color"})
            self.assertNotEqual(a["objects"][0]["color"], b["objects"][0]["color"])

    def test_render_size_and_determinism(self):
        for row in self.rows[::137]:
            im = render(row)
            self.assertEqual(im.size, (224, 224))
            self.assertEqual(im.mode, "RGB")
            self.assertEqual(im.tobytes(), render(row).tobytes())

    def test_ties_invalid_scores_and_conditioned_metrics(self):
        sample = self.rows[0]
        row = score_row(sample, [0.2] * 6, [0.0] * 9)
        self.assertEqual(row["prediction"], 0)
        self.assertTrue(row["exact_tie"])
        with self.assertRaises(ValueError):
            score_row(sample, [float("nan")] * 6, [0.0] * 9)
        stats = aggregate([row])
        self.assertEqual(stats["control_correct_samples"], 1)
        self.assertEqual(stats["accuracy_given_control_correct"], 1)
        self.assertEqual(bootstrap_ci([row], draws=100), [1.0, 1.0])
