import unittest

from vlm_probe import evidence_1117 as ev


class Evidence1117Tests(unittest.TestCase):
    def test_pet_species_rule(self):
        self.assertEqual(ev.pet_species("oxford_pet/test/Abyssinian_201.jpg"), "cat")
        self.assertEqual(ev.pet_species("oxford_pet/test/beagle_1.jpg"), "dog")

    def test_bootstrap_is_deterministic_and_bounded(self):
        flags = [1] * 4 + [0] * 96
        a = ev.bootstrap_rate(flags, seed=1, draws=200)
        self.assertEqual(a, ev.bootstrap_rate(flags, seed=1, draws=200))
        self.assertLessEqual(0.0, a[0])
        self.assertLessEqual(a[0], 4.0)
        self.assertLessEqual(4.0, a[1])

    def test_counting_matches_breakdown_not_old_map(self):
        c = ev.counting_summary()
        self.assertEqual((c["n"], c["correct"]), (1080, 355))
        self.assertEqual(c["groups"]["color"]["blue"]["correct"], 98)
        self.assertEqual(c["groups"]["color"]["red"]["correct"], 136)
        self.assertEqual(sum(map(sum, c["confusion_rows_true_cols_pred"])), 1080)

    def test_committed_inputs_pass_verification(self):
        v = ev.verify_inputs()
        self.assertEqual(v["probing"]["rows"], {"counting": 1080, "negation": 1440})
        self.assertEqual(v["pet"]["rows"], 3669)
        self.assertEqual(v["pet"]["classes"], {"cat": 12, "dog": 25})

    def test_probing_verifier_rejects_fields_not_matching_scores(self):
        row = {"sample_id": "c/1", "task": "counting", "scores": [0.1, 0.3, 0.2], "target": 1,
               "prediction": 1, "correct": True, "absolute_count_error": 0, "number": 2}
        neg = {"sample_id": "n/1", "task": "negation", "scores": [0.5, 0.5], "target": 0,
               "prediction": 0, "correct": True}  # exact tie -> lowest index
        ev.verify_probing_rows([row, neg])
        for field, value in (("prediction", 2), ("correct", False), ("absolute_count_error", 1), ("number", 3)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ev.verify_probing_rows([dict(row, **{field: value}), neg])

    def test_pet_species_mapping_rejects_conflicts_and_unknown_predictions(self):
        def r(sid, t, p):
            return {"sample_id": sid, "target_name": t, "prediction_name": p}
        ok = [r("x/Abyssinian_1.jpg", "Abyssinian", "Beagle"), r("x/beagle_1.jpg", "Beagle", "Beagle")]
        self.assertEqual(ev.pet_class_species(ok), {"Abyssinian": "cat", "Beagle": "dog"})
        with self.assertRaisesRegex(ValueError, "more than one species"):
            ev.pet_class_species(ok + [r("x/beagle_2.jpg", "Abyssinian", "Beagle")])
        with self.assertRaisesRegex(ValueError, "prediction_name"):
            ev.pet_class_species(ok + [r("x/beagle_3.jpg", "Beagle", "Sphynx")])
        with self.assertRaisesRegex(ValueError, "pinned cat-breed"):
            ev.pet_class_species([r("x/Pug_1.jpg", "Pug", "Pug")])

    def test_pet_row_verifier_rejects_inconsistent_correct_flag(self):
        rows = ev.load_pet_rows()
        classes = ev.json.loads(ev.PET_CLASSES.read_text(encoding="utf-8"))
        bad = [dict(rows[0], correct="0" if rows[0]["correct"] == "1" else "1")] + rows[1:]
        with self.assertRaisesRegex(ValueError, "correct"):
            ev.verify_pet_rows(bad, classes)

    def test_c07_amendment_matches_followup_and_base_claim(self):
        root = ev.ROOT
        amend = ev.json.loads((root / "docs/references/claim_evidence_amendments.json").read_text(encoding="utf-8"))
        base = ev.json.loads((root / "docs/references/claim_evidence_map.json").read_text(encoding="utf-8"))
        follow = ev.json.loads((root / "results/human_review_followup_v1/metrics.json").read_text(encoding="utf-8"))
        a = {x["amendment_id"]: x for x in amend["amendments"]}["A1118-01"]
        c07 = {c["claim_id"]: c for c in base["claims"]}["C07"]
        self.assertIn(a["supersedes_limitation"], c07["limitations"])
        self.assertIn(a["keeps_limitation"], c07["limitations"])
        after = follow["holistic_all_conditions_after"]["t2_o0"]
        s = a["sensitivity_metrics_after_review_98_sources"]
        self.assertEqual((s["t2_o0_2x2_correct"], s["n_sources"]), (after["correct"], after["n_sources"]))
        self.assertAlmostEqual(s["t2_o0_2x2_accuracy_percent"], after["accuracy_percent"])
        self.assertEqual(s["t2_o0_2x2_wilson_ci95_percent"], after["wilson_ci95_percent"])
        self.assertAlmostEqual(s["t2_o0_2x2_one_sided_binomial_p_greater_0_5"], after["binomial_p_greater_0_5"])
        self.assertAlmostEqual(s["t2_o0_two_way_accuracy_percent"], after["two_way_accuracy_percent"])
        self.assertAlmostEqual(s["t2_o0_both_images_correct_percent"], after["both_images_correct_percent"])

    def test_saved_outputs_match_recomputation(self):
        if not ev.OUT.exists():
            self.skipTest("outputs not created")
        self.assertEqual(ev.main(["audit"]), 0)


if __name__ == "__main__":
    unittest.main()
