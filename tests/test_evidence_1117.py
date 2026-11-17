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

    def test_saved_outputs_match_recomputation(self):
        if not ev.OUT.exists():
            self.skipTest("outputs not created")
        self.assertEqual(ev.main(["audit"]), 0)


if __name__ == "__main__":
    unittest.main()
