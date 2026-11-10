"""Unit tests for P0-P5 extensions: data/claim ledgers, blinded review, spatial PSR, prompt study, and Vietnamese BPE."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vlm_probe.depth.blinded_review import (
    classify_decision_branch,
    cohens_kappa,
    export_blinded_sheets,
    score_confirmation_consensus,
)
from vlm_probe.depth.common import ROOT, read_json
from vlm_probe.prompt_study.design import (
    build_counting_prompts,
    build_dataset_templates,
    build_negation_prompts,
)
from vlm_probe.prompt_study.metrics import (
    logit_average_ensemble,
    probability_average_ensemble,
    summarize_template_suite,
    text_embedding_ensemble,
)
from vlm_probe.prompt_study.offline_analysis import analyze_saved_probing_ensembles
from vlm_probe.spatial_diagnostics.mechanism import (
    analyze_frozen_spatial_runs,
    decompose_pair_matrix,
)
from vlm_probe.spatial_diagnostics.quadrant_mitigation import (
    evaluate_geometric_upper_bound,
    relational_scores_from_regions,
)
from vlm_probe.vietnamese_study.bpe_diagnostics import analyze_bpe_tokenization
from vlm_probe.vietnamese_study.lexicon import (
    CIFAR100_VI_LEXICON,
    build_cifar100_multilingual_prompts,
    strip_vietnamese_diacritics,
)


class ExtensionsP0ToP5Tests(unittest.TestCase):
    def test_p0_and_p1_reference_ledgers_exist_and_match_frozen_metrics(self):
        ledger = read_json(ROOT / "docs/references/data_ledger.json")
        claims = read_json(ROOT / "docs/references/claim_evidence_map.json")
        self.assertEqual(ledger["ledger_version"], 1)
        self.assertGreaterEqual(len(ledger["datasets"]), 9)
        claim_by_id = {c["claim_id"]: c for c in claims["claims"]}
        self.assertEqual(set(claim_by_id.keys()), {"C01", "C02", "C03", "C04", "C05", "C06", "C07"})
        conf_metrics = read_json(ROOT / "results/full_natural_confirmation_v1/metrics.json")
        self.assertAlmostEqual(
            claim_by_id["C07"]["metrics"]["new_100_confirmation_2x2_t2_forward_percent"],
            conf_metrics["t2_o0"]["accuracy_percent"],
            places=5,
        )

    def test_p1_2_blinded_review_export_and_full_retention_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = export_blinded_sheets(seed=202611, output_dir=tmp_path)
            self.assertEqual(manifest["confirmation_100"]["num_items"], 100)
            self.assertEqual(manifest["exploratory_200"]["num_items"], 200)

            sheet_path = tmp_path / "blinded_sheet_confirmation_100.csv"
            key_path = tmp_path / "answer_key_confirmation_100.json"
            with sheet_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 100)
            # Ensure no ground-truth source_image_id or CLIP score is leaked in the blinded CSV
            self.assertNotIn("source_image_id", rows[0])
            self.assertNotIn("gap", rows[0])

            # Fill all 100 as accepted by two identical annotators and verify parity with frozen run
            sheet_a = tmp_path / "completed_a.csv"
            sheet_b = tmp_path / "completed_b.csv"
            for r in rows:
                r["annotator_id"] = "R1"
                r["both_objects_visible"] = "1"
                r["class_names_correct"] = "1"
                r["relation_clear"] = "1"
                r["overall_accept"] = "1"
            with sheet_a.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            with sheet_b.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            scored = score_confirmation_consensus(sheet_a, key_path=key_path, sheet_b_path=sheet_b)
            self.assertEqual(scored["n_retained_consensus"], 100)
            self.assertAlmostEqual(scored["primary_t2_o0"]["accuracy_percent"], 53.0, places=5)
            self.assertAlmostEqual(scored["inter_annotator_agreement"]["cohens_kappa"], 1.0, places=5)
            self.assertEqual(scored["decision_branch"]["branch"], "A")

    def test_p1_2_cohens_kappa_and_decision_branches(self):
        k = cohens_kappa([True, True, False, False], [True, False, True, False])
        self.assertAlmostEqual(k["cohens_kappa"], 0.0, places=6)
        self.assertEqual(classify_decision_branch(75, 0.01, 55.0, 75.0)["branch"], "D")
        self.assertEqual(classify_decision_branch(95, 0.30, 43.0, 62.0)["branch"], "A")
        self.assertEqual(classify_decision_branch(95, 0.01, 52.0, 70.0)["branch"], "B")

    def test_p2_1_prior_to_signal_decomposition_identities(self):
        # Construct a matrix where assignment gap > 0 (Delta = +0.002), but Prior Bias B = +0.020 dominates
        mat = np.array([
            [0.322, 0.300],  # d0 = +0.022 > 0 (correct for target 0)
            [0.318, 0.300],  # d1 = +0.018 > 0 (wrong for target 1, which needs d1 < 0)
        ])
        dec = decompose_pair_matrix(mat)
        self.assertAlmostEqual(dec["d0"], 0.022, places=7)
        self.assertAlmostEqual(dec["d1"], 0.018, places=7)
        self.assertAlmostEqual(dec["prior_bias_B"], 0.020, places=7)
        self.assertAlmostEqual(dec["spatial_signal_delta"], 0.002, places=7)
        self.assertAlmostEqual(dec["assignment_gap"], 0.004, places=7)
        self.assertAlmostEqual(dec["prior_to_signal_ratio"], 10.0, places=5)
        self.assertTrue(dec["assignment_correct"])
        self.assertFalse(dec["both_images_correct"])
        self.assertTrue(dec["same_prediction_collapse"])

        frozen_psr = analyze_frozen_spatial_runs()
        self.assertIn("synthetic_depth", frozen_psr)
        self.assertIn("natural_confirmation", frozen_psr)
        syn_h = frozen_psr["synthetic_depth"]["conditions"]["t0_o0_horizontal"]
        self.assertAlmostEqual(syn_h["both_images_correct_percent"], 1.1111111, places=3)
        self.assertGreater(syn_h["aggregate_prior_to_signal_ratio"], 5.0)

    def test_p2_2_region_mitigation_antisymmetry_and_geometry_bound(self):
        sa = np.array([0.8, 0.2, 0.5, 0.5])
        sb = np.array([0.1, 0.9, 0.5, 0.5])
        rel = relational_scores_from_regions(sa, sb)
        self.assertAlmostEqual(rel[0], -rel[1], places=7)
        self.assertAlmostEqual(rel[2], -rel[3], places=7)
        self.assertEqual(int(rel.argmax()), 0)

        geom = evaluate_geometric_upper_bound()
        self.assertEqual(geom["synthetic_depth_1440"]["n_images"], 1440)
        self.assertAlmostEqual(geom["synthetic_depth_1440"]["four_way_accuracy_percent"], 100.0, places=5)

    def test_p3_templates_baseline_parity_and_ensembling(self):
        cifar_classes_json = read_json(ROOT / "results/full_cifar100_clip_b32_v1/classes.json")
        class_names = [c["dataset_name"] for c in cifar_classes_json]
        expected_prompts = [c["prompt"] for c in cifar_classes_json]

        tpl_map = build_dataset_templates("cifar100", class_names)
        self.assertEqual(tpl_map["T0_baseline"], expected_prompts)
        self.assertEqual(len(tpl_map), 7)

        cnt_map = build_counting_prompts("red", "circle")
        self.assertEqual(cnt_map["C0_baseline"][0], "a drawing of one red circle.")
        self.assertEqual(cnt_map["C0_baseline"][5], "a drawing of six red circles.")

        neg_map = build_negation_prompts("red", "circle")
        self.assertEqual(
            neg_map["N0_with_without"],
            ["a drawing with a red circle.", "a drawing without a red circle."],
        )

        # Test two-stage L2 text embedding ensemble
        rng = np.random.default_rng(42)
        raw_embs = [rng.normal(size=(10, 16)).astype(np.float32) for _ in range(4)]
        ens_vec, norms = text_embedding_ensemble(raw_embs)
        self.assertEqual(ens_vec.shape, (10, 16))
        np.testing.assert_allclose(np.linalg.norm(ens_vec, axis=1), np.ones(10), atol=1e-5)
        self.assertTrue(((norms > 0.0) & (norms <= 1.000001)).all())

        # Offline analysis on frozen runs
        offline = analyze_saved_probing_ensembles()
        self.assertAlmostEqual(
            offline["negation_probing_v1"]["best_single_accuracy_percent"],
            91.3888888,
            places=3,
        )
        self.assertAlmostEqual(
            offline["negation_probing_v1"]["worst_single_accuracy_percent"],
            80.8333333,
            places=3,
        )

    def test_p4_vietnamese_lexicon_and_bpe_diagnostics(self):
        cifar_classes_json = read_json(ROOT / "results/full_cifar100_clip_b32_v1/classes.json")
        class_names = [c["dataset_name"] for c in cifar_classes_json]
        self.assertEqual(list(CIFAR100_VI_LEXICON.keys()), class_names)

        self.assertEqual(strip_vietnamese_diacritics("đồng hồ"), "dong ho")
        self.assertEqual(strip_vietnamese_diacritics("Đám mây trắng"), "Dam may trang")

        conditions, records = build_cifar100_multilingual_prompts(class_names)
        self.assertEqual(len(records), 100)
        self.assertEqual(conditions["en_baseline"], [c["prompt"] for c in cifar_classes_json])

        bpe = analyze_bpe_tokenization()
        en_mean = bpe["cifar100_labels_summary"]["en_baseline"]["mean_subword_tokens"]
        vi_mean = bpe["cifar100_labels_summary"]["vi_diacritics"]["mean_subword_tokens"]
        nodia_mean = bpe["cifar100_labels_summary"]["vi_no_diacritics"]["mean_subword_tokens"]
        self.assertGreater(vi_mean, en_mean * 2.0)
        self.assertGreater(vi_mean, nodia_mean)


if __name__ == "__main__":
    unittest.main()

