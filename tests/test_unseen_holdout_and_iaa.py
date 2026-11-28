"""Unit tests for WS2: Set B R2 IAA validation/kappa and Set C unseen holdout gating/selection/audit."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vlm_probe import human_review_iaa as iaa
from vlm_probe.depth.common import ROOT, read_json
from vlm_probe.io_utils import sha256, write_json
from vlm_probe.spatial_diagnostics import unseen_holdout as set_c


def _make_blank_row(bid, rel='left'):
    return {
        'blind_id': bid,
        'object_a': 'dog',
        'object_b': 'chair',
        'expected_relation': rel,
        'annotator_id': '',
        'both_objects_visible': '',
        'class_names_correct': '',
        'relation_clear': '',
        'perceived_relation': '',
        'overall_accept': '',
        'exclusion_reason': '',
        'notes': '',
    }


def _fill_row(row, annotator, accept=True, perceived=None):
    rel = perceived or row['expected_relation']
    return {
        **row,
        'annotator_id': annotator,
        'both_objects_visible': 'yes' if accept else 'no',
        'class_names_correct': 'yes',
        'relation_clear': 'yes' if accept else 'no',
        'perceived_relation': rel if accept else 'unclear',
        'overall_accept': 'yes' if accept else 'no',
        'exclusion_reason': '' if accept else 'OCCLUDED_OR_TINY',
        'notes': '',
    }


class IAATests(unittest.TestCase):
    def test_multiclass_cohens_kappa_cases(self):
        perfect = iaa.multiclass_cohens_kappa(['yes', 'no', 'yes', 'no'], ['yes', 'no', 'yes', 'no'])
        self.assertAlmostEqual(perfect['cohens_kappa'], 1.0)
        self.assertAlmostEqual(perfect['raw_agreement_percent'], 100.0)

        degenerate = iaa.multiclass_cohens_kappa(['yes', 'yes', 'yes'], ['yes', 'yes', 'yes'])
        self.assertFalse(degenerate['kappa_defined'])
        self.assertIsNone(degenerate['cohens_kappa'])

        partial = iaa.multiclass_cohens_kappa(['left', 'right', 'left', 'unclear'], ['left', 'left', 'left', 'unclear'])
        self.assertTrue(partial['kappa_defined'])
        self.assertAlmostEqual(partial['raw_agreement_percent'], 75.0)
        self.assertLess(partial['cohens_kappa'], 1.0)

    def test_r2_validation_rejects_r1_tampering_and_contradiction(self):
        blanks = [_make_blank_row('CONFIRMATION_100_000', 'left'), _make_blank_row('CONFIRMATION_100_001', 'right')]
        valid = [_fill_row(blanks[0], 'R2', True), _fill_row(blanks[1], 'R2', False)]
        self.assertEqual(iaa.validate_r2_rows(valid, blanks)['annotator_id'], 'R2')

        same_as_r1 = [_fill_row(blanks[0], 'R1', True), _fill_row(blanks[1], 'R1', False)]
        with self.assertRaises(ValueError):
            iaa.validate_r2_rows(same_as_r1, blanks)

        tampered = [{**valid[0], 'object_a': 'cat'}, valid[1]]
        with self.assertRaises(ValueError):
            iaa.validate_r2_rows(tampered, blanks)

        contradictory = [{**valid[0], 'perceived_relation': 'right'}, valid[1]]
        with self.assertRaises(ValueError):
            iaa.validate_r2_rows(contradictory, blanks)


class SetCHoldoutTests(unittest.TestCase):
    def test_disjoint_selection_excludes_seen_ids_hashes_and_dhash_duplicates(self):
        possible = [
            {'rank': '01', 'source_image_id': 10, 'target': 0},  # seen ID
            {'rank': '02', 'source_image_id': 11, 'target': 0},  # seen SHA-256
            {'rank': '03', 'source_image_id': 12, 'target': 0},  # near-dup dHash to seen
            {'rank': '04', 'source_image_id': 13, 'target': 0},  # valid left
            {'rank': '05', 'source_image_id': 14, 'target': 1},  # near-dup to 13
            {'rank': '06', 'source_image_id': 15, 'target': 1},  # valid right
        ]
        sha_map = {11: 'sha_seen', 12: 'sha12', 13: 'sha13', 14: 'sha14', 15: 'sha15'}
        h_seen = np.zeros((8, 8), dtype=bool)
        h_distinct_a = np.ones((8, 8), dtype=bool)
        h_distinct_b = np.eye(8, dtype=bool)
        dhash_map = {
            'p12': h_seen,
            'p13': h_distinct_a,
            'p14': h_distinct_a,
            'p15': h_distinct_b,
        }
        seen_orig = np.stack([h_seen])
        seen_flip = np.stack([h_seen])
        kept, dups, quotas = set_c.select_disjoint_candidates(
            possible,
            {10},
            {'sha_seen'},
            seen_orig,
            seen_flip,
            lambda r: {**r, 'image_path': f'p{r["source_image_id"]}', 'image_sha256': sha_map[r['source_image_id']]},
            lambda path, mirror=False: dhash_map[Path(path).name],
            max_dist=6,
            per_relation=1,
        )
        self.assertEqual([r['source_image_id'] for r in kept], [13, 15])
        self.assertEqual(quotas, {0: 1, 1: 1})
        self.assertEqual([d['reason'] for d in dups], ['excluded_id', 'sha256_match_seen', 'near_duplicate_seen', 'near_duplicate_within_set_c'])

    def test_balanced_cohort_and_adjudication_freeze_and_offline_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rev_dir = root / 'reviews'
            rev_dir.mkdir()
            cfg = read_json(set_c.CONFIG)
            cfg = {**cfg, 'target_sources': 2, 'target_per_relation': 1, 'minimum_sources': 2, 'bootstrap_draws': 200}
            cfg_path = root / 'cfg.json'
            proto_path = root / 'proto.md'
            proto_path.write_text('proto', encoding='utf-8')
            cfg['protocol'] = 'proto.md'
            write_json(cfg_path, cfg)

            blanks = [_make_blank_row('UNSEEN_C_001', 'left'), _make_blank_row('UNSEEN_C_002', 'right')]
            for slot in ('RA', 'RB'):
                with (rev_dir / f'unseen_c_blank_{slot}.csv').open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=set_c.REVIEW_FIELDS)
                    w.writeheader()
                    w.writerows(blanks)

            cands = [
                {'blind_id': 'UNSEEN_C_001', 'source_image_id': 900001, 'rank': 'a', 'target': 0,
                 'category_a': 'dog', 'category_b': 'chair', 'width': 400, 'height': 400,
                 'bbox_a': [20, 100, 100, 150], 'bbox_b': [260, 100, 100, 150],
                 'image_path': 'img1.jpg', 'image_sha256': 'hash1'},
                {'blind_id': 'UNSEEN_C_002', 'source_image_id': 900002, 'rank': 'b', 'target': 1,
                 'category_a': 'dog', 'category_b': 'chair', 'width': 400, 'height': 400,
                 'bbox_a': [260, 100, 100, 150], 'bbox_b': [20, 100, 100, 150],
                 'image_path': 'img2.jpg', 'image_sha256': 'hash2'},
            ]
            write_json(rev_dir / 'candidates.json', {'candidates': cands})

            ra_path = root / 'ra.csv'
            rb_path = root / 'rb.csv'
            adj_path = root / 'adj.csv'
            with ra_path.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=set_c.REVIEW_FIELDS)
                w.writeheader()
                w.writerows([_fill_row(blanks[0], 'quan', True), _fill_row(blanks[1], 'quan', True)])
            # RB disagrees on 002
            with rb_path.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=set_c.REVIEW_FIELDS)
                w.writeheader()
                w.writerows([_fill_row(blanks[0], 'phong', True), _fill_row(blanks[1], 'phong', False)])

            with mock.patch.object(set_c, 'ROOT', root), \
                 mock.patch.object(set_c, 'REVIEWS', rev_dir), \
                 mock.patch.object(set_c, 'CONFIG', cfg_path):
                # Same annotator rejected
                with rb_path.open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=set_c.REVIEW_FIELDS)
                    w.writeheader()
                    w.writerows([_fill_row(blanks[0], 'quan', True), _fill_row(blanks[1], 'quan', False)])
                with self.assertRaises(ValueError):
                    set_c.adjudicate(ra_path, rb_path)

                # Restore distinct annotator
                with rb_path.open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=set_c.REVIEW_FIELDS)
                    w.writeheader()
                    w.writerows([_fill_row(blanks[0], 'phong', True), _fill_row(blanks[1], 'phong', False)])
                summary = set_c.adjudicate(ra_path, rb_path)
                self.assertEqual(summary['disagreements'], 1)

                # Run refuses before freeze
                with self.assertRaises(FileNotFoundError):
                    set_c.run(cfg)

                # Fill adjudication and freeze
                adj_blank = set_c.read_csv_sig(rev_dir / 'adjudication_blank.csv')
                adj_blank[0]['adjudicator_id'] = 'tung'
                adj_blank[0]['final_accept'] = 'yes'
                adj_blank[0]['adjudication_reason'] = 'object clearly visible on re-check'
                with adj_path.open('w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=set_c.ADJ_FIELDS)
                    w.writeheader()
                    w.writerows(adj_blank)

                record = set_c.freeze(ra_path, rb_path, adj_path, cfg)
                self.assertEqual(record['n_selected'], 2)

                # Test offline metric computation & geometric baseline directly
                from vlm_probe.depth import natural
                rows = natural.make_images(read_json(rev_dir / 'selected.json'))
                hol = np.array([[0.3, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1], [0.3, 0.2, 0.1, 0.1]], dtype=np.float32)
                nat = np.array([[0.4, 0.1, 0.0, 0.0], [0.1, 0.4, 0.0, 0.0], [0.1, 0.4, 0.0, 0.0], [0.4, 0.1, 0.0, 0.0]], dtype=np.float32)
                with mock.patch('vlm_probe.spatial_diagnostics.unseen_holdout.read_json', return_value=read_json(ROOT / 'configs/probing_depth_v1.json')):
                    metrics, preds, pairs = set_c.compute_set_c_metrics(rows, hol, nat, nat, cfg)
                self.assertAlmostEqual(metrics['native_half']['primary_accuracy_percent'], 100.0)
                self.assertAlmostEqual(metrics['bbox_geometric_baseline']['bbox_centroid_2way_accuracy_percent'], 100.0)
                self.assertAlmostEqual(metrics['bbox_geometric_baseline']['ideal_half_canvas_2way_accuracy_percent'], 100.0)

                # Test offline audit recomputation from scores.npz
                run_dir = root / 'results' / 'full_set_c_test'
                run_dir.mkdir(parents=True)
                write_json(run_dir / 'study_config.json', cfg)
                write_json(run_dir / 'images.json', rows)
                np.savez_compressed(run_dir / 'scores.npz', holistic=hol, native=nat, padded=nat)
                write_json(run_dir / 'metrics.json', metrics)
                from vlm_probe.depth.common import write_csv
                write_csv(run_dir / 'predictions.csv', preds)
                write_csv(run_dir / 'pairs.csv', pairs)
                art_hashes = {
                    p.name: sha256(p)
                    for p in run_dir.iterdir()
                    if p.is_file() and p.name != 'manifest.json'
                }
                write_json(run_dir / 'manifest.json', {
                    'status': 'completed',
                    'mode': 'full',
                    'started_at_utc': '2099-01-01T00:00:00+00:00',
                    'artifact_sha256': art_hashes,
                    'input_sha256': {},
                })
                with mock.patch.object(set_c, 'load_seen_exclusions', return_value=(set(), set(), [])), \
                     mock.patch.object(set_c, 'read_json', side_effect=lambda p: read_json(ROOT / 'configs/probing_depth_v1.json') if str(p).endswith('probing_depth_v1.json') else read_json(p)):
                    res = set_c.audit(directory=run_dir)
                self.assertTrue(res['verified'])
                self.assertTrue(res['recomputed_from_scores'])


if __name__ == '__main__':
    unittest.main()
