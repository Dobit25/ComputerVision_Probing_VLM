import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vlm_probe.depth.common import read_json
from vlm_probe.spatial_diagnostics import natural_recognition as nr

CATS = [{'name': n, 'supercategory': s} for n, s in [
    ('person', 'person'), ('dog', 'animal'), ('cat', 'animal'), ('horse', 'animal'), ('sheep', 'animal'),
    ('chair', 'furniture'), ('couch', 'furniture'), ('bed', 'furniture'), ('dining table', 'furniture'),
    ('apple', 'food'), ('orange', 'food'), ('car', 'vehicle'), ('truck', 'vehicle')]]
SRC = {'source_image_id': 7, 'category_a': 'dog', 'category_b': 'chair', 'width': 640, 'height': 480,
       'bbox_a': [10, 10, 50, 50], 'bbox_b': [300, 100, 40, 40]}


def cfg():
    c = copy.deepcopy(read_json(nr.CONFIG))
    c['contested_pairs'] = [['couch', 'chair'], ['car', 'truck']]
    return c


def fill(rows, decision='accepted', **over):
    out = []
    for r in rows:
        r = dict(r, reviewer_id='R1', foil_absent='yes', unambiguous='yes', decision=decision,
                 reason='' if decision == 'accepted' else 'foil visible')
        r.update(over)
        out.append(r)
    return out


class FoilTests(unittest.TestCase):
    def test_caption_articles_are_symmetric(self):
        c = cfg()
        self.assertEqual(nr.caption(c['caption_template'], 'apple', 'dining table'),
                         'a photo showing an apple and a dining table.')

    def test_foils_exclude_targets_present_and_contested_and_prefer_supercategory(self):
        c = cfg()
        a = nr.foil_candidates(SRC, 'replace_a', CATS, {'cat'}, c)
        self.assertEqual({x['name'] for x in a}, {'horse', 'sheep'})  # dog: same supercategory, cat present
        self.assertEqual({x['tier'] for x in a}, {1})
        b = nr.foil_candidates(SRC, 'replace_b', CATS, set(), c)
        self.assertEqual({x['name'] for x in b}, {'bed', 'dining table'})  # couch contested with chair

    def test_single_class_supercategory_falls_back_to_tier_2(self):
        src = dict(SRC, category_a='person')
        a = nr.foil_candidates(src, 'replace_a', CATS, set(), cfg())
        self.assertTrue(a and {x['tier'] for x in a} == {2})
        self.assertNotIn('chair', {x['name'] for x in a})

    def test_ranking_is_deterministic_and_order_independent(self):
        c = cfg()
        x = nr.foil_candidates(SRC, 'replace_b', CATS, set(), c)
        y = nr.foil_candidates(SRC, 'replace_b', list(reversed(CATS)), set(), c)
        self.assertEqual(x, y)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.c = cfg()
        cands = {(str(SRC['source_image_id']), r): nr.foil_candidates(SRC, r, CATS, set(), self.c) for r in nr.ROLES}
        self.cands = cands
        self.blank = [nr.review_row(SRC, r, 1, cands[('7', r)][0], 'h') for r in nr.ROLES]

    def test_valid_review_passes(self):
        self.assertEqual(nr.validate_review(fill(self.blank), self.blank, 'R1'), [])

    def test_review_logic_errors_are_caught(self):
        cases = {
            'accepted requires': fill(self.blank, foil_absent='unsure'),
            'requires a reason': fill(self.blank, decision='revise', reason=''),
            'presented fields changed': fill(self.blank, foil='cat'),
            'missing review_id': fill(self.blank[:1]),
            'duplicate review_id': fill(self.blank + self.blank[:1]),
            'reviewer_id R2': fill(self.blank, reviewer_id='R2'),
            'decision must be': fill(self.blank, decision='ok'),
        }
        for needle, rows in cases.items():
            with self.subTest(needle):
                self.assertTrue(any(needle in p for p in nr.validate_review(rows, self.blank, 'R1')),
                                nr.validate_review(rows, self.blank, 'R1'))

    def test_bom_csv_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.csv'
            p.write_text('﻿' + ','.join(nr.REVIEW_FIELDS) + '\n' + ','.join(['a'] * 15) + '\n', encoding='utf-8')
            self.assertEqual(list(nr.read_review_csv(p)[0])[0], 'review_id')

    def test_revise_uncertain_and_exhaustion(self):
        r1 = fill(self.blank[:1], decision='revise') + fill(self.blank[1:], decision='uncertain')
        nxt = nr.next_round_rows(r1, self.cands, {'7': 'h'}, {'7': SRC}, 3)
        self.assertEqual([x['candidate_version'] for x in nxt], ['2', '1'])
        self.assertEqual(nxt[0]['foil'], self.cands[('7', 'replace_a')][1]['name'])
        self.assertTrue(nxt[1]['review_id'].endswith('_adj'))
        final = nr.resolve([r1], 3)
        self.assertEqual({v['status'] for v in final.values()}, {'unresolved'})
        last = fill([dict(self.blank[0], candidate_version='3')], decision='revise')
        self.assertEqual(nr.resolve([r1, last], 3)[('7', 'replace_a')]['status'], 'no_valid_foil')
        done = nr.resolve([r1, fill(nxt)], 3)
        self.assertEqual({v['status'] for v in done.values()}, {'accepted'})
        keep, excluded = nr.analysed_sources(done, [7, 8])
        self.assertEqual(keep, [7])
        self.assertEqual(excluded, [{'source_image_id': 8, 'replace_a': 'missing', 'replace_b': 'missing'}])


class MetricTests(unittest.TestCase):
    def setUp(self):
        from vlm_probe.depth.natural import make_images
        self.c = cfg()
        src = [dict(SRC, source_image_id=i, target=0) for i in (1, 2)]
        self.images = make_images(src)
        self.foils = {(str(i), r): {'foil': 'horse' if r == 'replace_a' else 'bed', 'foil_tier': 1}
                      for i in (1, 2) for r in nr.ROLES}
        self.trials = nr.make_trials(self.images, self.foils, self.c)

    def test_trial_layout(self):
        self.assertEqual(len(self.trials), 8)
        t = self.trials[0]
        self.assertEqual(t['true_caption'], 'a photo showing a dog and a chair.')
        self.assertEqual(t['foil_caption'], 'a photo showing a horse and a chair.')
        self.assertEqual(t['captions'][t['true_index']], t['true_caption'])
        self.assertEqual(self.trials[1]['foil_caption'], 'a photo showing a dog and a bed.')

    def test_summary_counts_ties_as_wrong_and_crosstab(self):
        s = np.array([[.3, .2]] * 8)
        s[1] = [.25, .25]   # exact tie on source 1 original replace_b
        s[7] = [.1, .2]     # source 2 flip replace_b wrong
        rel = {im['image_id']: im['variant'] == 'original' for im in self.images}
        m, preds, images = nr.summarize(self.trials, s, rel, self.c)
        self.assertEqual((m['primary']['correct'], m['primary']['n']), (6, 8))
        self.assertEqual(m['primary']['exact_ties'], 1)
        self.assertEqual(m['by_role']['replace_b']['correct'], 2)
        self.assertEqual(m['both_roles_correct_per_image']['correct'], 2)
        self.assertEqual(m['all_four_correct_per_source']['correct'], 0)
        x = m['relation_crosstab_descriptive']
        self.assertEqual((x['recognition_both_correct_and_relation_correct'], x['recognition_both_correct_and_relation_wrong']), (1, 1))
        self.assertFalse(preds[1]['correct'])
        with self.assertRaises(ValueError):
            nr.summarize(self.trials, s, {}, self.c)

    def test_canvas_box_matches_baseline_canvas(self):
        from PIL import Image, ImageDraw
        from vlm_probe.spatial_diagnostics.natural_regions import baseline_canvas
        im = Image.new('RGB', (640, 427), 'black')
        ImageDraw.Draw(im).rectangle((300, 150, 380, 250), fill='white')
        canvas = np.asarray(baseline_canvas(im)).max(2) > 127
        ys, xs = np.nonzero(canvas)
        x0, y0, x1, y1 = nr.canvas_box([300, 150, 81, 101], 640, 427)
        self.assertLessEqual(abs(xs.min() - x0), 1.5); self.assertLessEqual(abs(xs.max() + 1 - x1), 1.5)
        self.assertLessEqual(abs(ys.min() - y0), 1.5); self.assertLessEqual(abs(ys.max() + 1 - y1), 1.5)


class GateAndDataTests(unittest.TestCase):
    def test_draft_config_refuses_inference_and_review_freeze(self):
        c = read_json(nr.CONFIG)
        if c['status'] == 'frozen':
            self.skipTest('config frozen by team')
        with self.assertRaises(RuntimeError):
            nr.run(c, smoke=True)
        if not (nr.REVIEWS / 'freeze.json').exists():
            with self.assertRaises(RuntimeError):
                nr.freeze(c)

    def test_cohort_is_98_reviewed_sources_with_relation_results(self):
        c = read_json(nr.CONFIG)
        sources = nr.cohort(c)
        self.assertEqual(len(sources), 98)
        rejected = [r for r in nr.csv_rows(nr.ROOT / c['cohort_membership']) if r['accepted'] != 'True']
        self.assertEqual(sorted(r['blind_id'][-3:] for r in rejected), ['006', '092'])
        self.assertFalse({int(r['source_image_id']) for r in rejected} & {s['source_image_id'] for s in sources})
        rel = nr.relation_correct_by_image(c)
        self.assertEqual(len(rel), 196)
        for s in sources:
            self.assertIn(f"coco/{s['source_image_id']}/original", rel)

    def test_config_contested_pairs_are_pairs(self):
        for p in read_json(nr.CONFIG)['contested_pairs']:
            self.assertEqual(len(set(p)), 2)


if __name__ == '__main__':
    unittest.main()
