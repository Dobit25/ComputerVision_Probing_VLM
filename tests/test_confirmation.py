import unittest
import numpy as np
from vlm_probe.depth.confirmation import wilson,paired_metrics,CONFIG,SELECTION
from vlm_probe.depth.common import ROOT,read_json


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(CONFIG)

    def rows(self):
        return [{'group_id':'source','template':t,'order':o,'target':y}
                for t in range(3) for o in range(2) for y in range(2)]

    def test_wilson_intervals_and_invalid_counts(self):
        low,high=wilson(50,100)
        self.assertAlmostEqual(low,40.383153,places=5)
        self.assertAlmostEqual(high,59.616847,places=5)
        self.assertAlmostEqual(wilson(0,100)[0],0)
        self.assertAlmostEqual(wilson(100,100)[1],100)
        with self.assertRaises(ValueError):wilson(1,0)

    def test_assignment_survives_caption_bias_but_individual_predictions_do_not(self):
        matrix=np.array([[.9,.8,.0,.0],[.3,.4,.0,.0]])
        scores=np.tile(matrix,(6,1))
        baseline,_=paired_metrics(self.rows(),scores,self.cfg)
        scores[:,0]+=1
        biased,_=paired_metrics(self.rows(),scores,self.cfg)
        self.assertEqual(baseline['t2_o0']['accuracy_percent'],100)
        self.assertEqual(biased['t2_o0']['accuracy_percent'],100)
        self.assertEqual(baseline['t2_o0']['both_images_correct_percent'],100)
        self.assertEqual(biased['t2_o0']['both_images_correct_percent'],0)

    def test_tie_rule_and_exact_binomial(self):
        scores=np.ones((12,4))
        m,_=paired_metrics(self.rows(),scores,self.cfg)
        self.assertEqual(m['t2_o0']['correct'],0)
        self.assertEqual(m['t2_o0']['exact_ties'],1)
        self.assertEqual(m['t2_o0']['near_ties'],1)
        self.assertEqual(m['t2_o0']['binomial_p_greater_0_5'],1)
        scores=np.tile([[.5,.1,0,0],[.1,.5,0,0]],(6,1))
        m,_=paired_metrics(self.rows(),scores,self.cfg)
        self.assertEqual(m['t2_o0']['binomial_p_greater_0_5'],.5)

    def test_missing_or_duplicate_relation_in_pair_rejected(self):
        rows=self.rows();scores=np.ones((12,4))
        with self.assertRaises(ValueError):paired_metrics(rows[:-1],scores[:-1],self.cfg)
        rows[1]['target']=0
        with self.assertRaises(ValueError):paired_metrics(rows,scores,self.cfg)

    def test_old_saved_scores_reproduce_exploratory_point_estimates(self):
        old=ROOT/'results/full_depth_natural_v1'
        with np.load(old/'scores.npz',allow_pickle=False) as z:scores=z['scores']
        m,_=paired_metrics(read_json(old/'trials.json'),scores,self.cfg)
        ref=read_json(ROOT/'reports/probing_depth_v1/exploratory_pair_metrics.json')['natural']
        for key,v in m.items():
            self.assertAlmostEqual(v['accuracy_percent'],ref[key+'_horizontal']['assignment_accuracy_percent'])
            self.assertAlmostEqual(v['two_way_accuracy_percent'],ref[key+'_horizontal']['two_way_accuracy_percent'])
        self.assertEqual(m['t2_o0']['correct'],54)

    def test_frozen_holdout_excludes_all_previously_screened_sources(self):
        old={r['source_image_id'] for r in read_json(ROOT/self.cfg['exclude_all_previous_candidates'])}
        new=read_json(SELECTION/'selected.json')
        ids=[r['source_image_id'] for r in new]
        self.assertEqual(len(ids),100)
        self.assertEqual(len(set(ids)),100)
        self.assertFalse(old.intersection(ids))
        self.assertEqual({r['target'] for r in new},{0,1})


if __name__=='__main__':unittest.main()
