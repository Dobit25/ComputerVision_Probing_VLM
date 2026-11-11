import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vlm_probe.extension_artifacts import prediction_tables, verify_manifest
from vlm_probe.prompt_study.design import build_dataset_templates
from vlm_probe.spatial_diagnostics.natural_regions import summarize, baseline_canvas
from vlm_probe.human_review import validate, META
from PIL import Image


class IntegrationReviewTests(unittest.TestCase):
    def test_dtd_baseline_keeps_texture_and_domain_does_not_duplicate_it(self):
        templates = build_dataset_templates('dtd', ['dotted'])
        self.assertEqual(templates['T0_baseline'], ['a photo of a dotted texture.'])
        self.assertEqual(templates['T4_domain'], ['a close-up surface texture of dotted.'])

    def test_region_primary_delta_is_paired_and_uses_source_groups(self):
        rows = [{'group_id':'one','target':0,'image_id':'a'}, {'group_id':'one','target':1,'image_id':'b'}]
        s = np.array([[.8,-.8,.1,-.1],[-.8,.8,.1,-.1]])
        base = np.tile([.2,.1,0,0], (2,1))
        m, pred, pairs = summarize(rows,s,base,2,{'bootstrap_draws':100,'bootstrap_seed':0})
        self.assertEqual(m['primary_accuracy_percent'],100)
        self.assertEqual(m['paired_delta_pp'],50)
        self.assertEqual(m['paired_delta_ci95_pp'],[50,50])
        self.assertEqual(m['n_sources'],1)
        self.assertEqual(m['both_images_correct_percent'],100)
        self.assertTrue(pairs[0]['correct'])

    def test_region_rejects_incomplete_pairs_and_wrong_axis_labels(self):
        cfg={'bootstrap_draws':10,'bootstrap_seed':0}
        rows=[{'group_id':'one','target':0,'image_id':'a'}]
        with self.assertRaises(ValueError): summarize(rows,np.ones((1,4)),np.ones((1,4)),2,cfg)
        rows=[{'group_id':'one','target':2,'image_id':'a'},{'group_id':'one','target':2,'image_id':'b'}]
        with self.assertRaises(ValueError): summarize(rows,np.ones((2,4)),np.ones((2,4)),2,cfg)

    def test_baseline_canvas_uses_center_and_preserves_square(self):
        im=Image.new('RGB',(448,224),'red')
        for x in range(112,336):
            for y in range(224): im.putpixel((x,y),(0,0,255))
        crop=baseline_canvas(im)
        self.assertEqual(crop.size,(224,224))
        self.assertEqual(crop.getpixel((0,100)),(0,0,255))
        self.assertEqual(crop.getpixel((223,100)),(0,0,255))

    def test_macro_and_image_transition_counts_are_distinct(self):
        samples=[{'sample_id':str(i),'target_id':y} for i,y in enumerate([0,0,0,1])]
        scores={'base':np.array([[2,0],[2,0],[2,0],[2,0]]),
                'changed':np.array([[2,0],[2,0],[0,2],[0,2]])}
        _,per_class,d=prediction_tables(scores,samples,['a','b'],'base')
        self.assertEqual(d['base']['macro_accuracy_percent'],50)
        self.assertAlmostEqual(d['changed']['macro_accuracy_percent'],100*(2/3+1)/2)
        self.assertEqual(d['changed']['wrong_to_correct'],1)
        self.assertEqual(d['changed']['correct_to_wrong'],1)

    def test_manifest_rejects_changed_artifact_and_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(root/'source.py').write_text('x=1')
            (run/'scores.txt').write_text('original');(root/'config.json').write_text('{}')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            with zipfile.ZipFile(root/'source.zip','w') as z:z.write(root/'source.py','source.py')
            m={'status':'completed','artifact_sha256':{'scores.txt':sha(run/'scores.txt')},
                'input_sha256':{'config.json':sha(root/'config.json')},
                'source_snapshot':{'path':'source.zip','sha256':sha(root/'source.zip')},
                'source_sha256':{'source.py':sha(root/'source.py')}}
            (run/'manifest.json').write_text(json.dumps(m))
            with patch('vlm_probe.extension_artifacts.ROOT',root):
                verify_manifest(run)
                (run/'scores.txt').write_text('changed')
                with self.assertRaises(ValueError):verify_manifest(run)
                (run/'scores.txt').write_text('original');(root/'config.json').write_text('{"changed":1}')
                with self.assertRaises(ValueError):verify_manifest(run)

    def test_blank_translation_review_cannot_be_marked_complete(self):
        with self.assertRaises(ValueError):validate(META/'vietnamese_100_blank.csv','translations')


if __name__=='__main__':unittest.main()
