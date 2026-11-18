import csv
import tempfile
import unittest
from pathlib import Path

from vlm_probe.human_review_followup import ROOT, normalize_sheet, subset_indices


class HumanReviewFollowupTests(unittest.TestCase):
    def test_normalization_preserves_all_labels_and_decisions(self):
        source=ROOT/'reviews/human_review_final_v1/history/received_confirmation_100_blank.csv'
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)/'normalized.csv'
            rows=normalize_sheet(source,dest,'images')
            self.assertEqual(len(rows),100)
            self.assertEqual(sum(r['overall_accept']=='no' for r in rows),2)
            self.assertEqual(rows[40]['overall_accept'],'yes')
            self.assertEqual(rows[92]['both_objects_visible'],'no')
            self.assertTrue(dest.read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_populated_unknown_column_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'bad.csv';dest=Path(folder)/'out.csv'
            source.write_text('blind_id,extra\nx,important note\n',encoding='utf-8')
            with self.assertRaises(ValueError):normalize_sheet(source,dest,'images')

    def test_subset_requires_complete_pairs_and_preserves_order(self):
        rows=[{'group_id':g,'variant':v,'target':t} for g in ['b','a']
              for v,t in [('original',0),('horizontal_flip',1)]]
        self.assertEqual(subset_indices(rows,{'a'}).tolist(),[2,3])
        with self.assertRaises(ValueError):subset_indices(rows[:-1],{'a'})
        with self.assertRaises(ValueError):subset_indices(rows,{'unknown'})


if __name__=='__main__':unittest.main()
