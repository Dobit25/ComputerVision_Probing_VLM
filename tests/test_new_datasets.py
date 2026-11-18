import json
import re
import unittest
from pathlib import Path

from vlm_probe.configuration import validate_config
from vlm_probe.dataset_adapters.dtd import PROTOCOL


class NewDatasetProtocolTests(unittest.TestCase):
    def test_dtd_hyphenated_category_is_valid(self):
        self.assertIsNotNone(re.fullmatch(PROTOCOL['sample_id_pattern'],
                                         'dtd/test/p1/polka-dotted/polka-dotted_0032.jpg'))
        self.assertIsNone(re.fullmatch(PROTOCOL['sample_id_pattern'],
                                      'dtd/test/p1/../polka-dotted_0032.jpg'))

    def test_new_configs_keep_official_split_and_frozen_model(self):
        root = Path(__file__).resolve().parents[1]
        for suffix, size, classes in [('oxford_pet', 3669, 37), ('dtd', 1880, 47), ('food101', 25250, 101)]:
            config = json.loads((root / f'configs/clip_b32_{suffix}.json').read_text())
            protocol = validate_config(config)
            self.assertEqual((protocol['num_samples'], protocol['num_classes']), (size, classes))
            wrong = dict(config, split='train')
            with self.assertRaises(ValueError):
                validate_config(wrong)
