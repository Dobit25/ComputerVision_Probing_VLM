import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from vlm_probe.configuration import validate_config
from vlm_probe.dataset_adapters.flowers102 import verify_test_images

ROOT = Path(__file__).resolve().parents[1]


class FlowersTests(unittest.TestCase):
    def test_flowers_requires_new_protocol_and_own_limit(self):
        config = json.loads((ROOT / "configs/clip_b32_flowers102.json").read_text())
        self.assertEqual(validate_config(config)["num_samples"], 6149)
        with self.assertRaisesRegex(ValueError, "v1"):
            validate_config({**config, "protocol_version": 1})
        with self.assertRaisesRegex(ValueError, "6149"):
            validate_config({**config, "limit": 6150, "selection": "first_n_test_indices"}, effective=True)

    def test_extracted_image_must_match_archive_bytes(self):
        cache = ROOT / ".cache" / "tests"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root = Path(tmp)
            image = root / "image_00001.jpg"
            image.write_bytes(b"original image bytes")
            archive = root / "images.tgz"
            with tarfile.open(archive, "w:gz") as stream:
                entry = tarfile.TarInfo("jpg/image_00001.jpg")
                entry.size = len(image.read_bytes())
                stream.addfile(entry, io.BytesIO(image.read_bytes()))
            self.assertEqual(len(verify_test_images(archive, [image])), 64)
            image.write_bytes(b"changed image bytes")
            with self.assertRaisesRegex(RuntimeError, "differs from original archive"):
                verify_test_images(archive, [image])
