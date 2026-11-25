"""Regressions for the mask and patient variants observed in real downloads."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from glioma_idh.utsw_imaging import UTSW_FILES, build_utsw_manifest


class UtswManifestTests(unittest.TestCase):
    def test_manifest_selects_fets_mask_and_records_incomplete_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "BT0001"
            folder.mkdir()
            for filename in [*UTSW_FILES.values(), "tumorseg_manual_correction.nii.gz"]:
                (folder / filename).write_bytes(b"fixture")
            pd.DataFrame({"patient_id": ["BT0001", "BT0002"]}).to_csv(root / "selected.csv", index=False)
            (root / "lock.json").write_text(json.dumps({"development_ids": ["BT0001", "BT0002"]}))
            result = build_utsw_manifest(root, root / "selected.csv", root / "manifest.csv", root / "lock.json")
            manifest = pd.read_csv(root / "manifest.csv")
            self.assertEqual(result["complete_file_sets"], 1)
            self.assertTrue(manifest.iloc[0].mask_path.endswith("tumorseg_FeTS.nii.gz"))
            self.assertTrue((root / "manifest.inventory.csv").is_file())

    def test_manifest_rejects_reserved_patient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"patient_id": ["BT0002"]}).to_csv(root / "selected.csv", index=False)
            (root / "lock.json").write_text(json.dumps({"development_ids": ["BT0001"]}))
            with self.assertRaisesRegex(ValueError, "locked development"):
                build_utsw_manifest(root, root / "selected.csv", root / "manifest.csv", root / "lock.json")

    def test_manifest_rejects_duplicate_patient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"patient_id": ["BT0001", "BT0001"]}).to_csv(root / "selected.csv", index=False)
            (root / "lock.json").write_text(json.dumps({"development_ids": ["BT0001"]}))
            with self.assertRaisesRegex(ValueError, "unique"):
                build_utsw_manifest(root, root / "selected.csv", root / "manifest.csv", root / "lock.json")
