"""MRI geometry, patient pooling, fusion and uncertainty edge-case checks."""
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import nn

from glioma_idh.evaluation import paired_auc_bootstrap
from glioma_idh.features import encode_views
from glioma_idh.imaging import intensity_bounds, load_case, normalise_volume, prepare_manifest, preprocess_case, select_slices
from glioma_idh.multimodal import PatientHead, fit_head, load_common_development, run_development


def save_fixture(directory: Path, patient="SYNTHETIC_TEST", mask_empty=False):
    rng = np.random.default_rng(8)
    shape = (32, 36, 20)
    mask = np.zeros(shape, dtype=np.uint8)
    if not mask_empty:
        mask[8:21, 10:24, 5:14] = 2
    row = {"patient_id": patient}
    for sequence in ["t1", "t1ce", "t2", "flair"]:
        volume = rng.normal(80, 15, shape).astype(np.float32)
        volume[:2] = 0
        path = directory / f"{patient}_{sequence}.nii.gz"
        nib.save(nib.Nifti1Image(volume, np.diag([1, 1, 2, 1])), path)
        row[f"{sequence}_path"] = str(path)
    path = directory / f"{patient}_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.diag([1, 1, 2, 1])), path)
    row["mask_path"] = str(path)
    return row


class ImagingTests(unittest.TestCase):
    def test_slice_normalisation_matches_full_volume_with_same_bounds(self):
        volume = np.random.default_rng(4).normal(size=(32, 36, 20)).astype(np.float32)
        volume[:2] = 0
        expected = normalise_volume(volume)[:, :, 7]
        actual = normalise_volume(volume[:, :, 7], intensity_bounds(volume))
        np.testing.assert_array_equal(actual, expected)

    def test_resume_checks_input_and_cache_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = save_fixture(root)
            manifest = root / 'manifest.csv'
            pd.DataFrame([row]).to_csv(manifest, index=False)
            output = root / 'processed'
            first = prepare_manifest(manifest, output, 'synthetic_software_test')
            second = prepare_manifest(manifest, output, 'synthetic_software_test')
            self.assertFalse(first.iloc[0].cache_reused)
            self.assertTrue(second.iloc[0].cache_reused)
            cache = output / 'SYNTHETIC_TEST.npz'
            cache.write_bytes(b'interrupted cache')
            repaired = prepare_manifest(manifest, output, 'synthetic_software_test')
            self.assertFalse(repaired.iloc[0].cache_reused)
            self.assertEqual(repaired.iloc[0].status, 'passed')
            image = nib.load(row['t1_path'])
            data = image.get_fdata()
            data[4, 4, 4] = np.nan
            nib.save(nib.Nifti1Image(data, image.affine), row['t1_path'])
            changed = prepare_manifest(manifest, output, 'synthetic_software_test')
            self.assertEqual(changed.iloc[0].status, 'failed')

    def test_case_produces_four_sequences_with_unique_axial_views(self):
        with tempfile.TemporaryDirectory() as temp:
            row = save_fixture(Path(temp))
            views, info, _ = preprocess_case(row, size=64)
            self.assertEqual(views.shape, (4, 4, 3, 64, 64))
            self.assertEqual(info["orientation"], ["R", "A", "S"])
            self.assertEqual(len(set(info["selected_global_axial_slices"])), 4)
            self.assertTrue(np.isfinite(views).all())
            self.assertGreaterEqual(views.min(), 0)
            self.assertLessEqual(views.max(), 1)

    def test_affine_mismatch_is_rejected_even_if_shapes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            row = save_fixture(Path(temp))
            image = nib.load(row["t2_path"])
            affine = image.affine.copy()
            affine[0, 3] += 5
            nib.save(nib.Nifti1Image(image.get_fdata(), affine), row["t2_path"])
            with self.assertRaisesRegex(ValueError, "grids differ"):
                load_case(row)

    def test_empty_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Empty tumour"):
                load_case(save_fixture(Path(temp), mask_empty=True))

    def test_fractional_mask_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            row = save_fixture(Path(temp))
            image = nib.load(row["mask_path"])
            nib.save(nib.Nifti1Image(image.get_fdata() / 3, image.affine), row["mask_path"])
            with self.assertRaisesRegex(ValueError, "discrete"):
                load_case(row)

    def test_nonfinite_mri_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            row = save_fixture(Path(temp))
            image = nib.load(row["t1_path"])
            data = image.get_fdata()
            data[4, 4, 4] = np.nan
            nib.save(nib.Nifti1Image(data, image.affine), row["t1_path"])
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                load_case(row)

    def test_single_slice_tumour_is_not_duplicated(self):
        mask = np.zeros((10, 10, 5), dtype=bool)
        mask[3:5, 3:5, 2] = True
        self.assertEqual(select_slices(mask, 4), [2])

    def test_constant_image_does_not_silently_normalise(self):
        with self.assertRaisesRegex(ValueError, "Constant"):
            normalise_volume(np.ones((10, 10, 10)))

    def test_pooling_is_sequence_specific_and_batch_independent(self):
        class FakeEncoder(nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2, 3))[:, None].repeat(1, 512)
        views = np.zeros((4, 3, 3, 32, 32), dtype=np.float32)
        for i in range(4):
            views[i] = i / 4
        a = encode_views(FakeEncoder(), views, batch_size=1)
        b = encode_views(FakeEncoder(), views, batch_size=4)
        np.testing.assert_allclose(a, b, rtol=1e-6)
        self.assertEqual(a.shape, (2048,))
        self.assertLess(a[0], a[512])


class ModelAndEvaluationTests(unittest.TestCase):
    def test_zero_epoch_budget_is_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, "epochs must be positive"):
            run_development(Path('unused'), Path('unused'), Path('unused'), Path('unused'), Path('unused'), max_epochs=0)

    def test_every_head_outputs_one_logit_per_patient(self):
        for mode in ["clinical", "mri", "fusion"]:
            self.assertEqual(tuple(PatientHead(2048, 3, mode)(torch.zeros(7, 2048), torch.zeros(7, 3)).shape), (7,))

    def test_fusion_head_receives_gradient_through_both_branches(self):
        model = PatientHead(10, 3, "fusion", dropout=0)
        image, clinical = torch.randn(4, 10, requires_grad=True), torch.randn(4, 3, requires_grad=True)
        model(image, clinical).sum().backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertGreater(clinical.grad.abs().sum().item(), 0)

    def test_real_experiment_rejects_synthetic_feature_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            info = Path(temp) / "feature_manifest.json"
            info.write_text(json.dumps({"data_scopes": ["synthetic_software_test"]}))
            with self.assertRaisesRegex(ValueError, "real UTSW"):
                run_development(Path(temp)/"features.npz", info, Path("unused"), Path("unused"), Path(temp)/"out")

    def test_feature_join_excludes_reserved_test_patient(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            ids = [f"P{i}" for i in range(12)]
            frame = pd.DataFrame({"patient_id": ids[:10], "age": 40, "sex": "male", "idh_label": [0,1]*5, "fold": [0,1]*5})
            frame.to_csv(path/"dev.csv", index=False)
            (path/"lock.json").write_text(json.dumps({"development_ids": ids[:10], "test_ids": ids[10:], "fold_assignments": dict(zip(ids[:10], [0,1]*5))}))
            np.savez(path/"features.npz", patient_ids=np.array(ids), features=np.zeros((12,2048), dtype=np.float32))
            common, image = load_common_development(path/"features.npz", path/"lock.json", path/"dev.csv")
            self.assertEqual(image.shape, (10,2048))
            self.assertTrue(set(common.patient_id).isdisjoint(ids[10:]))

    def test_checkpoint_round_trip_preserves_predictions(self):
        rng = np.random.default_rng(7)
        x, c, y = rng.normal(size=(12,10)), rng.normal(size=(12,3)), np.array([0,1]*6)
        model, before, history, epoch = fit_head(x,c,y,x,c,y,"fusion",42,fixed_epochs=2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"model.pt"
            torch.save(model.state_dict(), path)
            reloaded = PatientHead(10,3,"fusion")
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            reloaded.eval()
            with torch.inference_mode():
                after = torch.sigmoid(reloaded(torch.tensor(x,dtype=torch.float32),torch.tensor(c,dtype=torch.float32))).numpy()
            np.testing.assert_array_equal(before,after)

    def test_identical_paired_predictions_have_zero_auc_difference(self):
        result = paired_auc_bootstrap([0,1,0,1], [0.1,0.9,0.2,0.8], [0.1,0.9,0.2,0.8], samples=100)
        self.assertEqual(result["auc_difference"],0)
        self.assertEqual(result["ci95"],[0.0,0.0])

    def test_unpaired_array_sizes_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"matching shape"):
            paired_auc_bootstrap([0,1], [0.1], [0.2,0.8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
