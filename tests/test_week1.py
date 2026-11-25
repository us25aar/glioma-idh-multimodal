"""Scientific validity checks using deliberately small synthetic fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from glioma_idh.data import clinical_inputs, make_manifest, read_metadata
from glioma_idh.experiment import make_pipeline, train_baselines
from glioma_idh.splits import create_splits, persist_splits


def fixture_metadata(n=40):
    return pd.DataFrame({
        "Subject ID": [f"SYNTHETIC_TEST_{i:03d}" for i in range(n)],
        "Age at Imaging": [str(20 + i % 50) for i in range(n)],
        "Sex at birth": ["M" if i % 2 else "F" for i in range(n)],
        "IDH": ["mutated" if i % 3 == 0 else "wild type" for i in range(n)],
        "Operation Status": ["PRE"] * n,
        "Manually Refined Segmentation": ["Yes"] * n,
        "T1": ["Available"] * n, "T1GD": ["Available"] * n,
        "T2": ["Available"] * n, "T2FLAIR": ["Available"] * n,
        "Tumor Grade": ["DO_NOT_USE"] * n,
    })


class MetadataTests(unittest.TestCase):
    def read_fixture(self, frame):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.tsv"
            frame.to_csv(path, sep="\t", index=False)
            return read_metadata(path)

    def test_unknown_idh_is_excluded_not_converted_to_wildtype(self):
        raw = fixture_metadata()
        raw.loc[0, "IDH"] = "NA"
        manifest = make_manifest(self.read_fixture(raw))
        self.assertTrue(pd.isna(manifest.loc[0, "idh_label"]))
        self.assertFalse(manifest.loc[0, "eligible_metadata"])
        self.assertEqual(manifest.loc[0, "exclusion_reason"], "unknown_idh")

    def test_unreviewed_numeric_label_fails_closed(self):
        raw = fixture_metadata()
        raw.loc[0, "IDH"] = "1"
        with self.assertRaisesRegex(ValueError, "Unrecognized IDH"):
            self.read_fixture(raw)

    def test_duplicate_patient_ids_are_rejected(self):
        raw = fixture_metadata()
        raw.loc[1, "Subject ID"] = raw.loc[0, "Subject ID"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read_fixture(raw)

    def test_missing_sequence_is_not_eligible(self):
        raw = fixture_metadata()
        raw.loc[0, "T1GD"] = "NA"
        result = make_manifest(self.read_fixture(raw))
        self.assertFalse(result.loc[0, "eligible_metadata"])
        self.assertIn("metadata_missing_sequence", result.loc[0, "exclusion_reason"])

    def test_postoperative_data_is_excluded(self):
        raw = fixture_metadata()
        raw.loc[0, "Operation Status"] = "POST"
        self.assertFalse(make_manifest(self.read_fixture(raw)).loc[0, "eligible_metadata"])

    def test_invalid_age_requires_review(self):
        raw = fixture_metadata()
        raw.loc[0, "Age at Imaging"] = "not_an_age"
        with self.assertRaisesRegex(ValueError, "Unparseable"):
            make_manifest(self.read_fixture(raw))

    def test_only_age_and_sex_can_enter_feature_matrix(self):
        manifest = make_manifest(fixture_metadata())
        manifest["Tumor Grade"] = manifest.idh_label
        manifest["IDH"] = manifest.idh_label
        self.assertEqual(list(clinical_inputs(manifest)), ["age", "sex"])
        self.assertEqual(list(clinical_inputs(manifest, age_only=True)), ["age"])


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.cohort = make_manifest(fixture_metadata(80))
        self.cohort["idh_label"] = self.cohort.idh_label.astype(int)

    def test_partitions_and_folds_are_disjoint_and_complete(self):
        development, test, folds = create_splits(self.cohort, 42, 0.2, 5)
        self.assertTrue(set(development.patient_id).isdisjoint(test.patient_id))
        self.assertEqual(set(development.patient_id) | set(test.patient_id), set(self.cohort.patient_id))
        for fold in range(folds):
            train_ids = set(development.loc[development.fold.ne(fold), "patient_id"])
            val_ids = set(development.loc[development.fold.eq(fold), "patient_id"])
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertEqual(set(development.loc[development.fold.eq(fold), "idh_label"]), {0, 1})

    def test_split_is_reproducible_despite_input_row_order(self):
        first, test_first, _ = create_splits(self.cohort, 42, 0.2, 5)
        second, test_second, _ = create_splits(self.cohort.sample(frac=1, random_state=3), 42, 0.2, 5)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(test_first, test_second)

    def test_duplicate_ids_cannot_be_split(self):
        bad = pd.concat([self.cohort, self.cohort.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            create_splits(bad, 42, 0.2, 5)

    def test_saved_split_cannot_be_silently_replaced(self):
        development, test, _ = create_splits(self.cohort, 42, 0.2, 5)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            persist_splits(development, test, output, "fixture_hash", 42)
            persist_splits(development, test, output, "fixture_hash", 42)
            with self.assertRaisesRegex(ValueError, "differs"):
                persist_splits(development, test, output, "different_hash", 42)

    def test_imputation_and_scaling_do_not_fit_validation_data(self):
        train = pd.DataFrame({"age": [20.0, 30.0, np.nan, 40.0], "sex": ["male", "female", "male", "female"]})
        estimator = make_pipeline("clinical_logistic", pd.Series([0, 1, 0, 1]), 42)
        estimator.fit(train, pd.Series([0, 1, 0, 1]))
        age_transform = estimator.named_steps["preprocess"].named_transformers_["age"]
        self.assertEqual(age_transform.named_steps["impute"].statistics_[0], 30.0)
        fitted_mean = age_transform.named_steps["scale"].mean_.copy()
        estimator.predict_proba(pd.DataFrame({"age": [10000.0], "sex": ["unseen_category"]}))
        np.testing.assert_array_equal(age_transform.named_steps["scale"].mean_, fitted_mean)

    def test_oof_exports_never_contain_locked_test_patients(self):
        development, test, _ = create_splits(self.cohort, 42, 0.2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary, predictions = train_baselines(development, ["majority", "clinical_logistic"], 42, output)
            self.assertTrue(set(predictions.patient_id).isdisjoint(test.patient_id))
            for _, frame in predictions.groupby("model"):
                self.assertEqual(len(frame), len(development))
                self.assertFalse(frame.patient_id.duplicated().any())
            self.assertTrue(summary.roc_auc_mean.between(0, 1).all())
            self.assertFalse(any("test" in p.name for p in output.glob("*predictions*")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
