"""Regression checks for independent stopping, calibration and inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from glioma_idh.data import write_json
from glioma_idh.validation import (fit_model, predict_model, probability_logits,
                                   run_validation, select_threshold, split_training_roles)


class ValidationTests(unittest.TestCase):
    def test_training_roles_are_disjoint_and_cover_outer_training(self):
        labels = np.array([0, 1] * 25)
        roles = split_training_roles(labels, 42)
        fit, stop, calibration = [set(roles[name]) for name in ['fit', 'stopping', 'calibration']]
        self.assertTrue(fit.isdisjoint(stop) and fit.isdisjoint(calibration) and stop.isdisjoint(calibration))
        self.assertEqual(fit | stop | calibration, set(range(50)))
        self.assertEqual(fit | stop, set(roles['refit']))
        for values in roles.values():
            self.assertEqual(set(labels[values]), {0, 1})

    def test_training_roles_are_reproducible(self):
        first = split_training_roles(np.array([0, 1] * 25), 42)
        second = split_training_roles(np.array([0, 1] * 25), 42)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_constant_calibration_predictions_resolve_threshold_to_half(self):
        self.assertEqual(select_threshold([0, 1, 0, 1], [.4, .4, .4, .4]), .5)

    def test_probability_logits_handle_saturated_outputs(self):
        logits = probability_logits([0, .5, 1])
        self.assertTrue(np.isfinite(logits).all())
        self.assertEqual(logits.shape, (3, 1))
        with self.assertRaisesRegex(ValueError, 'between zero and one'):
            probability_logits([-.1, 1])

    def test_calibration_holdout_is_excluded_from_base_scaler(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({'patient_id': [f'SYNTHETIC_{i}' for i in range(40)],
                              'idh_label': [0, 1] * 20, 'age': np.arange(40) + 25,
                              'sex': ['male', 'female'] * 20})
        image = rng.normal(size=(40, 2048)).astype(np.float32)
        roles = split_training_roles(frame.idh_label, 42)
        image[roles['calibration']] += 100
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            info = fit_model('fusion', frame, image, roles, directory,
                              {'data_scopes': ['synthetic_software_test']}, 42, 2)
            transforms = joblib.load(directory / 'fusion_transforms.joblib')
            np.testing.assert_allclose(transforms['image_scaler'].mean_, image[roles['refit']].mean(axis=0), atol=1e-6)
            self.assertTrue(set(info['training_roles']['calibration']).isdisjoint(info['training_roles']['refit']))
            raw, calibrated, threshold = predict_model(directory, 'fusion', image, frame)
            raw_again, calibrated_again, threshold_again = predict_model(directory, 'fusion', image, frame)
            np.testing.assert_array_equal(raw, raw_again)
            np.testing.assert_array_equal(calibrated, calibrated_again)
            self.assertEqual(threshold, threshold_again)
            self.assertTrue(0 <= threshold <= 1)
            (directory / 'fusion_transforms.joblib').write_bytes(b'incomplete replacement')
            with self.assertRaisesRegex(ValueError, 'differs from the fitted calibration'):
                predict_model(directory, 'fusion', image, frame)

    def test_synthetic_provenance_cannot_enter_real_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            path.write_text(json.dumps({'data_scopes': ['synthetic_software_test']}))
            with self.assertRaisesRegex(ValueError, 'real UTSW'):
                run_validation(Path('unused'), path, Path('unused'), Path('unused'), Path(temporary))

    def test_failed_atomic_json_replace_preserves_previous_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'state.json'
            write_json(path, {'status': 'previous'})
            with patch.object(Path, 'replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    write_json(path, {'status': 'new'})
            self.assertEqual(json.loads(path.read_text())['status'], 'previous')
            self.assertEqual(list(path.parent.glob('*.tmp')), [])
