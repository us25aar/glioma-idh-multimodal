"""Checks for selective, resumable development-only MRI acquisition."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from download_development import map_destinations, missing_inputs, wait_for_transfer
from glioma_idh.utsw_imaging import UTSW_FILES


class AcquisitionTests(unittest.TestCase):
    def test_missing_previous_transfer_can_be_retried_after_client_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('download_development.local', return_value={'transfers': []}), patch('download_development.time.sleep'):
                status = wait_for_transfer('missing', {'request_id': 'missing'}, Path(temporary) / 'state.json', 1800)
            self.assertEqual(status, 'not_found')

    def test_directory_or_partial_file_is_not_a_complete_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / 'BT0001'
            case.mkdir()
            for name in UTSW_FILES.values():
                (case / (name + '.partial')).write_bytes(b'partial')
            self.assertEqual(len(missing_inputs(root, ['BT0001'])), 5)
            for name in UTSW_FILES.values():
                (case / name).write_bytes(b'fixture')
            self.assertEqual(missing_inputs(root, ['BT0001']), [])

    def test_transfer_mapping_preserves_patient_directories(self):
        expected = [{'patient_id': 'BT0001', 'filename': 'brain_t1.nii.gz'},
                    {'patient_id': 'BT0002', 'filename': 'brain_t1.nii.gz'}]
        spec = {'paths': [{'source': '/server/BT0001/brain_t1.nii.gz'},
                          {'source': '/server/BT0002/brain_t1.nii.gz'}]}
        map_destinations(spec, expected)
        self.assertEqual([path['destination'] for path in spec['paths']],
                         ['BT0001/brain_t1.nii.gz', 'BT0002/brain_t1.nii.gz'])

    def test_unrequested_transfer_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unexpected source'):
            map_destinations({'paths': [{'source': '/server/BT0002/brain_t1.nii.gz'}]},
                              [{'patient_id': 'BT0001', 'filename': 'brain_t1.nii.gz'}])
