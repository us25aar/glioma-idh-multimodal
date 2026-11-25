"""Mask review flags address remote components seen in the real MRI pilot."""
import unittest

import numpy as np

from glioma_idh.imaging import crop_bounds
from glioma_idh.quality import mask_quality


class MaskQualityTests(unittest.TestCase):
    def test_remote_small_component_expands_crop_and_flags_thin_view(self):
        mask = np.zeros((80, 80, 40), dtype=bool)
        mask[30:50, 30:50, 15:25] = True
        mask[2, 2, 2] = True
        bounds = crop_bounds(mask, [1, 1, 1], 2)
        details = {'spacing_mm': [1, 1, 1], 'margin_mm': 2,
                   'crop_bounds': [[part.start, part.stop] for part in bounds],
                   'selected_global_axial_slices': [2, 15, 20, 24]}
        result = mask_quality(mask, details)
        self.assertIn('crop_expansion_from_small_components', result['review_flags'])
        self.assertIn('low_mask_area_in_selected_slice', result['review_flags'])
        self.assertEqual(result['components_6_connected'], 2)
