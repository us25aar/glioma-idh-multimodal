"""Quantify mask and selected-slice properties for subsequent visual review."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .imaging import crop_bounds, read_image_manifest


def mask_quality(mask: np.ndarray, details: dict) -> dict:
    components, count = ndimage.label(mask)
    sizes = np.bincount(components.ravel())[1:]
    largest_label = int(sizes.argmax()) + 1
    largest = components == largest_label
    main_bounds = crop_bounds(largest, details['spacing_mm'], details['margin_mm'])
    main_volume = int(np.prod([part.stop - part.start for part in main_bounds]))
    actual_volume = int(np.prod([end - start for start, end in details['crop_bounds']]))
    slice_areas = mask.sum(axis=(0, 1))
    selected_areas = slice_areas[details['selected_global_axial_slices']]
    expansion = actual_volume / main_volume
    largest_fraction = float(sizes.max() / mask.sum())
    minimum_view_fraction = float(selected_areas.min() / slice_areas.max())
    flags = []
    if largest_fraction > .99 and expansion > 1.5:
        flags.append('crop_expansion_from_small_components')
    if minimum_view_fraction < .01:
        flags.append('low_mask_area_in_selected_slice')
    return {'components_6_connected': int(count), 'largest_component_fraction': largest_fraction,
            'crop_expansion_relative_to_largest_component': float(expansion),
            'smallest_selected_area_relative_to_peak_slice': minimum_view_fraction,
            'selected_slice_mask_voxels': selected_areas.astype(int).tolist(),
            'review_flags': flags}


def audit_masks(manifest_path: Path, processed: Path, output: Path) -> dict:
    manifest = read_image_manifest(manifest_path).set_index('patient_id')
    qc = pd.read_csv(processed / 'image_qc.csv', keep_default_na=False)
    rows = []
    for patient in qc.loc[qc.status.eq('passed'), 'patient_id']:
        details = json.loads((processed / f'{patient}.json').read_text())
        mask = nib.as_closest_canonical(nib.load(manifest.loc[patient, 'mask_path'])).get_fdata(dtype=np.float32) > 0
        statistics = mask_quality(mask, details)
        rows.append({'patient_id': patient, **statistics,
                     'review_flags': '|'.join(statistics['review_flags']),
                     'sampled_overlay': str((processed / 'qc' / f'{patient}.png').resolve())})
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'mask_quality_audit.csv', index=False)
    return {'patients_audited': len(frame), 'patients_flagged': int(frame.review_flags.ne('').sum()),
            'policy': 'Heuristic review flags only. Cases are retained pending visual assessment.'}
