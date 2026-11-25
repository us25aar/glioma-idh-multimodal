"""Build an explicit manifest for the supplied FeTS-preprocessed UTSW files."""

from pathlib import Path
import json

import pandas as pd


UTSW_FILES = {
    "t1_path": "brain_t1.nii.gz",
    "t1ce_path": "brain_t1ce.nii.gz",
    "t2_path": "brain_t2.nii.gz",
    "flair_path": "brain_flair.nii.gz",
    "mask_path": "tumorseg_FeTS.nii.gz",
}


def build_utsw_manifest(root: Path, selection_csv: Path, output_csv: Path, lock_path: Path) -> dict:
    """Use one supplied automatic mask variant consistently across the pilot."""
    selected = pd.read_csv(selection_csv, dtype={"patient_id": str})
    lock = json.loads(lock_path.read_text())
    ids = selected.patient_id.tolist()
    if len(ids) != len(set(ids)) or not set(ids).issubset(lock["development_ids"]):
        raise ValueError("MRI pilot selection must contain unique locked development patients.")
    if any(not patient.startswith("BT") or not patient[2:].isdigit() for patient in ids):
        raise ValueError("Unexpected UTSW patient directory identifier.")
    rows, inventory = [], []
    for patient in ids:
        directory = root / patient
        paths = {column: (directory / name).resolve() for column, name in UTSW_FILES.items()}
        missing = [path.name for path in paths.values() if not path.is_file()]
        inventory.append({"patient_id": patient, "complete_required_files": not missing,
                          "missing_files": "|".join(missing)})
        if not missing:
            rows.append({"patient_id": patient, **{key: str(path) for key, path in paths.items()}})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["patient_id", *UTSW_FILES]).to_csv(output_csv, index=False)
    pd.DataFrame(inventory).to_csv(output_csv.with_suffix(".inventory.csv"), index=False)
    return {"selected_patients": len(ids), "complete_file_sets": len(rows),
            "mask_policy": "supplied automatic tumorseg_FeTS.nii.gz for every patient",
            "image_variant": "brain_t1, brain_t1ce, brain_t2, brain_flair",
            "manifest": str(output_csv), "anatomical_qc_completed": False}
