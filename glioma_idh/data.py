"""Download, validate and describe the public UTSW clinical metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

METADATA_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/UTSW_Glioma_Metadata-2-1.tsv"
DATASET_DOI = "10.7937/DFAE-1B86"
MRI_COLUMNS = ("T1", "T1GD", "T2", "T2FLAIR")
REQUIRED_COLUMNS = (
    "Subject ID", "Sex at birth", "Age at Imaging", "IDH",
    "Operation Status", "Manually Refined Segmentation", *MRI_COLUMNS,
)
LABELS = {"mutated": 1, "wild type": 0}
MISSING = {"", "na", "n/a", "nan", "unknown", "not available", "indeterminate"}


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_metadata(destination: Path) -> dict:
    """Download only the small public TSV, preserving an existing local copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = Request(METADATA_URL, headers={"User-Agent": "Glioma-IDH-MS-Research/0.1"})
        with urlopen(request, timeout=45) as response:
            payload = response.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("Metadata response exceeds the 10 MiB safety limit.")
        if not payload.decode("utf-8-sig").startswith("Subject ID\t"):
            raise ValueError("Response is not the expected UTSW metadata TSV.")
        destination.write_bytes(payload)
    provenance = {
        "dataset": "UTSW-Glioma", "dataset_doi": DATASET_DOI,
        "source_url": METADATA_URL, "source_file": destination.name,
        "sha256": sha256(destination), "bytes": destination.stat().st_size,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0; cite the dataset and associated paper",
        "note": "Verification time is not necessarily the original download time.",
    }
    write_json(destination.with_suffix(".provenance.json"), provenance)
    return provenance


def read_metadata(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    for column in table:
        table[column] = table[column].str.strip()
    if table["Subject ID"].eq("").any():
        raise ValueError("Empty patient identifiers are not permitted.")
    if table["Subject ID"].duplicated().any():
        raise ValueError("Duplicate patient identifiers require manual reconciliation before splitting.")
    labels = set(table["IDH"].str.lower())
    unexpected = labels - set(LABELS) - MISSING
    if unexpected:
        raise ValueError(f"Unrecognized IDH values; update the reviewed mapping explicitly: {sorted(unexpected)}")
    sex_values = set(table["Sex at birth"].str.lower())
    if sex_values - {"m", "f"} - MISSING:
        raise ValueError("Unrecognized sex encoding; review the source data dictionary.")
    return table


def make_manifest(raw: pd.DataFrame) -> pd.DataFrame:
    age_text = raw["Age at Imaging"].str.lower()
    age = pd.to_numeric(age_text.mask(age_text.isin(MISSING)), errors="coerce")
    if ((age.isna()) & ~age_text.isin(MISSING)).any():
        raise ValueError("Unparseable ages require review; they will not be silently imputed.")
    if ((age < 18) | (age > 110)).any():
        raise ValueError("This adult-cohort protocol expects ages from 18 through 110; review eligibility.")
    label = raw["IDH"].str.lower().map(LABELS).astype("Int64")
    sequence_available = raw[list(MRI_COLUMNS)].apply(lambda s: s.str.lower().eq("available")).all(axis=1)
    preoperative = raw["Operation Status"].str.upper().eq("PRE")
    reasons = []
    for known, available, pre in zip(label.notna(), sequence_available, preoperative):
        flags = []
        if not known:
            flags.append("unknown_idh")
        if not available:
            flags.append("metadata_missing_sequence")
        if not pre:
            flags.append("not_preoperative")
        reasons.append(";".join(flags))
    manifest = pd.DataFrame({
        "patient_id": raw["Subject ID"],
        "age": age,
        "sex": raw["Sex at birth"].str.lower().map({"m": "male", "f": "female"}).fillna("unknown"),
        "idh_label": label,
        "eligible_metadata": label.notna() & sequence_available & preoperative,
        "exclusion_reason": reasons,
        "four_sequences_declared": sequence_available,
        "preoperative_declared": preoperative,
        "manually_refined_mask_declared": raw["Manually Refined Segmentation"].str.lower().eq("yes"),
        "image_files_verified": False,
        "source_collection": "UTSW-Glioma",
    })
    return manifest.sort_values("patient_id").reset_index(drop=True)


def audit_metadata(raw: pd.DataFrame, manifest: pd.DataFrame, source: Path) -> dict:
    eligible = manifest[manifest.eligible_metadata]
    ages = manifest.age.dropna()
    return {
        "source_sha256": sha256(source), "rows": len(raw),
        "unique_patients": int(manifest.patient_id.nunique()),
        "raw_idh_counts": {str(k): int(v) for k, v in raw.IDH.value_counts().items()},
        "metadata_eligible_patients": len(eligible),
        "eligible_idh_mutant": int(eligible.idh_label.eq(1).sum()),
        "eligible_idh_wildtype": int(eligible.idh_label.eq(0).sum()),
        "excluded_unknown_idh": int(manifest.idh_label.isna().sum()),
        "missing_age": int(manifest.age.isna().sum()),
        "unknown_sex": int(manifest.sex.eq("unknown").sum()),
        "age_range": [float(ages.min()), float(ages.max())] if len(ages) else None,
        "all_four_sequences_declared": int(manifest.four_sequences_declared.sum()),
        "preoperative_declared": int(manifest.preoperative_declared.sum()),
        "manually_refined_masks_declared": int(manifest.manually_refined_mask_declared.sum()),
        "images_locally_verified": 0,
        "clinical_feature_allowlist": ["age", "sex"],
        "excluded_from_model_inputs": [c for c in raw.columns if c not in {"Age at Imaging", "Sex at birth"}],
        "limitations": [
            "Metadata declarations do not verify the existence, alignment or quality of MRI files.",
            "Image-verified eligibility and consistent mask selection remain pending.",
            "These patients come from one institutional collection.",
        ],
    }


def clinical_inputs(frame: pd.DataFrame, age_only: bool = False) -> pd.DataFrame:
    """Strict allowlist: no identifier, molecular result, diagnosis or grade enters X."""
    columns = ["age"] if age_only else ["age", "sex"]
    return frame.loc[:, columns].copy()
