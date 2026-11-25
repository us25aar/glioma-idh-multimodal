"""Deterministic, patient-level MRI preprocessing with explicit grid checks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from .data import sha256, write_json

SEQUENCES = ("t1", "t1ce", "t2", "flair")
PATH_COLUMNS = tuple(f"{s}_path" for s in SEQUENCES) + ("mask_path",)
PREPROCESS_VERSION = "ras-crop12mm-percentile1-99-four-axial-v1"


def read_image_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"patient_id", *PATH_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Image manifest must contain {sorted(required)}")
    if frame.patient_id.eq("").any() or frame.patient_id.duplicated().any():
        raise ValueError("Image patient IDs must be nonempty and unique.")
    for column in PATH_COLUMNS:
        frame[column] = frame[column].map(lambda p: str((path.parent / p).resolve()) if p and not Path(p).is_absolute() else p)
    return frame


def load_case(row: dict) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Require anatomically registered inputs. Resampling is not registration."""
    arrays, images, file_hashes = [], [], {}
    for column in PATH_COLUMNS:
        path = Path(row[column])
        if not path.is_file():
            raise ValueError(f"Missing input {column}: {path.name}")
        image = nib.as_closest_canonical(nib.load(path))
        if len(image.shape) != 3 or min(image.shape) < 2:
            raise ValueError(f"{column} must be a nonempty three-dimensional NIfTI.")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite values in {column}.")
        images.append(image)
        arrays.append(data)
        file_hashes[column] = sha256(path)
    reference = images[0]
    for image in images[1:]:
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-3):
            raise ValueError("MRI/mask grids differ after RAS canonicalisation; register and inspect before use.")
    mask_data = arrays[-1]
    if not np.allclose(mask_data, np.round(mask_data), atol=1e-4) or mask_data.min() < 0:
        raise ValueError("Mask must contain nonnegative discrete region labels.")
    mask = mask_data > 0
    if not mask.any():
        raise ValueError("Empty tumour mask cannot define a tumour-focused crop.")
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Invalid voxel spacing.")
    return arrays[:-1], mask, {
        "shape": list(reference.shape), "spacing_mm": spacing.tolist(),
        "orientation": list(nib.aff2axcodes(reference.affine)), "mask_voxels": int(mask.sum()),
        "mask_region_values": np.unique(mask_data).tolist(), "input_sha256": file_hashes,
        "grid_check": "passed", "registration_claim": "common grid verified; anatomical alignment requires visual QC",
    }


def select_slices(mask: np.ndarray, count: int = 4) -> list[int]:
    if count < 1:
        raise ValueError("At least one slice must be requested.")
    valid = np.flatnonzero(mask.any(axis=(0, 1)))
    if not len(valid):
        raise ValueError("No tumour-containing axial slices.")
    indices = np.linspace(0, len(valid) - 1, min(count, len(valid))).round().astype(int)
    return valid[np.unique(indices)].tolist()


def crop_bounds(mask: np.ndarray, spacing: list[float], margin_mm: float = 12) -> tuple[slice, slice, slice]:
    if margin_mm < 0:
        raise ValueError("Crop margin cannot be negative.")
    coords = np.where(mask)
    margin = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    return tuple(slice(max(0, int(c.min()) - int(m)), min(size, int(c.max()) + int(m) + 1))
                 for c, m, size in zip(coords, margin, mask.shape))


def intensity_bounds(data: np.ndarray) -> tuple[float, float]:
    foreground = data[data != 0]
    if len(foreground) < 2:
        raise ValueError("Image has insufficient nonzero brain foreground.")
    low, high = np.percentile(foreground, [1, 99])
    if high <= low:
        raise ValueError("Constant foreground cannot be intensity-normalised.")
    return float(low), float(high)


def normalise_volume(data: np.ndarray, bounds: tuple[float, float] | None = None) -> np.ndarray:
    low, high = intensity_bounds(data) if bounds is None else bounds
    result = np.clip((data - low) / (high - low), 0, 1).astype(np.float32)
    result[data == 0] = 0
    return result


def resize_pad(slice_array: np.ndarray, size: int) -> np.ndarray:
    if size < 16:
        raise ValueError("Output image side must be at least 16 pixels.")
    # Transpose consistently for an axial display/encoder view.
    image = Image.fromarray(slice_array.T.astype(np.float32), mode="F")
    scale = size / max(image.size)
    dims = tuple(max(1, round(side * scale)) for side in image.size)
    image = image.resize(dims, resample=Image.Resampling.BILINEAR)
    result = np.zeros((size, size), dtype=np.float32)
    x, y = (size - dims[0]) // 2, (size - dims[1]) // 2
    result[y:y + dims[1], x:x + dims[0]] = np.asarray(image)
    return result


def preprocess_case(row: dict, count: int = 4, size: int = 224, margin_mm: float = 12) -> tuple[np.ndarray, dict, tuple]:
    started = time.perf_counter()
    volumes, mask, details = load_case(row)
    bounds = crop_bounds(mask, details["spacing_mm"], margin_mm)
    selected = select_slices(mask[bounds], count)
    z_global = [z + bounds[2].start for z in selected]
    views, intensity_ranges = [], []
    for volume in volumes:
        scale_bounds = intensity_bounds(volume)
        intensity_ranges.append(list(scale_bounds))
        cropped = volume[bounds]
        slices = [resize_pad(normalise_volume(cropped[:, :, z], scale_bounds), size) for z in selected]
        views.append(np.repeat(np.stack(slices)[:, None, :, :], 3, axis=1))
    result = np.stack(views).astype(np.float32)  # [sequence, valid_slice, RGB, H, W]
    details.update({
        "patient_id": row["patient_id"], "preprocess_version": PREPROCESS_VERSION,
        "selected_global_axial_slices": z_global, "tensor_shape": list(result.shape),
        "crop_bounds": [[b.start, b.stop] for b in bounds], "margin_mm": margin_mm,
        "intensity_bounds": intensity_ranges,
        "normalisation": "per-volume nonzero foreground percentiles 1 and 99; clipped to [0,1]",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    return result, details, (volumes, mask, bounds)


def save_qc_figure(volumes, mask, details: dict, destination: Path, data_scope: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.parent.mkdir(parents=True, exist_ok=True)
    z = details["selected_global_axial_slices"][len(details["selected_global_axial_slices"]) // 2]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), constrained_layout=True)
    ranges = details.get('intensity_bounds') or [intensity_bounds(volume) for volume in volumes]
    for axis, sequence, volume, scale_bounds in zip(axes, SEQUENCES, volumes, ranges):
        axis.imshow(normalise_volume(volume[:, :, z], scale_bounds).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axis.contour(mask[:, :, z].T, levels=[0.5], colors=["#f2b447"], linewidths=0.7)
        x, y = details["crop_bounds"][:2]
        from matplotlib.patches import Rectangle
        axis.add_patch(Rectangle((x[0], y[0]), x[1]-x[0], y[1]-y[0], fill=False, edgecolor="#62bcce", linewidth=0.8))
        axis.set_title(sequence.upper(), fontsize=10)
        axis.axis("off")
    fig.suptitle(f"{details['patient_id']} | {data_scope} | RAS axial slice {z}", fontsize=10)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def prepare_manifest(manifest_path: Path, output: Path, data_scope: str = "real_utsw") -> pd.DataFrame:
    if data_scope not in {"real_utsw", "synthetic_software_test"}:
        raise ValueError("Explicit data scope is required.")
    frame = read_image_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in frame.to_dict("records"):
        patient = row["patient_id"]
        if not patient.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Unsafe patient ID for cache path.")
        try:
            cache_path = output / f"{patient}.npz"
            details_path = output / f"{patient}.json"
            figure_path = output / "qc" / f"{patient}.png"
            input_hashes = {column: sha256(Path(row[column])) for column in PATH_COLUMNS}
            previous = {}
            if details_path.is_file():
                try:
                    previous = json.loads(details_path.read_text())
                except (ValueError, OSError):
                    previous = {}
            if (cache_path.is_file() and figure_path.is_file()
                    and previous.get("input_sha256") == input_hashes
                    and previous.get("data_scope") == data_scope
                    and previous.get("preprocess_version") == PREPROCESS_VERSION
                    and previous.get("cache_sha256") == sha256(cache_path)):
                rows.append({"patient_id": patient, "status": "passed", "reason": "",
                             "cache_path": str(cache_path.resolve()), "data_scope": data_scope,
                             "cache_reused": True})
                continue
            tensor, details, originals = preprocess_case(row)
            details["data_scope"] = data_scope
            np.savez_compressed(cache_path, views=tensor)
            details["cache_sha256"] = sha256(cache_path)
            write_json(details_path, details)
            save_qc_figure(originals[0], originals[1], details, figure_path, data_scope)
            rows.append({"patient_id": patient, "status": "passed", "reason": "", "cache_path": str(cache_path.resolve()),
                         "data_scope": data_scope, "cache_reused": False})
        except (ValueError, OSError, nib.filebasedimages.ImageFileError) as error:
            rows.append({"patient_id": patient, "status": "failed", "reason": str(error), "cache_path": "",
                         "data_scope": data_scope, "cache_reused": False})
    result = pd.DataFrame(rows)
    result.to_csv(output / "image_qc.csv", index=False)
    return result
