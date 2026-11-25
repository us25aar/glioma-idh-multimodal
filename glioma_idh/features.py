"""Frozen ImageNet ResNet-18 feature extraction with cache provenance."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import sha256, write_json
from .imaging import PREPROCESS_VERSION, SEQUENCES


def make_encoder(weights_dir: Path, pretrained: bool = True) -> nn.Module:
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(weights_dir.resolve()))
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    encoder.fc = nn.Identity()
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def encode_views(encoder: nn.Module, views: np.ndarray, batch_size: int = 4, device: str = "cpu") -> np.ndarray:
    if views.ndim != 5 or views.shape[0] != 4 or views.shape[2] != 3 or views.shape[1] < 1:
        raise ValueError("Expected four sequences with valid RGB slice views.")
    if not np.isfinite(views).all() or views.min() < 0 or views.max() > 1:
        raise ValueError("Encoder inputs must be finite and scaled to [0,1].")
    if batch_size < 1:
        raise ValueError("Image batch size must be positive.")
    encoder = encoder.to(device).eval()
    vectors = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for sequence_views in views:
            chunks = []
            for start in range(0, len(sequence_views), batch_size):
                inputs = torch.from_numpy(sequence_views[start:start+batch_size]).to(device)
                chunks.append(encoder((inputs - mean) / std).cpu())
            vectors.append(torch.cat(chunks).mean(dim=0).numpy())
    features = np.concatenate(vectors).astype(np.float32)
    if features.shape != (2048,) or not np.isfinite(features).all():
        raise ValueError("Unexpected encoder feature vector.")
    return features


def extract_features(qc_csv: Path, output: Path, weights_dir: Path, device: str = "cpu", batch_size: int = 4) -> dict:
    torch.set_num_threads(4)
    qc = pd.read_csv(qc_csv, keep_default_na=False)
    if qc.patient_id.duplicated().any():
        raise ValueError("Duplicate feature patient IDs.")
    selected = qc.loc[qc.status.eq("passed")]
    if selected.empty:
        raise ValueError("No QC-passed cases to encode.")
    encoder = make_encoder(weights_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, vectors = [], []
    started = time.perf_counter()
    checkpoint = weights_dir / "checkpoints/resnet18-f37072fd.pth"
    weights_hash = sha256(checkpoint)
    for row in selected.itertuples():
        cache = Path(row.cache_path)
        cache_details = json.loads(cache.with_suffix('.json').read_text())
        if (cache_details.get('patient_id') != row.patient_id or cache_details.get('data_scope') != row.data_scope
                or cache_details.get('preprocess_version') != PREPROCESS_VERSION):
            raise ValueError('Preprocessed cache provenance differs from the feature request.')
        signature = {"cache_sha256": sha256(cache), "weights_sha256": weights_hash, "sequence_order": list(SEQUENCES), "preprocess_version": PREPROCESS_VERSION}
        if cache_details.get('cache_sha256', signature['cache_sha256']) != signature['cache_sha256']:
            raise ValueError('Preprocessed cache checksum differs from its provenance.')
        vector_path = output / f"{row.patient_id}.npy"
        info_path = output / f"{row.patient_id}.json"
        previous = {}
        if info_path.is_file():
            try:
                previous = json.loads(info_path.read_text())
            except (ValueError, OSError):
                previous = {}
        cached = (vector_path.is_file() and all(previous.get(key) == value for key, value in signature.items())
                  and previous.get('vector_sha256') == sha256(vector_path))
        if cached:
            vector = np.load(vector_path, allow_pickle=False)
            if vector.shape != (2048,) or not np.isfinite(vector).all():
                raise ValueError('Invalid cached feature vector.')
        else:
            with np.load(cache, allow_pickle=False) as bundle:
                vector = encode_views(encoder, bundle["views"], batch_size, device)
            np.save(vector_path, vector, allow_pickle=False)
            write_json(info_path, {**signature, 'vector_sha256': sha256(vector_path)})
        vectors.append(vector)
        rows.append({"patient_id": row.patient_id, "data_scope": row.data_scope})
    np.savez_compressed(output / "features.npz", patient_ids=np.array([r["patient_id"] for r in rows]), features=np.stack(vectors))
    pd.DataFrame(rows).to_csv(output / "feature_patients.csv", index=False)
    info = {"patients": len(rows), "feature_dimensions": 2048, "sequence_order": list(SEQUENCES), "preprocess_version": PREPROCESS_VERSION,
            "encoder": "ResNet18 IMAGENET1K_V1 frozen", "weights_sha256": weights_hash,
            "elapsed_seconds": round(time.perf_counter()-started, 3), "device": device,
            "data_scopes": sorted(set(r["data_scope"] for r in rows)), "features_sha256": sha256(output / "features.npz")}
    write_json(output / "feature_manifest.json", info)
    return info
