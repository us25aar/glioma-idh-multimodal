"""Execute the locked 20-patient local MRI feasibility pilot after acquisition."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_idh.features import extract_features
from glioma_idh.imaging import prepare_manifest
from glioma_idh.multimodal import run_development
from glioma_idh.utsw_imaging import build_utsw_manifest
from analyse_real_pilot import main as analyse_pilot


def main():
    selection = Path('outputs/ipr/download_selection.csv')
    manifest = Path('data/image_manifest_real_pilot.csv')
    lock = Path('outputs/week1/split_lock.json')
    session_path = Path('outputs/ipr/download_session.json')
    session = json.loads(session_path.read_text()) if session_path.exists() else {}
    inventory = build_utsw_manifest(Path(session.get('destination', 'dataset')), selection, manifest, lock)
    if inventory['complete_file_sets'] != 20:
        raise ValueError('All 20 selected pilot file sets must be present before execution.')
    print(json.dumps(inventory), flush=True)
    qc = prepare_manifest(manifest, Path('data/processed/real_pilot'))
    if not qc.status.eq('passed').all():
        raise ValueError('Inspect data/processed/real_pilot/image_qc.csv before proceeding.')
    print('Preprocessed', len(qc), 'real patients', flush=True)
    feature_dir = Path('data/features/real_pilot')
    info = extract_features(Path('data/processed/real_pilot/image_qc.csv'), feature_dir, Path('outputs/weights'))
    print(json.dumps(info), flush=True)
    run = run_development(feature_dir / 'features.npz', feature_dir / 'feature_manifest.json',
                          Path('outputs/week1/development_patients.csv'), lock, Path('outputs/ipr/real_pilot'))
    print(json.dumps(run['models']), flush=True)
    analyse_pilot()


if __name__ == '__main__':
    main()
