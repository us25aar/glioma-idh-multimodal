"""Local MRI workflow: explicit manifests, preprocessing, features and fusion."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("root", type=Path)
    inventory.add_argument("selection_csv", type=Path)
    inventory.add_argument("output_csv", type=Path)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path)
    features = sub.add_parser("features")
    features.add_argument("qc_csv", type=Path)
    features.add_argument("output", type=Path)
    features.add_argument("--weights-dir", type=Path, default=Path("outputs/weights"))
    train = sub.add_parser("train")
    train.add_argument("features_dir", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.command == "inventory":
        from glioma_idh.utsw_imaging import build_utsw_manifest
        print(json.dumps(build_utsw_manifest(args.root, args.selection_csv, args.output_csv,
                         Path("outputs/week1/split_lock.json")), indent=2))
    elif args.command == "prepare":
        from glioma_idh.imaging import prepare_manifest
        print(prepare_manifest(args.manifest, args.output).to_string(index=False))
    elif args.command == "features":
        from glioma_idh.features import extract_features
        print(json.dumps(extract_features(args.qc_csv, args.output, args.weights_dir), indent=2))
    else:
        from glioma_idh.multimodal import run_development
        print(json.dumps(run_development(args.features_dir / "features.npz", args.features_dir / "feature_manifest.json",
            Path("outputs/week1/development_patients.csv"), Path("outputs/week1/split_lock.json"), args.output, args.epochs), indent=2))


if __name__ == "__main__":
    main()
