"""Command-line entry points for reproducible week-one work."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import audit_metadata, download_metadata, make_manifest, read_metadata, sha256, write_json
from .splits import create_splits, persist_splits


def run_week1(config_path: Path) -> dict:
    from .experiment import environment_info, train_baselines
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["allowed_clinical_features"] != ["age", "sex"]:
        raise ValueError("Week-one feature allowlist is fixed to age and sex.")
    if config.get("evaluation_scope") != "development_cross_validation_only":
        raise ValueError("The week-one runner does not evaluate the locked test set.")
    source = Path(config["metadata_path"])
    output = Path(config["output_dir"])
    raw = read_metadata(source)
    manifest = make_manifest(raw)
    audit = audit_metadata(raw, manifest, source)
    cohort = manifest.loc[manifest.eligible_metadata].copy()
    cohort["idh_label"] = cohort.idh_label.astype(int)
    development, test, folds = create_splits(cohort, config["seed"], config["test_fraction"], config["development_folds"])
    persist_splits(development, test, output, audit["source_sha256"], config["seed"])
    write_json(output / "run_status.json", {"status": "running", "locked_test_evaluated": False})
    manifest.to_csv(output / "patient_manifest.csv", index=False)
    write_json(output / "metadata_audit.json", audit)
    summary, oof = train_baselines(development, config["models"], config["seed"], output)
    code_hash = hashlib.sha256()
    for source_file in sorted(Path(__file__).parent.glob("*.py")):
        code_hash.update(source_file.name.encode())
        code_hash.update(source_file.read_bytes())
    run = {
        "status": "complete", "config": config,
        "started_at_utc": started_at,
        "duration_seconds": round(time.perf_counter() - started_clock, 3),
        "metadata_sha256": audit["source_sha256"], "config_sha256": sha256(config_path),
        "code_sha256": code_hash.hexdigest(), "environment": environment_info(),
        "development_patients": len(development), "locked_test_patients": len(test),
        "actual_development_folds": folds, "models_run": config["models"],
        "locked_test_evaluated": False, "images_downloaded_or_verified": False,
        "note": "No hyperparameter search. Scores are preliminary development CV estimates, not final or external validation.",
    }
    write_json(output / "run_manifest.json", run)
    write_json(output / "run_status.json", {"status": "complete", "locked_test_evaluated": False})
    write_run_summary(output, audit, run, summary)
    return run


def write_run_summary(output: Path, audit: dict, run: dict, summary) -> None:
    lines = [
        "# Week One Executed Results", "",
        "These are real UTSW clinical metadata experiments. MRI has not been downloaded or inspected.", "",
        f"Source metadata SHA256: `{audit['source_sha256']}`", "",
        f"Unique patients: {audit['unique_patients']}. Metadata-eligible labeled patients: {audit['metadata_eligible_patients']}.",
        f"IDH-mutant: {audit['eligible_idh_mutant']}; IDH-wildtype: {audit['eligible_idh_wildtype']}; excluded unknown IDH: {audit['excluded_unknown_idh']}.",
        f"Development patients: {run['development_patients']}; locked test patients: {run['locked_test_patients']}; development folds: {run['actual_development_folds']}.", "",
        "Only age and sex are model inputs. Labels, diagnosis, pathology, grade, treatment, identifiers and scanner fields are excluded.", "",
        "| Model | Mean fold ROC AUC | Fold SD | Mean average precision | Mean balanced accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.model} | {row.roc_auc_mean:.3f} | {row.roc_auc_std:.3f} | {row.average_precision_mean:.3f} | {row.balanced_accuracy_mean:.3f} |")
    lines += [
        "", "Scores use the development folds only. Fold SD is not a confidence interval. Threshold metrics use the preliminary fixed threshold 0.5.",
        "No final test or external evaluation has been performed. The clinical results do not establish an MRI or multimodal improvement.",
        "MRI availability is declared by the metadata, not verified from local files. Recheck a common image-eligible cohort in week two while preserving the fixed patient partitions.", "",
        "The run manifest records software versions, code and configuration hashes, the metadata hash, and the model list.", "",
    ]
    (output / "executed_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Glioma IDH week-one research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the small public clinical TSV only")
    download.add_argument("--destination", type=Path, default=Path("data/raw/utsw/UTSW_Glioma_Metadata-2-1.tsv"))
    run = commands.add_parser("week1", help="Audit, lock patient partitions, and run development-only clinical CV")
    run.add_argument("--config", type=Path, default=Path("configs/utsw_week1.json"))
    commands.add_parser("doctor", help="Record the active environment and detected GPU")
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download_metadata(args.destination), indent=2))
    elif args.command == "week1":
        result = run_week1(args.config)
        print(json.dumps({k: result[k] for k in ["status", "development_patients", "locked_test_patients", "actual_development_folds", "locked_test_evaluated"]}, indent=2))
    else:
        from .experiment import environment_info
        print(json.dumps(environment_info(), indent=2))
