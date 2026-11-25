"""Run the scientific validity tests and save machine-readable evidence."""

import hashlib
import argparse
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/ipr/validation_checks.json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "successful": result.wasSuccessful(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "test_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'tests').glob('test_*.py'))},
        "fixture_scope": "Synthetic fixtures only; test metrics are not research results.",
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
