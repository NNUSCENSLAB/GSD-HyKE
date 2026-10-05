#!/usr/bin/env python3
"""Compact command-line entry point for the public GSD-HyKE package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify() -> None:
    gold = json.loads((ROOT / "data/test_gold.json").read_text(encoding="utf-8"))
    extraction = read_jsonl(ROOT / "predictions/extraction_predictions.jsonl")
    descriptions = read_jsonl(ROOT / "predictions/description_predictions.jsonl")
    api_predictions = read_jsonl(ROOT / "predictions/api_extraction_predictions.jsonl")
    extraction_counts = Counter(row["setting_id"] for row in extraction)
    description_counts = Counter(row["setting_id"] for row in descriptions)
    api_counts = Counter(row["setting_id"] for row in api_predictions)
    problems = []
    if len(gold) != 24:
        problems.append(f"test gold rows: expected 24, found {len(gold)}")
    if any(value != 24 for value in extraction_counts.values()):
        problems.append(f"extraction setting counts: {dict(extraction_counts)}")
    if any(value != 24 for value in description_counts.values()):
        problems.append(f"description setting counts: {dict(description_counts)}")
    if len(api_counts) != 4 or any(value != 24 for value in api_counts.values()):
        problems.append(f"API setting counts: {dict(api_counts)}")
    images = list((ROOT / "data/test_images").glob("*"))
    if len([path for path in images if path.is_file()]) != 24:
        problems.append("expected 24 test images")
    for stage in ("stage1", "stage2"):
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (ROOT / "models" / stage / name).is_file():
                problems.append(f"missing models/{stage}/{name}")
    if problems:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(problems))
    print(f"PASS: 24 test diagrams, {len(extraction_counts)} extraction settings, {len(api_counts)} API settings, {len(description_counts)} description settings, and two adapters are present.")


def demo() -> None:
    examples = json.loads((ROOT / "data/examples.json").read_text(encoding="utf-8"))
    for task, rows in examples["tasks"].items():
        print(f"{task}: {len(rows)} example records")


def run_script(script: str, extra: list[str]) -> None:
    subprocess.run([sys.executable, str(ROOT / "code" / script), *extra], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Run the CPU-only package preflight")
    subparsers.add_parser("demo", help="List the released format examples")
    subparsers.add_parser("example", help="Run a lightweight frozen-prediction walkthrough")
    evaluate = subparsers.add_parser("evaluate", help="Recompute frozen-output metrics")
    evaluate.add_argument("--descriptions", action="store_true")
    subparsers.add_parser("infer", help="Run or validate final-model inference")
    subparsers.add_parser("bootstrap", help="Recompute paired-bootstrap intervals")
    args, extra = parser.parse_known_args()
    if args.command == "verify":
        verify()
    elif args.command == "demo":
        demo()
    elif args.command == "example":
        run_script("example.py", extra)
    elif args.command == "evaluate":
        run_script("evaluate_descriptions.py" if args.descriptions else "evaluate.py", extra)
    elif args.command == "infer":
        run_script("inference.py", extra)
    elif args.command == "bootstrap":
        defaults = ["--input-dir", str(ROOT / "data/bootstrap_inputs"), "--output-dir", str(ROOT / "results/bootstrap")]
        run_script("bootstrap.py", defaults + extra)


if __name__ == "__main__":
    main()
