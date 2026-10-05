#!/usr/bin/env python3
"""Run the frozen-output evaluator for all settings of manuscript Table 4–7."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions/extraction_predictions.jsonl"
API_PREDICTIONS = ROOT / "predictions/api_extraction_predictions.jsonl"


def setting_ids(path: Path, prefix: str) -> list[str]:
    """Read unique setting IDs from the frozen JSONL without changing its rows."""
    values = {
        json.loads(line)["setting_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return sorted(name for name in values if name.startswith(prefix))


def run(path: Path, settings: list[str], output: Path, model: str, device: str) -> None:
    """Evaluate one compatible prediction file into a new, non-overwritten result."""
    if not settings:
        raise ValueError(f"No settings found in {path}")
    if output.exists():
        raise FileExistsError(f"Result already exists; choose a new --output-dir: {output}")
    command = [
        sys.executable, str(ROOT / "code/evaluate.py"),
        "--predictions", str(path), "--output", str(output),
        "--model", model, "--device", device,
    ]
    if "table6/without_first_stage_training" in settings:
        command.extend(("--entity-unconstrained-setting", "table6/without_first_stage_training"))
    for name in settings:
        command.extend(("--setting", name))
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    prefix = f"table{args.table}/"
    open_output = args.output_dir / f"table{args.table}.json"
    api_output = args.output_dir / "table4_api.json"
    planned = [open_output, api_output] if args.table == 4 else [open_output]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"Result already exists; choose a new --output-dir: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(PREDICTIONS, setting_ids(PREDICTIONS, prefix), open_output, args.model, args.device)
    if args.table == 4:
        run(API_PREDICTIONS, setting_ids(API_PREDICTIONS, prefix), api_output, args.model, args.device)


if __name__ == "__main__":
    main()
