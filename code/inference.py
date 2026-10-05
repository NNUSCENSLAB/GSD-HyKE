#!/usr/bin/env python3
"""Run or validate the released Stage 1 -> Stage 2 inference workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "code/_inference_support"
STAGE1_INPUT = ROOT / "data/runtime/description_test24.json"
STAGE2_GOLD = ROOT / "data/runtime/stage2_gold.json"
IMAGE_ROOT = ROOT / "data/test_images"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate(stage1_adapter: Path, stage2_adapter: Path) -> None:
    for path in (
        STAGE1_INPUT,
        STAGE2_GOLD,
        stage1_adapter / "adapter_model.safetensors",
        stage1_adapter / "adapter_config.json",
        stage2_adapter / "adapter_model.safetensors",
        stage2_adapter / "adapter_config.json",
    ):
        require(path)
    stage1 = json.loads(STAGE1_INPUT.read_text(encoding="utf-8"))
    stage2 = json.loads(STAGE2_GOLD.read_text(encoding="utf-8"))
    if len(stage1) != 24 or len(stage2) != 24:
        raise RuntimeError("Expected 24 Stage 1 and Stage 2 rows")
    for row in stage1:
        for image in row.get("images", []):
            require(IMAGE_ROOT / image)
    print("Inference preflight passed: 24 test rows, 24 images, and both adapters are available.")


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--scibert", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--stage1-adapter", type=Path, default=ROOT / "models/stage1")
    parser.add_argument("--stage2-adapter", type=Path, default=ROOT / "models/stage2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/inference_test24")
    args = parser.parse_args()
    validate(args.stage1_adapter, args.stage2_adapter)
    if args.check_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_predictions = args.output_dir / "stage1_predictions.json"
    stage1_metrics = args.output_dir / "stage1_metrics.json"
    stage2_input = args.output_dir / "stage2_input.json"
    stage2_predictions = args.output_dir / "stage2_predictions.json"
    stage2_report = args.output_dir / "stage2_report.json"
    stage2_metrics = args.output_dir / "stage2_metrics.json"
    run([
        sys.executable, str(SUPPORT / "eval_stage1_description.py"),
        "--data-path", str(STAGE1_INPUT), "--image-root", str(IMAGE_ROOT),
        "--model-path", args.base_model, "--adapter-path", str(args.stage1_adapter),
        "--pred-path", str(stage1_predictions), "--result-path", str(stage1_metrics),
        "--bert-model-path", args.scibert, "--max-new-tokens", "1024",
        "--temperature", "0", "--repetition-penalty", "1.05", "--max-samples", "24",
        "--prompt-mode", "trained_paragraph_events", "--image-max-pixels", "262144",
    ])
    run([
        sys.executable, str(SUPPORT / "build_stage2b1_exact_input_from_stage1_desc.py"),
        "--stage1-pred-path", str(stage1_predictions), "--template-path", str(STAGE2_GOLD),
        "--out-path", str(stage2_input),
    ])
    run([
        sys.executable, str(SUPPORT / "run_stage2b1_exact_mention_pipeline.py"),
        "--input-path", str(stage2_input), "--gold-path", str(STAGE2_GOLD),
        "--adapter-path", str(args.stage2_adapter), "--model-path", args.base_model,
        "--out-dir", str(args.output_dir), "--prefix", "stage2",
        "--pred-path", str(stage2_predictions), "--report-path", str(stage2_report),
        "--eval-path", str(stage2_metrics), "--max-new-tokens", "2048",
        "--temperature", "0", "--repetition-penalty", "1.05", "--max-samples", "24",
        "--tau", "0.7", "--sim-model", args.scibert,
    ])


if __name__ == "__main__":
    main()
