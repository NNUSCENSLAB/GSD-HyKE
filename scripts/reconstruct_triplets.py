#!/usr/bin/env python3
"""Reconstruct historical triplet outputs into candidate Evolution Relations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from triplets import reconstruct_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "predictions/triplet_raw.jsonl")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test_gold.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/triplet_reconstructed_from_raw.jsonl")
    args = parser.parse_args()
    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    by_index = {row["index"]: row["sample_id"] for row in gold_rows}
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 24 or set(row["index"] for row in rows) != set(by_index):
        raise ValueError("Expected 24 unique test indices in raw triplet predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda item: item["index"]):
            index = row["index"]
            if row["sample_id"] != by_index[index]:
                raise ValueError(f"sample_id mismatch at index {index}")
            prediction, valid = reconstruct_text(row["prediction_text"])
            result = {
                "setting_id": "table5/triplet_reconstructed",
                "index": index,
                "sample_id": row["sample_id"],
                "parse_valid": valid,
                "evolutions": prediction["evolutions"],
            }
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Reconstructed {len(rows)} predictions; output: {args.output}")


if __name__ == "__main__":
    main()
