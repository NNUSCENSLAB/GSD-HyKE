#!/usr/bin/env python3
"""Export evaluation-ready API predictions without raw service responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODELS = {
    "gpt-4o": ("table4/gpt_4o", 0.2069),
    "gpt-5.4": ("table4/gpt_5_4", 0.2432),
    "gpt-5.5": ("table4/gpt_5_5", 0.2642),
    "gpt-5.6-terra": ("table4/gpt_5_6_terra", 0.2745),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(bundle: Path, gold_path: Path) -> tuple[list[dict], dict]:
    """Join selected API repetitions to public test IDs by checked index/image."""
    records_path = bundle / "inputs/revised_test24/records.json"
    records = read_json(records_path)
    gold = read_json(gold_path)
    if len(records) != 24 or len(gold) != 24:
        raise ValueError("Expected exactly 24 source and gold records")
    records_by_index = {row["index"]: row for row in records}
    gold_by_index = {row["index"]: row for row in gold}
    if set(records_by_index) != set(gold_by_index) or len(records_by_index) != 24:
        raise ValueError("Source and gold indices are not unique and identical")
    for index in records_by_index:
        if Path(records_by_index[index]["image"]).name != Path(gold_by_index[index]["image"]).name:
            raise ValueError(f"Image mismatch at index {index}")

    output = []
    provenance = {
        "schema_version": "1.0",
        "scope": "Table 4 API-model main repetitions; 24 revised test diagrams",
        "mapping": "index joined only after exact source/public image-basename validation",
        "records_sha256": sha256(records_path),
        "gold_sha256": sha256(gold_path),
        "source_paths_relative_to_bundle": True,
        "excludes": ["raw API response", "response identifiers", "request payload", "stage-one output"],
        "settings": {},
    }
    for source_name, (setting_id, expected_f1) in MODELS.items():
        root = bundle / "outputs" / source_name / "revised_test24"
        summary_path = root / "formal_evaluation_summary.json"
        summary = read_json(summary_path)
        repeat = summary["main_repeat"]
        actual_f1 = summary["main_result"]["relation_relaxed"]["f1"]
        if abs(actual_f1 - expected_f1) > 0.00005:
            raise ValueError(f"Published Relation F1 mismatch for {source_name}: {actual_f1}")
        stage_path = root / f"repeat_{repeat}" / "stage2.json"
        eval_path = root / f"repeat_{repeat}" / "stage2_for_eval.json"
        stage_rows = read_json(stage_path)
        eval_rows = read_json(eval_path)
        stage_by_index = {row["index"]: row for row in stage_rows}
        eval_by_index = {row["index"]: row for row in eval_rows}
        if (len(stage_rows) != 24 or len(eval_rows) != 24
                or set(stage_by_index) != set(gold_by_index)
                or set(eval_by_index) != set(gold_by_index)):
            raise ValueError(f"Incomplete or duplicate test indices for {source_name}")
        for index in sorted(gold_by_index):
            stage = stage_by_index[index]
            prediction = json.loads(eval_by_index[index]["prediction"])
            if stage["record_id"] != records_by_index[index]["record_id"]:
                raise ValueError(f"Record ID mismatch for {source_name} index {index}")
            if stage["parse_error"] is not None or prediction != stage["prediction"]:
                raise ValueError(f"Prediction mismatch or parse failure for {source_name} index {index}")
            output.append({
                "setting_id": setting_id,
                "index": index,
                "sample_id": gold_by_index[index]["sample_id"],
                "parse_valid": True,
                "evolutions": prediction["evolutions"],
            })
        provenance["settings"][setting_id] = {
            "source_model": source_name,
            "model_returned": sorted({row["model_returned"] for row in stage_by_index.values()}),
            "main_repeat": repeat,
            "selection_rule": summary["selection_rule"],
            "rows": 24,
            "reported_relation_f1": actual_f1,
            "summary_file": str(summary_path.relative_to(bundle)),
            "summary_sha256": sha256(summary_path),
            "stage2_file": str(stage_path.relative_to(bundle)),
            "stage2_sha256": sha256(stage_path),
            "stage2_for_eval_file": str(eval_path.relative_to(bundle)),
            "stage2_for_eval_sha256": sha256(eval_path),
        }
    return output, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    rows, provenance = export(args.bundle, args.gold)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.provenance.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} prediction rows to {args.output}")


if __name__ == "__main__":
    main()
