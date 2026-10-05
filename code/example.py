#!/usr/bin/env python3
"""Run a lightweight walkthrough of one released GSD-HyKE prediction.

The example uses only Python's standard library. It joins one test diagram to
one frozen prediction by ``sample_id``, validates the referenced image and
Evolution Relation role structure, and writes a compact JSON result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLE_FIELDS = (
    "input_mentions",
    "output_mentions",
    "mechanism_mentions",
    "time_mentions",
    "location_mentions",
)
CORE_ROLE_FIELDS = ("input_mentions", "output_mentions", "mechanism_mentions")


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSON Lines records from *path*."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_unique(rows: list[dict], *, sample_id: str, setting_id: str | None = None) -> dict:
    """Return exactly one matching record or raise a descriptive error."""

    matches = [
        row
        for row in rows
        if row.get("sample_id") == sample_id
        and (setting_id is None or row.get("setting_id") == setting_id)
    ]
    if len(matches) != 1:
        qualifier = f" and setting_id={setting_id!r}" if setting_id else ""
        raise ValueError(f"Expected one row for sample_id={sample_id!r}{qualifier}; found {len(matches)}")
    return matches[0]


def summarize_relation(relation: dict) -> dict:
    """Keep role lists unchanged and report whether all core roles are present."""

    roles = {}
    for field in ROLE_FIELDS:
        value = relation.get(field, [])
        roles[field] = value if isinstance(value, list) else []
    return {
        "complete_core_roles": all(roles[field] for field in CORE_ROLE_FIELDS),
        **roles,
    }


def build_example(
    gold_path: Path,
    predictions_path: Path,
    sample_id: str,
    setting_id: str,
) -> dict:
    """Join and validate one gold record and one frozen prediction."""

    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    prediction_rows = read_jsonl(predictions_path)
    gold = find_unique(gold_rows, sample_id=sample_id)
    prediction = find_unique(prediction_rows, sample_id=sample_id, setting_id=setting_id)

    image_path = gold_path.parent / gold["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Referenced image does not exist: {image_path}")

    predicted_relations = [summarize_relation(row) for row in prediction.get("evolutions", [])]
    return {
        "schema_version": "1.0",
        "purpose": "Lightweight walkthrough of one released frozen prediction; not a new model run.",
        "sample": {
            "sample_id": sample_id,
            "image": gold["image"],
            "figure_caption": gold.get("figure_caption", ""),
            "related_text": gold.get("related_text", ""),
        },
        "system_output": {
            "setting_id": setting_id,
            "parse_valid": bool(prediction.get("parse_valid", False)),
            "evolution_relation_count": len(predicted_relations),
            "complete_relation_count": sum(row["complete_core_roles"] for row in predicted_relations),
            "evolutions": predicted_relations,
        },
        "reference_summary": {
            "gold_evolution_relation_count": len(gold.get("gold_evolutions", [])),
            "description_gold": gold.get("description_gold", ""),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="FIG-041")
    parser.add_argument("--setting", default="table5/full_hypergraph")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/test_gold.json")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "predictions/extraction_predictions.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/example/FIG-041.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = build_example(args.gold, args.predictions, args.sample_id, args.setting)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output = result["system_output"]
    print(f"Sample: {args.sample_id}")
    print(f"Setting: {args.setting}")
    print(f"Parse valid: {output['parse_valid']}")
    print(
        "Predicted Evolution Relations: "
        f"{output['evolution_relation_count']} "
        f"({output['complete_relation_count']} with Input, Output, and Driver)"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
