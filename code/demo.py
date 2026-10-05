#!/usr/bin/env python3
"""Show a synthetic two-stage example and score an identical relation fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import SemanticSimilarity, collect_mentions, evaluate_setting


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    relation = json.loads((EXAMPLES / "demo_relations.json").read_text(encoding="utf-8"))
    gold = [{"sample_id": "SYNTHETIC-DEMO", "gold_evolutions": relation["evolutions"]}]
    prediction = [{"sample_id": "SYNTHETIC-DEMO", "evolutions": relation["evolutions"]}]

    print("DESCRIPTION")
    print((EXAMPLES / "demo_description.txt").read_text(encoding="utf-8").strip())
    print("\nEVIDENCE SENTENCE")
    print((EXAMPLES / "demo_evidence.txt").read_text(encoding="utf-8").strip())
    print("\nEVOLUTION RELATION JSON")
    print(json.dumps(relation, ensure_ascii=False, indent=2))

    similarity = SemanticSimilarity(args.model, args.device)
    similarity.precompute(collect_mentions(gold, prediction), batch_size=8)
    result = evaluate_setting(gold, prediction, similarity, tau=0.7)
    print("\nSynthetic self-match: Relation Relaxed F1 = " + f"{result['relation_relaxed']['f1']:.4f}")
    print("This checks the schema and metric path; it is not a model-performance result.")


if __name__ == "__main__":
    main()
