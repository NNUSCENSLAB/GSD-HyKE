#!/usr/bin/env python3
"""Recompute entity, relation, and graph metrics from frozen predictions."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer


ROLE_FIELDS = (
    "input_mentions",
    "output_mentions",
    "location_mentions",
    "time_mentions",
    "mechanism_mentions",
)
ROLE_LABELS = {
    "input_mentions": "Object",
    "output_mentions": "Object",
    "mechanism_mentions": "Mechanism",
    "location_mentions": "Location",
    "time_mentions": "Time",
}
GRAPH_WEIGHTS = {
    "input_mentions": 0.35,
    "output_mentions": 0.35,
    "mechanism_mentions": 0.15,
    "location_mentions": 0.10,
    "time_mentions": 0.05,
}


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def as_dict(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": self.tp, "fp": self.fp, "fn": self.fn}


class SemanticSimilarity:
    """Mean-pooled SciBERT cosine similarity with an in-memory cache."""

    def __init__(self, model_name: str, device: str) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.cache: dict[str, torch.Tensor] = {}

    def precompute(self, texts: Iterable[str], batch_size: int) -> None:
        values = sorted({str(text).strip() for text in texts if str(text).strip()} - self.cache.keys())
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                token_embeddings = self.model(**encoded)[0]
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                vectors = (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1).cpu()
            self.cache.update(dict(zip(batch, vectors)))

    def score(self, left: str, right: str) -> float:
        return float(torch.dot(self.cache[str(left).strip()], self.cache[str(right).strip()]).item())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_events(events: Sequence[dict], require_transition: bool = True) -> list[dict]:
    output = []
    for raw in events or []:
        event = {}
        for field in ROLE_FIELDS:
            values = raw.get(field, []) if isinstance(raw, dict) else []
            if isinstance(values, list):
                cleaned = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
                if cleaned:
                    event[field] = cleaned
        if not require_transition or (event.get("input_mentions") and event.get("output_mentions")):
            output.append(event)
    return output


def assignment(matrix: Sequence[Sequence[float]]) -> list[tuple[int, int, float]]:
    if not matrix or not matrix[0]:
        return []
    values = np.asarray(matrix, dtype=float)
    rows, columns = linear_sum_assignment(-values)
    return [(int(row), int(column), float(values[row, column])) for row, column in zip(rows, columns)]


def mention_matches(predicted: Sequence[str], gold: Sequence[str], sim: SemanticSimilarity, tau: float) -> int:
    if not predicted or not gold:
        return 0
    matrix = [[sim.score(left, right) for right in gold] for left in predicted]
    return sum(score >= tau for _, _, score in assignment(matrix))


def role_overlap(predicted: Sequence[str], gold: Sequence[str], sim: SemanticSimilarity, tau: float) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    matched = mention_matches(predicted, gold, sim, tau)
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def historical_role_score(predicted: Sequence[str], gold: Sequence[str], sim: SemanticSimilarity, tau: float) -> float:
    if not predicted or not gold:
        return 0.0
    used = set()
    matched = 0
    for pred_text in predicted:
        candidates = [(sim.score(pred_text, gold_text), index) for index, gold_text in enumerate(gold) if index not in used]
        if candidates:
            score, index = max(candidates)
            if score >= tau:
                used.add(index)
                matched += 1
    return matched / max(len(predicted), len(gold))


def relation_event_score(predicted: dict, gold: dict, sim: SemanticSimilarity, tau: float) -> float:
    scores = []
    for field in ROLE_FIELDS:
        # The manuscript evaluator formed role signatures with sorted sets
        # before its one-to-one greedy mention matching. Preserve that order:
        # greedy matching can otherwise differ for semantically overlapping
        # mentions even when the underlying strings are identical.
        pred_values = sorted(set(predicted.get(field, [])))
        gold_values = sorted(set(gold.get(field, [])))
        if not pred_values and not gold_values:
            continue
        scores.append(historical_role_score(pred_values, gold_values, sim, tau))
    return sum(scores) / len(scores) if scores else 0.0


def relation_tp(predicted: Sequence[dict], gold: Sequence[dict], sim: SemanticSimilarity, tau: float) -> int:
    if not predicted or not gold:
        return 0
    matrix = [[relation_event_score(pred, ref, sim, tau) for ref in gold] for pred in predicted]
    return sum(score >= tau for _, _, score in assignment(matrix))


def entities(events: Sequence[dict]) -> list[tuple[str, str]]:
    result = []
    for event in events:
        for field, label in ROLE_LABELS.items():
            result.extend((label, text) for text in event.get(field, []))
    return result


def entity_tp(predicted: Sequence[tuple[str, str]], gold: Sequence[tuple[str, str]], sim: SemanticSimilarity, tau: float, strict_label: bool = True) -> int:
    if not predicted or not gold:
        return 0
    matrix = []
    for pred_label, pred_text in predicted:
        matrix.append([
            sim.score(pred_text, gold_text) if not strict_label or pred_label == gold_label else 0.0
            for gold_label, gold_text in gold
        ])
    return sum(score >= tau for _, _, score in assignment(matrix))


def chain_edges(events: Sequence[dict], sim: SemanticSimilarity, tau: float) -> list[tuple[int, int]]:
    edges = []
    for left_index, left in enumerate(events):
        for right_index, right in enumerate(events):
            if left_index != right_index and mention_matches(
                left.get("output_mentions", []), right.get("input_mentions", []), sim, tau
            ):
                edges.append((left_index, right_index))
    return sorted(set(edges))


def graph_event_score(predicted: dict, gold: dict, sim: SemanticSimilarity, tau: float) -> float:
    return sum(
        GRAPH_WEIGHTS[field] * role_overlap(predicted.get(field, []), gold.get(field, []), sim, tau)
        for field in GRAPH_WEIGHTS
    )


def graph_sample(predicted: Sequence[dict], gold: Sequence[dict], sim: SemanticSimilarity, tau: float) -> dict:
    matrix = [[graph_event_score(pred, ref, sim, tau) for ref in gold] for pred in predicted]
    pairs = []
    for pred_index, gold_index, score in assignment(matrix):
        core = (
            mention_matches(predicted[pred_index].get("input_mentions", []), gold[gold_index].get("input_mentions", []), sim, tau)
            and mention_matches(predicted[pred_index].get("output_mentions", []), gold[gold_index].get("output_mentions", []), sim, tau)
        )
        if score > 0 and core:
            pairs.append((pred_index, gold_index, score))
    mapping = {pred_index: gold_index for pred_index, gold_index, _ in pairs}
    pred_edges = chain_edges(predicted, sim, tau)
    gold_edges = set(chain_edges(gold, sim, tau))
    mapped_edges = {(mapping[left], mapping[right]) for left, right in pred_edges if left in mapping and right in mapping}
    node_cost = len(predicted) + len(gold) - 2 * len(pairs) + sum(1 - score for _, _, score in pairs)
    edge_cost = len(gold_edges - mapped_edges) + len(mapped_edges - gold_edges)
    ged = node_cost + edge_cost
    normalizer = max(1, len(predicted) + len(gold) + len(pred_edges) + len(gold_edges))
    valid = [all(event.get(field) for field in ("input_mentions", "output_mentions", "mechanism_mentions")) for event in predicted]
    return {
        "ged": ged,
        "nged": ged / normalizer,
        "ecc_recovered": len(gold_edges & mapped_edges),
        "ecc_gold": len(gold_edges),
        "epv": sum(valid) / len(valid) if valid else None,
    }


def collect_mentions(gold_rows: list[dict], prediction_rows: list[dict], include_incomplete: bool = False) -> list[str]:
    values = []
    for row in gold_rows:
        for event in clean_events(row["gold_evolutions"]):
            for field in ROLE_FIELDS:
                values.extend(event.get(field, []))
    for row in prediction_rows:
        for event in clean_events(row["evolutions"], require_transition=not include_incomplete):
            for field in ROLE_FIELDS:
                values.extend(event.get(field, []))
    return values


def evaluate_setting(
    gold_rows: list[dict], prediction_rows: list[dict], sim: SemanticSimilarity,
    tau: float,
) -> dict:
    gold = {row["sample_id"]: clean_events(row["gold_evolutions"]) for row in gold_rows}
    predicted = {row["sample_id"]: clean_events(row["evolutions"]) for row in prediction_rows}
    raw_predicted = {row["sample_id"]: clean_events(row["evolutions"], require_transition=False) for row in prediction_rows}
    if set(gold) != set(predicted):
        raise ValueError("Gold and prediction sample_id sets differ")
    relation = Counts()
    entity = Counts()
    graph_rows = []
    for sample_id in sorted(gold):
        gold_events, pred_events = gold[sample_id], predicted[sample_id]
        tp = relation_tp(pred_events, gold_events, sim, tau)
        relation.tp += tp
        relation.fp += len(pred_events) - tp
        relation.fn += len(gold_events) - tp
        entity_events = raw_predicted[sample_id]
        gold_entities, pred_entities = entities(gold_events), entities(entity_events)
        tp = entity_tp(
            pred_entities, gold_entities, sim, tau,
            strict_label=True,
        )
        entity.tp += tp
        entity.fp += len(pred_entities) - tp
        entity.fn += len(gold_entities) - tp
        graph_events = raw_predicted[sample_id]
        graph_rows.append(graph_sample(graph_events, gold_events, sim, tau))
    recovered = sum(row["ecc_recovered"] for row in graph_rows)
    gold_edges = sum(row["ecc_gold"] for row in graph_rows)
    epv = [row["epv"] for row in graph_rows if row["epv"] is not None]
    return {
        "samples": len(gold),
        "entity_label_constrained_relaxed": entity.as_dict(),
        "relation_relaxed": relation.as_dict(),
        "graph": {
            "ged": sum(row["ged"] for row in graph_rows) / len(graph_rows),
            "nged": sum(row["nged"] for row in graph_rows) / len(graph_rows),
            "ecc": recovered / gold_edges if gold_edges else None,
            "ecc_recovered_edges": recovered,
            "ecc_gold_edges": gold_edges,
            "epv": sum(epv) / len(epv) if epv else None,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=root / "data/test_gold.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions/extraction_predictions.jsonl")
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=root / "results/recomputed_extraction_metrics.json")
    args = parser.parse_args()

    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    all_predictions = read_jsonl(args.predictions)
    grouped = defaultdict(list)
    for row in all_predictions:
        grouped[row["setting_id"]].append(row)
    selected = args.setting or sorted(grouped)
    unknown = sorted(set(selected) - set(grouped))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    selected_rows = [row for name in selected for row in grouped[name]]
    similarity = SemanticSimilarity(args.model, args.device)
    similarity.precompute(collect_mentions(gold_rows, selected_rows, include_incomplete=True), args.batch_size)
    results = {
        "config": {"similarity_model": args.model, "threshold": args.threshold},
        "settings": {
            name: evaluate_setting(gold_rows, grouped[name], similarity, args.threshold)
            for name in selected
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, result in results["settings"].items():
        relation = result["relation_relaxed"]
        graph = result["graph"]
        epv = f"{graph['epv']:.4f}" if graph["epv"] is not None else "N/A"
        print(f"{name}: Relation F1={relation['f1']:.4f}, GED={graph['ged']:.4f}, nGED={graph['nged']:.4f}, ECC={graph['ecc']:.4f}, EPV={epv}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
