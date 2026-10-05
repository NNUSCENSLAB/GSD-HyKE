"""Reconstruct Evolution Relations from the direct-triplet baseline format."""

from __future__ import annotations

import json
from typing import Any


FIELDS = (
    "input_mentions",
    "output_mentions",
    "mechanism_mentions",
    "location_mentions",
    "time_mentions",
)


def direct_triples_to_events(obj: dict[str, Any]) -> dict[str, Any]:
    """Group triples by evidence IDs and exact output anchor, preserving order.

    This is the published comparison's event reconstruction rule. Missing
    evidence IDs fall back to grouping by output anchor alone.
    """
    by_output: dict[tuple[tuple[int, ...], str], dict[str, Any]] = {}
    order: list[tuple[tuple[int, ...], str]] = []

    def evidence_ids(triple: dict[str, Any]) -> tuple[int, ...]:
        result = []
        raw = triple.get("evidence_sent_ids", [])
        for value in raw if isinstance(raw, list) else []:
            try:
                sentence_id = int(value)
            except (TypeError, ValueError):
                continue
            if sentence_id not in result:
                result.append(sentence_id)
        return tuple(result)

    def event(output: str, evidence: tuple[int, ...]) -> dict[str, Any]:
        key = (evidence, output)
        if key not in by_output:
            by_output[key] = {field: [] for field in FIELDS}
            if evidence:
                by_output[key]["evidence_sent_ids"] = list(evidence)
            order.append(key)
            by_output[key]["output_mentions"].append(output)
        return by_output[key]

    for triple in obj.get("triples", []):
        if not isinstance(triple, dict):
            continue
        head = str(triple.get("head", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        evidence = evidence_ids(triple)
        if not head or not tail:
            continue
        if relation in {"evolves_to", "drives"}:
            item = event(tail, evidence)
            field = "input_mentions" if relation == "evolves_to" else "mechanism_mentions"
            if head not in item[field]:
                item[field].append(head)
        elif relation in {"hasLocation", "hasTime"}:
            item = event(head, evidence)
            field = "location_mentions" if relation == "hasLocation" else "time_mentions"
            if tail not in item[field]:
                item[field].append(tail)

    return {"evolutions": [by_output[key] for key in order]}


def reconstruct_text(raw_prediction: str) -> tuple[dict[str, Any], bool]:
    """Return an empty prediction for a non-JSON raw response without inventing content."""
    try:
        parsed = json.loads(raw_prediction)
    except (TypeError, json.JSONDecodeError):
        return {"evolutions": []}, False
    if not isinstance(parsed, dict) or not isinstance(parsed.get("triples"), list):
        return {"evolutions": []}, False
    return direct_triples_to_events(parsed), True
