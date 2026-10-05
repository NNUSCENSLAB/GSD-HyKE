import argparse
import importlib.util
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASE_EVAL_PATH = Path(__file__).resolve().parent / "eval_stage2b1_evolutions_template_text.py"

ROLE_FIELDS = ("input_ids", "output_ids", "location_ids", "time_ids", "mechanism_ids")
ROOT_LIST_FIELDS = {
    "evidence_sent_ids": "evidence_sent_ids",
    "evidences": "evidence_sent_ids",
    "input_mentions": "input_mentions",
    "input": "input_mentions",
    "inputs": "input_mentions",
    "output_mentions": "output_mentions",
    "output": "output_mentions",
    "outputs": "output_mentions",
    "mechanism_mentions": "mechanism_mentions",
    "mechanism": "mechanism_mentions",
    "mechanisms": "mechanism_mentions",
    "location_mentions": "location_mentions",
    "location": "location_mentions",
    "locations": "location_mentions",
    "time_mentions": "time_mentions",
    "time": "time_mentions",
    "times": "time_mentions",
}
SENT_ID_PATTERN = re.compile(r"S?(\d+)")
CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```\s*$", re.S)
PAREN_QUOTED_SPAN_PATTERN = re.compile(r'\("([^"\n]+)"\)')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


template_eval = load_module(BASE_EVAL_PATH, "stage2b1_template_eval_for_exact_proto")


def canonical_root(obj):
    if isinstance(obj, list):
        return {"evolutions": obj}
    if isinstance(obj, dict) and isinstance(obj.get("evolutions"), list):
        return obj
    return None


def sanitize_prediction_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    match = CODE_FENCE_PATTERN.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    # Repair common JSON breakage such as:
    # "location_mentions": ["Fall face (">45°")"]
    # where inner quotes are unescaped but only decorate a parenthetical span.
    cleaned = PAREN_QUOTED_SPAN_PATTERN.sub(lambda m: f'({m.group(1)})', cleaned)
    return cleaned


def normalize_sent_ids(values) -> List[int]:
    if not isinstance(values, list):
        return []
    out = []
    for val in values:
        if isinstance(val, int):
            out.append(int(val))
            continue
        match = SENT_ID_PATTERN.fullmatch(str(val).strip())
        if match:
            out.append(int(match.group(1)))
    return sorted(set(v for v in out if v > 0))


def normalize_mentions(values) -> List[str]:
    if not isinstance(values, list):
        return []
    out = []
    seen = set()
    for val in values:
        text = str(val).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def extract_balanced_json_list(text: str, start_idx: int) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "[":
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def extract_field_list_fragment(text: str, aliases: Sequence[str]) -> Optional[List]:
    for alias in aliases:
        pattern = re.compile(rf'"{re.escape(alias)}"\s*:\s*\[', re.I)
        match = pattern.search(text)
        if not match:
            continue
        start = text.find("[", match.start())
        fragment = extract_balanced_json_list(text, start)
        if fragment is None:
            continue
        try:
            values = json.loads(fragment)
        except Exception:
            continue
        if isinstance(values, list):
            return values
    return None


def recover_truncated_single_event(text: str) -> Optional[Dict]:
    field_aliases = {
        "evidence_sent_ids": ("evidence_sent_ids", "evidences"),
        "input_mentions": ("input_mentions", "input", "inputs"),
        "output_mentions": ("output_mentions", "output", "outputs"),
        "mechanism_mentions": ("mechanism_mentions", "mechanism", "mechanisms"),
        "location_mentions": ("location_mentions", "location", "locations"),
        "time_mentions": ("time_mentions", "time", "times"),
    }

    norm = {}
    for dst, aliases in field_aliases.items():
        values = extract_field_list_fragment(text, aliases)
        if values is None:
            continue
        if dst == "evidence_sent_ids":
            clean_vals = normalize_sent_ids(values)
        else:
            clean_vals = normalize_mentions(values)
        if clean_vals:
            norm[dst] = clean_vals

    if not norm.get("output_mentions"):
        return None
    if len(norm.get("evidence_sent_ids", [])) > 3:
        return None
    if len(norm.get("input_mentions", [])) > 8:
        return None
    if len(norm.get("output_mentions", [])) > 12:
        return None
    if len(norm.get("mechanism_mentions", [])) > 8:
        return None
    return {"evolutions": [norm]}


def parse_prediction_text(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    text = sanitize_prediction_text(text)
    obj = template_eval.base.extract_json_object(text)
    root = canonical_root(obj)
    if root is None:
        stripped = (text or "").strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                root = canonical_root(json.loads(stripped))
            except Exception:
                root = None
    if root is None:
        root = recover_truncated_single_event(text)
    if root is None:
        return None, "invalid_json"

    cleaned = []
    for evo in root["evolutions"]:
        if not isinstance(evo, dict):
            continue
        norm = {}
        for src, dst in ROOT_LIST_FIELDS.items():
            if src not in evo:
                continue
            if dst == "evidence_sent_ids":
                vals = normalize_sent_ids(evo.get(src))
            else:
                vals = normalize_mentions(evo.get(src))
            if vals:
                norm[dst] = vals
        if norm.get("output_mentions"):
            cleaned.append(norm)
    return {"evolutions": cleaned}, None


def load_gold_samples(path: str) -> List[Dict]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    samples = []
    for idx, row in enumerate(rows):
        user_text = row["messages"][1]["content"]
        assistant_text = row["messages"][2]["content"]

        if "Input passage:\n" in user_text:
            gold_proto, invalid_reason = parse_prediction_text(assistant_text)
            if gold_proto is None:
                gold_obj = {"evolutions": []}
                gold_entities = []
            else:
                gold_obj, gold_entities = convert_proto_to_eval_graph(gold_proto)
        else:
            gold_entities, _ = template_eval.builder.extract_entities_and_text(user_text)
            gold_obj, invalid_reason = template_eval.parse_prediction_from_gold_json(assistant_text)
        samples.append(
            {
                "index": row.get("index", idx),
                "gold_entities": gold_entities,
                "gold_obj": gold_obj,
                "gold_invalid_reason": invalid_reason,
            }
        )
    return samples


def make_pred_entity(label: str, text: str, entity_id: int) -> Dict:
    return {"id": entity_id, "text": text, "label": label}


def convert_proto_to_eval_graph(proto_obj: Dict) -> Tuple[Dict, List[Dict]]:
    next_id = 1
    pred_entities = []
    evolutions = []
    role_to_label = {
        "input_mentions": "Object",
        "output_mentions": "Object",
        "mechanism_mentions": "Mechanism",
        "location_mentions": "Location",
        "time_mentions": "Time",
    }

    for evo in proto_obj.get("evolutions", []):
        norm_evo = {}
        for mention_field, role_field in (
            ("input_mentions", "input_ids"),
            ("output_mentions", "output_ids"),
            ("mechanism_mentions", "mechanism_ids"),
            ("location_mentions", "location_ids"),
            ("time_mentions", "time_ids"),
        ):
            ids = []
            for mention in evo.get(mention_field, []):
                pred_entities.append(make_pred_entity(role_to_label[mention_field], mention, next_id))
                ids.append(next_id)
                next_id += 1
            if ids:
                norm_evo[role_field] = ids
        if template_eval.has_valid_core_roles(norm_evo):
            evolutions.append(norm_evo)
    return {"evolutions": evolutions}, pred_entities


def normalized_exact_mention(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).strip().split())


def exact_event_signature(obj: Dict, entities: Sequence[Dict]) -> Tuple[Tuple[str, ...], ...]:
    entity_by_id = {int(entity["id"]): entity for entity in entities if "id" in entity}
    fields = (
        "input_ids",
        "output_ids",
        "mechanism_ids",
        "location_ids",
        "time_ids",
    )
    signature = []
    for field in fields:
        mentions = []
        for entity_id in obj.get(field, []):
            entity = entity_by_id.get(int(entity_id))
            if entity:
                mentions.append(normalized_exact_mention(entity.get("text", "")))
        signature.append(tuple(sorted(set(mentions))))
    return tuple(signature)


def exact_evolution_match(pred_obj: Dict, gold_obj: Dict, pred_entities: Sequence[Dict], gold_entities: Sequence[Dict]) -> int:
    gold_remaining = [exact_event_signature(event, gold_entities) for event in gold_obj.get("evolutions", [])]
    matches = 0
    for pred_event in pred_obj.get("evolutions", []):
        pred_signature = exact_event_signature(pred_event, pred_entities)
        try:
            gold_index = gold_remaining.index(pred_signature)
        except ValueError:
            continue
        gold_remaining.pop(gold_index)
        matches += 1
    return matches


def evaluate(gold_samples: Sequence[Dict], pred_rows: Sequence[Dict], tau: float, sim_model: str, exact_match: bool = False) -> Dict:
    pred_map = {int(row["index"]): row.get("prediction", "") for row in pred_rows}
    sim = template_eval.base.SemanticSimilarity(sim_model)
    evo_metric = template_eval.base.MetricCounter()
    per_sample = []

    for sample in gold_samples:
        pred_text = pred_map.get(int(sample["index"]), "")
        proto_obj, invalid_reason = parse_prediction_text(pred_text)
        if proto_obj is None:
            evo_metric.fn += len(sample["gold_obj"]["evolutions"])
            per_sample.append({"index": sample["index"], "format_valid": False, "invalid_reason": invalid_reason})
            continue

        pred_obj, pred_entities = convert_proto_to_eval_graph(proto_obj)
        if exact_match:
            evo_tp = exact_evolution_match(
                pred_obj=pred_obj,
                gold_obj=sample["gold_obj"],
                pred_entities=pred_entities,
                gold_entities=sample["gold_entities"],
            )
        else:
            evo_tp = template_eval.evolution_soft_match(
                pred_obj=pred_obj,
                gold_obj=sample["gold_obj"],
                pred_entities=pred_entities,
                gold_entities=sample["gold_entities"],
                sim=sim,
                tau=tau,
            )
        evo_metric.tp += evo_tp
        evo_metric.fp += max(0, len(pred_obj["evolutions"]) - evo_tp)
        evo_metric.fn += max(0, len(sample["gold_obj"]["evolutions"]) - evo_tp)
        per_sample.append(
            {
                "index": sample["index"],
                "format_valid": True,
                "pred_evolution_count": len(pred_obj["evolutions"]),
                "gold_evolution_count": len(sample["gold_obj"]["evolutions"]),
                "evolution_tp": evo_tp,
            }
        )

    return {
        "tau": tau,
        "match_mode": "normalized_exact" if exact_match else "semantic_soft",
        "evolution_soft": {
            "precision": evo_metric.precision(),
            "recall": evo_metric.recall(),
            "f1": evo_metric.f1(),
            "tp": evo_metric.tp,
            "fp": evo_metric.fp,
            "fn": evo_metric.fn,
        },
        "num_samples": len(gold_samples),
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate exact-mention prototype outputs against Stage2B1 gold.")
    parser.add_argument("--gold-path", required=True, type=str)
    parser.add_argument("--pred-path", required=True, type=str)
    parser.add_argument("--result-path", required=True, type=str)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--sim-model", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--exact-match", action="store_true", help="Require normalized exact mention sets for every event role.")
    parser.add_argument("--merge-object-mechanism", action="store_true")
    parser.add_argument("--allow-partial-core-events", action="store_true")
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional comma-separated original sample indices to evaluate.",
    )
    args = parser.parse_args()

    template_eval.MERGE_OBJECT_MECHANISM = bool(args.merge_object_mechanism)
    template_eval.ALLOW_PARTIAL_CORE_EVENTS = bool(args.allow_partial_core_events)

    gold_samples = load_gold_samples(args.gold_path)
    selected_indices = {int(value.strip()) for value in args.indices.split(",") if value.strip()}
    if selected_indices:
        gold_samples = [sample for sample in gold_samples if int(sample["index"]) in selected_indices]
        found_indices = {int(sample["index"]) for sample in gold_samples}
        if found_indices != selected_indices:
            raise ValueError(f"Requested indices missing from gold: {sorted(selected_indices - found_indices)}")
    pred_rows = json.load(open(args.pred_path, "r", encoding="utf-8"))
    result = evaluate(gold_samples, pred_rows, tau=args.tau, sim_model=args.sim_model, exact_match=args.exact_match)

    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["evolution_soft"], ensure_ascii=False, indent=2))
    print(f"Saved to: {args.result_path}")


if __name__ == "__main__":
    main()
