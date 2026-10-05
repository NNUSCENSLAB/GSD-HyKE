import argparse
import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Avoid protobuf C-extension incompatibilities when importing AutoProcessor
# under the system python environment.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

try:
    from transformers import MllamaForConditionalGeneration
except Exception:
    MllamaForConditionalGeneration = None

try:
    from transformers import InternVLForConditionalGeneration
except Exception:
    InternVLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLConfig = None
    Qwen2VLForConditionalGeneration = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


ENTITY_TYPES = {
    "Time",
    "Location",
    "Object",
    "Attribute",
    "Evolution",
    "Mechanism",
    "Sub-Mechanism",
}

RELATION_TYPES = {
    "hasLocation",
    "hasTime",
    "hasComponent",
    "hasAttribute",
    "Input",
    "Output",
    "Cause",
    "Drive",
}

DUAL_TOP_LEVEL_KEYS = {
    "semantic_entities",
    "hyper_nodes",
    "semantic_relations",
    "incidence_relations",
    "hyper_relations",
}


def _encode_non_empty(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids] if ids else []


def _build_json_start_token_ids(tokenizer) -> Set[int]:
    start_ids: Set[int] = set()
    for s in ["{", " {", "\n{", "\n {", "\t{", "\t {"]:
        ids = _encode_non_empty(tokenizer, s)
        if ids:
            start_ids.add(ids[0])
    return start_ids


def _build_bad_words_ids(tokenizer) -> List[List[int]]:
    # Ban common markdown/typography tokens that frequently corrupt strict JSON.
    ban_texts = [
        "```",
        "```json",
        "```JSON",
        "“",
        "”",
        "‘",
        "’",
        "（",
        "）",
        "【",
        "】",
    ]
    bad_ids: List[List[int]] = []
    seen = set()
    for t in ban_texts:
        ids = _encode_non_empty(tokenizer, t)
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        bad_ids.append(ids)
    return bad_ids


class JsonConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        allowed_start_ids: Set[int],
        banned_ids: Set[int],
    ):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.allowed_start_ids = set(int(x) for x in allowed_start_ids)
        self.banned_ids = set(int(x) for x in banned_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1) Global ban on high-risk tokens (smart quotes / markdown fence).
        if self.banned_ids:
            banned = [i for i in self.banned_ids if 0 <= i < scores.shape[-1]]
            if banned:
                scores[:, banned] = -float("inf")

        for row in range(input_ids.shape[0]):
            gen_ids = input_ids[row, self.prompt_len :]
            gen_len = int(gen_ids.numel())

            # 2) First generated token should start JSON directly.
            if gen_len == 0 and self.allowed_start_ids:
                allowed = [i for i in self.allowed_start_ids if 0 <= i < scores.shape[-1]]
                if allowed:
                    row_scores = scores[row]
                    mask = torch.ones_like(row_scores, dtype=torch.bool)
                    mask[allowed] = False
                    row_scores[mask] = -float("inf")
                continue

            # Keep this processor lightweight: avoid per-step decode/parse checks.

        return scores


@dataclass
class MetricCounter:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def extract_json_object(raw_text: str) -> Optional[Dict]:
    if not raw_text:
        return None
    raw_text = raw_text.strip()
    try:
        obj = json.loads(raw_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    left = raw_text.find("{")
    right = raw_text.rfind("}")
    if left == -1 or right == -1 or left >= right:
        return None

    chunk = raw_text[left : right + 1]
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _norm_id(v):
    if v is None:
        return None
    # Handle numeric ids such as 12965.0 in model output.
    if isinstance(v, (int, float)):
        if float(v).is_integer():
            return str(int(v))
        return str(v)
    s = str(v).strip()
    return s if s else None


def normalize_graph(obj: Dict) -> Dict:
    entities = []
    relations = []
    if isinstance(obj, dict):
        obj_keys = set(obj.keys())
        if {"entities", "relations"}.issubset(obj_keys):
            entities = obj.get("entities", [])
            relations = obj.get("relations", [])
        elif DUAL_TOP_LEVEL_KEYS.issubset(obj_keys):
            # Convert dual-layer output to single-layer for evaluation:
            # entities := semantic_entities + hyper_nodes
            # relations := semantic_relations + incidence_relations + hyper_relations
            entities = (obj.get("semantic_entities", []) or []) + (obj.get("hyper_nodes", []) or [])
            relations = (
                (obj.get("semantic_relations", []) or [])
                + (obj.get("incidence_relations", []) or [])
                + (obj.get("hyper_relations", []) or [])
            )

    norm_entities = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        if "id" not in e or "text" not in e:
            continue
        norm_entities.append(
            {
                "id": str(e.get("id")),
                "text": str(e.get("text", "")).strip(),
                "label": str(e.get("label", "")).strip(),
            }
        )

    norm_relations = []
    for r in relations:
        if not isinstance(r, dict):
            continue
        rel_type = str(r.get("type", "")).strip()

        # Primary schema.
        from_id = _norm_id(r.get("from_id"))
        to_id = _norm_id(r.get("to_id"))

        # Common alternative keys from imperfect generations.
        if from_id is None:
            from_id = _norm_id(r.get("head_id"))
        if to_id is None:
            to_id = _norm_id(r.get("tail_id"))

        # Another common variant: from_to: [from, to].
        if (from_id is None or to_id is None) and isinstance(r.get("from_to"), list):
            pair = r.get("from_to")
            if len(pair) >= 2:
                from_id = _norm_id(pair[0]) if from_id is None else from_id
                to_id = _norm_id(pair[1]) if to_id is None else to_id

        if from_id is None or to_id is None or not rel_type:
            continue
        norm_relations.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "type": rel_type,
            }
        )

    return {"entities": norm_entities, "relations": norm_relations}


def parse_and_validate_prediction(pred_text: str, strict_output: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    pred_obj = extract_json_object(pred_text)
    if pred_obj is None:
        return None, "invalid_json"

    pred_graph = normalize_graph(pred_obj)
    if not strict_output:
        return pred_graph, None

    obj_keys = set(pred_obj.keys())
    is_single = obj_keys == {"entities", "relations"}
    is_dual = obj_keys == DUAL_TOP_LEVEL_KEYS
    if not (is_single or is_dual):
        return None, "top_level_keys_must_be_single_or_dual_schema"

    if is_single:
        raw_entities = pred_obj.get("entities")
        raw_relations = pred_obj.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            return None, "entities_relations_must_be_lists"
        if len(raw_entities) != len(pred_graph.get("entities", [])):
            return None, "malformed_entity_items"
        if len(raw_relations) != len(pred_graph.get("relations", [])):
            return None, "malformed_relation_items"
        raw_entity_lists = [raw_entities]
        raw_relation_lists = [raw_relations]
    else:
        raw_sem_entities = pred_obj.get("semantic_entities")
        raw_hyper_nodes = pred_obj.get("hyper_nodes")
        raw_sem_rel = pred_obj.get("semantic_relations")
        raw_inc_rel = pred_obj.get("incidence_relations")
        raw_hyper_rel = pred_obj.get("hyper_relations")
        if not all(
            isinstance(x, list)
            for x in [raw_sem_entities, raw_hyper_nodes, raw_sem_rel, raw_inc_rel, raw_hyper_rel]
        ):
            return None, "dual_schema_values_must_be_lists"
        raw_entity_lists = [raw_sem_entities, raw_hyper_nodes]
        raw_relation_lists = [raw_sem_rel, raw_inc_rel, raw_hyper_rel]

    entity_ids = set()
    for raw_entities in raw_entity_lists:
        for e in raw_entities:
            if not isinstance(e, dict):
                return None, "entity_not_object"
            if set(e.keys()) != {"id", "text", "label"}:
                return None, "entity_keys_must_be_id_text_label"

    for e in pred_graph["entities"]:
        if e["label"] not in ENTITY_TYPES:
            return None, f"invalid_entity_label:{e['label']}"
        if e["id"] in entity_ids:
            return None, "duplicate_entity_id"
        entity_ids.add(e["id"])

    for raw_relations in raw_relation_lists:
        for r in raw_relations:
            if not isinstance(r, dict):
                return None, "relation_not_object"
            if set(r.keys()) != {"from_id", "to_id", "type"}:
                return None, "relation_keys_must_be_from_id_to_id_type"

    for r in pred_graph["relations"]:
        if r["type"] not in RELATION_TYPES:
            return None, f"invalid_relation_type:{r['type']}"
        if r["from_id"] not in entity_ids or r["to_id"] not in entity_ids:
            return None, "dangling_relation_id"

    return pred_graph, None


def load_stage2_gold(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict] = []
    for i, item in enumerate(data):
        messages = item.get("messages", [])
        system_text = "You are a helpful assistant."
        user_text = ""
        assistant_text = ""

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content

        gold_obj = extract_json_object(assistant_text)
        gold_graph = normalize_graph(gold_obj) if gold_obj else {"entities": [], "relations": []}

        image_rel = item.get("images", [None])[0]
        samples.append(
            {
                "index": i,
                "image_rel": image_rel,
                "system": system_text,
                "user": user_text,
                "gold_raw": assistant_text,
                "gold": gold_graph,
            }
        )
    return samples


class SemanticSimilarity:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache: Dict[str, torch.Tensor] = {}
        self.use_bert = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[WARN] Failed to load embedding model '{model_name}'. Fall back to lexical similarity. Error: {e}")
            self.use_bert = False

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

    def _embed(self, text: str) -> torch.Tensor:
        if text in self.cache:
            return self.cache[text]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
            vec = self._mean_pooling(output, encoded["attention_mask"])  # (1, hidden)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1).squeeze(0).cpu()

        self.cache[text] = vec
        return vec

    def score(self, a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0

        if not self.use_bert:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        va = self._embed(a)
        vb = self._embed(b)
        return float(torch.dot(va, vb).item())


def best_bipartite_matches(sim_matrix: List[List[float]]) -> List[Tuple[int, int, float]]:
    if not sim_matrix or not sim_matrix[0]:
        return []

    rows = len(sim_matrix)
    cols = len(sim_matrix[0])

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        import numpy as np

        cost = -np.array(sim_matrix, dtype=float)
        r_idx, c_idx = linear_sum_assignment(cost)
        return [(int(r), int(c), float(sim_matrix[int(r)][int(c)])) for r, c in zip(r_idx, c_idx)]
    except Exception:
        # Greedy fallback if scipy is unavailable.
        pairs = []
        used_r = set()
        used_c = set()

        flat = []
        for r in range(rows):
            for c in range(cols):
                flat.append((sim_matrix[r][c], r, c))
        flat.sort(key=lambda x: x[0], reverse=True)

        for score, r, c in flat:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c, float(score)))
        return pairs


def entity_soft_match(
    pred_entities: List[Dict],
    gold_entities: List[Dict],
    sim: SemanticSimilarity,
    tau: float,
    strict_label: bool = True,
) -> int:
    if not pred_entities or not gold_entities:
        return 0

    matrix = []
    for p in pred_entities:
        row = []
        for g in gold_entities:
            if strict_label and p.get("label") != g.get("label"):
                row.append(0.0)
                continue
            row.append(sim.score(p.get("text", ""), g.get("text", "")))
        matrix.append(row)

    matched = best_bipartite_matches(matrix)
    return sum(1 for _, _, s in matched if s >= tau)


def relation_soft_match(
    pred_graph: Dict,
    gold_graph: Dict,
    sim: SemanticSimilarity,
    tau: float,
    strict_type: bool = True,
) -> int:
    pred_entities = {e["id"]: e for e in pred_graph.get("entities", [])}
    gold_entities = {e["id"]: e for e in gold_graph.get("entities", [])}

    pred_rels = pred_graph.get("relations", [])
    gold_rels = gold_graph.get("relations", [])

    if not pred_rels or not gold_rels:
        return 0

    matrix = []
    for pr in pred_rels:
        p_from = pred_entities.get(pr.get("from_id", ""), {}).get("text", "")
        p_to = pred_entities.get(pr.get("to_id", ""), {}).get("text", "")
        p_type = pr.get("type", "")

        row = []
        for gr in gold_rels:
            g_from = gold_entities.get(gr.get("from_id", ""), {}).get("text", "")
            g_to = gold_entities.get(gr.get("to_id", ""), {}).get("text", "")
            g_type = gr.get("type", "")

            if strict_type and p_type != g_type:
                row.append(0.0)
                continue

            hs = sim.score(p_from, g_from)
            ts = sim.score(p_to, g_to)
            row.append((hs + ts) / 2.0)
        matrix.append(row)

    matched = best_bipartite_matches(matrix)
    return sum(1 for _, _, s in matched if s >= tau)


def compute_structure_metrics(pred_graphs: List[Optional[Dict]]) -> Dict[str, float]:
    total = len(pred_graphs)
    json_valid = 0

    total_rel = 0
    valid_rel_ref = 0
    dangling_rel = 0
    duplicate_rel = 0

    for graph in pred_graphs:
        if graph is None:
            continue

        json_valid += 1
        entities = graph.get("entities", [])
        relations = graph.get("relations", [])

        entity_ids = {e.get("id") for e in entities}
        rel_keys = []

        for r in relations:
            total_rel += 1
            key = (r.get("from_id"), r.get("to_id"), r.get("type"))
            rel_keys.append(key)

            ok = r.get("from_id") in entity_ids and r.get("to_id") in entity_ids
            if ok:
                valid_rel_ref += 1
            else:
                dangling_rel += 1

        duplicate_rel += max(0, len(rel_keys) - len(set(rel_keys)))

    id_valid_rate = valid_rel_ref / total_rel if total_rel > 0 else 1.0
    dangling_edge_rate = dangling_rel / total_rel if total_rel > 0 else 0.0
    duplicate_edge_rate = duplicate_rel / total_rel if total_rel > 0 else 0.0

    return {
        "json_valid_rate": json_valid / total if total > 0 else 0.0,
        "id_valid_rate": id_valid_rate,
        "dangling_edge_rate": dangling_edge_rate,
        "duplicate_edge_rate": duplicate_edge_rate,
    }


def resolve_image_path(image_root: str, image_rel: str) -> str:
    if image_rel is None:
        raise ValueError("Sample has no image path")
    if os.path.isabs(image_rel):
        return image_rel
    return os.path.join(image_root, image_rel)


def build_qwen_messages(
    system_text: str,
    user_text: str,
    image_path: str,
    strict_output: bool = False,
    retry_reason: Optional[str] = None,
) -> List[Dict]:
    cleaned_user = user_text.replace("<image>", "").strip()

    if strict_output:
        is_dual_prompt = (
            "semantic_entities" in cleaned_user
            and "hyper_nodes" in cleaned_user
            and "incidence_relations" in cleaned_user
        )
        if is_dual_prompt:
            strict_block = (
                "\n\nStrict output constraints:\n"
                "1) Output ONLY one valid JSON object with keys: \"semantic_entities\", \"hyper_nodes\", \"semantic_relations\", \"incidence_relations\", \"hyper_relations\".\n"
                "2) Each entity/node item must be exactly: {\"id\": int, \"text\": str, \"label\": str}.\n"
                "3) Each relation item must be exactly: {\"from_id\": int, \"to_id\": int, \"type\": str}.\n"
                "4) label must be one of: [\"Time\", \"Location\", \"Object\", \"Attribute\", \"Evolution\", \"Mechanism\", \"Sub-Mechanism\"].\n"
                "5) relation.type must be one of: [\"hasLocation\", \"hasTime\", \"hasComponent\", \"hasAttribute\", \"Input\", \"Output\", \"Cause\", \"Drive\"].\n"
                "6) from_id and to_id must refer to existing ids in semantic_entities or hyper_nodes.\n"
                "7) Do NOT output any additional keys, explanation text, markdown, or comments.\n"
                "8) Keep the output compact (single-line minified JSON, no pretty print).\n"
                "9) Extract only information grounded in the given image/text; do NOT invent extra placeholder chains.\n"
            )
        else:
            strict_block = (
                "\n\nStrict output constraints:\n"
                "1) Output ONLY one valid JSON object with top-level keys: \"entities\" and \"relations\".\n"
                "2) Each entity must be exactly: {\"id\": int, \"text\": str, \"label\": str}.\n"
                "3) entity.label must be one of: [\"Time\", \"Location\", \"Object\", \"Attribute\", \"Evolution\", \"Mechanism\", \"Sub-Mechanism\"].\n"
                "4) Each relation must be exactly: {\"from_id\": int, \"to_id\": int, \"type\": str}.\n"
                "5) relation.type must be one of: [\"hasLocation\", \"hasTime\", \"hasComponent\", \"hasAttribute\", \"Input\", \"Output\", \"Cause\", \"Drive\"].\n"
                "6) from_id and to_id must refer to existing entity ids in the same output.\n"
                "7) Do NOT output any additional keys, explanation text, markdown, or comments.\n"
                "8) Keep the output compact (single-line minified JSON, no pretty print).\n"
                "9) Extract only information grounded in the given image/text; do NOT invent extra placeholder chains.\n"
                "10) Keep the graph concise: entities <= 80 and relations <= 120.\n"
            )
        cleaned_user = cleaned_user + strict_block

    if retry_reason:
        cleaned_user = (
            cleaned_user
            + f"\n\nPrevious output was invalid ({retry_reason}). Regenerate and strictly follow the constraints."
        )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": cleaned_user},
            ],
        },
    ]


def _detect_vl_model_type(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return str(getattr(config, "model_type", "") or "").lower()
    except Exception:
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
                return str(raw.get("model_type", "") or "").lower()
            except Exception:
                pass
        raise


def _load_qwen2_vl_compat_config(model_path: str):
    if Qwen2VLConfig is None:
        raise RuntimeError("Current transformers version does not provide Qwen2VLConfig.")
    config_path = Path(model_path) / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["model_type"] = "qwen2_vl"
    architectures = raw.get("architectures")
    if isinstance(architectures, list):
        raw["architectures"] = [
            "Qwen2VLForConditionalGeneration" if arch == "Qwen2_5_VLForConditionalGeneration" else arch
            for arch in architectures
        ]
    return Qwen2VLConfig.from_dict(raw)


def load_vl_model_and_processor(
    model_path: str,
    adapter_path: str = "",
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = True,
    gpu_max_memory_gib: Optional[int] = None,
    cpu_max_memory_gib: int = 200,
    offload_folder: Optional[str] = None,
):
    if AutoProcessor is None or PeftModel is None:
        raise RuntimeError(
            "Inference dependencies are unavailable. "
            "Make sure the installed transformers/peft versions support VL inference."
        )

    model_type = _detect_vl_model_type(model_path)
    config_override = None
    if model_type == "mllama":
        if MllamaForConditionalGeneration is None:
            raise RuntimeError("Current transformers version does not provide MllamaForConditionalGeneration.")
        model_cls = MllamaForConditionalGeneration
    elif model_type in {"qwen2_5_vl", "qwen2_vl"}:
        if Qwen2_5_VLForConditionalGeneration is not None:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif model_type == "qwen2_5_vl" and Qwen2VLForConditionalGeneration is not None:
            model_cls = Qwen2VLForConditionalGeneration
            config_override = _load_qwen2_vl_compat_config(model_path)
        else:
            if AutoModelForVision2Seq is None:
                raise RuntimeError(
                    "Current transformers version does not provide Qwen2_5_VLForConditionalGeneration "
                    "and AutoModelForVision2Seq is unavailable."
                )
            model_cls = AutoModelForVision2Seq
    elif model_type == "internvl":
        if InternVLForConditionalGeneration is None:
            raise RuntimeError("Current transformers version does not provide InternVLForConditionalGeneration.")
        model_cls = InternVLForConditionalGeneration
    else:
        if AutoModelForVision2Seq is None:
            raise RuntimeError(
                f"Unsupported VL model_type={model_type!r} and AutoModelForVision2Seq is unavailable."
            )
        model_cls = AutoModelForVision2Seq

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if gpu_max_memory_gib is not None:
        load_kwargs["max_memory"] = {0: f"{gpu_max_memory_gib}GiB", "cpu": f"{cpu_max_memory_gib}GiB"}
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder
            load_kwargs["offload_state_dict"] = True
    if config_override is not None:
        load_kwargs["config"] = config_override

    model = model_cls.from_pretrained(model_path, **load_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return model, processor, model_type


def run_inference(
    samples: List[Dict],
    model_path: str,
    adapter_path: str,
    image_root: str,
    max_new_tokens: int,
    temperature: float,
    strict_output: bool,
    retry_on_invalid: int,
    retry_max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    constrained_decoding: bool,
    gpu_max_memory_gib: Optional[int] = None,
    cpu_max_memory_gib: int = 200,
    offload_folder: Optional[str] = None,
) -> List[str]:
    print("[INFO] Loading stage2 model...")
    model, processor, model_type = load_vl_model_and_processor(
        model_path=model_path,
        adapter_path=adapter_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_max_memory_gib=gpu_max_memory_gib,
        cpu_max_memory_gib=cpu_max_memory_gib,
        offload_folder=offload_folder,
    )
    print(f"[INFO] Loaded VL model_type={model_type or 'unknown'}")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None

    # Get a stable target device for input tensors.
    first_param = next(model.parameters())
    target_device = first_param.device

    outputs = []
    retried = 0
    repaired = 0
    unresolved_invalid = 0
    # temperature <= 0 means greedy decoding; sampling params must be disabled.
    if temperature is None or temperature <= 0:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": repetition_penalty,
        }
    else:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": repetition_penalty,
        }
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    start_token_ids: Set[int] = set()
    banned_token_ids: Set[int] = set()
    bad_words_ids: List[List[int]] = []
    if constrained_decoding and tokenizer is not None:
        start_token_ids = _build_json_start_token_ids(tokenizer)
        bad_words_ids = _build_bad_words_ids(tokenizer)
        for seq in bad_words_ids:
            for tid in seq:
                banned_token_ids.add(int(tid))
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids
        print(
            "[INFO] Constrained decoding enabled: "
            f"start_token_candidates={len(start_token_ids)}, bad_word_sequences={len(bad_words_ids)}"
        )

    for i, s in enumerate(samples):
        image_path = resolve_image_path(image_root, s["image_rel"])
        image = Image.open(image_path).convert("RGB")

        final_text = ""
        retry_reason = None
        made_retry = False
        valid_after_retry = False

        for attempt in range(max(0, retry_on_invalid) + 1):
            messages = build_qwen_messages(
                s["system"],
                s["user"],
                image_path,
                strict_output=strict_output,
                retry_reason=retry_reason if attempt > 0 else None,
            )

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            call_gen_kwargs = dict(gen_kwargs)
            # Optional shorter decoding budget on retry.
            if attempt > 0 and retry_max_new_tokens > 0:
                call_gen_kwargs["max_new_tokens"] = min(max_new_tokens, retry_max_new_tokens)

            if constrained_decoding and tokenizer is not None:
                prompt_len = int(inputs["input_ids"].shape[1])
                call_gen_kwargs["logits_processor"] = LogitsProcessorList(
                    [
                        JsonConstrainedLogitsProcessor(
                            tokenizer=tokenizer,
                            prompt_len=prompt_len,
                            allowed_start_ids=start_token_ids,
                            banned_ids=banned_token_ids,
                        )
                    ]
                )
                call_gen_kwargs["renormalize_logits"] = True

            with torch.no_grad():
                generated = model.generate(**inputs, **call_gen_kwargs)

            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
            pred_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            final_text = pred_text

            if not strict_output:
                break

            _, retry_reason = parse_and_validate_prediction(pred_text, strict_output=True)
            if retry_reason is None:
                if made_retry:
                    valid_after_retry = True
                break

            if attempt < max(0, retry_on_invalid):
                made_retry = True
                continue
            unresolved_invalid += 1

        if made_retry:
            retried += 1
        if valid_after_retry:
            repaired += 1

        outputs.append(final_text)

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"[INFO] Inference progress: {i + 1}/{len(samples)}")

    if strict_output and retry_on_invalid > 0:
        print(
            "[INFO] Strict retry stats: "
            f"retried_samples={retried}, repaired_after_retry={repaired}, still_invalid={unresolved_invalid}"
        )

    return outputs


def save_predictions(path: str, samples: List[Dict], pred_texts: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for s, p in zip(samples, pred_texts):
        rows.append(
            {
                "index": s["index"],
                "image_rel": s["image_rel"],
                "prediction": p,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_predictions(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    pred_map = {}
    for r in rows:
        idx = int(r["index"])
        pred_map[idx] = r.get("prediction", "")
    return pred_map


def evaluate(samples: List[Dict], pred_texts: List[str], tau: float, sim_model: str, strict_output: bool) -> Dict:
    sim = SemanticSimilarity(sim_model)

    entity_metric = MetricCounter()
    relation_metric = MetricCounter()

    pred_graphs: List[Optional[Dict]] = []
    per_sample = []

    for s, pred_text in zip(samples, pred_texts):
        pred_graph, invalid_reason = parse_and_validate_prediction(pred_text, strict_output=strict_output)
        pred_graphs.append(pred_graph)

        gold_graph = s["gold"]

        if pred_graph is None:
            entity_metric.fn += len(gold_graph["entities"])
            relation_metric.fn += len(gold_graph["relations"])
            per_sample.append(
                {
                    "index": s["index"],
                    "json_valid": False,
                    "invalid_reason": invalid_reason,
                }
            )
            continue

        e_tp = entity_soft_match(pred_graph["entities"], gold_graph["entities"], sim, tau, strict_label=True)
        entity_metric.tp += e_tp
        entity_metric.fp += max(0, len(pred_graph["entities"]) - e_tp)
        entity_metric.fn += max(0, len(gold_graph["entities"]) - e_tp)

        r_tp = relation_soft_match(pred_graph, gold_graph, sim, tau, strict_type=True)
        relation_metric.tp += r_tp
        relation_metric.fp += max(0, len(pred_graph["relations"]) - r_tp)
        relation_metric.fn += max(0, len(gold_graph["relations"]) - r_tp)

        per_sample.append(
            {
                "index": s["index"],
                "json_valid": True,
                "pred_entity_count": len(pred_graph["entities"]),
                "gold_entity_count": len(gold_graph["entities"]),
                "pred_relation_count": len(pred_graph["relations"]),
                "gold_relation_count": len(gold_graph["relations"]),
                "entity_tp": e_tp,
                "relation_tp": r_tp,
            }
        )

    structure = compute_structure_metrics(pred_graphs)

    return {
        "tau": tau,
        "entity_soft": {
            "precision": entity_metric.precision(),
            "recall": entity_metric.recall(),
            "f1": entity_metric.f1(),
            "tp": entity_metric.tp,
            "fp": entity_metric.fp,
            "fn": entity_metric.fn,
        },
        "relation_soft": {
            "precision": relation_metric.precision(),
            "recall": relation_metric.recall(),
            "f1": relation_metric.f1(),
            "tp": relation_metric.tp,
            "fp": relation_metric.fp,
            "fn": relation_metric.fn,
        },
        "structure": structure,
        "num_samples": len(samples),
        "per_sample": per_sample,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stage2 model with soft F1 + structure metrics")
    parser.add_argument(
        "--gold-path",
        type=str,
        default="data/runtime/stage2_gold.json",
        help="Path to stage2_fold0_test json",
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        default="results/inference_test24/stage2_predictions.json",
        help="Where to load/save predictions",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default="results/inference_test24/stage2_metrics.json",
        help="Where to write final metrics json",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run model inference before evaluation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="models/stage2",
        help="Stage2 adapter/checkpoint path",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/test_images",
        help="Root folder to resolve relative image paths",
    )
    parser.add_argument("--tau", type=float, default=0.8, help="Soft match threshold")
    parser.add_argument(
        "--sim-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for semantic similarity",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty >1 discourages repetitive loops in generation",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=6,
        help="Prevent repeating n-grams of this size (0 to disable)",
    )
    parser.add_argument(
        "--strict-output",
        action="store_true",
        help="Enable strict JSON/schema validation (keys, type set, and ID references)",
    )
    parser.add_argument(
        "--retry-on-invalid",
        type=int,
        default=0,
        help="When strict output is enabled, retry generation this many times if output is invalid",
    )
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=0,
        help="Optional max_new_tokens for retry attempts only (0 means no extra cap)",
    )
    parser.add_argument(
        "--constrained-decoding",
        action="store_true",
        help="Enable hard decoding constraints (JSON start token + bad token bans)",
    )
    parser.add_argument("--gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=200)
    parser.add_argument("--offload-folder", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_stage2_gold(args.gold_path)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if args.run_inference:
        pred_texts = run_inference(
            samples=samples,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            image_root=args.image_root,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            strict_output=args.strict_output,
            retry_on_invalid=args.retry_on_invalid,
            retry_max_new_tokens=args.retry_max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            constrained_decoding=args.constrained_decoding,
            gpu_max_memory_gib=args.gpu_max_memory_gib,
            cpu_max_memory_gib=args.cpu_max_memory_gib,
            offload_folder=args.offload_folder,
        )
        save_predictions(args.pred_path, samples, pred_texts)
        print(f"[INFO] Predictions saved to: {args.pred_path}")
    else:
        pred_map = load_predictions(args.pred_path)
        pred_texts = [pred_map.get(s["index"], "") for s in samples]

    result = evaluate(samples, pred_texts, tau=args.tau, sim_model=args.sim_model, strict_output=args.strict_output)

    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== Evaluation Summary =====")
    print(f"Samples: {result['num_samples']}")
    print(f"Entity Soft F1: {result['entity_soft']['f1']:.4f}")
    print(f"Relation Soft F1: {result['relation_soft']['f1']:.4f}")
    print(f"JSON valid rate: {result['structure']['json_valid_rate']:.4f}")
    print(f"ID valid rate: {result['structure']['id_valid_rate']:.4f}")
    print(f"Dangling edge rate: {result['structure']['dangling_edge_rate']:.4f}")
    print(f"Duplicate edge rate: {result['structure']['duplicate_edge_rate']:.4f}")
    print(f"Results saved to: {args.result_path}")


if __name__ == "__main__":
    main()
