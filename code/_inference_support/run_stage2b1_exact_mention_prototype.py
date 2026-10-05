import argparse
import importlib.util
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_EVAL_PATH = SCRIPT_DIR / "eval_stage2_model.py"
BUILD_PATH = SCRIPT_DIR / "build_stage2b1_evolution_template_text_chunked_dataset.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_EVAL_PATH, "stage2_model_base_for_exact_mention")
builder = load_module(BUILD_PATH, "stage2b1_chunk_builder_for_exact_mention")


def extract_input_text(user_text: str) -> str:
    marker = "Input passage:\n"
    pos = user_text.find(marker)
    if pos != -1:
        tagged = user_text[pos + len(marker):].strip()
        # New exact-mention datasets already carry sentence-tagged passages.
        # Strip tags and recover the raw passage text for re-segmentation.
        return re.sub(r"\[/?S\d+\]", "", tagged)
    _, input_text = builder.extract_entities_and_text(user_text)
    return input_text


def normalize_input_text(input_text: str) -> str:
    text = unicodedata.normalize("NFKC", input_text or "")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_sentence_only_tagged_text(input_text: str) -> Tuple[str, List[str]]:
    sentence_spans = builder.build_sentence_spans(input_text)
    tagged_sentences = []
    for idx, (s0, s1) in enumerate(sentence_spans, start=1):
        tagged_sentences.append(f"[S{idx}]{input_text[s0:s1]}[/S{idx}]")
    return "".join(tagged_sentences), tagged_sentences


def build_messages(tagged_text: str, prompt_variant: str = "default") -> List[Dict]:
    system = "You are a careful geomorphology event extractor."
    rules = [
        "1) Each mention must be copied exactly from the evidence sentence text. Do not normalize, paraphrase, shorten, or expand mentions.",
        "2) evidence_sent_ids must contain only sentence ids that appear in the input, such as 1, 2, 3.",
        "3) Use one sentence if one sentence is enough. Use two consecutive sentences only when they clearly describe the same single evolution event.",
        "4) Do not combine unrelated information across distant sentences.",
        "5) Extract an event only when the sentence explicitly states that some input entity or entities generate, form, produce, transform into, or change into some output entity or entities.",
        "6) Do not output taxonomy, classification, composition, framework summary, comparison, co-adjustment, process listing, or background statements as evolution events.",
        "7) If one event has multiple outputs, keep them in one event.",
        "8) Do not create an event if output_mentions would be empty.",
        '9) Use location_mentions only for the core location noun phrase, not surrounding prepositions or function words. For example, use "valley floor", not "at the valley floor".',
        "10) Use time_mentions only for the core time phrase, not surrounding prepositions.",
        "11) Do not use clause headers, scale names, section labels, or framework names as input_mentions or output_mentions unless they themselves are explicitly transformed.",
        "12) Prefer geomorphic objects, materials, landforms, or process participants for input_mentions and output_mentions.",
        "13) Extract at most one main evolution event from one sentence unless that same sentence explicitly states two separate full transformations.",
        '14) If no explicit evolution event exists, output exactly {"evolutions": []}.',
        "15) Do not output explanations, markdown, or any text outside the JSON object.",
    ]
    if prompt_variant == "retry_strict":
        rules.extend(
            [
                "16) Return compact valid JSON only. Stop immediately after the closing bracket and brace.",
                "17) Do not produce long enumerations. If one candidate event would need more than 2 evidence sentences, discard it.",
                "18) If a field would contain many repeated or taxonomy-like items, discard that event instead of listing them.",
                "19) Keep each field concise: usually <= 4 mentions, and never repeat the same mention within a field.",
                "20) If unsure whether a passage is a real transformation event, prefer omitting it.",
            ]
        )
    elif prompt_variant == "precision_guard":
        rules.extend(
            [
                "16) Extract only relations explicitly stated in one complete input sentence; ignore incomplete trailing sentences.",
                "17) Do not turn panel labels, legends, categories, leader lines, spatial adjacency, or visual similarity into events.",
                "18) If an input, output, or process would need to be inferred, omit the event.",
                "19) Do not output duplicate or semantically equivalent events.",
                "20) Return compact valid JSON only, escape JSON special characters correctly, and stop immediately after the closing brace.",
            ]
        )
    user = (
        "You are given a geomorphology passage annotated only with sentence tags.\n\n"
        "Task: extract geomorphic evolution events directly from the passage without any provided entity list.\n\n"
        "Output ONLY one JSON object with key:\n"
        "- evolutions\n\n"
        "Each evolution object may contain:\n"
        '- evidence_sent_ids: [int]\n'
        '- input_mentions: [str]\n'
        '- output_mentions: [str]\n'
        '- mechanism_mentions: [str]\n'
        '- location_mentions: [str]\n'
        '- time_mentions: [str]\n\n'
        "Rules:\n"
        + "\n".join(rules)
        + "\n\nInput passage:\n"
        + tagged_text
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


def parse_prediction(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    obj = base.extract_json_object(text)
    if not isinstance(obj, dict):
        return None, "invalid_json"
    evolutions = obj.get("evolutions")
    if not isinstance(evolutions, list):
        return None, "missing_evolutions"
    cleaned = []
    for evo in evolutions:
        if not isinstance(evo, dict):
            continue
        norm = {}
        sent_ids = evo.get("evidence_sent_ids", [])
        if isinstance(sent_ids, list):
            keep_ids = []
            for val in sent_ids:
                try:
                    keep_ids.append(int(val))
                except Exception:
                    continue
            if keep_ids:
                norm["evidence_sent_ids"] = sorted(set(keep_ids))
        for field in (
            "input_mentions",
            "output_mentions",
            "mechanism_mentions",
            "location_mentions",
            "time_mentions",
        ):
            vals = evo.get(field, [])
            if not isinstance(vals, list):
                continue
            keep_vals = []
            seen = set()
            for val in vals:
                text_val = str(val).strip()
                if not text_val or text_val in seen:
                    continue
                seen.add(text_val)
                keep_vals.append(text_val)
            if keep_vals:
                norm[field] = keep_vals
        if norm.get("output_mentions"):
            cleaned.append(norm)
    return {"evolutions": cleaned}, None


def gather_evidence_text(tagged_sentences: Sequence[str], sent_ids: Sequence[int]) -> str:
    parts = []
    for sid in sent_ids:
        idx = int(sid) - 1
        if 0 <= idx < len(tagged_sentences):
            sent = tagged_sentences[idx]
            sent = re.sub(r"\[/?S\d+\]", "", sent)
            parts.append(sent)
    return "".join(parts)


def mention_grounded(mention: str, evidence_text: str, full_text: str) -> str:
    if mention in evidence_text:
        return "evidence_exact"
    if mention in full_text:
        return "fulltext_only"
    return "missing"


def analyze_prediction(pred_obj: Optional[Dict], tagged_sentences: Sequence[str], input_text: str) -> Dict:
    if pred_obj is None:
        return {
            "event_count": 0,
            "mention_total": 0,
            "mention_evidence_exact": 0,
            "mention_fulltext_only": 0,
            "mention_missing": 0,
            "per_event": [],
        }
    per_event = []
    total = 0
    exact = 0
    full_only = 0
    missing = 0
    for evo in pred_obj.get("evolutions", []):
        evidence_text = gather_evidence_text(tagged_sentences, evo.get("evidence_sent_ids", []))
        event_rows = []
        for field in (
            "input_mentions",
            "output_mentions",
            "mechanism_mentions",
            "location_mentions",
            "time_mentions",
        ):
            for mention in evo.get(field, []):
                status = mention_grounded(mention, evidence_text, input_text)
                total += 1
                if status == "evidence_exact":
                    exact += 1
                elif status == "fulltext_only":
                    full_only += 1
                else:
                    missing += 1
                event_rows.append({"field": field, "mention": mention, "grounding": status})
        per_event.append(
            {
                "evidence_sent_ids": evo.get("evidence_sent_ids", []),
                "mentions": event_rows,
            }
        )
    return {
        "event_count": len(pred_obj.get("evolutions", [])),
        "mention_total": total,
        "mention_evidence_exact": exact,
        "mention_fulltext_only": full_only,
        "mention_missing": missing,
        "per_event": per_event,
    }


def generate_predictions(
    samples: Sequence[Dict],
    model_path: str,
    adapter_path: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    prompt_variant: str,
    gpu_max_memory_gib: Optional[int] = None,
    cpu_max_memory_gib: int = 200,
    offload_folder: Optional[str] = None,
) -> List[str]:
    print("[INFO] Loading stage2 model...")
    model, processor, model_type = base.load_vl_model_and_processor(
        model_path=model_path,
        adapter_path=adapter_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_max_memory_gib=gpu_max_memory_gib,
        cpu_max_memory_gib=cpu_max_memory_gib,
        offload_folder=offload_folder,
    )
    print(f"[INFO] Loaded VL model_type={model_type or 'unknown'}")

    first_param = next(model.parameters())
    target_device = first_param.device

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

    outputs = []
    for i, sample in enumerate(samples, start=1):
        messages = build_messages(sample["tagged_text"], prompt_variant=prompt_variant)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        pred_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        outputs.append(pred_text)
        print(f"[INFO] Inference progress: {i}/{len(samples)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def parse_indices(text: str) -> Optional[set]:
    if not text:
        return None
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def load_samples(path: str, max_samples: int, indices: Optional[set], normalize_input: bool) -> List[Dict]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    out = []
    for idx, row in enumerate(rows):
        row_index = row.get("index", idx)
        if indices is not None and int(row_index) not in indices:
            continue
        user_text = row["messages"][1]["content"]
        input_text = extract_input_text(user_text)
        if normalize_input:
            input_text = normalize_input_text(input_text)
        tagged_text, tagged_sentences = build_sentence_only_tagged_text(input_text)
        out.append(
            {
                "index": row_index,
                "input_text": input_text,
                "tagged_text": tagged_text,
                "tagged_sentences": tagged_sentences,
            }
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage2B1 exact-mention prototype without external entity list.")
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--pred-path", required=True, type=str)
    parser.add_argument("--report-path", required=True, type=str)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", required=True, type=str)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="default",
        choices=["default", "retry_strict", "precision_guard"],
    )
    parser.add_argument("--gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=200)
    parser.add_argument("--offload-folder", type=str, default=None)
    args = parser.parse_args()

    samples = load_samples(
        args.input_path,
        args.max_samples,
        indices=parse_indices(args.indices),
        normalize_input=bool(args.normalize_input),
    )
    pred_texts = generate_predictions(
        samples=samples,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        prompt_variant=args.prompt_variant,
        gpu_max_memory_gib=args.gpu_max_memory_gib,
        cpu_max_memory_gib=args.cpu_max_memory_gib,
        offload_folder=args.offload_folder,
    )

    pred_rows = []
    report_rows = []
    totals = {
        "samples": len(samples),
        "parsed_ok": 0,
        "invalid": 0,
        "event_count": 0,
        "mention_total": 0,
        "mention_evidence_exact": 0,
        "mention_fulltext_only": 0,
        "mention_missing": 0,
    }

    for sample, pred_text in zip(samples, pred_texts):
        pred_obj, invalid_reason = parse_prediction(pred_text)
        analysis = analyze_prediction(pred_obj, sample["tagged_sentences"], sample["input_text"])
        if pred_obj is None:
            totals["invalid"] += 1
        else:
            totals["parsed_ok"] += 1
        totals["event_count"] += analysis["event_count"]
        totals["mention_total"] += analysis["mention_total"]
        totals["mention_evidence_exact"] += analysis["mention_evidence_exact"]
        totals["mention_fulltext_only"] += analysis["mention_fulltext_only"]
        totals["mention_missing"] += analysis["mention_missing"]
        pred_rows.append({"index": sample["index"], "prediction": pred_text})
        report_rows.append(
            {
                "index": sample["index"],
                "invalid_reason": invalid_reason,
                "parsed_prediction": pred_obj,
                "analysis": analysis,
            }
        )

    Path(args.pred_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, ensure_ascii=False, indent=2)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump({"totals": totals, "per_sample": report_rows}, f, ensure_ascii=False, indent=2)

    print("\n===== Prototype Summary =====")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"Predictions saved to: {args.pred_path}")
    print(f"Report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
