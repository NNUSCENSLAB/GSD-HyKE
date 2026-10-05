import argparse
import json
import re
from pathlib import Path


MARKER = "Input passage:\n"
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dedupe_repeated_sentences(text: str):
    """Keep the first occurrence of each complete sentence, ignoring whitespace variation."""
    original = str(text or "")
    sentences = SENTENCE_BOUNDARY_RE.split(original.strip()) if original.strip() else []
    seen = set()
    kept = []
    removed = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        key = " ".join(sentence.split())
        if key in seen:
            removed.append(sentence)
            continue
        seen.add(key)
        kept.append(sentence)
    # A no-op dedupe must be byte-preserving.  Otherwise the comparison would
    # conflate exact-repeat removal with a prompt-formatting change.
    if not removed:
        return original, removed
    return " ".join(kept), removed


def build_rows(
    stage1_rows,
    template_rows,
    dedupe_sentences=False,
    fallback_to_evaluation_text=False,
    match_by_position=False,
):
    if match_by_position and len(stage1_rows) != len(template_rows):
        raise ValueError(
            "Position matching requires equal row counts: "
            f"stage1={len(stage1_rows)}, template={len(template_rows)}"
        )
    template_by_index = {int(row.get("index", i)): row for i, row in enumerate(template_rows)}
    out_rows = []
    diagnostics = []
    for i, desc_row in enumerate(stage1_rows):
        index = int(desc_row.get("index", i))
        template = template_rows[i] if match_by_position else template_by_index.get(index)
        if template is None:
            raise KeyError(f"Missing template row for index={index}")

        user_text = str(template["messages"][1]["content"])
        marker_pos = user_text.find(MARKER)
        if marker_pos < 0:
            raise ValueError(f"Template row index={index} does not contain '{MARKER.strip()}' marker.")

        if "stage2_input_text" in desc_row:
            passage = desc_row.get("stage2_input_text", "")
            if not str(passage).strip() and fallback_to_evaluation_text:
                passage = desc_row.get("evaluation_text") or desc_row.get("prediction") or ""
        else:
            passage = desc_row.get("evaluation_text") or desc_row.get("prediction") or ""
        passage = str(passage).strip()
        original_passage = passage
        removed = []
        if dedupe_sentences:
            passage, removed = dedupe_repeated_sentences(passage)

        new_user = user_text[: marker_pos + len(MARKER)] + passage
        out_rows.append(
            {
                "index": index,
                "messages": [
                    {"role": template["messages"][0]["role"], "content": template["messages"][0]["content"]},
                    {"role": template["messages"][1]["role"], "content": new_user},
                ],
            }
        )
        diagnostics.append(
            {
                "index": index,
                "original_char_count": len(original_passage),
                "filtered_char_count": len(passage),
                "removed_sentence_count": len(removed),
                "removed_sentences": removed,
            }
        )
    return out_rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Build Stage2B1 exact-mention input from Stage1 description predictions.")
    parser.add_argument("--stage1-pred-path", required=True, type=str)
    parser.add_argument("--template-path", required=True, type=str)
    parser.add_argument("--out-path", required=True, type=str)
    parser.add_argument(
        "--dedupe-repeated-sentences",
        action="store_true",
        help="Keep only the first exact sentence occurrence after whitespace normalization.",
    )
    parser.add_argument(
        "--dedupe-report-path",
        type=str,
        help="Optional JSON path for per-record sentence-removal diagnostics.",
    )
    parser.add_argument(
        "--fallback-to-evaluation-text",
        action="store_true",
        help="Use the generated description when a two-section Stage 1 output lacks Event Sentences.",
    )
    parser.add_argument(
        "--match-by-position",
        action="store_true",
        help="Match Stage 1 and template rows by list position when legacy templates have no indices.",
    )
    args = parser.parse_args()

    stage1_rows = load_json(args.stage1_pred_path)
    template_rows = load_json(args.template_path)
    out_rows, diagnostics = build_rows(
        stage1_rows,
        template_rows,
        dedupe_sentences=args.dedupe_repeated_sentences,
        fallback_to_evaluation_text=args.fallback_to_evaluation_text,
        match_by_position=args.match_by_position,
    )
    dump_json(args.out_path, out_rows)
    if args.dedupe_report_path:
        dump_json(args.dedupe_report_path, diagnostics)
    print(f"[INFO] Saved {len(out_rows)} rows to: {args.out_path}")
    if args.dedupe_repeated_sentences:
        affected = sum(row["removed_sentence_count"] > 0 for row in diagnostics)
        removed = sum(row["removed_sentence_count"] for row in diagnostics)
        print(f"[INFO] Sentence dedupe affected {affected} rows and removed {removed} repetitions.")


if __name__ == "__main__":
    main()
