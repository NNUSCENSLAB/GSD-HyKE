#!/usr/bin/env python3
"""Recompute ROUGE-L, SciBERT BERTScore, and CLIPScore."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers import AutoTokenizer


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
CLIPSCORE_WEIGHT = 2.5


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(str(text or "").lower())


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    values = [0] * (len(right) + 1)
    for left_value in left:
        previous = 0
        for index, right_value in enumerate(right, start=1):
            current = values[index]
            values[index] = previous + 1 if left_value == right_value else max(values[index], values[index - 1])
            previous = current
    return values[-1]


def rouge_l(prediction: str, reference: str) -> tuple[float, float, float]:
    predicted, gold = tokens(prediction), tokens(reference)
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    common = lcs_length(predicted, gold)
    precision, recall = common / len(predicted), common / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_text(predictions: list[str], references: list[str], model: str, batch_size: int) -> dict:
    from bert_score import score as bert_score

    rouge = [rouge_l(prediction, reference) for prediction, reference in zip(predictions, references)]
    tokenizer = AutoTokenizer.from_pretrained(model)
    max_length = min(getattr(tokenizer, "model_max_length", 512), 512)

    def truncate(text: str) -> str:
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_attention_mask=False, return_token_type_ids=False)
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    precision, recall, f1 = bert_score(
        [truncate(text) for text in predictions],
        [truncate(text) for text in references],
        model_type=model,
        num_layers=12,
        lang="en",
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "samples": len(predictions),
        "rouge_l": {
            "precision": sum(value[0] for value in rouge) / len(rouge),
            "recall": sum(value[1] for value in rouge) / len(rouge),
            "f1": sum(value[2] for value in rouge) / len(rouge),
        },
        "bertscore": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
        },
    }


def clipscore_from_cosine(cosine: torch.Tensor) -> torch.Tensor:
    """Convert CLIP cosine similarities to the manuscript CLIPScore scale."""
    return CLIPSCORE_WEIGHT * cosine.clamp(min=0)


class ClipScorer:
    """Image-text CLIPScore using the released Table 3 convention."""

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score(self, texts: list[str], image_paths: list[Path], batch_size: int) -> dict:
        """Return the mean and per-sample CLIPScore for aligned text-image pairs."""
        values: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                inputs = self.processor(
                    text=batch_texts,
                    images=images,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                    text_features = self.model.get_text_features(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                    )
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                    scores = clipscore_from_cosine((image_features * text_features).sum(dim=-1))
                values.extend(float(value) for value in scores.cpu())
            finally:
                for image in images:
                    image.close()
        return {
            "mean": sum(values) / len(values) if values else 0.0,
            "per_sample": values,
        }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=root / "data/test_gold.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions/description_predictions.jsonl")
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=4)
    parser.add_argument("--skip-clipscore", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "results/recomputed_description_metrics.json")
    args = parser.parse_args()

    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    gold = {row["sample_id"]: row["description_gold"] for row in gold_rows}
    image_paths = {row["sample_id"]: root / "data" / row["image"] for row in gold_rows}
    grouped = defaultdict(dict)
    for row in read_jsonl(args.predictions):
        grouped[row["setting_id"]][row["sample_id"]] = row["prediction"]
    selected = args.setting or sorted(grouped)
    clip_scorer = None if args.skip_clipscore else ClipScorer(args.clip_model, args.clip_device)
    results = {
        "config": {
            "bertscore_model": args.model,
            "clip_model": None if args.skip_clipscore else args.clip_model,
            "clipscore_formula": None if args.skip_clipscore else "2.5 * max(cosine(image_embedding, text_embedding), 0)",
        },
        "settings": {},
    }
    for setting in selected:
        if set(grouped[setting]) != set(gold):
            raise ValueError(f"Gold and prediction sample_id sets differ for {setting}")
        ids = sorted(gold)
        predictions = [grouped[setting][sample_id] for sample_id in ids]
        results["settings"][setting] = evaluate_text(
            predictions,
            [gold[sample_id] for sample_id in ids],
            args.model,
            args.batch_size,
        )
        if clip_scorer is not None:
            results["settings"][setting]["clipscore"] = clip_scorer.score(
                predictions,
                [image_paths[sample_id] for sample_id in ids],
                args.clip_batch_size,
            )
        result = results["settings"][setting]
        clip_text = "" if clip_scorer is None else f", CLIPScore={result['clipscore']['mean']:.4f}"
        print(
            f"{setting}: ROUGE-L={result['rouge_l']['f1']:.4f}, "
            f"BERTScore={result['bertscore']['f1']:.4f}{clip_text}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
