import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


CODE_DIR = Path(__file__).resolve().parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: List[str]) -> None:
    print("[INFO] Running:", " ".join(args))
    env = dict(os.environ)
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    subprocess.run(args, check=True, env=env)


def metric_triplet(obj: Dict) -> Dict:
    evo = obj.get("evolution_soft", {})
    return {
        "precision": evo.get("precision", 0.0),
        "recall": evo.get("recall", 0.0),
        "f1": evo.get("f1", 0.0),
        "tp": evo.get("tp", 0),
        "fp": evo.get("fp", 0),
        "fn": evo.get("fn", 0),
    }


def build_markdown(summary: Dict) -> str:
    lines = []
    lines.append("# Stage2B1 Exact-Mention Pipeline Summary")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- prefix: `{summary['prefix']}`")
    lines.append(f"- input_path: `{summary['input_path']}`")
    lines.append(f"- gold_path: `{summary['gold_path']}`")
    lines.append(f"- pred_path: `{summary['files']['pred_path']}`")
    lines.append(f"- report_path: `{summary['files']['report_path']}`")
    lines.append(f"- eval_path: `{summary['files']['eval_path']}`")
    lines.append("")

    generation = summary.get("generation_report", {})
    gen_totals = generation.get("totals", {})
    lines.append("## Generation")
    lines.append("")
    lines.append(f"- samples: `{gen_totals.get('samples', 0)}`")
    lines.append(f"- parsed_ok: `{gen_totals.get('parsed_ok', 0)}`")
    lines.append(f"- invalid: `{gen_totals.get('invalid', 0)}`")
    lines.append(f"- event_count: `{gen_totals.get('event_count', 0)}`")
    lines.append(f"- mention_total: `{gen_totals.get('mention_total', 0)}`")
    lines.append(f"- mention_evidence_exact: `{gen_totals.get('mention_evidence_exact', 0)}`")
    lines.append(f"- mention_fulltext_only: `{gen_totals.get('mention_fulltext_only', 0)}`")
    lines.append(f"- mention_missing: `{gen_totals.get('mention_missing', 0)}`")
    lines.append("")

    eval_metrics = metric_triplet(summary.get("eval_result", {}))
    lines.append("## Evaluation")
    lines.append("")
    lines.append("| Precision | Recall | F1 | TP | FP | FN |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| {eval_metrics['precision']:.4f} | {eval_metrics['recall']:.4f} | {eval_metrics['f1']:.4f} | "
        f"{eval_metrics['tp']} | {eval_metrics['fp']} | {eval_metrics['fn']} |"
    )
    lines.append("")

    comparisons = summary.get("comparisons", [])
    if comparisons:
        lines.append("## Comparisons")
        lines.append("")
        lines.append("| Name | Precision | Recall | F1 | TP | FP | FN | Delta F1 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in comparisons:
            m = row["metrics"]
            lines.append(
                f"| {row['name']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['tp']} | {m['fp']} | {m['fn']} | {row['delta_f1_vs_exact']:+.4f} |"
            )
        lines.append("")

    invalid_rows = [
        row["index"]
        for row in summary.get("eval_result", {}).get("per_sample", [])
        if row.get("format_valid") is False
    ]
    if invalid_rows:
        lines.append("## Remaining Invalid Samples")
        lines.append("")
        lines.append("- indices: `" + ", ".join(str(i) for i in invalid_rows) + "`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage2B1 exact-mention generation + evaluation pipeline.")
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--gold-path", required=True, type=str)
    parser.add_argument("--adapter-path", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--prefix", type=str, default="exact_mention")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--prompt-variant", type=str, default="default", choices=["default", "retry_strict", "precision_guard"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--sim-model", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--gpu-max-memory-gib", type=int, default=None)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=200)
    parser.add_argument("--offload-folder", type=str, default=None)
    parser.add_argument("--merge-object-mechanism", action="store_true")
    parser.add_argument("--allow-partial-core-events", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--pred-path", type=str, default="")
    parser.add_argument("--report-path", type=str, default="")
    parser.add_argument("--eval-path", type=str, default="")
    parser.add_argument("--compare-eval-path", action="append", default=[])
    parser.add_argument("--compare-name", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.pred_path) if args.pred_path else out_dir / f"{args.prefix}_predictions.json"
    report_path = Path(args.report_path) if args.report_path else out_dir / f"{args.prefix}_report.json"
    eval_path = Path(args.eval_path) if args.eval_path else out_dir / f"{args.prefix}_eval.json"
    summary_json_path = out_dir / f"{args.prefix}_summary.json"
    summary_md_path = out_dir / f"{args.prefix}_summary.md"

    if not args.skip_generate:
        gen_cmd = [
            sys.executable,
            str(CODE_DIR / "run_stage2b1_exact_mention_prototype.py"),
            "--input-path",
            args.input_path,
            "--pred-path",
            str(pred_path),
            "--report-path",
            str(report_path),
            "--adapter-path",
            args.adapter_path,
            "--model-path",
            args.model_path,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--max-samples",
            str(args.max_samples),
            "--prompt-variant",
            args.prompt_variant,
        ]
        if args.gpu_max_memory_gib is not None:
            gen_cmd.extend(
                [
                    "--gpu-max-memory-gib",
                    str(args.gpu_max_memory_gib),
                    "--cpu-max-memory-gib",
                    str(args.cpu_max_memory_gib),
                ]
            )
        if args.offload_folder:
            gen_cmd.extend(["--offload-folder", args.offload_folder])
        run_cmd(gen_cmd)
    elif not pred_path.exists():
        raise FileNotFoundError(f"--skip-generate was set but pred_path does not exist: {pred_path}")

    if not report_path.exists():
        raise FileNotFoundError(f"Missing generation report: {report_path}")

    eval_cmd = [
        sys.executable,
        str(CODE_DIR / "eval_stage2b1_exact_mention_prototype.py"),
        "--gold-path",
        args.gold_path,
        "--pred-path",
        str(pred_path),
        "--result-path",
        str(eval_path),
        "--tau",
        str(args.tau),
        "--sim-model",
        args.sim_model,
    ]
    if args.merge_object_mechanism:
        eval_cmd.append("--merge-object-mechanism")
    if args.allow_partial_core_events:
        eval_cmd.append("--allow-partial-core-events")
    run_cmd(eval_cmd)

    generation_report = load_json(report_path)
    eval_result = load_json(eval_path)

    comparisons = []
    compare_names = list(args.compare_name)
    if compare_names and len(compare_names) != len(args.compare_eval_path):
        raise ValueError("--compare-name count must match --compare-eval-path count")

    exact_f1 = metric_triplet(eval_result)["f1"]
    for idx, path_str in enumerate(args.compare_eval_path):
        path = Path(path_str)
        obj = load_json(path)
        name = compare_names[idx] if idx < len(compare_names) else path.stem
        metrics = metric_triplet(obj)
        comparisons.append(
            {
                "name": name,
                "path": str(path),
                "metrics": metrics,
                "delta_f1_vs_exact": exact_f1 - metrics["f1"],
            }
        )

    summary = {
        "prefix": args.prefix,
        "input_path": args.input_path,
        "gold_path": args.gold_path,
        "adapter_path": args.adapter_path,
        "generation_config": {
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "max_samples": args.max_samples,
        },
        "eval_config": {
            "tau": args.tau,
            "sim_model": args.sim_model,
            "merge_object_mechanism": bool(args.merge_object_mechanism),
            "allow_partial_core_events": bool(args.allow_partial_core_events),
        },
        "files": {
            "pred_path": str(pred_path),
            "report_path": str(report_path),
            "eval_path": str(eval_path),
        },
        "generation_report": generation_report,
        "eval_result": eval_result,
        "comparisons": comparisons,
    }
    save_json(summary_json_path, summary)
    save_text(summary_md_path, build_markdown(summary))

    print(f"[INFO] Summary JSON saved to: {summary_json_path}")
    print(f"[INFO] Summary Markdown saved to: {summary_md_path}")


if __name__ == "__main__":
    main()
