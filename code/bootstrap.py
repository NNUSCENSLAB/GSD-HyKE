#!/usr/bin/env python3
"""Run paired, diagram-level bootstrap comparisons from released count tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PACKAGE_ROOT / "data/bootstrap_inputs"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results/bootstrap"
SETTINGS = {
    "full": "Full GSD-HyKE",
    "triplet": "Triplet-based representation",
    "without_evidence": "w/o Evidence",
    "without_description": "w/o Description generation",
}
SUMMARY_FIELDS = (
    "full_setting",
    "comparator",
    "metric",
    "observed_full",
    "observed_comparator",
    "observed_delta",
    "ci_lower",
    "ci_upper",
    "probability_delta_gt_zero",
    "bootstrap_iterations",
    "seed",
    "number_of_test_diagrams",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer a repository-relative path in persisted reports."""
    try:
        return path.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_counts(path: Path, expected_setting: str) -> dict[str, dict[str, int]]:
    """Load one count table while enforcing unique, valid sample_id keys."""
    result: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"setting", "sample_id", "index", "tp", "fp", "fn", "predicted_count", "gold_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {path}: {sorted(required-set(reader.fieldnames or []))}")
        for row in reader:
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"Empty sample_id in {path}")
            if sample_id in result:
                raise ValueError(f"Duplicate sample_id {sample_id} in {path}")
            if row["setting"] != expected_setting:
                raise ValueError(f"Unexpected setting {row['setting']!r} in {path}; expected {expected_setting!r}")
            counts = {field: int(row[field]) for field in ("tp", "fp", "fn", "predicted_count", "gold_count", "index")}
            if min(counts.values()) < 0:
                raise ValueError(f"Negative count in {path} for {sample_id}: {counts}")
            if counts["tp"] + counts["fp"] != counts["predicted_count"]:
                raise ValueError(f"TP+FP does not equal predicted_count in {path} for {sample_id}")
            if counts["tp"] + counts["fn"] != counts["gold_count"]:
                raise ValueError(f"TP+FN does not equal gold_count in {path} for {sample_id}")
            result[sample_id] = counts
    if not result:
        raise ValueError(f"No rows in {path}")
    return result


def micro_f1(tp: int | np.ndarray, fp: int | np.ndarray, fn: int | np.ndarray) -> float | np.ndarray:
    """Calculate micro F1 from summed TP, FP, and FN counts."""
    denominator = 2 * tp + fp + fn
    if isinstance(denominator, np.ndarray):
        return np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return (2 * tp / denominator) if denominator else 0.0


def validate_tables(tables: dict[str, dict[str, dict[str, int]]]) -> list[str]:
    """Require exact diagram alignment and common gold support across settings."""
    sample_ids = sorted(tables["full"])
    if len(sample_ids) != 24:
        raise ValueError(f"Expected exactly 24 diagrams, found {len(sample_ids)}")
    for setting, table in tables.items():
        if set(table) != set(sample_ids):
            raise ValueError(
                f"sample_id mismatch in {setting}: missing={sorted(set(sample_ids)-set(table))}, "
                f"extra={sorted(set(table)-set(sample_ids))}"
            )
    full_gold = {sample_id: tables["full"][sample_id]["gold_count"] for sample_id in sample_ids}
    for setting, table in tables.items():
        for sample_id in sample_ids:
            if table[sample_id]["gold_count"] != full_gold[sample_id]:
                raise ValueError(
                    f"gold_count mismatch for {sample_id}: full={full_gold[sample_id]}, "
                    f"{setting}={table[sample_id]['gold_count']}"
                )
    return sample_ids


def aggregate(table: dict[str, dict[str, int]], sample_ids: list[str]) -> tuple[int, int, int]:
    return tuple(sum(table[sample_id][key] for sample_id in sample_ids) for key in ("tp", "fp", "fn"))


def aggregate_resample(
    table: dict[str, dict[str, int]], sample_ids: list[str], draw_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum each bootstrap row, counting a diagram repeatedly when redrawn."""
    sampled_ids = np.asarray(sample_ids, dtype=object)[draw_indices]
    values = []
    for field in ("tp", "fp", "fn"):
        per_draw = np.asarray([[table[str(sample_id)][field] for sample_id in draw] for draw in sampled_ids], dtype=np.int64)
        values.append(per_draw.sum(axis=1))
    return values[0], values[1], values[2]


def build_results(
    tables: dict[str, dict[str, dict[str, int]]], sample_ids: list[str], iterations: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use shared paired draws and resample diagram counts with replacement."""
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(sample_ids), size=(iterations, len(sample_ids)))
    pairings = [("triplet", "Full GSD-HyKE vs triplet-based representation"),
                ("without_evidence", "Full GSD-HyKE vs w/o Evidence"),
                ("without_description", "Full GSD-HyKE vs w/o Description generation")]
    summaries: list[dict[str, Any]] = []
    distributions: dict[str, Any] = {
        "schema_version": "1.0",
        "metric": "Relation Relaxed F1 (micro from summed per-diagram TP/FP/FN)",
        "bootstrap_method": "paired percentile bootstrap over 24 test diagrams; sample with replacement",
        "bootstrap_iterations": iterations,
        "seed": seed,
        "number_of_test_diagrams": len(sample_ids),
        "sample_id_order": sample_ids,
        "shared_draw_indices": draw_indices.tolist(),
        "comparisons": {},
    }
    full_tp, full_fp, full_fn = aggregate(tables["full"], sample_ids)
    observed_full = float(micro_f1(full_tp, full_fp, full_fn))
    full_boot = micro_f1(*aggregate_resample(tables["full"], sample_ids, draw_indices))

    for comparator, label in pairings:
        comp_tp, comp_fp, comp_fn = aggregate(tables[comparator], sample_ids)
        observed_comp = float(micro_f1(comp_tp, comp_fp, comp_fn))
        comp_counts = aggregate_resample(tables[comparator], sample_ids, draw_indices)
        comp_boot = micro_f1(*comp_counts)
        delta = full_boot - comp_boot
        ci_lower, ci_upper = np.quantile(delta, [0.025, 0.975], method="linear")
        summary = {
            "full_setting": SETTINGS["full"],
            "comparator": SETTINGS[comparator],
            "metric": "Relation Relaxed F1",
            "observed_full": observed_full,
            "observed_comparator": observed_comp,
            "observed_delta": observed_full - observed_comp,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "probability_delta_gt_zero": float(np.mean(delta > 0)),
            "bootstrap_iterations": iterations,
            "seed": seed,
            "number_of_test_diagrams": len(sample_ids),
        }
        summaries.append(summary)
        distributions["comparisons"][comparator] = {
            "label": label,
            "full_f1": full_boot.tolist(),
            "comparator_f1": comp_boot.tolist(),
            "delta": delta.tolist(),
        }
    return summaries, distributions


def load_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing provenance file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(
    report_path: Path,
    summaries: list[dict[str, Any]],
    input_paths: dict[str, Path],
    provenance_path: Path,
    iterations: int,
    seed: int,
) -> None:
    """Write a compact report with sources, method, and reproducibility limits."""
    provenance = load_provenance(provenance_path)
    lines = [
        "# Paired bootstrap comparison report",
        "",
        "The analysis resamples the 24 held-out GSD diagrams with replacement. All three comparisons use the same 10,000 draws (seed 42 by default). Each replicate sums per-diagram TP, FP, and FN before calculating micro Relation Relaxed F1. Confidence intervals are percentile intervals for `F1_full - F1_comparator`; `P(delta > 0)` is a bootstrap proportion, not a conventional p-value.",
        "",
        f"- Iterations: {iterations}",
        f"- Seed: {seed}",
        f"- Resampling unit: `sample_id` (24 diagrams; tables joined by `sample_id`, never row position)",
        f"- Derived count-table provenance: `{display_path(provenance_path)}` (SHA-256 `{sha256(provenance_path)}`)",
        "",
        "## Inputs",
        "",
        "| Setting | Input table | SHA-256 |",
        "|---|---|---|",
    ]
    for setting, table_path in input_paths.items():
        lines.append(f"| {SETTINGS[setting]} | `{display_path(table_path)}` | `{sha256(table_path)}` |")
    lines += [
        "",
        "Source-artifact hashes and derivation notes are recorded in `data/bootstrap_inputs/provenance.json`. This release supports independent reproduction of the bootstrap from the released per-diagram count tables. Frozen comparator predictions are also provided in `predictions/extraction_predictions.jsonl`; comparator model weights are not included.",
        "",
        "## Results",
        "",
        "| Full setting | Comparator | Observed full F1 | Observed comparator F1 | Observed delta | 95% CI for delta | P(delta > 0) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['full_setting']} | {row['comparator']} | {row['observed_full']:.4f} | "
            f"{row['observed_comparator']:.4f} | {row['observed_delta']:.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {row['probability_delta_gt_zero']:.4f} |"
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "python code/bootstrap.py --input-dir data/bootstrap_inputs --output-dir results/bootstrap --iterations 10000 --seed 42",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing bootstrap outputs")
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    paths = {setting: args.input_dir / f"{setting}_per_diagram_counts.csv" for setting in SETTINGS}
    provenance_path = args.input_dir / "provenance.json"
    for path in [*paths.values(), provenance_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    tables = {setting: read_counts(path, setting) for setting, path in paths.items()}
    sample_ids = validate_tables(tables)
    summaries, distributions = build_results(tables, sample_ids, args.iterations, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": args.output_dir / "bootstrap_summary.csv",
        "distributions": args.output_dir / "bootstrap_distributions.json",
        "report": args.output_dir / "bootstrap_report.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing output(s): " + ", ".join(existing))

    with outputs["summary"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    distributions["inputs"] = {
        setting: {"path": display_path(path), "sha256": sha256(path)} for setting, path in paths.items()
    }
    distributions["provenance"] = {"path": display_path(provenance_path), "sha256": sha256(provenance_path)}
    outputs["distributions"].write_text(json.dumps(distributions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(outputs["report"], summaries, paths, provenance_path, args.iterations, args.seed)
    for row in summaries:
        print(
            f"{row['comparator']}: delta={row['observed_delta']:.6f}, "
            f"95% CI=[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}], "
            f"P(delta>0)={row['probability_delta_gt_zero']:.4f}"
        )


if __name__ == "__main__":
    main()
