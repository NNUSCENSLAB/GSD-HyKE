"""CPU-only tests for the compact public release."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = load_module("release_bootstrap", ROOT / "code/bootstrap.py")
example = load_module("release_example", ROOT / "code/example.py")
evaluation = load_module("release_evaluate", ROOT / "code/evaluate.py")
description_evaluation = load_module("release_description_evaluate", ROOT / "code/evaluate_descriptions.py")
triplets = load_module("release_triplets", ROOT / "code/triplets.py")


class ReleaseIntegrityTests(unittest.TestCase):
    def test_complete_test_set_and_images(self) -> None:
        rows = json.loads((ROOT / "data/test_gold.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["sample_id"] for row in rows}), 24)
        self.assertTrue(all((ROOT / "data" / row["image"]).is_file() for row in rows))

    def test_every_prediction_setting_has_24_unique_samples(self) -> None:
        for filename in ("extraction_predictions.jsonl", "api_extraction_predictions.jsonl", "description_predictions.jsonl"):
            rows = [json.loads(line) for line in (ROOT / "predictions" / filename).read_text().splitlines()]
            counts = Counter(row["setting_id"] for row in rows)
            self.assertTrue(counts)
            self.assertTrue(all(count == 24 for count in counts.values()))
            for setting in counts:
                ids = [row["sample_id"] for row in rows if row["setting_id"] == setting]
                self.assertEqual(len(ids), len(set(ids)))

    def test_api_predictions_match_test_indices(self) -> None:
        gold = json.loads((ROOT / "data/test_gold.json").read_text(encoding="utf-8"))
        by_index = {row["index"]: row["sample_id"] for row in gold}
        rows = [json.loads(line) for line in (ROOT / "predictions/api_extraction_predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 96)
        self.assertTrue(all(row["sample_id"] == by_index[row["index"]] for row in rows))
        self.assertTrue(all(row["parse_valid"] and isinstance(row["evolutions"], list) for row in rows))

    def test_six_negative_training_records_are_not_test_predictions(self) -> None:
        data = json.loads((ROOT / "data/negative_training_examples.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["records"]), 6)
        self.assertFalse(data["frozen_predictions_available"])
        self.assertTrue(all(not row["input_passage"] and not row["gold_evolutions"] for row in data["records"]))

    def test_raw_triplets_reconstruct_released_event_lists(self) -> None:
        raw = [json.loads(line) for line in (ROOT / "predictions/triplet_raw.jsonl").read_text().splitlines()]
        frozen = {
            row["index"]: row["evolutions"]
            for row in map(json.loads, (ROOT / "predictions/extraction_predictions.jsonl").read_text().splitlines())
            if row["setting_id"] == "table5/triplet_reconstructed"
        }
        self.assertEqual(len(raw), len(frozen))
        for row in raw:
            reconstructed, valid = triplets.reconstruct_text(row["prediction_text"])
            self.assertEqual(reconstructed["evolutions"], frozen[row["index"]])
            self.assertEqual(valid, row["index"] != 18)

    def test_synthetic_demo_contains_all_core_roles(self) -> None:
        relation = json.loads((ROOT / "examples/demo_relations.json").read_text(encoding="utf-8"))["evolutions"][0]
        self.assertTrue(all(relation[field] for field in ("input_mentions", "output_mentions", "mechanism_mentions")))
        self.assertTrue((ROOT / "examples/demo_gsd.svg").is_file())

    def test_example_image_references_resolve(self) -> None:
        data = json.loads((ROOT / "data/examples.json").read_text(encoding="utf-8"))
        for task in ("description_generation", "visual_dialogue"):
            for row in data["tasks"][task]:
                self.assertTrue((ROOT / "data" / row["image"]).is_file())

    def test_lightweight_example_joins_by_sample_id(self) -> None:
        result = example.build_example(
            ROOT / "data/test_gold.json",
            ROOT / "predictions/extraction_predictions.jsonl",
            "FIG-041",
            "table5/full_hypergraph",
        )
        self.assertEqual(result["sample"]["sample_id"], "FIG-041")
        self.assertTrue(result["system_output"]["parse_valid"])
        self.assertEqual(result["system_output"]["evolution_relation_count"], 3)
        self.assertEqual(result["system_output"]["complete_relation_count"], 3)


class BootstrapTests(unittest.TestCase):
    def test_duplicate_sample_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "setting,sample_id,index,tp,fp,fn,predicted_count,gold_count\n"
                "full,A,0,1,0,0,1,1\n"
                "full,A,1,1,0,0,1,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                bootstrap.read_counts(path, "full")

    def test_repeated_bootstrap_draw_is_counted_repeatedly(self) -> None:
        table = {
            "A": {"tp": 1, "fp": 0, "fn": 0},
            "B": {"tp": 0, "fp": 2, "fn": 3},
        }
        import numpy as np

        tp, fp, fn = bootstrap.aggregate_resample(table, ["A", "B"], np.asarray([[0, 0], [1, 1]]))
        self.assertEqual(tp.tolist(), [2, 0])
        self.assertEqual(fp.tolist(), [0, 4])
        self.assertEqual(fn.tolist(), [0, 6])


class MetricConventionTests(unittest.TestCase):
    def test_incomplete_event_can_be_retained_for_graph_audit(self) -> None:
        events = [{"input_mentions": ["a"], "output_mentions": []}]
        self.assertEqual(evaluation.clean_events(events), [])
        self.assertEqual(len(evaluation.clean_events(events, require_transition=False)), 1)

    def test_entity_matching_is_role_constrained_by_default(self) -> None:
        class ExactSimilarity:
            def score(self, left, right):
                return float(left == right)

        predicted = [("Mechanism", "terrace")]
        gold = [("Object", "terrace")]
        self.assertEqual(evaluation.entity_tp(predicted, gold, ExactSimilarity(), 0.7), 0)
        self.assertEqual(evaluation.entity_tp(predicted, gold, ExactSimilarity(), 0.7, strict_label=False), 1)

    def test_manuscript_clipscore_scale(self) -> None:
        cosine = description_evaluation.torch.tensor([0.324575, -0.1])
        scores = description_evaluation.clipscore_from_cosine(cosine)
        self.assertAlmostEqual(float(scores[0]), 0.8114375, places=6)
        self.assertEqual(float(scores[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
