# User guide

## Inputs and outputs

The full two-stage inference input is one diagram image, its caption, and
source-context text. The released 24 test inputs are in
`data/runtime/description_test24.json`; image paths resolve under
`data/test_images/`. Stage 1 writes a description and an ordered list of
evidence sentences. Stage 2 uses those sentences to emit a JSON object with
an `evolutions` array. Each element has Input (`input_mentions`), Output
(`output_mentions`), Driver (`mechanism_mentions`), and optional Time and
Location roles. `evidence_sent_ids` refer to Stage 1 sentence numbers. See
`schema/evolution_relation.json` and `examples/demo_relations.json`.

An empty `evolutions` array means no supported relation was extracted. It
does not imply a prediction file is missing. Incomplete predicted relations
can occur and are counted according to the selected evaluation convention.

## Common commands

| Command | Input | Output | Purpose |
| --- | --- | --- | --- |
| `bash scripts/run_demo.sh` | Synthetic SVG/text fixture | Printed description, evidence, JSON, self-match Relation F1 | Check the schema and metric path without Qwen weights |
| `python main.py example` | One released test gold/prediction pair | `results/example/FIG-041.json` | Inspect a real frozen result without any model download |
| `python scripts/evaluate.py --table 5` | Gold + all Table 5 frozen predictions | `results/tables/table5.json` | Recalculate selected manuscript metrics |
| `python main.py infer --check-only` | Test inputs, images, adapters | Console preflight | Check inference assets without loading Qwen |
| `python main.py infer` | 24 test images/context + Qwen base + two adapters | Stage 1 and Stage 2 files under `results/inference_test24/` | Re-run final-model inference on the released test set |

`scripts/evaluate.py` also accepts `--table 4`, `6`, or `7`, `--device`,
`--model` (SciBERT name or local path), and `--output-dir`. It refuses to
overwrite existing result files. The default SciBERT model is
`allenai/scibert_scivocab_uncased`, and the similarity threshold is 0.7.
Table 4 API outputs require `--api-table4-convention`; the wrapper applies it
automatically for the four API rows.

`scripts/reconstruct_triplets.py` converts the released raw triplet strings
to the event candidates used for the Table 5 comparison. It refuses to
overwrite its output. See `predictions/triplet_provenance.json` for the
historical grouping rule and the one malformed raw record.

The demo is a self-match smoke test, not evidence of extraction accuracy.
The test-metric commands evaluate already-frozen outputs; they do not retrain
or re-query any comparator model. See `data_statement.md` for the exact
release scope and limitations.
