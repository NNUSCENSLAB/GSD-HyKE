# GSD-HyKE: compact reproducibility release

**Manuscript:** “A process-centered hypergraph data model and extraction method for landform transformations in geomorphological schematic diagrams” (submission draft for
*Computers & Geosciences*). **Authors:** Yekang Zhou, Teng Zhong, Pei Xu,
Songshan Yue, Run Shi, Jiahao Sun, Bingxian Lin, Liangchen Zhou, and Guonian
Lü. **Contact:** Teng Zhong, tzhong27@njnu.edu.cn. This repository corresponds
to the revised 24-diagram test set and manuscript Tables 3–7; it is not an
accepted-paper archive.

This repository is the compact public reproduction package for **GSD-HyKE**.
It is intentionally organized around two reviewer-facing tasks: recomputing the
reported test-set metrics from frozen outputs and running the final two-stage
model on the 24 held-out geomorphological schematic diagrams (GSDs).

The release includes the complete 24-diagram test set, rights-reviewed test
images, frozen predictions for the released open-source comparisons and
ablations and four API-model comparisons, two final LoRA adapters, small examples of all data formats, metric
code, and paired-bootstrap inputs. It does **not** include the complete training
corpus or every intermediate checkpoint.

The task is specific: given one GSD image, its caption, and associated source
context, Stage 1 generates a geomorphological description and evidence
sentences; Stage 2 extracts role-specific Evolution Relations. It is not a
general-purpose chat assistant.

## Repository layout

```text
.
├── main.py                    # Small command-line entry point
├── environment.yml            # Conda evaluation/inference environment
├── configs/                   # Historical training settings and limitations
├── schema/                    # Evolution Relation JSON schema
├── examples/                  # Original synthetic diagram and text fixtures
├── scripts/                   # One-command demo and per-table evaluation
├── code/                      # Example, evaluation, inference, and bootstrap code
├── data/
│   ├── examples.json          # Small examples of the five task formats
│   ├── test_gold.json         # Gold annotations for all 24 test GSDs
│   ├── test_images/           # The corresponding 24 test images
│   ├── example_images/        # Images used by the format examples
│   ├── runtime/               # Exact final-model test inputs
│   └── bootstrap_inputs/      # Per-diagram TP/FP/FN tables
├── predictions/               # Frozen description and extraction outputs
├── models/                    # Final Stage 1 and Stage 2 LoRA adapters
├── expected_results.json      # Expected recomputed values
└── tests/                     # CPU-only integrity and unit tests
```

## Installation

Python 3.10 or 3.11 is recommended. The released end-to-end workflow was
verified with Python 3.10, PyTorch 2.6.0 (CUDA 12.4 build), Transformers 4.49.0,
PEFT 0.15.2, and Accelerate 1.7.0. Clone with Git LFS because the two adapter
files are tracked as LFS objects.

```bash
git lfs install
git clone https://github.com/NNUSCENSLAB/GSD-HyKE.git
cd GSD-HyKE
conda env create -f environment.yml
conda activate gsd-hyke
```

Alternatively, use pip without Conda:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Conda specification installs the evaluation/inference stack. The
historical training environment also used LLaMA-Factory `0.9.4.dev0`, but
training is not runnable from this compact corpus. CUDA 12.4 and one A100
80 GB are the tested reference for full inference, not a guaranteed minimum.

Metric recomputation can run on CPU; CUDA is optional for this step. Final
inference uses `Qwen/Qwen2.5-VL-7B-Instruct` in bfloat16 with automatic device
mapping and therefore requires access to the base model plus a CUDA-capable
GPU. The released end-to-end 24-diagram run was verified on one NVIDIA A100
80 GB GPU. This is a tested reference configuration, not a measured minimum:
the minimum VRAM, peak RAM, and wall-clock runtime have not been benchmarked.
The released adapters do not contain the base-model weights. SciBERT is
downloaded as `allenai/scibert_scivocab_uncased` unless a local path is
supplied.

The inference wrapper processes 24 test diagrams, limits image inputs to
262,144 pixels, and uses maximum generation lengths of 1,024 tokens for Stage
1 and 2,048 tokens for Stage 2. The two LoRA adapters are approximately 77 MiB
each; the repository excluding downloaded base models is approximately 185
MiB.

## Quick verification

The following commands do not load a large model:

```bash
python main.py verify
python main.py demo
python -m unittest discover -s tests -v
python main.py infer --check-only
```

## Run the lightweight example

To inspect an original synthetic two-stage river-valley diagram and print a
description, evidence sentence, role-specific JSON, and a SciBERT-based
Relation Relaxed F1 self-match smoke test, run:

```bash
bash scripts/run_demo.sh
```

The fixture is in `examples/`; it is not a new study observation or a trained
model prediction. This command needs SciBERT but **not** the 7B Qwen weights.
To use a local SciBERT cache, pass `--model /path/to/scibert`. The output
schema is [schema/evolution_relation.json](schema/evolution_relation.json):
`input_mentions` = Input, `output_mentions` = Output, and
`mechanism_mentions` = Driver; Time and Location are optional role lists.

For a real frozen test case, the existing CPU-only walkthrough requires no
model download:

This CPU-only example requires no model download. It joins one test diagram to
the released full-model prediction by `sample_id`, verifies the referenced
image and Evolution Relation role structure, and writes a human-readable JSON
walkthrough:

```bash
python main.py example
```

The default input is sample `FIG-041` from `data/test_gold.json` and setting
`table5/full_hypergraph` from `predictions/extraction_predictions.jsonl`. The
output is written to `results/example/FIG-041.json` and contains the source
caption and context, the predicted Input/Output/Driver/Time/Location roles, and
a short gold-reference summary. It is a walkthrough of a frozen prediction,
not a new inference run and not a replacement for the semantic evaluation.

Alternative released samples or settings can be selected explicitly:

```bash
python main.py example \
  --sample-id FIG-079 \
  --setting table7/without_evidence \
  --output results/example/FIG-079-without-evidence.json
```

The principal public interfaces are:

| Command | Input | Output | Expected behavior |
| --- | --- | --- | --- |
| `python main.py example` | One gold row and one frozen prediction joined by `sample_id` | One walkthrough JSON | Validates and displays the structured Evolution Relations without loading a model |
| `python main.py evaluate` | The 24 gold records and selected frozen prediction settings | `results/recomputed_extraction_metrics.json` | Recomputes Entity/Relation relaxed scores, GED, nGED, ECC, and EPV |
| `python main.py bootstrap` | Four released per-diagram TP/FP/FN tables | Summary CSV and distribution JSON under `results/bootstrap/` | Repeats paired diagram-level bootstrap comparisons |
| `python main.py infer --check-only` | Test inputs, images, and both adapters | Console preflight result | Verifies inference assets without loading the base model |
| `python main.py infer` | 24 images, contextual text, base model, and two adapters | Stage 1 and Stage 2 predictions and metrics under `results/inference_test24/` | Runs the released two-stage model workflow on a CUDA-capable system |

## Recompute extraction results

For a table-oriented entry point, use `python scripts/evaluate.py --table 4`
(or `5`, `6`, `7`). It reads the setting IDs recorded in the released
prediction files, writes a separate JSON result per table under
`results/tables/`, and refuses to overwrite an existing result. Table 4 also
runs the API rows using their historical metric convention. For example:

```bash
python scripts/evaluate.py --table 4 --device cpu
python scripts/evaluate.py --table 5 --device cpu
python scripts/evaluate.py --table 6 --device cpu
python scripts/evaluate.py --table 7 --device cpu
```

Use a different `--output-dir` for a second run. `predictions/README.md`
maps each `setting_id` to the relevant manuscript row.

This reproduces the released 24-diagram results using role-constrained
SciBERT matching at the manuscript threshold, `tau = 0.7`:

```bash
python main.py evaluate --setting table5/full_hypergraph --device cpu
python main.py evaluate --setting table5/triplet_reconstructed --device cpu
```

Omit `--setting` to evaluate all released open-source and ablation settings.
The output contains Entity Relaxed F1, Relation Relaxed precision/recall/F1,
GED, nGED, ECC, and EPV. See `predictions/provenance.json` for the setting IDs.

The four API-model rows of Table 4 use a separate sanitized prediction file:

```bash
python main.py evaluate \
  --predictions predictions/api_extraction_predictions.jsonl \
  --api-table4-convention \
  --output results/recomputed_api_extraction_metrics_historical.json \
  --device cpu
```

`--api-table4-convention` is necessary to reproduce the historical Table 4
scoring: Entity matching is not type-constrained, GED/nGED use all parsed
events including incomplete ones, and EPV is the Input-and-Output core rate.
Without this flag, the compact evaluator's default conventions give different
Entity and graph scores. The released records reproduce the selected main
repetition for each model;
they do not permit new API inference or the three-repeat selection procedure
to be rerun without the original service and unreleased raw responses. See
`predictions/api_provenance.json` for the selected repetition and source hashes.

The Table 5 triplet arm additionally releases the raw triplet text in
`predictions/triplet_raw.jsonl`. Its historical event reconstruction groups
triples by evidence-sentence IDs and exact Output anchor (or Output alone when
IDs are absent). Recreate the frozen event candidates with
`python scripts/reconstruct_triplets.py`; the 24 reconstructed relation lists
have been checked against the released Table 5 rows. One raw output is
malformed at test index 18 and is preserved as such, rather than silently
repaired. See `predictions/triplet_provenance.json`.

Description ROUGE-L and SciBERT BERTScore are recomputed with:

```bash
python main.py evaluate --descriptions --setting table3/qwen_gsd_hyke
```

## Recompute paired-bootstrap intervals

```bash
python main.py bootstrap --iterations 10000 --seed 42
```

The resampling unit is the test diagram. Each replicate sums the selected
diagrams' TP, FP, and FN before computing micro Relation Relaxed F1. The three
comparisons share the same paired draws.

## Run final-model inference

First perform a non-GPU preflight:

```bash
python main.py infer --check-only
```

Then run the released Stage 1 to Stage 2 workflow:

```bash
python main.py infer \
  --base-model Qwen/Qwen2.5-VL-7B-Instruct \
  --scibert allenai/scibert_scivocab_uncased
```

Outputs are written to `results/inference_test24/`. Model download and GPU
requirements are the user's responsibility.

## Scope and known limitations

- This compact repository supports independent metric recomputation from the
  frozen outputs and final-model inference on the released 24-image test set.
- It is not a full training release: only format examples are provided for the
  training tasks, so the original training runs cannot be repeated from this
  repository alone.
- The original Stage 2 launch YAML stated dropout 0.1, but resumed a Stage 1
  adapter whose effective PEFT dropout was 0.2; both released adapters record
  0.2. `configs/README.md` documents why the portable effective configuration
  uses 0.2.
- Six relation-negative records belong to the original **training** split,
  not the 24-image test split. Their gold-only listing is released, but no
  frozen predictions for them have been found; no negative-test metric is
  claimed. See `docs/data_statement.md`.
- The four API-model Table 4 rows can be re-evaluated from frozen per-diagram
  predictions, but the repository does not reproduce their API generation.
- The historical manuscript CLIPScore convention is still under author review.
  This release therefore recomputes Table 3 ROUGE-L and BERTScore but does not
  claim reproduction of CLIPScore.
- Frozen predictions are provided for the triplet and ablation comparisons;
  their model checkpoints are not included. Their evaluation is reproducible,
  but their inference is not reproduced end to end.

These boundaries should also be stated in the manuscript's Code and Data
Availability section.

The complete release scope and image-sharing rationale are in
`docs/data_statement.md`; the input/output user guide is in
`docs/user_guide.md`.

## Data, licences, and attribution

Original project code is licensed under the MIT License (`LICENSE`). Original
annotations and documentation are released under CC BY 4.0
(`DATA_LICENSE.md`). Diagram images and source text remain subject to their
source-article terms; item-level provenance is in
`data/attribution_manifest.csv` and is explained in
`THIRD_PARTY_NOTICES.md`. The Qwen base model, SciBERT, and LLaMA-Factory retain
their respective upstream licences.

Repository: https://github.com/NNUSCENSLAB/GSD-HyKE

Citation metadata for the submission-stage repository are in `CITATION.cff`.
The final article journal reference and DOI must be added after publication.
