# Data and reproduction scope

The study partitioned 120 geomorphological schematic diagrams (GSDs) into 84
training, 12 validation, and 24 test diagrams. This public package is designed
for test-metric recomputation and final-adapter inference and therefore
provides all 24 test diagrams, their gold annotations, frozen per-diagram
predictions, and 12 training-format examples. `data/split_manifest.csv`
enumerates these released files.
The complete 120-record study manifest was audited by DOI before release: its
96 source articles had zero train/validation/test overlaps. Thus, diagrams
from the same source article were assigned to the same split.

Six of the 84 unique Stage 2 **training** records have empty `input_passage`
and empty `gold_evolutions`. They were repeated ten times in the original
training data (60 of 840 Stage 2 training rows). A gold listing is provided in
`data/negative_training_examples.json`, separate from the test evaluation.
These six records document how no-relation cases were represented during
training and are excluded from test-set metrics by design.

The 24 public test images and 12 illustrative training images have item-level
source and licence records in `data/attribution_manifest.csv`. Third-party
images and article text retain their source terms under their respective
source licences. Public redistribution details are recorded in
`THIRD_PARTY_NOTICES.md`.

The released final LoRA adapters permit test-time inference with the external
Qwen base model. API-based comparator reproducibility is provided at the
frozen-output evaluation level. The frozen outputs, gold annotations, and
metric code permit the reported test-metric comparisons to be recalculated.

This release includes synthetic diagram and text fixtures under `examples/`
for a usage demonstration without relying on a third-party study figure.
