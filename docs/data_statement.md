# Data and reproduction scope

The study originally partitioned 120 geomorphological schematic diagrams
(GSDs) into 84 training, 12 validation, and 24 test diagrams. The compact
public package releases all 24 test diagrams, their gold annotations, and
frozen per-diagram predictions for the manuscript comparisons. Its
`data/split_manifest.csv` covers only files included here (24 test diagrams
and 12 training-format examples); it is **not** the complete 120-diagram split
manifest. The full training corpus and validation images are not included, so
this package supports test-metric recomputation and final-adapter inference,
not full retraining or independent checkpoint selection.
The compact manifest by itself also does not audit whether every pair of
figures from the same paper was assigned to one split; consult the complete
study split records before making that stronger claim.

Six of the 84 unique Stage 2 **training** records have empty `input_passage`
and empty `gold_evolutions`. They were repeated ten times in the original
training data (60 of 840 Stage 2 training rows). They are not negative test
diagrams. A small gold-only listing is in `data/negative_training_examples.json`.
Predictions for those six training-only cases are outside the compact release;
the package does not claim a negative-case accuracy result.

The 24 public test images and 12 illustrative training images have item-level
source and licence records in `data/attribution_manifest.csv`. Third-party
images and article text retain their source terms and are not covered by the
project's code or original-annotation licences. The remaining training and
validation images are not present in this compact package. This is a release
scope limitation, not a claim that every omitted image is legally
unshareable. See `THIRD_PARTY_NOTICES.md` before further redistribution.

The released final LoRA adapters permit test-time inference when the external
Qwen base model is available. The full training corpus and intermediate
checkpoints are not released. API-based comparator predictions are sanitized
parsed outputs, not raw service responses; their generation cannot be rerun
from this repository. Frozen outputs plus gold and metric code permit the
reported test-metric comparisons to be recalculated.

This release includes synthetic diagram and text fixtures under `examples/`
for a usage demonstration without relying on a third-party study figure.
