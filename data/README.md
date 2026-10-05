# Data

- `examples.json` contains small format examples only. It is not the complete
  training corpus.
- `test_gold.json` and `test_images/` are the complete released 24-diagram test
  set used for frozen-output evaluation.
- `runtime/` contains the exact schemas consumed by the final inference
  wrapper.
- `bootstrap_inputs/` contains derived per-diagram counts sufficient to repeat
  the paired bootstrap without comparator checkpoints.
- `negative_training_examples.json` lists the six gold-only, no-relation Stage
  2 training cases. It contains no model predictions and is not a test set.
- `split_manifest.csv` and `attribution_manifest.csv` cover the public records
  in this compact package.

Image use is governed by the source-level terms recorded in the attribution
manifest, not by the repository's MIT licence.
See `../docs/data_statement.md` for the 84/12/24 study split and the narrower
scope of the files in this compact package.
