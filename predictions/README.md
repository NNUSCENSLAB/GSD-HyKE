# Frozen predictions

Each JSONL row contains a `setting_id`, `sample_id`, test index, and the frozen
prediction for one test diagram. Every released setting has exactly 24 rows.

`extraction_predictions.jsonl` covers the open-source model comparison, the
hypergraph-versus-triplet comparison, and the dataset and reasoning ablations.
`description_predictions.jsonl` covers the six open-source Stage 1 rows.
`extraction_predictions.jsonl` uses these Table 4–7 setting IDs:

| Manuscript table | Frozen-output setting IDs |
| --- | --- |
| 4, open-source rows | `table4/{qwen,llama,internvl}_{base,gsd_hyke}` |
| 5, representation | `table5/full_hypergraph`, `table5/triplet_reconstructed` |
| 6, data/Stage 1 ablations | `table6/full`, `table6/without_first_stage_training`, `table6/without_classification`, `table6/without_completion`, `table6/without_visual_dialogue`, `table6/without_description_generation` |
| 7, reasoning/description | `table7/full`, `table7/without_description`, `table7/without_evidence` |

The API file uses `table4/gpt_4o`, `table4/gpt_5_4`, `table4/gpt_5_5`, and
`table4/gpt_5_6_terra`. Every setting has 24 rows. `table5/full_hypergraph`,
`table6/full`, and `table7/full` refer to the same full-model test output,
not independent runs.
`api_extraction_predictions.jsonl` contains 24 frozen Stage 2 predictions for
each of the four API-based Table 4 models. These rows are the selected main
repetitions, not averages across three repetitions. The index-to-`sample_id`
join was validated against all 24 image filenames. Only parsed Evolution
Relations are included; raw API responses, request payloads, and response IDs
are not released. Source paths and SHA-256 hashes are in `api_provenance.json`.
Use `python main.py evaluate --predictions predictions/api_extraction_predictions.jsonl
--device cpu` for the manuscript Table 4 metrics. API and open-source rows use
the same role-constrained Entity/Relation definitions and full EPV definition.

`triplet_raw.jsonl` contains the 24 direct-triplet model outputs before event
reconstruction, including one malformed raw output. Its exact source hash,
index mapping, and grouping rule are in `triplet_provenance.json`.

The files enable evaluation of these outputs but do not replace the omitted
comparator checkpoints. See `provenance.json` for setting IDs, source-artifact
hashes, and known limitations.
