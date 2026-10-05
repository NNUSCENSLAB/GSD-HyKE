# Third-party materials

## Diagram images and source text

The files in `data/test_images/` and `data/example_images/` are redistributed from source articles rather than
licensed by the GSD-HyKE authors. Item-level article URLs, DOIs, figure numbers,
source filenames, and the completed CC BY 4.0 review status are recorded in
`data/attribution_manifest.csv`. The manifest review covered the
article-level licence and the target figure caption or credit line.

Users must retain source attribution and indicate modifications. Renaming and
dataset packaging do not transfer copyright to the GSD-HyKE authors. If a
manifest record conflicts with the publisher record, the publisher record
controls and the repository maintainers should be notified.

The 24 files in `data/test_images/` are used by the released test-time input
schema. The smaller `data/example_images/` set supports format and usage
examples. Both directories retain the terms of their respective sources.

## Models and frameworks

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`; obtain it from its official model
  repository and comply with its licence.
- Semantic encoder: `allenai/scibert_scivocab_uncased`; obtain it from its
  official model repository and comply with its terms.
- Training framework: LLaMA-Factory; comply with its upstream licence.
- `models/stage1/` and `models/stage2/` contain project-trained LoRA adapters,
  not copies of the Qwen base-model weights.

The repository's MIT `LICENSE` applies to original project code only.
The CC BY 4.0 grant in `DATA_LICENSE.md` applies only to original annotations
and documentation.
