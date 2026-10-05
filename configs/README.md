# Effective training-setting specifications

These files document the effective portable LLaMA-Factory SFT settings,
including Stage 1 learning rate `2e-5` and Stage 2 learning rate `2e-6`.
They document the published settings. Full training additionally requires the
complete named training and validation datasets.

The released Stage 1 and Stage 2 adapter configurations both record LoRA
dropout 0.2. Accordingly, both portable YAML files use 0.2, consistent with
the final checkpoints and manuscript Table A.1.
