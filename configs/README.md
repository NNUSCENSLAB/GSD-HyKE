# Historical training specifications

These files document the effective portable LLaMA-Factory SFT settings,
including Stage 1 learning rate `2e-5` and Stage 2 learning rate `2e-6`.
They are documentation, **not runnable training commands in this compact
release**: the named training and validation datasets are not included.

The original Stage 2 launch YAML contained `lora_dropout: 0.1`, but Stage 2
resumed the Stage 1 adapter with LLaMA-Factory `create_new_adapter: false` (the
default). In LLaMA-Factory 0.9.4.dev0, that path loads the existing PEFT adapter
configuration instead of creating new LoRA weights from the YAML parameters.
The loaded Stage 1 adapter and the saved Stage 2 adapter both record dropout
0.2. Accordingly, `stage2_sft.yaml` states the effective value 0.2, consistent
with the released adapter and manuscript Table A.1. The original value 0.1 is
retained here as provenance rather than presented as the effective setting.

These files still do not assert that checkpoint selection used the highest validation
Relation Relaxed F1; no such rule is established by the YAML alone.
