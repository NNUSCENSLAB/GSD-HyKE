# Released adapters

`stage1/` and `stage2/` contain the final project-trained LoRA adapters used by
the released inference workflow. Both expect the upstream base model
`Qwen/Qwen2.5-VL-7B-Instruct` and do not include its weights.

The adapters use LoRA rank 8, alpha 16, and dropout 0.2. Users must comply with
the upstream Qwen model licence as well as this repository's notices.
