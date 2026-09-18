# Optional official SAM 3 integration

Authority: `https://github.com/facebookresearch/sam3`, pinned by `SAM3_REVISION` for repeatable project work.

The provider imports `build_sam3_image_model` and `Sam3Processor` directly from the official package. `set_image` is cached by image path/mtime. Each new proposal clears prompt state but reuses the embedding, then applies official text and/or normalized center-`xywh` box prompts.

Configuration is environment-only:

- `SAM3_REPO_PATH`: optional checkout used to verify the configured Git revision.
- `SAM3_CHECKPOINT_PATH`: optional official checkpoint path.
- `SAM3_LOAD_FROM_HF`: allow the official builder to use the workstation's normal Hugging Face authentication.
- `SAM3_DEVICE`: `cuda`, `cpu`, or `auto`.
- `SAM3_CONFIDENCE_THRESHOLD`: proposal filter threshold.

No token, checkpoint, username, or machine path is in source control. Load/prediction failures are returned as actionable API errors and the manual canvas remains available.
