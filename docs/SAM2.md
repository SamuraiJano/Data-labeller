# Default SAM 2.1 initial-mask provider

Data Labeller uses Meta's official [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) image predictor as its default initial-mask assistant. The launcher installs the pinned package and uses the verified repository-local small checkpoint. SAM 3 remains an optional text-prompt provider documented in [SAM3.md](SAM3.md).

Configure the optional provider through environment variables rather than repository paths or embedded credentials:

```text
JANO_SEGMENTATION_PROVIDER=sam2
SAM2_REPO_PATH=<optional-path-to-official-sam2-checkout>
SAM2_REVISION=2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_CHECKPOINT_PATH=<optional-path-to-sam2.1-checkpoint>
SAM2_MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml
SAM2_DEVICE=auto
```

When `SAM2_CHECKPOINT_PATH` is unset, the provider discovers `checkpoints/sam2.1_hiera_small.pt` and verifies its SHA-256 before loading. It loads lazily, caches the current image embedding, supports positive/negative points and boxes, and keeps manual annotation available when dependencies, weights, CUDA, or memory are unavailable. It does not provide text prompting.
