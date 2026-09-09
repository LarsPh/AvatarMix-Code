# AvatarMix pipeline

This module orchestrates avatar reconstruction, composition, GSReshape, paired training-data generation, image refinement, and Gaussian fine-tuning.

Start with the [main README](../README.md), [installation](../docs/installation.md), and [data preparation](../docs/data_preparation.md). Use `src/configs/thuman2.yaml` for SeamFix or `src/configs/thuman2_fullbody.yaml` for FullbodyFix. Both configurations use the shared asset and data layout described there.

From the repository root:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py --help
```

Refiner training is documented in [Training](../docs/training.md). Weights and datasets are not bundled. Third-party implementations retain their own terms; see [Third-party notices](../THIRD_PARTY_NOTICES.md).
