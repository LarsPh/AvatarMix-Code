# Training SeamFix and FullbodyFix

Prepare the component environments, body models, and input avatars as described in [Data preparation](data_preparation.md). Training images and pretrained AvatarMix weights are not included.

## Generate paired training data

The pipeline performs a first swap, reconstructs and resegments the resulting avatar, then swaps back to produce examples aligned with the original avatar. Keep the generated portraits, semantic masks, full-body views, and metadata together; the data loader uses their relationship to recover training targets.

Use [thuman2_training.yaml](../swapvton/src/configs/thuman2_training.yaml) to generate training data with SAM voting enabled. This configuration uses the `labeled` outputs throughout both swaps. The composition configuration, `thuman2.yaml`, uses no-SAM labels instead. Generate training data in a separate working copy of the avatar dataset to keep the two sets of processed meshes and labels separate.

Run the data-generation pipeline on a pair whose multi-view images are already rendered:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py \
  --config swapvton/src/configs/thuman2_training.yaml --subjects 0365,0228 \
  --pipeline-mode training --start-at convert_to_avatarrex
```

This includes reconstruction and can be expensive. If the initial avatars are already prepared with this training configuration, start at `head_donator_reposing` instead. The training-data mode performs the double swap; it does not train the diffusion model. Output paths and `skip_existing` behavior are controlled by the stage configuration.

Create separate pair manifests for training and validation under `data/thuman2_avatarrex/pairs/`. For example:

```yaml
pairs:
  - pair_id: 0
    subject_a: '0365'
    subject_b: '0228'
```

Save your training list as `train.yaml` and your validation list as `val.yaml`; use disjoint subjects when measuring generalization. Preserve quoted IDs so that leading zeros survive YAML parsing.

The pipeline also accepts a manifest through `--load-pairs`:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py \
  --config swapvton/src/configs/thuman2_training.yaml \
  --load-pairs /absolute/path/to/data/thuman2_avatarrex/pairs/train.yaml \
  --pipeline-mode training --start-at convert_to_avatarrex
```

The refiner configs expect double-swap data in `head_swapped_back_renders/`, original avatars in the per-subject directories, and first-swap validation renders in `head_swapped_renders/`. Use `data.combined_data_root`, `data.gt_dataset_root`, and `data.validation_data_root` to override those locations. Do not flatten or rename the generated camera and portrait directories.

## Train a refiner

Run from `difix/` so that relative data paths resolve consistently:

```sh
cd difix
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_swapfix_lora.py experiment=seamfix
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_swapfix_lora.py experiment=fullbodyfix
```

Run these jobs separately on one available GPU. Both configurations train new adapters on the pretrained Difix backbone. The public configurations retain UNet ranks 8/16 for SeamFix/FullbodyFix, respectively, and VAE decoder rank 4. The modes share the training entrypoint and use different image/mask preparation.

The default configs start a new training run with W&B offline logging for scalar metrics and image previews. Outputs and checkpoints are written under the Hydra output directory. Enable online W&B if desired.

Inspect the resolved settings before a run:

```sh
uv run python scripts/train_swapfix_lora.py --cfg job --resolve experiment=seamfix
```

Adjust batch size to available VRAM with `data.batch_size`. For a short check using your prepared data:

```sh
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_swapfix_lora.py \
  experiment=seamfix trainer.max_steps=2 trainer.limit_train_batches=2 \
  trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0 data.num_workers=0
```

This short run loads the diffusion backbone and checks that your prepared data can be used for training.

Keep `model.model_name=nvidia/difix` when using the base model: this identifier enables the repository's serializable VAE setup. For offline use, populate the Hugging Face cache and set `HF_HUB_OFFLINE=1`; do not replace this identifier with an arbitrary cache snapshot path.

FullbodyFix training uses 448×896 images. The `fullbody_target_resolution` option applies to validation/inference preprocessing, not the training dataset.

## Resume and inference

Resume training with `ckpt_path=/absolute/path/to/checkpoint.ckpt`. To run the refiner's inference/export mode on a prepared pair manifest:

```sh
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_swapfix_lora.py \
  experiment=seamfix train=false test=true \
  ckpt_path=/absolute/path/to/seamfix.ckpt \
  data.test_pairs_yaml=/absolute/path/to/pairs.yaml \
  data.test_output_root=/absolute/path/to/refined_views
```

For a complete composition/refinement run, prefer the pipeline's `difix_refinement` stage, which supplies the per-pair input paths. Use `experiment=fullbodyfix` and the matching checkpoint for full-body refinement. The `test` switch invokes image inference/export.
