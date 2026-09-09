# Data preparation

The THUman2.0 configuration is the main example. Obtain the scans and fitted body parameters under the dataset's terms. Dataset files, SMPL/SMPL-X models, pretrained networks, and AvatarMix checkpoints are not included in this repository.

## External assets

By default, shared assets live under `external_assets/` at the repository root. Set `AVATARMIX_ASSET_ROOT` to an absolute directory to keep them elsewhere. The pipeline and SMPL helpers use this location.

```text
external_assets/
  smpl/
    smpl/                         # SMPL models if required by your inputs
    smplx/
      SMPLX_MALE.npz
      SMPLX_FEMALE.npz
      SMPLX_NEUTRAL.npz
    J_regressor_body25_smplx.txt   # Required by SMPL-X fitting
    J_regressor_body25_smplx_lite.txt  # Only for the optional lite-model path
  smplx_vert_segmentation.json     # SMPL-X vertex-to-body-part index lists
  seamfix.ckpt                     # Your trained checkpoint; release pending
  fullbodyfix.ckpt                 # Optional; release pending
```

Obtain body models from the [SMPL-X project](https://smpl-x.is.tue.mpg.de/). Use the model files and vertex segmentation that match your SMPL-X topology. Some tools also use auxiliary body-model correspondence/regressor files; preserve their filenames when placing them under `smpl/`.

The semantic parser is acquired separately with `scripts/setup_4ddress.py`, as described in [Installation](installation.md). Follow the [4D-DRESS model instructions](https://github.com/eth-ait/4d-dress#model-installation) for Graphonomy, RAFT, and SAM weights. Their expected locations remain:

```text
4d-dress/4dhumanparsing/checkpoints/
  graphonomy/inference.pth
  raft/models/raft-things.pth
  sam/sam_vit_h_4b8939.pth
```

SAM3 hand parsing uses its upstream model loader and may require access to the model's Hugging Face repository. The SMPL-X fitting configuration uses `yolov8l-pose.pt`; obtain it through Ultralytics. Refiner training starts from `nvidia/difix` and may download the backbone and perceptual-loss weights.

## Render raw scans

The [THuman renderer](../external_tools/thuman2_render/README.md) is included in `external_tools/thuman2_render/` and uses ICON-derived rendering utilities.

Arrange its input as follows, using symlinks if preferred:

```text
external_tools/thuman2_render/data/thuman2/
  all.txt
  scans/
    0365/0365.obj
    0365/material0.jpeg
    ...
  smplx/
    0365.pkl
    0365.obj
    ...
```

`all.txt` contains subject IDs in order. The renderer's `--start_subject` and `--end_subject` select indices in that list, with an exclusive end. The pipeline's render stage assumes the conventional full list ordered by numeric subject ID; for a custom subset list, render explicitly with the appropriate indices.

Example for an input list where subject 0365 is at index 365:

```sh
cd external_tools/thuman2_render
CUDA_VISIBLE_DEVICES=0 uv run python render_batch.py \
  --headless True --size 1024 --additional_pitch_views --no_parent_dir \
  --start_subject 365 --end_subject 366 --out_dir /absolute/path/to/data/thuman2_repose/render
cd ../..
```

Use one available GPU with working NVIDIA OpenGL/EGL support. Multiprocessing is disabled by default.

The converter expects the rendered images and fitted parameters together:

```text
data/thuman2_repose/
  render/0365/
    render/000_p+00.png
    calib/000_p+00.txt
    ...
  smplx/0365.pkl
  smplx/0365.obj
  ...
```

The PNG alpha channel supplies foreground masks. Each calibration file contains a 4×4 world-to-camera transform, a 4×4 intrinsic block, and a near/far line. The example renderer produces the camera names expected by the converter, including the additional pitch views. Keep fitted parameters in `smplx/`, or link that directory from the raw dataset.

## Convert and reconstruct avatars

`AVATARMIX_DATA_ROOT` points to the parent directory containing `thuman2_repose/` and `thuman2_avatarrex/`. Without an override, it is the repository's `data/` directory. Update the paths and subject IDs in `swapvton/src/configs/thuman2.yaml` for your dataset.

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py \
  --config swapvton/src/configs/thuman2.yaml --subjects 0365,0228 \
  --run-stages convert_to_avatarrex,convert_to_neus2,neus2_train,copy_mesh,clean_mesh,splatting_avatar,smplx_fitting,hands_transplant,parse_mesh,process_mesh,lbs_transfer
```

Run reconstruction only after the environments and external models are available. Some preprocessing stages replace derived meshes or fitted parameters inside the selected data tree. Use a working copy of your processed data when experimenting with different settings.

The converted per-subject structure is:

```text
data/thuman2_avatarrex/
  0365/
    calibration_full.json
    smpl_params.npz
    thuman2_metadata.json
    000_p+00/0000.jpg
    000_p+00/mask/pha/0000.jpg
    ...
    mesh/
      trimesh_cleaned/0000.obj
      labeled/
      labeled_no_sam/
      processed/
  output-splatting/
    neusclean_sub0365/point_cloud/iteration_10000/
      point_cloud.ply
      embedding.json
```

The iteration directory is configurable; use the checkpoint iteration selected in the pipeline configuration. Mesh topology, Gaussian embeddings, semantic labels, and SMPL parameters must come from the same avatar preparation run.

## Composition and refinement artifacts

The [README](../README.md#usage) gives the composition command. GSReshape consumes the processed meshes and skinning information, runs the cloth-fit solver, and deforms the attached Gaussians. The pipeline writes composed Gaussians under `swapped/` and rendered views under `head_swapped_renders/` in the data root.

SeamFix/FullbodyFix require those renders and their masks/calibration, not just a standalone image. The `difix_refinement` stage builds the refiner arguments for the selected pair. `splatting_avatar_gs_finetune` then uses the refined views to update the composed avatar. Keep the output suffix consistent across the stages; the shipped examples use a common suffix.

`thuman2.yaml` selects SeamFix. `thuman2_fullbody.yaml` selects optional FullbodyFix with the same input composition. Set `paths.seamfixed_test_dir` and the refiner checkpoint path explicitly if maintaining separate outputs for the two modes. Here, `test` names refer to inference/render export.

See [Training](training.md) to generate double-swap examples and train the refiners.
