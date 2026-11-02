<div align="center">

# AvatarMix

### Identity-Preserving Cross-Avatar Composition for Outfit Personalization

[Zhaorong Wang](https://larsph.github.io/) · [Yoshihiro Kanamori](https://kanamori.cs.tsukuba.ac.jp/) · [Yuki Endo](https://endo-yuki-t.github.io/)

University of Tsukuba · **CVPR 2026 Findings**

[Paper](https://arxiv.org/abs/2606.03506) · [Project Page](https://larsph.github.io/avatarmix/) · [Video](https://larsph.github.io/avatarmix/static/videos/AvatarMix_CVPR2026F_video.mp4)

</div>

![AvatarMix outfit personalization results](assets/teaser.jpg)

AvatarMix combines a user's identity with another avatar's outfit in a mesh-based 3D Gaussian representation. GSReshape adapts the clothed body to the user's physique; SeamFix repairs the head–neck join, and optional FullbodyFix restores appearance after reshaping.

## Method

![AvatarMix pipeline](assets/method_overview.jpg)

The workflow reconstructs mesh-based Gaussian avatars, composes the user's head with the model's reshaped body, refines rendered views, and updates the composed Gaussians. See the [paper](https://arxiv.org/abs/2606.03506) for the method and implementation details.

## Installation

Use Linux, an NVIDIA GPU, and separate environments for the pipeline components. The code uses `uv`; CUDA extensions require a compatible CUDA toolkit and C++ compiler.

Follow [Installation](docs/installation.md), then [Data preparation](docs/data_preparation.md). Commands below run from the repository root unless stated otherwise. Set the data directory once in your shell, or use the default `data/` directory:

```sh
export AVATARMIX_DATA_ROOT=/absolute/path/to/data
```

For fish, use `set -gx AVATARMIX_DATA_ROOT /absolute/path/to/data` instead.

## Usage

Inspect the pipeline options:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py --help
```

After preparing two avatars and editing the input/output settings in [thuman2.yaml](swapvton/src/configs/thuman2.yaml), compose a user–model pair:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py \
  --config swapvton/src/configs/thuman2.yaml --subjects 0365,0228 \
  --run-stages head_donator_reposing,cloth_fit_reshaping,direct_swapping,render_swapped_gaussians
```

Subject IDs are examples: use subjects available in your own dataset. The pair configuration controls direction, shape fitting, camera selection, and intermediate output paths.

With the [SeamFix adapter](https://drive.google.com/file/d/1qKqfQWfRzX0QiHRktLNODKjYxHTTXjA5/view) placed at `external_assets/seamfix.safetensors`, refine the views and update the avatar:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py \
  --config swapvton/src/configs/thuman2.yaml --subjects 0365,0228 \
  --run-stages difix_refinement,splatting_avatar_gs_finetune
```

FullbodyFix is optional and is selected by visual inspection of reshaping artifacts. Use [thuman2_fullbody.yaml](swapvton/src/configs/thuman2_fullbody.yaml) with the [FullbodyFix adapter](https://drive.google.com/file/d/1rJYdTG93PtAgVWjV0vQSChjU_SpsS-vc/view) at `external_assets/fullbodyfix.safetensors` for full-body refinement. The data guide explains the intermediate artifacts these stages consume.

## Pretrained refiners

| Adapter | Size | SHA-256 |
| --- | ---: | --- |
| [SeamFix](https://drive.google.com/file/d/1qKqfQWfRzX0QiHRktLNODKjYxHTTXjA5/view) | 12.9 MB | `bb3ac12f2aa5bb32aa5b0175147b67513b4c99bd27cf653698e9668a05337ea4` |
| [FullbodyFix](https://drive.google.com/file/d/1rJYdTG93PtAgVWjV0vQSChjU_SpsS-vc/view) | 19.5 MB | `678aada4b5fa8c52ecbcc653ccb45a6efdb5d45697efaaa854545599f0bc20df` |

These adapters require the [NVIDIA Difix backbone](https://huggingface.co/nvidia/difix) at revision `2b0c6fb5797c26b01154dfdeb19d36e5e2eaf388`; the code loads it separately. Download the adapters and place them in `external_assets/`, or set `AVATARMIX_ASSET_ROOT` to their directory. The [download folder](https://drive.google.com/drive/folders/1Xzh3jSFpdgenvDf2-8WYkN4LTdws8zZL) includes the model license and notice. Use is limited to non-commercial research/evaluation under the applicable model terms.

## Training

![Training data generation for the refiners](assets/training_artifact_refiners.jpg)

See [Training](docs/training.md) for double-swap data generation, pair manifests, and refiner training commands. The refiner configurations are [seamfix.yaml](difix/configs/experiment/seamfix.yaml) and [fullbodyfix.yaml](difix/configs/experiment/fullbodyfix.yaml).

## Code layout

| Directory | Role |
| --- | --- |
| `swapvton/` | Pipeline, preprocessing, cross-avatar composition, and training-data generation |
| `splatting/` | Mesh-based Gaussian reconstruction, deformation, rendering, and fine-tuning |
| `cloth-fit/` | Mesh retargeting used by GSReshape |
| `difix/` | SeamFix and FullbodyFix training and inference |
| `neus2/` | Initial surface reconstruction |
| `4d-dress/`, `sam3/` | Semantic parsing and hand segmentation |
| `lbs_transfer/` | Skinning-weight transfer |

## Release TODO

- [ ] Example assets

## Citation

```bibtex
@inproceedings{wang2026avatarmix,
  author    = {Wang, Zhaorong and Kanamori, Yoshihiro and Endo, Yuki},
  title     = {{AvatarMix}: Identity-Preserving Cross-Avatar Composition for Outfit Personalization},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026},
  pages     = {425-435}
}
```

## Acknowledgements and license

AvatarMix builds on SplattingAvatar, Gaussian Splatting, Difix3D+, NeuS2, 4D-DRESS, SAM/SAM3, Intersection-Free Garment Retargeting, and Robust Skin Weights Transfer. We thank their authors for sharing their work.

The optional SeamFix and FullbodyFix adapters build on NVIDIA Difix and its Stable Diffusion Turbo backbone. Powered by Stability AI.

AvatarMix's original code is provided for **non-commercial research** under [LICENSE](LICENSE). Third-party code and modifications derived from it remain subject to their respective terms; see [Third-party notices](THIRD_PARTY_NOTICES.md). Model weights, body models, and datasets have separate licenses and are not bundled here.

For questions, contact [Zhaorong Wang](mailto:zhaorong.wang1997@gmail.com).
