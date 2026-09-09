# Third-party notices

The top-level AvatarMix license covers the authors' original contributions only. It does not relicense any of the components below. Their existing notices and restrictions continue to apply, including to derivative code where required by their terms. Model and dataset licenses are separate from code licenses.

| Component | Source | Applicable notice |
| --- | --- | --- |
| SplattingAvatar and Gaussian Splatting | [SplattingAvatar](https://github.com/initialneil/SplattingAvatar) | [splatting/LICENSE.md](splatting/LICENSE.md): SplattingAvatar CC BY-NC-SA 4.0 and the Gaussian Splatting research license |
| Gaussian rasterization / nearest-neighbor extensions | Included source in `splatting/submodules/` | Licenses and copyright headers within each extension |
| Difix3D+ | [NVIDIA Difix3D](https://github.com/nv-tlabs/Difix3D) | [difix/LICENSE.txt](difix/LICENSE.txt), including NVIDIA and backbone-related terms |
| NeuS2 | [NeuS2](https://github.com/19reborn/NeuS2) | [neus2/LICENSE.txt](neus2/LICENSE.txt); bundled libraries retain their own notices |
| Garment retargeting / PolyFEM | [cloth-fit](https://github.com/Huangzizhou/cloth-fit) | [cloth-fit/LICENSE](cloth-fit/LICENSE), MIT; fetched dependencies have their own licenses |
| Robust Skin Weights Transfer | [Source fork](https://github.com/LarsPh/Experiments_RobustSkinWeightsTransfer) | [lbs_transfer/LICENSE](lbs_transfer/LICENSE), MIT |
| 4D-DRESS parsing | [4D-DRESS](https://github.com/eth-ait/4d-dress) | Upstream source is not bundled. Fixed-version acquisition and adaptation patches are provided in [4d-dress/](4d-dress/README.md). No explicit top-level code license was found; obtain any necessary permission from its authors. AvatarMix does not license upstream or derivative portions of the patches. |
| Graphonomy | [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) | Acquired separately by the parser setup; preserve the fetched repository's LICENSE |
| RAFT | [RAFT](https://github.com/princeton-vl/RAFT) | Acquired separately with its LICENSE; the supplied patch preserves local compatibility changes |
| pygco / GCO | [pygco](https://github.com/yujiali/pygco) | GCO must be obtained separately under its own terms; preserve its notices when installing it |
| SAM3 | [SAM3](https://github.com/facebookresearch/sam3) | [sam3/LICENSE](sam3/LICENSE); model access and model terms are separate |
| THuman renderer / ICON-derived utilities | [THuman2_renderer](https://github.com/LarsPh/THuman2_renderer), [ICON](https://github.com/YuliangXiu/ICON) | Source headers in `external_tools/thuman2_render/lib/`, including MPG notices; the included SMPL-X implementation has its own LICENSE |
| SMPL/SMPL-X implementations | Included under the corresponding `smplx_utils/` and renderer directories | Existing MPG/SMPL-X copyright and licensing notices; obtain body models separately |
| PyTorch3D / tiny-cuda-nn | [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) | Installed from pinned upstream revisions; retain their licenses with any redistribution |

AvatarMix includes modifications to the reconstruction, retargeting, and refinement components, and supplies parsing modifications as patches. These modifications remain subject to the applicable upstream terms.

Before redistributing a component with unspecified or separately granted rights, obtain the applicable permission from its rights holder. In particular, the original AvatarMix license is not an authorization to redistribute 4D-DRESS or the ICON/MPG-derived utilities under different terms.
