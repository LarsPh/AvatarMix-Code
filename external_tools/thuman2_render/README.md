# THuman2 renderer

Renders multi-view THUman2.0 scans and camera calibration for AvatarMix preprocessing.

Follow [Installation](../../docs/installation.md) to set up the renderer environment and [Data preparation](../../docs/data_preparation.md#render-raw-scans) for the input layout and rendering command. Headless GPU rendering requires NVIDIA OpenGL/EGL support. Subject range arguments select indices in `all.txt`, with an exclusive end index.

The renderer uses [ICON](https://github.com/YuliangXiu/ICON)-derived utilities. See [Third-party notices](../../THIRD_PARTY_NOTICES.md) and the license headers in `lib/`.
