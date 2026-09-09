# Semantic parsing dependency

AvatarMix uses a modified [4D-DRESS](https://github.com/eth-ait/4d-dress) pipeline for semantic mesh and image parsing.

This directory contains environment metadata, a fixed source manifest, and adaptation patches. From the AvatarMix root:

```sh
uv run --no-project --python 3.10 python scripts/setup_4ddress.py
bash scripts/install_env.sh 4d-dress
```

The setup script fetches the revisions in [sources.json](sources.json), applies the patches, and places the source in `4dhumanparsing/`, including Graphonomy, RAFT, and pygco. Obtain GCO and model weights separately as described in [Installation](../docs/installation.md) and [Data preparation](../docs/data_preparation.md).

Setup stops if the destination already exists. To install another copy, pass `--output-dir /absolute/path/to/an/empty/component-root`.

## Source terms

4D-DRESS source is downloaded separately. The pinned revision has no explicit top-level code license; obtain any necessary permission from its authors before use or redistribution. AvatarMix's license does not grant rights to upstream source or derivative portions of the patches. See [Third-party notices](../THIRD_PARTY_NOTICES.md).
