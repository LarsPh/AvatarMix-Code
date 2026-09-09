#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
module="${1:-}"
case "$module" in
  swapvton|splatting|difix|4d-dress|neus2) torch_version=2.7.1; vision_version=0.22.1; cuda_index=cu128 ;;
  external_tools/thuman2_render) torch_version=2.4.1; vision_version=0.19.1; cuda_index=cu124 ;;
  lbs_transfer) uv sync --directory "$release_root/lbs_transfer" --python 3.10; exit 0 ;;
  *) echo 'Usage: bash scripts/install_env.sh {swapvton|splatting|difix|4d-dress|neus2|lbs_transfer|external_tools/thuman2_render}' >&2; exit 2 ;;
esac

module_root="$release_root/$module"
if [[ "$module" == "4d-dress" && ! -d "$module_root/4dhumanparsing" ]]; then
  echo 'Fetch parsing sources first: uv run --no-project --python 3.10 python scripts/setup_4ddress.py' >&2
  exit 2
fi
if [[ ! -x "$module_root/.venv/bin/python" ]]; then
  uv venv --python 3.10 "$module_root/.venv"
fi
uv pip install --python "$module_root/.venv/bin/python" \
  "torch==$torch_version" "torchvision==$vision_version" \
  --index-url "https://download.pytorch.org/whl/$cuda_index"
uv pip install --python "$module_root/.venv/bin/python" 'setuptools<81' wheel pip packaging ninja numpy pybind11 requests
uv sync --directory "$module_root" --python 3.10
