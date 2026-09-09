# Installation

Run commands from the AvatarMix repository root. Use a Linux machine with an NVIDIA GPU for CUDA builds and execution. `uv`, Git, CMake, a C++ compiler, and the CUDA toolkit must be available. Rendering raw scans additionally requires OpenGL/EGL support from the NVIDIA driver.

Each component uses its own environment.

| Component | Python | PyTorch / toolkit family | Notes |
| --- | --- | --- | --- |
| Composition, Gaussian avatars, refiners, parsing, NeuS2 | 3.10 | PyTorch 2.7.1 / CUDA 12.8 | CUDA extensions build from source |
| THuman renderer | 3.10 | PyTorch 2.4.1 / CUDA 12.4 | Matches its source requirements; needs EGL for headless rendering |
| Skinning-weight transfer | 3.10 | CPU | NumPy and libigl |

Dependency versions are defined in each component's `pyproject.toml` and `uv.lock`.

## Python environments

Check the toolchain before building:

```sh
nvidia-smi
nvcc --version
gcc --version
cmake --version
uv --version
```

Set `CUDA_HOME` to the toolkit matching the component you are building. For example, in fish:

```fish
set -gx CUDA_HOME /usr/local/cuda-12.8
fish_add_path $CUDA_HOME/bin
```

For semantic parsing, first read [the upstream-source notice](../4d-dress/README.md), then fetch the pinned sources and apply the supplied compatibility patches:

```sh
uv run --no-project --python 3.10 python scripts/setup_4ddress.py
```

This downloads the parsing source and applies the required patches. If the destination already exists, setup stops without replacing it. Model weights are installed separately.

Install the components you need:

```sh
bash scripts/install_env.sh swapvton
bash scripts/install_env.sh splatting
bash scripts/install_env.sh difix
bash scripts/install_env.sh 4d-dress
bash scripts/install_env.sh neus2
bash scripts/install_env.sh lbs_transfer
```

For raw-scan rendering, select a CUDA 12.4 toolchain and run:

```sh
bash scripts/install_env.sh external_tools/thuman2_render
```

The installer installs PyTorch and build prerequisites into the component's `.venv`, then runs `uv sync`. Install the system prerequisites listed above first. Use `MAX_JOBS=4` to limit parallel compiler processes if needed.

PyTorch3D and tiny-cuda-nn use fixed source revisions. The modified rasterizer, nearest-neighbor extension, Phong surface binding, and `reshape_ops` are included as local source dependencies. Build them on the machine where they will run; do not copy a `.venv` or a binary wheel across machines with different CUDA, Python, or C++ runtimes.

SAM3 is installed with the parsing environment.

## Native tools

Build NeuS2 without a GUI or DLSS:

```sh
cmake -S neus2 -B neus2/build -DCMAKE_BUILD_TYPE=Release \
  -DNGP_BUILD_WITH_GUI=OFF -DNGP_BUILD_WITH_VULKAN=OFF -DNGP_BUILD_WITH_OPTIX=OFF \
  -DPython_EXECUTABLE="$PWD/neus2/.venv/bin/python"
cmake --build neus2/build --parallel 4
```

Use the Python interpreter from the NeuS2 environment when configuring its bindings. If CMake detects another Python, check its configure output before proceeding. GUI/DLSS binary dependencies are not bundled in this release.

Build the retargeting solver:

```sh
cmake -S cloth-fit -B cloth-fit/build -DCMAKE_BUILD_TYPE=Release -DPOLYFEM_WITH_TESTS=OFF
cmake --build cloth-fit/build --target PolyFEM_bin --parallel 4
```

The pipeline expects `cloth-fit/build/PolyFEM_bin`. CMake fetches the solver's upstream dependencies during configuration.

The parsing component uses pygco. Obtain GCO 3.0 under its own terms, then build the wrapper as described in the [upstream parsing installation](https://github.com/eth-ait/4d-dress#model-installation):

```sh
cd 4d-dress/4dhumanparsing/lib/pygco
curl -fL https://vision.cs.uwaterloo.ca/files/gco-v3.0.zip -o gco-v3.0.zip
unzip -o gco-v3.0.zip -d gco_source
make all
cd ../../../..
```

## Check the entrypoints

These commands inspect options/configuration without starting training:

```sh
uv run --project swapvton python swapvton/src/scripts/pipeline.py --help
cd difix
uv run python scripts/train_swapfix_lora.py --cfg job experiment=seamfix
uv run python scripts/train_swapfix_lora.py --cfg job experiment=fullbodyfix
cd ..
```

Next, obtain the [external assets and prepare the data](data_preparation.md). Check GPU availability again before a render or training job, and select one GPU with `CUDA_VISIBLE_DEVICES`.

### Validation scope

Gaussian rendering/backpropagation and a FullbodyFix training step were checked with PyTorch 2.7.1+cu128 on an RTX 6000 Ada. Fresh-machine installation and the complete pipeline have not been validated.
