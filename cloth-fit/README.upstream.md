# [Siggraph 2025] Intersection-free Garment Retargeting

> Upstream documentation with retargeting extensions. For AvatarMix setup and usage, see [README.md](README.md).
[![Build](https://github.com/Huangzizhou/cloth-fit/actions/workflows/continuous.yml/badge.svg)](https://github.com/Huangzizhou/cloth-fit/actions/workflows/continuous.yml)

This is the opensource reference implementation of the SIGGRAPH 2025 paper [Intersection-free Garment Retargeting](https://huangzizhou.github.io/assets/img/research/cloth/paper.pdf). This code is modified based on [PolyFEM](https://github.com/polyfem/polyfem). It uses the nonlinear optimizers, linear solvers, and collision handling in [PolyFEM](https://github.com/polyfem/polyfem).

![Output sample](./garment-data/output.gif)

## Files

* `src/`: source code
* `cmake/` and `CMakeLists.txt`: CMake files
* `json-specs/`: input JSON configuration specifications
* `garment-data/`: input data and scripts needed to run the code
* `tests/`: unit-tests

## Build

### Dependencies

- use conda to install gxx=11.3.0, gcc=11.3.0, boost-cpp=1.83, tbb, tbb-devel, blosc, zlib, (cmake)


The code can be compiled with
```
mkdir build
# linux
cd build
# remember to enable tensorboard and threading (TBB or CPP) at cmake config if needed
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"  -DPOLYFEM_WITH_TENSORBOARD=ON -DPOLYFEM_THREADING=TBB ..
make -j4
# windows
# copy .dll files from build/_deps/openvdb-build to build/Release/
cmake -S . -B build --config Release -DOPENVDB_CORE_SHARED=OFF
cmake --build build --config Release -j 8 -- /maxcpucount:8
cmake --build build --config Release --target unit_tests -j 8
```
gcc 7 or later is recommended.


## Run

The folder `garment-data` contains four basic examples of retargeting garments to target avatars: `foxgirl_skirt`, `Goblin_Jacket`, `Goblin_Jumpsuit`, and `Trex_Jacket`. To run the examples, e.g. for `foxgirl_skirt`,
```
cd garment-data/foxgirl_skirt
../../build/PolyFEM_bin -j setup.json --max_threads 16 > log
```

### Optional: Per-vertex fit weighting
You can locally tighten or loosen the final shrink stage by providing `fit_weight_masks` in your setup json. Each entry specifies a set of garment vertex indices (0-based) and a `multiplier` applied on top of the global `fit_weight`.

Example:

```json
{
  "fit_weight": 5.0,
  "fit_weight_masks": [
    { "indices_path": "tight_hem.txt", "multiplier": 2.0 },
    { "indices": [123,124,125], "multiplier": 0.2 }
  ]
}
```

- Larger than 1.0 tightens locally; 1.0 keeps default; 0.0 disables shrinking at those vertices.
- If the same vertex appears in multiple masks, the last entry wins.
- Indices refer to the garment mesh vertex IDs (not avatar or collision indices).
- This complements `no_fit_spec_path` (face-level exclusion). Faces fully listed in `no_fit_spec_path` are excluded first, then per-vertex multipliers act on remaining faces.

## Output Files
* `step_avatar_<n>.obj`: optimization sequence of the target avatar geometry
* `step_garment_<n>.obj`: optimization sequence of the retargeted garment surface
* `sdf.obj`: avatar surface converted from the signed distance field
* `source_skeleton.obj`: skeleton of the source garment
* `target_skeleton.obj`: skeleton of the target avatar
* `target_avatar.obj`: target avatar geometry
* `projected_avatar.obj`: projected target avatar on its skeleton

## Script Settings

The specification of each JSON configuration is described in `json-specs/input-spec.json`.

### Meshes

* `avatar_mesh_path`: the target avatar mesh
* `garment_mesh_path`: the source garment mesh
* `source_skeleton_path`: the skeleton edge mesh of the source garment in `.obj` format
* `target_skeleton_path`: the skeleton edge mesh of the target avatar in `.obj` format
* `avatar_skin_weights_path`: the skinning weights of the target avatar, a matrix of size `#number_of_skeleton_nodes` times `#number_of_vertices` (optional)

For avatars and garments, only `.obj` triangular mesh is supported. The source and target skeletons should share the same mesh connectivity, and skeleton joints should be ordered in the same way. The skinning weights will be used to project the target avatar to its skeleton. If `avatar_skin_weights_path` is not provided, the vertices will be simply projected to the closest bone in distance. Ideally, the source and target skeletons are in the same pose, so that the difference in pose does not cause unecessary geometric distortion on the garment.

### Objectives and Weights

There are multiple objectives used in the method, each has its own weight:

* `similarity_penalty_weight`: $L_\text{surf}$ preserves the surface shape, always set to 1
* `curvature_penalty_weight`: $L_\text{curvature}$ preserves the curve curvature
* `twist_penalty_weight`: $L_\text{torsion}$ preserves the curve torsion
* `curve_center_target_weight`: $L_\text{pos}$ preserves the relative position of curve loops (e.g. hem, cuff)
* `solver/contact/barrier_stiffness`: $L_\text{contact}$ avoids contact by adding a barrier

The choice of `barrier_stiffness` depends on the mesh resolution, geometry scale, and the specific problem setup. The general rule is that when the min distance reported in the log is much smaller than `dhat` (say 1/100 of `dhat`), then it should probably be increased.

### Optimizer

The nonlinear optimizer lives in a separate Github repository called [PolySolve](https://github.com/Huangzizhou/polysolve/tree/garment). The optimizer parameters are specified under JSON entry `solver/nonlinear`, whose specification is in [nonlinear-solver-spec.json](https://github.com/Huangzizhou/polysolve/blob/garment/nonlinear-solver-spec.json).
