# Optional Multi-View / Gaussian Backends

The core PhotoToPhysical environment already supports these native mesh paths:

- TSDF
- Poisson
- Consistent Surfel + Poisson

It also supports **native DA3 Gaussian Splat PLY output** on the Gaussian-capable DA3 models without requiring an external mesh backend.

Additional optional components are kept separate so their toolchains do not destabilize the known-good DA3 environment.

---

# 1. Native DA3 Gaussian Splat output

Supported models:

```text
depth-anything/DA3-GIANT-1.1
depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

In PhotoToPhysical:

1. open **Multi-Image 3D**,
2. expand **Additional DA3 outputs**,
3. enable **Export native DA3 Gaussian Splat .ply**.

The `.ply` is a 3D Gaussian Splat representation, not a conventional mesh PLY.

A Gaussian-specific viewer is required. SuperSplat is a convenient reference viewer:

https://playcanvas.com/supersplat/editor

The PLY export itself does not require the optional `gsplat` package.

## Optional Gaussian preview video

DA3 can also rasterize the Gaussian representation into a preview video. That path requires the upstream-pinned `gsplat` renderer.

From the activated DA3/PhotoToPhysical environment:

```cmd
python -m pip install ninja
python -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

Why `--no-build-isolation`?

The pinned `gsplat` source imports torch during extension setup. Pip's isolated build environment may therefore fail with:

```text
ModuleNotFoundError: No module named 'torch'
```

even though torch is correctly installed in the active venv.

Windows source compilation may also require Visual Studio C++ Build Tools and a compatible CUDA development toolkit.

If `gsplat` is missing, leave **Gaussian preview video** off and export only the PLY.

---

# 2. OpenMVS

Upstream:

https://github.com/cdcseacave/openMVS

OpenMVS is used as an external executable pipeline. PhotoToPhysical prepares a COLMAP-formatted bridge from DA3, then invokes:

```text
InterfaceCOLMAP
DensifyPointCloud
ReconstructMesh
```

For **OpenMVS + RefineMesh**, it additionally invokes:

```text
RefineMesh
```

## Windows setup

Use a suitable prebuilt Windows package if available, or build OpenMVS according to the upstream project.

Locate the directory containing executables such as:

```text
InterfaceCOLMAP.exe
DensifyPointCloud.exe
ReconstructMesh.exe
RefineMesh.exe
```

In PhotoToPhysical:

1. open **Multi-Image 3D**,
2. expand **OpenMVS external backend**,
3. set **OpenMVS executable folder** to that directory,
4. choose **OpenMVS** or **OpenMVS + RefineMesh**.

If the field is blank, PhotoToPhysical searches `PATH`.

### Recommended first OpenMVS test

```text
DA3 model:                 DA3-GIANT-1.1
DA3 process resolution:    504
Ray pose:                  ON
OpenMVS init confidence:   95
Resolution level:          1
Neighbor views:            6
Minimum agreeing views:    2
Refine resolution level:   1
Shared smoothing:          0 while comparing
Target triangles:          0 while comparing
```

The imported result is re-exported through PhotoToPhysical as OBJ/STL/PLY/preview GLB. Backend console output is saved to a log file.

---

# 3. 2D Gaussian Splatting / 2DGS Neural Surface

Official upstream implementation:

https://github.com/hbb1/2d-gaussian-splatting

This is **not** DA3's native feed-forward Gaussian output. It is a separate scene-specific optimization/training pipeline that PhotoToPhysical can use as a neural surface backend.

Because 2DGS has a different CUDA/PyTorch extension stack, do **not** install it into the core PhotoToPhysical DA3 environment. Use a separate environment.

## Clone with submodules

```powershell
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git
cd 2d-gaussian-splatting
```

Follow the upstream installation instructions. Their documented environment is typically created with conda:

```powershell
conda env create --file environment.yml
conda activate surfel_splatting
```

Because compiled CUDA extensions are involved, exact Windows setup can depend on:

- Visual Studio build tools,
- CUDA compiler/toolkit,
- PyTorch/CUDA compatibility,
- the exact 2DGS revision.

Treat the upstream repository as authoritative for those dependencies.

## Configure PhotoToPhysical

In **Multi-Image 3D -> 2DGS neural surface backend** set:

- **2DGS repository folder** to the cloned repository,
- **2DGS Python executable** to the Python from the separate 2DGS environment.

PhotoToPhysical will:

1. export DA3 cameras and a high-confidence point initialization to COLMAP format,
2. call 2DGS `train.py`,
3. call 2DGS `render.py` mesh extraction,
4. import the resulting `fuse*.ply`,
5. export it through the normal PhotoToPhysical mesh flow.

### Recommended first 2DGS test

```text
DA3 model:                    DA3-GIANT-1.1
DA3 process resolution:       504
Ray pose:                     ON
Initialization confidence:    95
Iterations:                   7000 smoke test
                              15000 quality comparison
                              30000 full run
Depth statistic:              0 / mean
Mesh extraction resolution:   1024
Unbounded:                    OFF for compact foreground subject
Shared smoothing:             0 while comparing
Target triangles:             0 while comparing
```

2DGS may take many minutes because it optimizes the representation for the specific scene.

---

# 4. Compare selected methods

PhotoToPhysical can compare:

```text
TSDF
Poisson
Consistent Surfel + Poisson
OpenMVS
OpenMVS + RefineMesh
2DGS Neural Surface
```

The native meshers reuse the normal DA3 prediction. OpenMVS and 2DGS use the same DA3 camera solution through the COLMAP bridge and then run their own image-based stages.

For practical testing:

1. compare the three native mesh methods first,
2. test OpenMVS separately,
3. test OpenMVS + RefineMesh,
4. only then add 2DGS if its longer optimization time is justified.

Native DA3 Gaussian PLY is an **additional output**, not a mesh method, so it can be enabled alongside any selected mesh reconstruction.

---

# 5. Video capture guidance

**Video sampling FPS** means extracted frames per second, not the source video's native FPS.

**Maximum views** is applied after sampling.

For environment / walk-through Gaussian capture, start with:

```text
Sampling FPS:          1 or 2
Maximum views:         0 for uncapped experiment
DA3 model:             DA3-GIANT-1.1
Process resolution:    504
Ray pose:              ON
Native Gaussian PLY:   ON
Gaussian video:        OFF unless gsplat is installed
```

A 36-second clip at 2 FPS produces about 72 candidate views. The same clip at 30 FPS with Max 0 produces about 1080 views, which is a stress test and contains many near-duplicate frames.

---

# 6. Licensing

These optional tools retain their own licenses and are not redistributed by PhotoToPhysical:

- `gsplat`: Apache-2.0
- OpenMVS: GNU AGPL v3
- official 2DGS code: Gaussian-Splatting research license, non-commercial research/evaluation unless separately permitted

Review upstream license text before redistribution, hosted services, or commercial use.
