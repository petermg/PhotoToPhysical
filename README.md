# PhotoToPhysical

**PhotoToPhysical** is a Windows/NVIDIA Gradio application for turning photos, depth maps, multi-view image sets, and video into 3D-printable meshes, fused 3D reconstructions, and native Depth Anything 3 Gaussian splats.

The project uses **Depth Anything 3 (DA3)** as the core geometry estimator and adds practical downstream workflows for:

- single-image depth-to-relief generation,
- external depth-map meshing,
- multi-image / video reconstruction,
- several interchangeable mesh backends,
- native DA3 Gaussian Splat output,
- camera-pose diagnostics,
- printable OBJ/STL export.

> **Primary tested platform:** Windows 11 + NVIDIA CUDA GPU.

---

## What PhotoToPhysical can produce

### Single image

From one image, PhotoToPhysical can generate:

- DA3 depth maps,
- confidence maps when available,
- a closed 2.5D heightfield solid,
- textured OBJ + MTL + albedo,
- STL for slicers / 3D printing,
- display-only GLB preview,
- metadata and ZIP packaging.

### Multi-image / video

From multiple overlapping views or video frames, PhotoToPhysical can generate:

- DA3 multi-view depth,
- DA3 camera intrinsics and extrinsics,
- DA3 fused point-cloud GLB,
- raw DA3 reconstruction arrays,
- camera-pose diagnostic GLB,
- several alternative triangle-mesh reconstructions,
- **native DA3 3D Gaussian Splat `.ply` output** on supported models,
- optional Gaussian preview video when `gsplat` is installed,
- OBJ / STL / PLY / GLB mesh exports,
- metadata, backend logs, and comparison outputs.

---

# Features

## Single-image depth generation

Selectable DA3 models:

- `depth-anything/DA3MONO-LARGE`
- `depth-anything/DA3-LARGE-1.1`
- `depth-anything/DA3-GIANT-1.1`
- `depth-anything/DA3NESTED-GIANT-LARGE-1.1`

Features include:

- CUDA-accelerated inference,
- automatic unload/reload when changing models,
- adjustable DA3 process resolution,
- pixel-coordinate crop controls,
- exact crop preview before inference,
- configurable depth normalization,
- raw float32 `.npy` and 32-bit `.exr`,
- normalized 32-bit EXR, 16-bit PNG, and 8-bit preview outputs,
- confidence-map export when supplied by the selected DA3 model,
- timing, GPU, CUDA, and metadata reporting.

For normal single-image depth-to-relief work, start with:

```text
depth-anything/DA3MONO-LARGE
```

In current testing, the dedicated monocular model has generally produced cleaner heightfields than the any-view models for this specific single-image relief workflow.

---

## Single-image printable relief / heightfield

The mesh stage converts normalized depth into a closed solid with:

- configurable grid density,
- configurable object width,
- aspect-preserving height,
- adjustable Z / relief scale,
- adjustable flat backing thickness,
- optional depth inversion,
- independent mesh-stage normalization,
- perimeter walls,
- flat back,
- watertightness and Euler-number reporting,
- OBJ / MTL / albedo export,
- STL export,
- Gradio GLB preview.

The mesh can be rebuilt repeatedly without rerunning DA3.

### Preview orientation note

The single-image control **Correct horizontal mirroring in Gradio preview** is display-only and does not alter OBJ/STL/depth files. In the current tested setup, leaving this option **unchecked** gives the correct orientation for both the Gradio preview and exported OBJ.

---

## Depth Curves / foreground emphasis

The Depth Curve controls remap normalized depth before Z displacement. This lets a small foreground portion use most of the physical relief range while compressing the background.

Example:

```text
Nearest 25% of depth  -> 85% of Z range
Remaining 75%         -> 15% of Z range
```

Available controls include:

- foreground percentage,
- background Z allocation,
- depth-range or nearest-pixels selection,
- smooth monotonic remapping,
- before/after depth previews.

Depth Curves operate during mesh generation, so they can be changed without rerunning DA3.

---

## External depth-map support

DA3 can be bypassed completely and the printable relief can be built from an existing depth map.

Supported formats:

```text
.exr
.png
.npy
.tif
.tiff
```

An optional external image can be supplied as OBJ color/albedo.

This is useful for comparing DA3 with Depth Anything V2, edited depth maps, or other depth-estimation tools.

---

# Multi-Image 3D

The Multi-Image 3D workflow is for overlapping views of the same subject or scene.

Supported input modes:

1. unordered uploaded photos,
2. ordered uploaded frames,
3. direct video,
4. local COLMAP datasets with known camera intrinsics/extrinsics.

Recommended DA3 models:

```text
depth-anything/DA3-GIANT-1.1
depth-anything/DA3NESTED-GIANT-LARGE-1.1
depth-anything/DA3-LARGE-1.1
```

Start at **process resolution 504** for the any-view models before experimenting with larger values.

## Video sampling FPS vs Maximum Views

These are separate controls:

- **Video sampling FPS** = how many frames per second PhotoToPhysical extracts from the source video for DA3. It is **not** asking for the native frame rate of the video.
- **Maximum views** = the maximum number of those sampled frames actually sent into DA3.

Examples for a 36-second video:

| Video sampling FPS | Maximum views | Approximate views sent to DA3 |
|---:|---:|---:|
| 1 | 18 | 18 |
| 1 | 0 | 36 |
| 2 | 18 | 18 |
| 2 | 0 | 72 |
| 5 | 0 | 180 |
| 30 | 0 | ~1080 |

`Maximum views = 0` means **no PhotoToPhysical view cap**.

The default of 18 is a conservative starting value based on DA3's documented 2–18-view training regime at the 504 base resolution. It is **not a hard DA3 inference limit**.

For normal video reconstruction, start around **1–2 sampling FPS** rather than feeding every native video frame. Very high frame rates create many nearly identical views and can dramatically increase memory and compute cost.

---

## Pose estimation and reference view

For pose-free inputs, PhotoToPhysical enables DA3's ray-based pose path by default.

Automatic reference-view strategy:

- ordered frames / video -> `middle`
- unordered photographs -> `saddle_balanced`

Manual choices remain available:

- `saddle_balanced`
- `saddle_sim_range`
- `middle`
- `first`

COLMAP input can supply known intrinsics/extrinsics directly. The optional scale-alignment control lets DA3 depth align to the known external camera scale.

---

# Multi-view mesh backends

PhotoToPhysical currently exposes several independent surface-reconstruction paths.

## 1. TSDF

Directly fuses DA3 depth maps using DA3 confidence, intrinsics, extrinsics, and optional sky masking.

Controls include:

- voxel detail,
- signed-distance truncation,
- far-depth percentile,
- shared smoothing,
- decimation,
- largest-component cleanup.

TSDF is useful as a direct, camera-aware depth-fusion baseline.

## 2. Poisson

Uses DA3's exported fused GLB point cloud and reconstructs a Screened Poisson surface after point cleanup and normal estimation.

This path is useful as a generic point-cloud-to-surface comparison.

## 3. Consistent Surfel + Poisson

This DA3-aware path uses the organized multi-view data rather than treating DA3 output as an anonymous point cloud.

It:

1. back-projects DA3 depth pixels into 3D,
2. calculates normals from the organized depth maps,
3. transforms positions and normals through DA3 cameras,
4. reprojects observations into neighboring views,
5. rejects depth-inconsistent samples,
6. optionally checks normal agreement,
7. requires configurable multi-view support,
8. confidence-weights and merges redundant surfels,
9. runs Screened Poisson on the cleaned oriented surfel cloud.

The status output reports sampled observations, accepted observations, acceptance percentage, merged surfels, and median supporting views.

## 4. OpenMVS

Optional external backend.

PhotoToPhysical exports the DA3 camera solution through DA3's COLMAP exporter and can invoke:

```text
InterfaceCOLMAP
DensifyPointCloud
ReconstructMesh
```

## 5. OpenMVS + RefineMesh

Runs the same OpenMVS geometric reconstruction and then uses image-based `RefineMesh` to refine the surface against the original views.

## 6. 2DGS Neural Surface

Optional external neural backend using the official 2D Gaussian Splatting implementation.

PhotoToPhysical supplies DA3 cameras and a high-confidence initialization in COLMAP form, then launches the external 2DGS training and mesh-extraction workflow in its own Python/CUDA environment.

See **[`INSTALL_MULTIVIEW_BACKENDS.md`](INSTALL_MULTIVIEW_BACKENDS.md)** for setup details.

---

# Native DA3 Gaussian Splat output

PhotoToPhysical can export DA3's own learned Gaussian representation **in addition to a mesh**.

Supported models:

```text
depth-anything/DA3-GIANT-1.1
depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

In **Multi-Image 3D -> Additional DA3 outputs**:

- enable **Export native DA3 Gaussian Splat .ply**,
- optionally enable **Also export DA3 Gaussian preview video**.

## Gaussian PLY

The `.ply` is a **3D Gaussian Splat file**, not a normal triangle mesh and not an ordinary XYZ point-cloud PLY. It contains Gaussian attributes such as position, scale, rotation, opacity, and appearance coefficients.

Use a Gaussian-splat viewer such as **SuperSplat** or another viewer that supports standard 3DGS PLY files.

The native Gaussian `.ply` export itself does **not** require the `gsplat` renderer package.

## Gaussian preview video

DA3's `gs_video` output rasterizes the predicted splats and therefore requires the pinned `gsplat` dependency.

Install it in the PhotoToPhysical/DA3 environment with:

```cmd
python -m pip install ninja
python -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

The `--no-build-isolation` flag is important because this pinned `gsplat` build imports PyTorch during extension setup. On Windows, source compilation may also require Visual Studio C++ Build Tools and a compatible CUDA development toolkit.

If `gsplat` is not installed, leave **Gaussian preview video** off and export only the Gaussian PLY.

### Current implementation note

The current PhotoToPhysical script performs the normal DA3 multi-view inference and, when native Gaussian export is requested, performs a second DA3 inference with `infer_gs=True`. Large uncapped video sets therefore cost substantially more time and memory when Gaussian export is enabled.

---

# Gaussian splats vs meshes

They solve different problems.

### Gaussian splats are well suited for:

- visually rich static scenes,
- rooms,
- streets,
- buildings,
- environments captured by a moving camera,
- interactive novel-view visualization.

### Triangle meshes are still required for:

- 3D printing,
- slicers,
- watertight physical objects,
- conventional mesh editing / CAD workflows.

For a static video walkthrough of an environment, the native DA3 Gaussian output may preserve visual quality better than forcing the entire reconstruction into a single triangle surface.

---

# Recommended first tests

## Multi-view mesh comparison

```text
Model:                  DA3-GIANT-1.1
Process resolution:     504
Ray pose:               ON
Confidence percentile:  40
Methods:                 TSDF + Poisson + Consistent Surfel
Smoothing:               0 for diagnostic comparison
Target triangles:        0 for diagnostic comparison
Largest component only:  OFF for diagnostic comparison
```

Keep the same source images for all methods so the mesher is the only variable.

## Native Gaussian scene/video test

```text
Model:                  DA3-GIANT-1.1
Video sampling FPS:     1 or 2
Maximum views:           0 for an uncapped experiment
Process resolution:      504
Ray pose:                ON
Native Gaussian PLY:     ON
Gaussian preview video:  OFF unless gsplat is installed
```

For a 30–60 second moving-camera clip, 1–2 FPS is a good first experiment before increasing frame density.

For a detailed testing sequence, see **[`MULTIVIEW_TESTING.md`](MULTIVIEW_TESTING.md)**.

---

# Quick install — Windows / NVIDIA

## Known-good core environment

```text
Python 3.10.6
PyTorch 2.10.0+cu128
torchvision 0.25.0+cu128
xformers 0.0.35
Gradio 6.25.0
NumPy 1.26.4
Open3D 0.19.0
OpenCV 4.11.0.86
Pillow 12.3.0
trimesh 5.0.0
```

Pinned Depth Anything 3 revision:

```text
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
```

## 1. Clone PhotoToPhysical

```cmd
git clone https://github.com/petermg/PhotoToPhysical.git
cd PhotoToPhysical
```

## 2. Create and activate the tested Python environment

```cmd
py -3.10 -m venv env
env\Scripts\activate
```

## 3. Upgrade packaging tools

```cmd
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install CUDA PyTorch and xFormers

```cmd
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
```

## 5. Install the pinned DA3 revision

```cmd
python -m pip install "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
```

## 6. Install PhotoToPhysical direct dependencies

```cmd
python -m pip install -r requirements-app.txt
```

## 7. Verify CUDA

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

## 8. Run

```cmd
python da3_print_gui_v2.py
```

For detailed setup, optional Gaussian rendering, and troubleshooting, see **[`INSTALL_WINDOWS_NVIDIA.txt`](INSTALL_WINDOWS_NVIDIA.txt)**.

For OpenMVS and 2DGS, see **[`INSTALL_MULTIVIEW_BACKENDS.md`](INSTALL_MULTIVIEW_BACKENDS.md)**.

---

# Output files

All generated jobs are written beneath:

```text
gradio_outputs/
```

Typical single-image outputs include:

```text
input_used.png
depth_raw_float32.npy
depth_raw_float32.exr
depth_near_white_norm32.exr
depth_far_white_norm32.exr
depth_near_white_16bit.png
depth_far_white_16bit.png
confidence_raw_float32.npy
confidence_raw_float32.exr
heightfield_solid.obj
heightfield_solid.mtl
heightfield_solid.stl
heightfield_solid_preview.glb
albedo.png
mesh_metadata.json
depth_metadata.json
```

Typical multi-view outputs include:

```text
da3_multiview/scene.glb
da3_prediction_for_reconstruction.npz
multiview_input_order.txt
camera_poses.json
camera_pose_diagnostic.glb
multiview_metadata.json

multiview_tsdf.*
multiview_poisson.*
multiview_consistent_surfel.*
multiview_openmvs.*
multiview_openmvs_refined.*
multiview_2dgs.*
```

When native DA3 Gaussian output is enabled, the job also includes the exported Gaussian `.ply`, and optionally a rasterized Gaussian video.

External OpenMVS / 2DGS workspaces can be large and are intentionally not packed into the compact downloadable ZIP; final imported meshes and backend logs are included.

Generated output directories should normally not be committed to Git.

---

# Troubleshooting

## CUDA not detected

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Collect these when reporting environment problems:

```cmd
python --version
python -m pip freeze
nvidia-smi
git --version
```

## Gaussian preview video crashes with `NameError: rasterization is not defined`

That means DA3 attempted `gs_video` rendering without the optional `gsplat` renderer successfully installed.

Either:

1. turn off **Also export DA3 Gaussian preview video** and keep Gaussian PLY enabled, or
2. install the pinned renderer:

```cmd
python -m pip install ninja
python -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

If the build then fails on `cl.exe`, MSVC, or CUDA compilation, install/repair the appropriate Visual Studio Build Tools and CUDA development toolchain rather than changing the known-good PyTorch stack at random.

---

# Repository hygiene

Recommended ignores:

```text
env/
.venv/
__pycache__/
gradio_outputs/
*.pyc
```

Do not commit:

- virtual environments,
- Hugging Face model caches,
- generated mesh/depth/splat jobs,
- large OpenMVS / 2DGS workspaces,
- local temporary files.

---

# Licensing

PhotoToPhysical uses multiple third-party projects and model checkpoints with different licenses.

Important examples:

- Depth Anything 3 source: Apache-2.0
- DA3MONO-LARGE checkpoint: Apache-2.0 according to the upstream model table at the time documented
- DA3-LARGE-1.1 / DA3-GIANT-1.1 / DA3NESTED-GIANT-LARGE-1.1 checkpoints: CC BY-NC 4.0 according to the upstream model table at the time documented
- `gsplat`: Apache-2.0
- OpenMVS: GNU AGPL v3
- official 2DGS implementation: Gaussian-Splatting research license / non-commercial research and evaluation terms

Review upstream licenses before redistribution or commercial use.

See **[`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt)**.

---

# Acknowledgements

Depth Anything 3 by ByteDance-Seed:

https://github.com/ByteDance-Seed/Depth-Anything-3

PhotoToPhysical currently pins DA3 to:

```text
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
```
