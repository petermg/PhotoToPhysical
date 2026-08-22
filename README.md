# DA3 Depth → Printable 3D

A Windows/NVIDIA Gradio application for turning Depth Anything 3 depth estimates—or externally generated depth maps—into 3D-printable relief meshes.

## Features

### Single-image depth generation

- GPU-accelerated Depth Anything 3 inference
- Selectable DA3 models:
  - `depth-anything/DA3MONO-LARGE`
  - `depth-anything/DA3-LARGE-1.1`
  - `depth-anything/DA3-GIANT-1.1`
- Automatic unloading/reloading when switching DA3 models
- Adjustable DA3 internal processing resolution
- Pixel-coordinate cropping before inference
- Crop preview showing the exact image that will be sent to DA3
- Configurable depth normalization percentiles
- Raw float32 depth export as:
  - `.npy`
  - 32-bit `.exr`
- Normalized near-white and far-white depth exports as:
  - 32-bit `.exr`
  - 16-bit `.png`
  - 8-bit preview `.png`
- Confidence-map export when the selected DA3 model provides confidence:
  - raw float32 `.npy`
  - raw float32 `.exr`
  - normalized 32-bit `.exr`
  - 16-bit `.png`
  - 8-bit preview `.png`
- Per-run metadata including model, CUDA/GPU information, processing resolution, image size, normalization settings, and inference time

### Single-image mesh / relief generation

- Converts a depth map into a dense 2.5D heightfield
- Configurable mesh grid resolution
- Configurable object width / aspect-preserving height
- Adjustable relief / Z scale
- Adjustable minimum backing thickness
- Optional depth inversion
- Independent mesh-stage depth normalization, so mesh shaping can be changed without rerunning the neural network
- Closed perimeter walls and flat backing for a printable solid
- Watertightness and Euler-number reporting
- Textured **OBJ + MTL + albedo** output using the original source image
- Geometry-only **STL** output for slicers / 3D printing
- Display-only **GLB** mesh preview for Gradio
- Rebuild the mesh repeatedly from an already generated depth map without rerunning DA3
- Automatic ZIP packaging of the generated job

### Depth Curves / foreground emphasis

- A depth-domain equivalent of image **Curves**
- Compresses background depth while reserving more physical Z range for foreground detail
- Adjustable foreground percentage
- Adjustable percentage of total Z range allocated to the background
- Two foreground-selection modes:
  - **Depth range** — e.g. normalized depth `0.75–1.0`
  - **Nearest pixels (percentile)** — chooses the cutoff so approximately the requested percentage of image pixels is foreground
- Smooth monotonic remapping rather than hard clipping
- Generates both pre-curve and post-curve 16-bit mesh-depth previews
- Depth Curves can be changed and the mesh rebuilt without rerunning DA3

### External depth-map support

- Bypass DA3 completely and build a mesh from an existing depth map
- Supported depth formats:
  - `.exr`
  - `.png`
  - `.npy`
  - `.tif`
  - `.tiff`
- Optional external color/albedo image for texturing the OBJ
- Useful for Depth Anything V2, edited depth maps, or output from other depth-estimation tools

### Multi-image / multi-view 3D reconstruction

- Four input modes:
  - unordered uploaded photographs
  - naturally sorted ordered image/frame sequences
  - direct video input with automatic frame extraction
  - local COLMAP datasets with known camera intrinsics/extrinsics
- Direct video sampling defaults to **1 FPS**, matching DA3's documented CLI default
- Deterministic natural filename sorting for ordered frame sets (`frame2` before `frame10`)
- Optional view cap with an **18-view default**, matching DA3's documented 2–18-view training regime at the 504 base resolution; larger sets are sampled evenly
- Multi-view model choices:
  - `depth-anything/DA3-GIANT-1.1`
  - `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
  - `depth-anything/DA3-LARGE-1.1`
- DA3 multi-view processing-resolution presets, with **504** as the recommended starting point
- **High-quality ray-based pose estimation** enabled by default for pose-free input (`use_ray_pose=True`)
- Fast camera-decoder pose estimation remains available by disabling ray pose
- Automatic reference-view selection by input type:
  - `middle` for ordered sequences / video frames
  - `saddle_balanced` for general unordered photos
- Manual reference-view strategies remain available:
  - `saddle_balanced`
  - `saddle_sim_range`
  - `middle`
  - `first`
- **COLMAP pose-conditioned depth** mode using DA3's own COLMAP loader
- Optional alignment of DA3 depth to known COLMAP camera scale
- Configurable DA3 confidence filtering
- Configurable maximum GLB point count
- DA3 fused `scene.glb` export
- Raw DA3 reconstruction arrays (`depth`, confidence, intrinsics, extrinsics, sky) saved for inspection/reproducibility
- Camera-pose diagnostic GLB showing DA3's point cloud plus camera centers and forward directions
- Camera matrices saved to JSON
- Input-order manifest showing the exact image sequence sent to DA3
- **TSDF fusion (recommended)** that directly integrates DA3 depth maps, confidence, intrinsics, and extrinsics
- Configurable TSDF voxel detail, signed-distance truncation, and far-depth outlier percentile
- Automatic DA3 sky masking when available
- Existing **Poisson surface reconstruction** retained as an alternate method
- Controlled **A/B: TSDF + Poisson** mode that runs both meshers from the exact same DA3 inference
- Poisson point-cloud voxel downsampling, normal estimation/orientation, reconstruction depth, and density trimming controls
- Shared Taubin smoothing, triangle-count reduction/decimation, and largest-component cleanup
- Multi-view mesh export as `.obj`, `.stl`, `.ply`, plus neutral double-sided preview `.glb` files
- Multi-view metadata, quality statistics, watertightness reporting, and downloadable ZIP package

### GUI / workflow conveniences

- Local Gradio web UI
- CUDA/GPU status reporting
- Model inference timing
- VRAM reporting
- 2D depth previews
- Interactive 3D mesh previews
- Separate **Depth**, **3D / Print**, **Multi-Image 3D**, and information workflows
- Output files exposed directly in the UI
- Downloadable ZIP packages
- Generated jobs stored separately under `gradio_outputs/`

> **Preview-orientation note:** the single-image option **“Correct horizontal mirroring in Gradio preview”** affects only the display-only preview GLB; it never changes the OBJ, STL, source texture, or depth maps. In the current tested setup, leaving this option **unchecked** gives the correct orientation for both the Gradio preview and exported OBJ.

> **Tested platform:** Windows + NVIDIA CUDA GPU

---

## Quick install

### Known-good environment

The working development environment used:

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

Depth Anything 3 was installed from the official ByteDance-Seed repository at this pinned revision:

```text
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
```

### 1. Clone this repository

```cmd
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>
```

### 2. Create a virtual environment

For maximum reproducibility, Python **3.10.6** is the tested version.

```cmd
py -3.10 -m venv env
env\Scripts\activate
```

PowerShell users can activate with:

```powershell
.\env\Scripts\Activate.ps1
```

### 3. Upgrade packaging tools

```cmd
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install the tested CUDA PyTorch stack

```cmd
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install xFormers:

```cmd
python -m pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install the pinned Depth Anything 3 revision

```cmd
python -m pip install "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
```

### 6. Install this app's dependencies

```cmd
python -m pip install -r requirements-app.txt
```

### 7. Verify CUDA

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Optional import check:

```cmd
python -c "import gradio, cv2, trimesh, open3d, xformers; from depth_anything_3.api import DepthAnything3; print('All required imports OK')"
```

### 8. Run the GUI

```cmd
python da3_print_gui_v2.py
```

The application should open in your browser.

---

## Detailed Windows / NVIDIA installation notes

The quick-start above is enough for a normal setup.

For a more detailed explanation of:

- manual DA3 cloning
- reproducibility
- troubleshooting
- CUDA verification
- output directories
- model downloads
- dependency notes

see:

**[`INSTALL_WINDOWS_NVIDIA.txt`](INSTALL_WINDOWS_NVIDIA.txt)**

The repository also includes:

- **[`requirements-app.txt`](requirements-app.txt)** — direct application dependencies
- **[`KNOWN_GOOD_ENVIRONMENT.txt`](KNOWN_GOOD_ENVIRONMENT.txt)** — tested environment snapshot
- **[`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt)** — upstream/model licensing notes

---

# Single-image workflow

The single-image workflow is intended for producing a 2.5D relief / heightfield from one image.

Typical flow:

```text
Image
  ↓
Optional crop
  ↓
DA3 depth inference
  ↓
Depth normalization
  ↓
Optional Depth Curves remapping
  ↓
Heightfield generation
  ↓
Flat backing + perimeter walls
  ↓
OBJ / STL
```

## Recommended model

For normal single-image depth-to-relief work, start with:

```text
depth-anything/DA3MONO-LARGE
```

In testing, the dedicated monocular model has often produced smoother, cleaner heightfields than the larger any-view models.

## Single-image controls

The mesh can be reshaped after depth generation without rerunning DA3. Available controls include:

- mesh grid maximum dimension
- object width
- relief / Z scale
- minimum backing thickness
- depth inversion
- low/high mesh normalization percentiles
- Depth Curves / foreground emphasis
- display-only Gradio preview orientation correction

The source image is retained as the OBJ albedo/texture when the normal DA3 single-image workflow is used.

---

# Depth Curves / foreground emphasis

The Depth Curves controls remap normalized depth before Z displacement.

This is useful when you want to preserve the full scene but devote much more of the available physical relief to foreground detail.

For example:

```text
Nearest 25% of depth  → 85% of the available Z range
Remaining 75%         → 15% of the available Z range
```

The background does not have to become flat—it can simply be compressed.

A useful starting point is:

```text
Enable depth curve:       ON
Foreground:               25%
Background Z allocation:  15%
```

Increase the background allocation for a gentler effect; reduce it for stronger foreground emphasis.

Depth Curves operate during mesh generation, so you can rebuild the mesh repeatedly without rerunning DA3.

---

# External depth maps

You can bypass DA3 inference and build a mesh directly from an existing depth map.

Supported external depth formats include:

```text
.exr
.png
.npy
.tif
.tiff
```

This is useful for comparing:

- Depth Anything V2
- Depth Anything 3
- other depth-estimation tools
- manually edited depth maps

An optional color/albedo image can also be supplied for OBJ texturing.

---

# Multi-image 3D reconstruction

The multi-image workflow is intended for multiple views of the same subject or scene.

It is especially useful for:

- frames extracted from a video
- a camera moving around a stationary subject
- multiple overlapping photographs of the same scene

The larger DA3 any-view models are the appropriate models for this workflow:

```text
depth-anything/DA3-LARGE-1.1
depth-anything/DA3-GIANT-1.1
depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

A recommended first quality test is:

```text
Model: DA3-GIANT-1.1
Process resolution: 504
Ray pose: ON
Reconstruction: TSDF
Confidence percentile: 40
```

For ordered frames and direct video input, the app automatically uses DA3's `middle` reference strategy by default. For unordered photo sets it uses `saddle_balanced`. You can disable automatic reference selection and manually choose `saddle_balanced`, `saddle_sim_range`, `middle`, or `first`.

Direct video input defaults to **1 FPS**, matching the upstream DA3 CLI. The app also defaults to a maximum of 18 selected views and evenly samples larger sequences.

Higher processing resolution is not automatically better for these models. Start at **504** before experimenting with 756 or 1008.

## Ray pose and known camera poses

For pose-free images, **ray-based pose estimation** is enabled by default. DA3 documents this mode as somewhat slower but generally more accurate than the camera decoder. Disable it when you specifically want the faster camera-head path for comparison.

For maximum control, choose **COLMAP local dataset (known poses)** and point the GUI at a standard COLMAP directory containing `images/` and `sparse/`. The app uses DA3's own COLMAP loader and supplies those intrinsics/extrinsics to DA3 for pose-conditioned depth estimation.

## TSDF vs Poisson

**TSDF is now the recommended reconstruction method.** It integrates DA3's organized depth maps together with confidence, camera intrinsics, and world-to-camera extrinsics. This preserves the camera-ray structure that is discarded when depth is first flattened into a generic point cloud.

Poisson reconstruction is still included because it can work well on some point clouds and provides a valuable comparison. Select **A/B: TSDF + Poisson** to run both methods from one DA3 inference. The app produces separate meshes/previews and reports geometry statistics for both.

## Multi-image quality controls

The UI exposes controls for:

- direct-video sampling FPS
- maximum view count
- ray-pose quality mode
- automatic/manual reference-view strategy
- COLMAP scale alignment
- DA3 confidence-filter percentile
- maximum GLB point count
- TSDF voxel detail
- TSDF signed-distance truncation
- TSDF far-depth truncation percentile
- Poisson point-cloud voxel downsampling
- Poisson reconstruction depth
- low-density Poisson trimming
- Taubin smoothing iterations
- target triangle count / full-resolution output
- keeping only the largest connected component

The app saves `scene.glb`, DA3's raw reconstruction arrays, exact input order, camera matrices, and `camera_pose_diagnostic.glb`. The diagnostic GLB is particularly useful for deciding whether a bad mesh originates in DA3's estimated camera path or in the selected surface-reconstruction method.

---

# Output files

Outputs are written under:

```text
gradio_outputs/
```

Depending on the workflow, jobs may contain files such as:

```text
input_used.png

depth_raw_float32.npy
depth_raw_float32.exr
depth_near_white_norm32.exr
depth_far_white_norm32.exr
depth_near_white_16bit.png
depth_far_white_16bit.png
depth_near_white_preview_8bit.png
depth_far_white_preview_8bit.png

confidence_raw_float32.npy
confidence_raw_float32.exr
confidence_norm32.exr
confidence_16bit.png
confidence_preview_8bit.png

mesh_depth_precurve_preview_16bit.png
mesh_depth_preview_16bit.png
heightfield_solid.obj
heightfield_solid.mtl
heightfield_solid.stl
heightfield_solid_preview.glb
albedo.png
mesh_metadata.json
depth_metadata.json

da3_multiview/scene.glb
da3_prediction_for_reconstruction.npz
multiview_input_order.txt
camera_poses.json
camera_pose_diagnostic.glb

multiview_mesh.obj
multiview_mesh.stl
multiview_mesh.ply
multiview_mesh_preview.glb

# In A/B mode:
multiview_tsdf.obj
multiview_tsdf.stl
multiview_tsdf.ply
multiview_tsdf_preview.glb
multiview_tsdf_surface_point_cloud.ply
multiview_poisson.obj
multiview_poisson.stl
multiview_poisson.ply
multiview_poisson_preview.glb
multiview_poisson_point_cloud_raw.ply
multiview_metadata.json
```

Not every file is produced by every model or workflow—for example, confidence files are only written when DA3 provides confidence output.

Generated output directories should normally **not** be committed to Git.

---

# Model downloads

Model weights are downloaded from Hugging Face the first time each model is used.

Internet access is therefore required for first use unless the model is already present in the local Hugging Face cache.

The model weights are **not included in this repository**.

---

# Licensing

This repository should contain a license for the code in **this project**.

Depth Anything 3 and its model checkpoints are third-party software/assets and have their own licenses.

Model checkpoint licenses may differ from one model to another. Review the upstream Depth Anything 3 repository and each Hugging Face model card before redistributing model weights or using a model in a commercial context.

See:

**[`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt)**

---

# Troubleshooting

If CUDA is not detected:

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

You can also collect:

```cmd
python --version
python -m pip freeze
nvidia-smi
git --version
```

The known-good development environment used Python **3.10.6** and PyTorch **2.10.0+cu128**.

---

# Repository hygiene

Recommended `.gitignore` entries include:

```text
env/
.venv/
__pycache__/
gradio_outputs/
*.pyc
```

Do not commit:

- Python virtual environments
- Hugging Face model caches
- generated mesh/depth jobs
- local temporary files

---

# Acknowledgements

This project uses **Depth Anything 3** by ByteDance-Seed:

https://github.com/ByteDance-Seed/Depth-Anything-3

The tested environment pins the upstream repository to:

```text
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
```
