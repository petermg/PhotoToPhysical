# PhotoToPhysical / `da3_print_gui_v2.py` — Exhaustive GUI Guide

**Scope of this guide**

This document is a field-by-field operator reference for the PhotoToPhysical GUI. It explains:

- what every visible control, button, field, tab, and viewer does;
- where each control sits in its pipeline;
- how changing the value affects output quality, speed, memory use, or behavior;
- which settings only affect preview/export formatting versus which settings change the underlying reconstruction;
- which settings force a new DA3 inference and which settings only rebuild downstream geometry;
- what each reconstruction method is actually doing.

This guide is based on the current PhotoToPhysical GUI in `da3_print_gui_v2.py`, including:

- the **single-image depth → printable solid** workflow;
- the **multi-image / video / COLMAP** workflow;
- **TSDF**, **Poisson**, **Consistent Surfel + Poisson**, **OpenMVS**, **OpenMVS + RefineMesh**, and **2DGS** mesh backends;
- **native DA3 Gaussian output** (`.ply`) and optional Gaussian preview video.

---

# 1. Big-picture mental model

PhotoToPhysical contains **two major workflows**:

## A. Single-image workflow
Purpose: turn one image and its DA3 depth map into a **closed printable relief solid** with a flat back.

Pipeline:

1. Load image.
2. Crop image if needed.
3. Run **DA3** on that image.
4. Save raw depth + normalized preview files.
5. Convert depth into a **closed heightfield solid**.
6. Export **OBJ / MTL / STL / preview GLB**.

Best for:

- ultrasound keepsakes;
- portraits or relief-style art;
- any single-image displacement-style print.

## B. Multi-image workflow
Purpose: reconstruct a subject or scene from **multiple overlapping views**.

Pipeline:

1. Collect multiple views (photos, ordered frames, sampled video, or known COLMAP dataset).
2. Run **DA3 any-view inference**.
3. Estimate depth, confidence, and camera poses (unless known COLMAP poses are supplied).
4. Build a 3D representation with one or more of the selectable methods:
   - **TSDF**
   - **Poisson**
   - **Consistent Surfel + Poisson**
   - **OpenMVS**
   - **OpenMVS + RefineMesh**
   - **2DGS**
5. Optionally export **native DA3 Gaussian splat output**.
6. Preview the result and download files / ZIP package.

Best for:

- multiple photos of an object;
- frames from a video;
- scene capture;
- testing multiple surface-reconstruction backends from one DA3 run.

---

# 2. Important foundational concepts

Before diving into individual controls, these concepts matter a lot.

## 2.1 DA3 process resolution vs mesh resolution are different things

There are **two different resolution ideas** in the app:

### DA3 process resolution
This is the resolution at which the neural network itself processes the input.

- Higher values can capture finer image detail.
- Higher values are slower and use more VRAM.
- This affects the **quality of DA3's predicted depth / geometry**.

### Mesh grid / surface resolution
This controls how dense the downstream mesh is built.

- It does **not** rerun DA3.
- It affects triangle count, mesh smoothness, and file size.
- It is a geometry-conversion setting, not a neural inference setting.

## 2.2 Some settings rerun DA3; others only rebuild geometry

This is one of the most important operational distinctions in the whole app.

### Settings that require a new DA3 inference
These change the neural prediction itself.

Examples:

- source image / crop;
- DA3 model;
- DA3 process resolution;
- multi-image input selection;
- multi-image input mode;
- video sampling FPS;
- maximum views;
- ray-pose setting;
- reference strategy;
- known COLMAP dataset path;
- native Gaussian output request.

### Settings that usually only rebuild geometry / exports
These reuse the existing depth or reconstruction result.

Examples:

- single-image mesh width;
- Z scale;
- base thickness;
- inversion;
- mesh near/far percentiles;
- depth curve controls;
- single-image preview mirror;
- triangle decimation / smoothing in multi-view;
- some mesher-specific settings when the DA3 prediction has already been generated.

## 2.3 Preview-only options vs real export options

A few things only affect what you see in the GUI, not the actual exported geometry.

### Most important example: `Correct horizontal mirroring in Gradio preview`

That option affects the **preview GLB only**.
It does **not** change the exported OBJ, STL, texture, or depth maps.

## 2.4 “Video sampling FPS” is **not** the source video's frame rate

This is easy to misread.

It means:

> **How many frames per second should PhotoToPhysical extract from the source video and feed into DA3?**

Examples for a 36-second video:

- `1 FPS` ≈ 36 extracted frames
- `2 FPS` ≈ 72 extracted frames
- `5 FPS` ≈ 180 extracted frames
- `30 FPS` ≈ 1080 extracted frames

Then, after extraction, **Maximum Views** may cap that set further.

## 2.5 `Maximum views` is applied after frame extraction

In multi-view mode:

1. video is sampled according to **Video sampling FPS**;
2. then the extracted set is reduced using **Maximum views**;
3. then DA3 uses the resulting final set.

So:

- `FPS = 30`, `Max Views = 18` → many frames extracted, but only 18 evenly spaced frames used.
- `FPS = 30`, `Max Views = 0` → the full extracted set is used.

---

# 3. Overall GUI layout

The main screen is organized as:

1. **Source image / crop controls** (left side, top)
2. **Depth settings** and **Printable mesh settings** (right side, top)
3. Tabs:
   - **Depth**
   - **3D / Print**
   - **Multi-Image 3D**
   - **About**

Single-image controls live mostly above the tabs.
Multi-image controls live inside the **Multi-Image 3D** tab.

---

# 4. Single-image workflow: complete control reference

## 4.1 `1. Source image`

**Control type:** image input

**Purpose:**
Upload or paste the single source image used for the monocular / single-image pipeline.

**Role in pipeline:**
This is the image that DA3 will process after optional cropping.

**How it affects output:**
The source image is the root input for:

- depth generation;
- printable mesh generation;
- OBJ texture (unless external depth mode is used with a custom external texture).

**Best practices:**

- Use a clean, high-resolution image.
- Remove overlays, rulers, logos, and UI chrome using crop controls.
- Keep the framing focused on the subject you want to turn into a relief.

---

## 4.2 `Upload an image to initialize crop coordinates.` / source dimensions display

**Control type:** markdown / status readout

**Purpose:**
Shows source image dimensions and helps initialize crop bounds.

**Role in pipeline:**
Informational only.

**How it affects output:**
It does not directly affect output, but it helps you enter valid crop coordinates.

---

## 4.3 `Crop before DA3` accordion

This section controls pre-inference cropping.

### 4.3.1 `Enable crop`

**Type:** checkbox

**Purpose:**
Turns cropping on or off.

**Role in pipeline:**
Cropping is applied **before DA3 inference**.

**When ON:**
Only the cropped region is sent to DA3.

**When OFF:**
The full image is used.

**Effect on output:**

- Changes the image DA3 sees.
- Can greatly improve depth quality by removing non-subject clutter.
- Reduces influence of text, rulers, borders, logos, or black padding.
- Usually changes the final mesh significantly because the depth map changes.

**Recommended use:**
Usually ON when the source image contains overlays or irrelevant margins.

### 4.3.2 `Left / X1`
### 4.3.3 `Top / Y1`
### 4.3.4 `Right / X2`
### 4.3.5 `Bottom / Y2`

**Type:** numeric fields

**Purpose:**
Define the crop rectangle in pixel coordinates.

**Role in pipeline:**
These coordinates determine the exact image region sent to DA3 when cropping is enabled.

**Effect on output:**

- Changing them changes the framing.
- Too tight a crop may remove useful subject context.
- Too loose a crop may include distracting UI or background.

**Operational notes:**

- `X1`, `Y1` are the top-left corner.
- `X2`, `Y2` are the bottom-right corner.
- They should describe the rectangle you want to keep.

### 4.3.6 `Preview Crop`

**Type:** button

**Purpose:**
Shows the exact image that will be sent to DA3.

**Role in pipeline:**
Preview only. It does not run DA3.

**Effect on output:**
No direct output change. It helps verify the crop before inference.

**Why it matters:**
This is the safest way to check whether the crop is correct before spending time on DA3 inference.

### 4.3.7 `Crop preview / exact DA3 input`

**Type:** image viewer

**Purpose:**
Displays the exact post-crop image that DA3 would receive.

**Role in pipeline:**
Preview / confirmation.

**Effect on output:**
None directly; diagnostic only.

---

## 4.4 `Depth settings`

This section controls the single-image DA3 inference.

### 4.4.1 `DA3 model`

**Type:** dropdown

**Choices:** typically from `MODEL_CHOICES`, defaulting to the monocular-oriented default.

**Purpose:**
Selects which DA3 model is used for single-image depth estimation.

**Role in pipeline:**
This is the neural model that predicts the depth map.

**Output impact:**
Very high. This changes the underlying depth prediction.

**Typical interpretation of choices:**

- **DA3Mono-Large**: best dedicated monocular choice.
- **DA3-Large / DA3-Giant any-view models**: useful for comparison, but usually not the default first choice for single-image relief work.

**Tradeoffs:**

- Better models may give cleaner depth but can use more VRAM and time.
- Different models may handle fine structures or smooth regions differently.

### 4.4.2 `DA3 internal process resolution`

**Type:** dropdown

**Choices:** `504, 756, 1008, 1260, 1512, 1764, 2016`

**Purpose:**
Sets the resolution DA3 uses internally during inference.

**Role in pipeline:**
Directly affects the neural prediction.

**Low values:**

- faster;
- lower VRAM;
- less fine detail.

**High values:**

- slower;
- more VRAM;
- can capture more detail, but may not always improve practical mesh quality.

**Important note:**
Higher is not always better. Some models may become noisier or less stable at very high process resolution.

**Good default:**
`1260` is the documented strong starting point for your RTX 4090 setup.

### 4.4.3 `Depth normalization` accordion

This does **not** change the raw DA3 depth values. It changes how the normalized depth outputs and previews are created.

#### `Low percentile`
#### `High percentile`

**Type:** sliders

**Purpose:**
Define the percentile range used when converting raw depth into normalized preview / export maps.

**Role in pipeline:**
Post-inference depth normalization for output maps.

**Effect on output:**

- Does **not** alter the raw saved float32 depth.
- Does change the near-white and far-white normalized EXRs / PNGs and previews.
- Affects how much contrast is allocated across the visible depth range.

**Lower low percentile:**
Includes more of the nearest outliers before clipping.

**Higher high percentile:**
Includes more of the far range before clipping.

**If the range is too narrow:**
Preview may clip or saturate.

**If the range is too broad:**
Useful contrast may be compressed.

### 4.4.4 `Generate DA3 Depth`

**Type:** button

**Purpose:**
Runs the single-image DA3 depth pipeline.

**Role in pipeline:**
This is the main action button for the depth stage.

**What it does:**

1. applies crop if enabled;
2. loads the chosen DA3 model;
3. runs DA3 inference;
4. saves raw depth and normalized outputs;
5. updates the **Depth** tab outputs.

**Outputs created:**

- `input_used.png`
- `depth_raw_float32.npy`
- `depth_raw_float32.exr`
- normalized EXRs and 16-bit PNGs
- preview PNGs
- confidence outputs if available
- metadata JSON

---

## 4.5 `Printable mesh settings`

This section controls the downstream single-image printable solid.

### 4.5.1 `Mesh grid max dimension`

**Type:** dropdown

**Choices:** `256, 384, 512, 640, 768, 1024`

**Purpose:**
Controls the maximum width/height of the resampled depth grid used to build the mesh.

**Role in pipeline:**
This affects the geometric resolution of the heightfield mesh.

**Low values:**

- faster;
- smaller files;
- smoother / less detailed mesh.

**High values:**

- denser mesh;
- more triangles;
- larger files;
- may preserve more detail.

**Important:**
This does **not** rerun DA3.
It only changes the mesh conversion stage.

**Default reasoning:**
`768` is a strong balance between detail and practicality.

### 4.5.2 `Object width`

**Type:** number

**Purpose:**
Sets the width of the generated object in arbitrary mesh units.

**Role in pipeline:**
Controls the physical / scene scale of the printable relief.

**Effect on output:**

- Changes overall object size.
- Does not change relative detail or depth proportions.
- Affects mesh dimensions in exported OBJ/STL.

**Note:**
STL is unitless. Final real-world size is usually chosen later in Blender or the slicer.

### 4.5.3 `Relief / Z scale`

**Type:** number

**Purpose:**
Controls how much the normalized depth range protrudes out of the backing.

**Role in pipeline:**
Determines the amplitude of the relief.

**Low values:**
Flatter relief.

**High values:**
More pronounced depth / stronger embossing.

**Too low:**
Subject may look flat.

**Too high:**
Can exaggerate noise or create an over-dramatic relief.

### 4.5.4 `Minimum backing thickness`

**Type:** number

**Purpose:**
Adds a flat base thickness behind the relief.

**Role in pipeline:**
This is what makes the mesh a **closed solid** with a physical back.

**Effect on output:**

- Thicker backing = sturdier object.
- Thinner backing = lighter object but potentially weaker print.

**Does not affect:**
foreground relief contrast directly.

### 4.5.5 `Invert DA3 raw depth (nearer = farther outward)`

**Type:** checkbox

**Purpose:**
Flips the interpreted depth direction before building the mesh.

**Role in pipeline:**
Controls whether near regions protrude outward or inward relative to the relief interpretation.

**When ON:**
Useful when the DA3 depth direction needs to be reversed for intuitive relief output.

**When OFF:**
Uses the depth orientation directly after normalization.

**Effect on output:**
Major visual impact on the shape. This is a real geometry change.

### 4.5.6 `Correct horizontal mirroring in Gradio preview`

**Type:** checkbox

**Purpose:**
Corrects only the on-screen preview orientation in Gradio.

**Role in pipeline:**
Preview-only transformation.

**Very important:**
This does **not** modify the exported OBJ/STL/texture. It only affects the preview GLB shown in the GUI.

**Use case:**
Gradio's 3D viewer convention can make the relief look horizontally mirrored even when the exported mesh is correct.

### 4.5.7 `Mesh depth normalization` accordion

These controls reshape the normalized depth **used for the mesh** without rerunning DA3.

#### `Mesh low percentile`
#### `Mesh high percentile`

**Purpose:**
Set the percentile window used to normalize the raw depth before relief generation.

**Role in pipeline:**
This is a downstream remapping stage for the printable mesh.

**Effect on output:**

- Changes depth contrast in the physical relief.
- Useful for suppressing outliers or stretching useful depth range.
- Does not rerun DA3.

**Lower low percentile:**
Preserves more of the near outliers.

**Lower high percentile:**
Clips more of the far range sooner.

**Good use cases:**

- recovering better facial depth contrast;
- clipping unusable extreme background values;
- making the relief more printable.

### 4.5.8 `Depth curve / foreground emphasis` accordion

This is one of the most powerful mesh-shaping sections.

It lets you compress the background and reserve more physical relief range for the foreground.

#### `Enable depth curve`

**Type:** checkbox

**Purpose:**
Turns the depth curve remapping on or off.

**When ON:**
A nonlinear curve remaps the normalized depth before building the mesh.

**Effect on output:**
Can dramatically increase foreground emphasis while compressing background depth.

#### `Foreground depth to emphasize (%)`

**Type:** slider

**Purpose:**
Defines what portion of the nearest depth range should be treated as the important foreground.

**Interpretation:**

- `25%` means the nearest/top quarter is treated as the emphasis region.

**Effect:**
Higher values classify more of the scene as foreground.

#### `Z range allocated to background (%)`

**Type:** slider

**Purpose:**
Controls how much of the physical relief range is reserved for the background.

**Example:**
`15%` means the background only gets the bottom 15% of the Z range, leaving 85% for foreground structure.

**Effect:**

- Lower values = stronger background compression.
- Higher values = background retains more relief.

#### `Foreground selection basis`

**Type:** dropdown

**Choices:**
- `Depth range`
- `Nearest pixels (percentile)`

**Purpose:**
Determines how the foreground threshold is computed.

**Depth range:**
The nearest percentage of the normalized depth range is foreground.

**Nearest pixels (percentile):**
The threshold is chosen so that roughly that fraction of image pixels are foreground.

**How to choose:**

- `Depth range` is simple and predictable.
- `Nearest pixels (percentile)` is useful when the subject occupies a limited image area and you want foreground emphasis based on subject coverage rather than abstract depth scale.

### 4.5.9 `Optional: use an external depth map` accordion

This bypasses DA3 and lets you build the printable mesh from your own depth file.

#### `Bypass DA3 and build the mesh from an uploaded depth file`

**Type:** checkbox

**Purpose:**
When enabled, PhotoToPhysical uses the uploaded external depth map instead of the DA3 result in the current job folder.

**Role in pipeline:**
This changes the source of the mesh stage.

**Use cases:**

- compare DA3 with DA2 or another model;
- use a hand-edited EXR/PNG/NPY depth map;
- test custom depth workflows.

#### `External depth file`

**Type:** file input

**Supported:** `.exr`, `.png`, `.npy`, `.tif`, `.tiff`

**Purpose:**
Supplies the depth source for external-depth mode.

**Effect on output:**
Directly replaces the normal DA3-produced depth source.

#### `Optional color/albedo image for OBJ texture`

**Type:** file input

**Purpose:**
Supplies a texture for the exported OBJ when using external depth.

**Effect on output:**
Changes the OBJ texture image only; the STL remains geometry-only.

### 4.5.10 `Build / Rebuild Printable Mesh`

**Type:** button

**Purpose:**
Builds the printable solid from either:

- the last DA3 depth result in the job folder, or
- the uploaded external depth file if external mode is enabled.

**Role in pipeline:**
This is the main geometry-generation button for the single-image mesh stage.

**What it does:**

1. loads raw depth;
2. resamples to the selected mesh grid;
3. normalizes depth using mesh near/far percentiles;
4. applies inversion if enabled;
5. optionally applies depth curve;
6. builds a **closed solid**;
7. exports OBJ / MTL / STL / preview GLB / metadata / previews / ZIP.

### 4.5.11 `Run Entire Pipeline`

**Type:** button

**Purpose:**
Runs the single-image depth stage and printable mesh stage in one click.

**Role in pipeline:**
Convenience action.

**What it does:**

1. runs `Generate DA3 Depth` internally;
2. then immediately runs `Build / Rebuild Printable Mesh` using that result.

**When to use:**
When you already know your crop and just want a complete result quickly.

**When not to use:**
If you want to inspect depth output first before deciding on mesh settings.

---

# 5. `Depth` tab: outputs and meanings

This tab shows the outputs of the single-image depth stage.

## 5.1 `Exact image sent to DA3`

Displays the actual cropped image used in inference.
Useful for verifying that DA3 saw what you intended.

## 5.2 `Near-white depth preview`

Displays the normalized depth preview where near regions are white.
Useful for a quick human-readable check.

## 5.3 `Depth output files`

This file list contains the saved depth outputs, typically including:

- raw float32 NPY;
- raw float32 EXR;
- normalized near-white EXR / 16-bit PNG / preview PNG;
- normalized far-white counterparts;
- confidence outputs if present;
- metadata JSON.

## 5.4 `depth_status`

A summary of:

- input size;
- model used;
- process resolution;
- inference time;
- GPU;
- output folder.

---

# 6. `3D / Print` tab: outputs and meanings

This tab shows the printable single-image mesh result.

## 6.1 `Printable mesh preview (display-only GLB)`

**Purpose:**
Displays the printable relief as a GLB in the viewer.

**Important:**
This preview can be horizontally mirrored if preview correction is enabled.
The real exported OBJ/STL are not altered by that preview-only transform.

## 6.2 `Mesh output files`

Usually includes:

- `heightfield_solid.obj`
- `heightfield_solid.mtl`
- `heightfield_solid.stl`
- preview GLB
- mesh-depth preview images
- mesh metadata JSON
- optional texture/albedo PNG

## 6.3 `Download complete ZIP package`

Downloads the entire job folder as a ZIP archive.

## 6.4 `mesh_status`

Shows key geometry statistics:

- mesh grid size;
- vertex count;
- triangle count;
- width / height / thickness;
- preview orientation correction status;
- depth curve status and parameters;
- watertight status;
- Euler number.

---

# 7. Multi-Image 3D tab: complete control reference

This is the most complex section of the GUI.

## 7.1 What the multi-image tab is for

This tab reconstructs a subject or scene from more than one view.

It supports four input styles:

1. **Unordered uploaded photos**
2. **Ordered uploaded frames**
3. **Video file**
4. **COLMAP local dataset (known poses)**

It then runs one or more downstream reconstruction methods.

---

## 7.2 `Multi-view input mode`

**Type:** dropdown

**Choices:**

- `Unordered uploaded photos`
- `Ordered uploaded frames`
- `Video file`
- `COLMAP local dataset (known poses)`

**Purpose:**
Tells the app what kind of multi-view input it is receiving.

**Role in pipeline:**
This affects frame ordering assumptions, camera-pose handling, and automatic reference-strategy selection.

### `Unordered uploaded photos`
Use this when you have multiple photos of the same subject but they are not a chronological frame sequence.

**Typical use:**
hand-captured photos around an object.

**Automatic reference behavior:**
usually `saddle_balanced`.

### `Ordered uploaded frames`
Use this when you already extracted frames from a video and their order matters.

**Typical use:**
a frame sequence from a moving camera.

**Automatic reference behavior:**
usually `middle`.

### `Video file`
Use this when you want the app to extract frames from a video itself.

**Typical use:**
one video clip from a phone or camera.

**Automatic reference behavior:**
usually `middle`.

### `COLMAP local dataset (known poses)`
Use this when you already have a COLMAP dataset with known camera intrinsics/extrinsics.

**Typical use:**
advanced calibrated reconstruction workflows.

**Important effect:**
Known camera poses are supplied, so DA3's ray-pose estimation is bypassed / ignored.

---

## 7.3 `Uploaded photos / ordered frames`

**Type:** multi-file input

**Purpose:**
Supplies the images for unordered-photo or ordered-frame modes.

**Effect on output:**
These are the actual DA3 views.

**Important:**
If you choose `Ordered uploaded frames`, file order matters.

---

## 7.4 `Optional source video`

**Type:** single file input

**Purpose:**
Supplies a video for the `Video file` input mode.

**Effect on output:**
The app samples frames from this video according to **Video sampling FPS**, then optionally reduces them with **Maximum views**.

---

## 7.5 `Video sampling FPS`

**Type:** number

**Purpose:**
Controls how many frames per second are extracted from the source video.

**Role in pipeline:**
Frame-selection stage before DA3 inference.

**Low values (e.g. 0.5–2):**

- fewer views;
- wider spacing between views;
- less redundancy;
- lighter compute;
- often better for testing.

**High values (e.g. 10–30):**

- many more frames;
- heavy compute;
- lots of near-duplicate views;
- useful mainly when motion is very fast or as a stress test.

**Recommended starting range:**
`1–2 FPS` for ordinary moving-camera clips.

---

## 7.6 `COLMAP / known-camera input` accordion

Used only for the `COLMAP local dataset (known poses)` mode.

### 7.6.1 `Local COLMAP dataset folder`

**Type:** textbox

**Purpose:**
Points to a local dataset folder containing at least `images/` and `sparse/`.

**Role in pipeline:**
Provides known camera/image structure to DA3's COLMAP input handler.

### 7.6.2 `COLMAP sparse subdirectory`

**Type:** textbox

**Purpose:**
Tells the app which sparse subfolder to use when COLMAP stores data under `sparse/0/` or similar.

**Use blank:**
If cameras/images are directly under `sparse/`.

**Use `0`:**
For the common `sparse/0/` layout.

### 7.6.3 `Align DA3 depth to COLMAP camera scale`

**Type:** checkbox

**Purpose:**
Aligns DA3 depth to the scale implied by the COLMAP extrinsics.

**Role in pipeline:**
Scale alignment stage when known poses are supplied.

**Effect:**
Important when you want DA3's depth to respect the metric or relative scale of the provided camera solution.

**Usually:** keep ON unless you have a specific reason not to.

---

## 7.7 `Maximum views (0 = no limit)`

**Type:** dropdown

**Choices:** `0, 4, 6, 8, 12, 18`

**Purpose:**
Caps how many views DA3 actually receives.

**Role in pipeline:**
Final view-selection stage before inference.

**Behavior:**

- If more source views exist than the cap, the app selects an evenly spaced subset.
- `0` disables the cap and uses all sampled/uploaded views.

**Low values:**

- faster;
- less memory use;
- less cross-view complexity;
- may lose coverage.

**High values / 0:**

- more coverage;
- more compute;
- can improve reconstruction if the views are informative;
- can also worsen things if the extra views are redundant, blurry, or inconsistent.

**Important nuance:**
The 18-view default is a conservative operational cap, not a proof that DA3 cannot use more views.

---

## 7.8 `Multi-view DA3 model`

**Type:** dropdown

**Choices:**

- `DA3-Giant-1.1 (quality any-view)`
- `DA3Nested-Giant-Large-1.1 (metric-scale nested)`
- `DA3-Large-1.1 (lighter/faster)`

**Purpose:**
Selects the DA3 any-view model used for multi-view inference.

### `DA3-Giant-1.1`
Best general quality-first choice.

### `DA3Nested-Giant-Large-1.1`
Adds nested / metric-scale behavior.
Potentially useful when scale fidelity matters.

### `DA3-Large-1.1`
Lower VRAM and faster iteration, but generally lower ceiling than Giant.

**Special note for Gaussian output:**
Native DA3 Gaussian export requires a Gaussian-capable model, specifically Giant or Nested Giant-Large.

---

## 7.9 `DA3 multi-view process resolution`

**Type:** dropdown

**Choices:** `504, 756, 1008`

**Purpose:**
Sets DA3's internal multi-view process resolution.

**Role in pipeline:**
Directly affects the neural reconstruction.

**504:**
Documented base/default regime and safest starting point.

**756 / 1008:**
Higher detail potential, but more experimental / heavier.

**Recommended first test:**
`504`.

---

## 7.10 `Pose estimation / reference view` accordion

These settings control how DA3 chooses or estimates the coordinate frame and camera relations.

### 7.10.1 `Use high-quality ray-based camera pose estimation`

**Type:** checkbox

**Purpose:**
Enables DA3's more accurate ray-based pose estimation.

**Role in pipeline:**
Camera-estimation stage.

**When ON:**
Slightly slower but generally better camera pose quality.

**When OFF:**
Uses DA3 without this higher-quality pose mode.

**Ignored when:**
Known COLMAP poses are supplied.

### 7.10.2 `Automatically choose reference strategy from input type`

**Type:** checkbox

**Purpose:**
Lets the app choose the reference-view strategy based on the input mode.

**Typical automatic behavior:**

- ordered frames / video → `middle`
- unordered photos → `saddle_balanced`

**Why it matters:**
A good reference strategy helps DA3 stabilize its multi-view reasoning.

### 7.10.3 `Manual reference-view strategy`

**Type:** dropdown

**Choices:**
- `saddle_balanced`
- `saddle_sim_range`
- `middle`
- `first`

**Purpose:**
Forces a specific reference strategy when automatic selection is disabled.

#### `saddle_balanced`
Best general choice for unordered photos.
Balanced baseline and overlap behavior.

#### `saddle_sim_range`
Often useful when the set has wider viewpoint variation.

#### `middle`
Best for ordered sequences or videos where a central reference frame is natural.

#### `first`
Supported, but usually not the best default unless you have a reason to anchor the reconstruction to the first image.

---

## 7.11 `DA3 point-cloud / confidence quality` accordion

These control the DA3-native point geometry and confidence filtering.

### 7.11.1 `Confidence filter percentile`

**Type:** slider

**Range:** `0–80`, default `40`

**Purpose:**
Filters out low-confidence DA3 samples.

**Role in pipeline:**
Affects the DA3 point-cloud/GLB path and the confidence masking used by related mesh comparisons.

**Lower values:**

- retain more geometry;
- also retain more uncertain / noisy points.

**Higher values:**

- cleaner, more selective geometry;
- more holes or missing weak regions.

**Good baseline:**
`40` is DA3's documented GLB default.

### 7.11.2 `Maximum DA3 GLB points`

**Type:** dropdown

**Choices:** `200k, 400k, 600k, 800k, 1,000,000`

**Purpose:**
Caps the point count in the DA3 GLB / point-cloud export.

**Role in pipeline:**
Affects point-cloud-based visualization and the Poisson path's source density.

**Lower values:**

- smaller files;
- lighter visualization;
- less dense point support.

**Higher values:**

- denser point cloud;
- larger files;
- potentially better Poisson detail if the points are good.

**Note:**
TSDF does **not** use the point cloud directly; it fuses depth maps.

---

## 7.12 `Additional DA3 outputs` accordion

These are additive exports produced by DA3 itself.
They do **not** replace the mesh method; they sit alongside it.

### 7.12.1 `Export native DA3 Gaussian Splat .ply`

**Type:** checkbox

**Purpose:**
Exports DA3's native Gaussian-splat representation as a `.ply`.

**Role in pipeline:**
Runs the DA3 Gaussian head (`infer_gs=True`) and writes a splat `.ply`.

**Important:**
This is not a mesh. It is a Gaussian-splat representation.

**Best use:**

- viewing scene reconstructions in a dedicated splat viewer;
- preserving the DA3 scene representation without forcing a triangle mesh.

**Requirements:**
Gaussian-capable DA3 model (Giant or Nested Giant-Large).

### 7.12.2 `Also export DA3 Gaussian preview video`

**Type:** checkbox

**Purpose:**
Asks DA3 to render a preview video of the Gaussian splat.

**Role in pipeline:**
Adds DA3 Gaussian rendering output.

**Important dependency:**
This requires **`gsplat`**. Without `gsplat`, the Gaussian `.ply` can still be exported, but the preview video will fail.

**Operational advice:**
If you are not sure `gsplat` is installed correctly, test with the `.ply` box ON and the video box OFF first.

---

## 7.13 `Surface reconstruction method`

**Type:** dropdown

**Choices:**

- `TSDF`
- `Poisson`
- `Consistent Surfel + Poisson`
- `OpenMVS`
- `OpenMVS + RefineMesh`
- `2DGS`
- `Compare selected methods`

**Purpose:**
Selects how the DA3 multi-view output is converted into a surface representation.

This is one of the most important controls in the multi-view workflow.

A full method-by-method explanation appears later in this guide.

---

## 7.14 `Methods to run when Compare selected methods is chosen`

**Type:** checkbox group

**Purpose:**
Defines which methods are run in compare mode.

**Role in pipeline:**
Lets you A/B/C methods from a single DA3 inference instead of rerunning DA3 separately.

**Best use:**
Controlled comparisons such as:

- TSDF vs Poisson vs Consistent Surfel;
- adding OpenMVS after installing it;
- comparing a mature external method against the native methods.

**Caution:**
2DGS can be very slow and may create large temporary data.

---

# 8. Surface reconstruction methods: what each one does

## 8.1 `TSDF`

### What it is
TSDF means **Truncated Signed Distance Field** fusion.

### What it uses
- DA3 depth maps
- DA3 confidence masks
- DA3 camera intrinsics/extrinsics

### Pipeline role
It fuses many per-view depth maps into a volumetric field, then extracts a triangle mesh.

### Strengths
- directly uses organized depth maps;
- often preserves local shape structure faithfully;
- good baseline for geometric fidelity.

### Weaknesses
- can show corrugation / terracing / layered surface artifacts when views disagree;
- can produce patchiness if confidence filtering is too strict.

### Best for
Users who want a direct depth-fusion baseline or a more “honest” representation of DA3's per-view surfaces.

---

## 8.2 `Poisson`

### What it is
A traditional **Screened Poisson surface reconstruction** from a point cloud.

### What it uses
- DA3 point cloud / GLB export
- estimated normals from the point cloud neighborhood

### Pipeline role
Converts the point cloud into a smooth continuous mesh.

### Strengths
- smooth, visually pleasing surfaces;
- often fills holes gracefully;
- can look better immediately than TSDF for organic shapes.

### Weaknesses
- can inflate / bloat the surface;
- can invent smooth bridges or extrapolated surfaces;
- depends heavily on point quality and normal estimation.

### Best for
Quick smooth-looking results or when the point cloud already looks great and you want a skin over it.

---

## 8.3 `Consistent Surfel + Poisson`

### What it is
A DA3-aware mesher added specifically for PhotoToPhysical.

### What it uses
- organized DA3 depth maps
- per-view geometry
- cross-view reprojection consistency
- depth-derived normals
- then Screened Poisson

### Pipeline role
It tries to preserve the strengths of DA3's multiview geometry while rejecting inconsistent observations before Poisson.

### Strengths
- more informed than plain Poisson;
- uses real multiview support checks;
- can reduce false surfaces and improve coherence.

### Weaknesses
- more parameters to tune;
- can get holes if the consistency checks are too strict;
- slower than naive Poisson.

### Best for
The main native PhotoToPhysical quality path when you want something better than legacy Poisson without leaving the DA3 environment.

---

## 8.4 `OpenMVS`

### What it is
An external mature multi-view stereo and meshing backend.

### What it uses
- DA3-exported COLMAP-compatible camera/seed data
- original source images
- OpenMVS dense reconstruction + meshing

### Pipeline role
Uses a more traditional visibility-aware MVS pipeline to reconstruct the surface.

### Strengths
- mature conventional MVS approach;
- explicitly reasons over multiview geometry;
- often robust for scene/object reconstruction.

### Weaknesses
- requires external installation;
- slower and more complex than native methods.

### Best for
Users who want a stronger conventional reconstruction backend than simple Open3D Poisson/TSDF.

---

## 8.5 `OpenMVS + RefineMesh`

### What it is
OpenMVS mesh reconstruction plus image-based mesh refinement.

### What it adds beyond plain OpenMVS
After building an initial mesh, it refines the mesh against the original images.

### Strengths
- can improve the surface using photo consistency;
- potentially stronger than plain OpenMVS for final geometry.

### Weaknesses
- more compute time;
- requires working OpenMVS install.

### Best for
Users seeking the highest-quality conventional OpenMVS result.

---

## 8.6 `2DGS`

### What it is
An external **2D Gaussian Splatting** neural surface-reconstruction backend.

### What it uses
- DA3-exported cameras and seed geometry
- the source images
- an iterative neural optimization process

### Pipeline role
Optimizes surface-oriented Gaussian primitives and then extracts a mesh.

### Strengths
- high potential quality ceiling;
- more aligned with advanced neural reconstruction methods.

### Weaknesses
- external setup required;
- much slower than native methods;
- training/optimization style workflow.

### Best for
Higher-end experiments where you are willing to spend more time for potentially better scene/object geometry.

---

## 8.7 `Compare selected methods`

### What it is
A meta-mode rather than a mesher.

### Purpose
Runs several selected methods from the same DA3 inference.

### Best use
Controlled evaluation and tuning.

### Why it matters
It prevents DA3 from being rerun for every method, which makes comparisons fairer and faster.

---

## 8.8 Native DA3 Gaussian output (not a mesh)

### What it is
DA3's own native Gaussian-splat representation.

### What it outputs
- `.ply` splat file
- optional rendered preview video

### Why it matters
This is often a better fit than a mesh for scene-scale or environment reconstruction because it avoids forcing everything into one triangle surface.

### Best viewing method
A Gaussian-splat viewer such as SuperSplat or another dedicated splat viewer.

---

# 9. Method-specific control reference

## 9.1 `TSDF quality settings`

### 9.1.1 `TSDF detail (voxels per median scene depth)`

**Type:** dropdown

**Choices:** `128, 192, 256, 384, 512`

**Purpose:**
Controls voxel density relative to median scene depth.

**Lower values:**

- larger voxels;
- smoother / coarser fusion;
- less RAM / CPU.

**Higher values:**

- smaller voxels;
- finer detail;
- more RAM / CPU;
- can reveal more noise.

**Recommended start:**
`256`.

### 9.1.2 `TSDF truncation distance (voxels)`

**Type:** slider

**Range:** `2–10`, default `5`

**Purpose:**
Sets how wide the integration band is around each observed surface.

**Lower values:**

- sharper / stricter integration;
- less smoothing across disagreement.

**Higher values:**

- more blending / averaging;
- may reduce seams but can soften structure.

**Recommended start:**
`4–6`.

### 9.1.3 `Ignore farthest depth outliers above percentile`

**Type:** slider

**Range:** `90–100`, default `99.5`

**Purpose:**
Removes extremely far depth outliers when setting TSDF integration limits.

**Effect:**
Protects the reconstruction from wild far-depth or sky pixels.

**Lower values:**
More aggressive outlier suppression.

**Higher values:**
Retains more far geometry but may become more sensitive to outliers.

---

## 9.2 `Poisson settings`

### 9.2.1 `Poisson point-cloud voxel downsample (0 = off)`

**Type:** number

**Purpose:**
Optionally downsamples the point cloud before Poisson.

**0:**
No downsampling.

**Higher values:**
Fewer points, which can reduce noise and memory but also lose detail.

**Use case:**
Only increase if the cloud is huge or noisy.

### 9.2.2 `Poisson reconstruction depth`

**Type:** dropdown

**Choices:** `8, 9, 10, 11`

**Purpose:**
Controls the octree depth / resolution of Poisson reconstruction.

**Lower values:**
Smoother, coarser mesh.

**Higher values:**
Finer detail, but more noise and more memory use.

**Recommended start:**
`9`.

### 9.2.3 `Trim lowest-density Poisson vertices (%)`

**Type:** slider

**Range:** `0–10`, default `0`

**Purpose:**
Trims the lowest-density regions of the Poisson output.

**0:**
Keep everything.

**1–3:**
Useful for removing weak skirts, wisps, or edge debris.

**Too high:**
Can delete real geometry.

---

## 9.3 `Consistent Surfel + Poisson quality`

### 9.3.1 `Surfel pixel stride`

**Type:** dropdown

**Choices:** `1, 2, 3, 4`

**Purpose:**
Controls how densely the DA3 depth pixels are sampled to create surfels.

**1:**
Use every depth pixel. Highest detail, slowest.

**2:**
Strong default balance.

**3–4:**
Lighter / faster, but lower sample density.

### 9.3.2 `Neighbor cameras tested per source view`

**Type:** dropdown

**Choices:** `2, 3, 4, 6, 8`

**Purpose:**
How many nearby cameras are checked when validating a surfel by reprojection.

**Lower values:**
Faster, less support evidence.

**Higher values:**
Stronger cross-view checking, slower.

### 9.3.3 `Minimum total supporting views`

**Type:** dropdown

**Choices:** `1, 2, 3, 4`

**Purpose:**
How many views must support a surfel for it to survive.

**1:**
Very permissive.

**2:**
Source + at least one agreeing neighbor. Strong default.

**3–4:**
Stricter; may improve fidelity but can create holes.

### 9.3.4 `Cross-view depth agreement tolerance (%)`

**Type:** slider

**Range:** `0.25–10`, default `2`

**Purpose:**
Controls how closely neighbor-view depth must match the reprojected surfel depth.

**Lower values:**
Stricter agreement; cleaner but can create holes.

**Higher values:**
More permissive; retains more data but may allow doubles or ripples.

**Main tuning range:**
`1–3%`.

### 9.3.5 `Cross-view normal disagreement allowed (degrees)`

**Type:** slider

**Range:** `10–180`, default `55`

**Purpose:**
Controls the maximum allowed angular disagreement between normals.

**Lower values:**
Strict normal consistency.

**Higher values:**
More permissive.

**180:**
Effectively disables the normal-consistency check.

### 9.3.6 `Surfel merge voxel size (0 = auto)`

**Type:** number

**Purpose:**
Sets the voxel size used when merging redundant supported surfels.

**0:**
Auto, derived from geometry scale.

**Smaller explicit values:**
Preserve more detail but merge less aggressively.

**Larger explicit values:**
Smooth/merge more aggressively.

### 9.3.7 `Maximum oriented surfels before merge`

**Type:** dropdown

**Choices:** `100k, 250k, 500k, 750k, 1,000,000`

**Purpose:**
Caps the surfel count before the merge stage.

**Lower values:**
Faster, smaller, potentially less detail.

**Higher values:**
More detail, heavier compute.

---

## 9.4 `OpenMVS external backend`

### 9.4.1 `OpenMVS executable folder (blank = search PATH)`

**Type:** textbox

**Purpose:**
Specifies where the OpenMVS executables are located.

**Blank:**
Search system PATH.

**Use explicit path:**
If OpenMVS is installed elsewhere.

### 9.4.2 `DA3 sparse COLMAP initialization confidence percentile`

**Type:** slider

**Range:** `50–99`, default `95`

**Purpose:**
Controls how selective the initial DA3 seed geometry is before OpenMVS re-densifies.

**Lower values:**
More initial seed points, more noise risk.

**Higher values:**
Cleaner, more compact seed.

### 9.4.3 `OpenMVS DensifyPointCloud resolution level`

**Type:** dropdown

**Choices:** `0, 1, 2`

**Purpose:**
Controls the image downscale level for dense reconstruction.

**0:**
Full resolution. Highest detail, heaviest.

**1:**
Half-scale. Good safe default.

**2:**
Lower resolution / lighter.

### 9.4.4 `OpenMVS neighbor views per depth estimate (0 = all)`

**Type:** dropdown

**Choices:** `0, 4, 6, 8, 10`

**Purpose:**
How many neighboring views OpenMVS uses per depth estimate.

**Lower values:**
Faster, less support.

**Higher values:**
Potentially better support, slower.

**0:**
Use all available.

### 9.4.5 `OpenMVS minimum agreeing views for fusion`

**Type:** dropdown

**Choices:** `2, 3, 4, 5`

**Purpose:**
How many views must agree before depth is fused.

**Lower values:**
More permissive.

**Higher values:**
Stricter, can improve robustness but increase holes.

### 9.4.6 `OpenMVS RefineMesh resolution level`

**Type:** dropdown

**Choices:** `0, 1, 2`

**Purpose:**
Resolution level used during RefineMesh.

**0:**
Highest detail / heaviest.

**1:**
Good balance.

**2:**
Lighter but less detail.

---

## 9.5 `2DGS neural surface backend`

### 9.5.1 `2DGS repository folder`

**Type:** textbox

**Purpose:**
Points to the installed 2DGS codebase.

### 9.5.2 `2DGS Python executable (blank = current Python)`

**Type:** textbox

**Purpose:**
Lets you run 2DGS in a separate Python environment.

**Why this matters:**
Strongly recommended so 2DGS dependencies cannot destabilize the main DA3 environment.

### 9.5.3 `DA3 initialization confidence percentile`

**Type:** slider

**Range:** `80–99.5`, default `95`

**Purpose:**
Controls how selective the DA3 seed geometry is for 2DGS initialization.

**Lower values:**
More points, more clutter risk.

**Higher values:**
Cleaner, smaller initialization.

### 9.5.4 `2DGS optimization iterations`

**Type:** dropdown

**Choices:** `7000, 15000, 30000`

**Purpose:**
Controls how long the 2DGS optimization runs.

**7000:**
Fast smoke test.

**15000:**
Good quality default.

**30000:**
Full-scale higher-cost run.

### 9.5.5 `2DGS mesh depth statistic (0 = mean, 1 = median)`

**Type:** slider

**Current practical interpretation:**
This behaves like a binary selector:

- `0` = mean
- `1` = median

**Purpose:**
Chooses the statistic used during depth/mesh derivation.

**Mean:**
Often smoother and more averaged.

**Median:**
Potentially more robust to outliers.

### 9.5.6 `2DGS mesh extraction resolution`

**Type:** dropdown

**Choices:** `512, 768, 1024, 1536`

**Purpose:**
Controls the resolution used when extracting the mesh from the optimized 2DGS representation.

**Lower values:**
Faster, coarser mesh.

**Higher values:**
More detail, more memory/time.

### 9.5.7 `Use 2DGS unbounded mesh extraction`

**Type:** checkbox

**Purpose:**
Switches between bounded compact extraction and unbounded scene-style extraction.

**OFF:**
Better for compact foreground subjects.

**ON:**
Better for larger rooms or scene-scale captures.

---

## 9.6 `Shared mesh post-processing`

These apply to whichever mesh reconstruction method(s) are run.

### 9.6.1 `Taubin smoothing iterations`

**Type:** slider

**Range:** `0–20`, default `3`

**Purpose:**
Applies smoothing after reconstruction.

**0:**
No smoothing; best for raw comparisons.

**Higher values:**
Smoother surface, but detail loss increases.

**Recommended use:**
Set to `0` when diagnosing differences between methods.
Use small values like `1–3` for final cleanup.

### 9.6.2 `Target triangle count (0 = full mesh)`

**Type:** dropdown

**Choices:** `0, 100k, 200k, 400k, 800k`

**Purpose:**
Decimates the mesh to a target triangle budget.

**0:**
Keep full mesh.

**Lower counts:**
Smaller files, faster downstream use, less detail.

**Higher counts / 0:**
Retain more detail.

**Recommended use:**
Use `0` while comparing methods; decimate only after choosing a winner.

### 9.6.3 `Keep only the largest connected mesh component`

**Type:** checkbox

**Purpose:**
Removes disconnected floating pieces or debris.

**When ON:**
Usually cleaner final result.

**When OFF:**
Useful for diagnosis, because you can see all fragments instead of silently losing them.

---

## 9.7 `Run Multi-View Reconstruction`

**Type:** button

**Purpose:**
Runs the multi-view workflow.

**What it does at a high level:**

1. gathers images / sampled frames / COLMAP dataset;
2. optionally caps the view count;
3. runs DA3 multi-view inference;
4. exports diagnostics and point geometry;
5. optionally exports native Gaussian output;
6. runs the selected mesher(s);
7. applies shared post-processing;
8. saves files and ZIP package.

---

# 10. Multi-view output viewers and what they mean

## 10.1 `Primary selected mesh preview`

Shows the main selected reconstruction result as a preview mesh.

## 10.2 `Second selected mesh preview (compare mode)`

Used when compare mode runs multiple methods. Lets you inspect a second method alongside the primary one.

## 10.3 `DA3 camera-pose diagnostic (point cloud + camera directions)`

One of the most important diagnostic viewers.

It shows the DA3 point cloud together with camera positions/directions.

**Why it matters:**
If reconstruction is bad, this helps you distinguish between:

- pose failure;
- good pose but poor meshing;
- weird view distribution.

## 10.4 `Native DA3 Gaussian preview video`

Shown when native Gaussian preview video was requested and rendered successfully.

**Purpose:**
Quickly inspect the splat output without leaving the GUI.

**Dependency:**
Requires `gsplat`.

## 10.5 `Multi-view status`

Summarizes:

- input mode;
- number of requested/used views;
- model;
- process resolution;
- Gaussian export info if applicable;
- per-method result stats;
- timing and output folder info.

## 10.6 `Multi-view output files`

Contains all exported artifacts from the run, which may include:

- DA3 scene GLB / point cloud;
- prediction NPZ;
- input order file;
- metadata JSON;
- camera diagnostic GLB / JSON;
- native Gaussian `.ply` and optional video;
- per-method OBJ/PLY/GLB/log files;
- ZIP package.

## 10.7 `Download multi-view ZIP package`

Downloads the entire multi-view job folder as a ZIP archive.

---

# 11. Buttons, downloads, and action summary

## Single-image buttons

### `Preview Crop`
Preview only. Does not run DA3.

### `Generate DA3 Depth`
Runs the depth stage only.

### `Build / Rebuild Printable Mesh`
Runs the mesh stage only, using either the current DA3 depth or an external depth file.

### `Run Entire Pipeline`
Runs depth and mesh end-to-end.

## Multi-image button

### `Run Multi-View Reconstruction`
Runs DA3 and the selected multi-view reconstructor(s).

## Download buttons

### `Download complete ZIP package`
Single-image printable-mesh package.

### `Download multi-view ZIP package`
Multi-image reconstruction package.

---

# 12. Which controls affect what? Quick matrix

## 12.1 Controls that change the **single-image DA3 prediction**

- Source image
- Crop enable / crop coordinates
- DA3 model
- DA3 internal process resolution
- Depth normalization preview percentiles (for exported normalized maps, not raw depth itself)
- Generate DA3 Depth button

## 12.2 Controls that change the **single-image printable mesh without rerunning DA3**

- Mesh grid max dimension
- Object width
- Relief / Z scale
- Minimum backing thickness
- Invert depth
- Mesh near/far percentiles
- Depth curve controls
- External depth mode / external depth file / external texture
- Build / Rebuild Printable Mesh

## 12.3 Controls that change the **multi-image DA3 reconstruction**

- Input mode
- Uploaded images / frames
- Video file
- Video sampling FPS
- COLMAP dataset path / sparse subdir / align-to-scale
- Maximum views
- Multi-view DA3 model
- Multi-view process resolution
- Ray pose
- Auto/manual reference strategy
- Confidence filter percentile
- Maximum DA3 GLB points
- Native Gaussian export options

## 12.4 Controls that mostly change **downstream multi-view meshing**

- Surface reconstruction method
- Compare-selected-methods list
- TSDF settings
- Poisson settings
- Surfel settings
- OpenMVS settings
- 2DGS settings
- Shared smoothing / target faces / largest component cleanup

---

# 13. Tuning by symptom: what to adjust when something looks wrong

## 13.1 Single-image relief looks too flat

Try:

- raise `Relief / Z scale`;
- narrow mesh near/far percentile range;
- enable `Depth curve` and compress background more.

## 13.2 Single-image relief exaggerates noise

Try:

- lower `Relief / Z scale`;
- widen mesh percentile range slightly;
- reduce how aggressively the foreground is emphasized.

## 13.3 Single-image preview looks mirrored but exported mesh is fine

That is exactly what `Correct horizontal mirroring in Gradio preview` is for.

## 13.4 Multi-view TSDF looks stripey / terraced / patchy

Try:

- lower `TSDF detail`;
- increase `TSDF truncation distance` slightly;
- lower or tune the confidence filtering;
- compare against Consistent Surfel and Poisson.

## 13.5 Multi-view Poisson looks bloated / melted

Try:

- lower `Poisson depth` if noise amplification is the issue;
- trim very lightly only if skirts/debris exist;
- compare against Consistent Surfel;
- try a cleaner point cloud via confidence filtering.

## 13.6 Consistent Surfel creates holes

Try:

- increase `Cross-view depth agreement tolerance`;
- lower `Minimum total supporting views`;
- raise allowed `Cross-view normal disagreement`;
- reduce strictness before concluding the geometry is bad.

## 13.7 Consistent Surfel looks too permissive / double-surfaced

Try:

- decrease `Cross-view depth agreement tolerance`;
- increase `Minimum total supporting views`;
- lower allowed normal disagreement.

## 13.8 OpenMVS is too slow or too heavy

Try:

- use `resolution level = 1 or 2`;
- reduce neighbor views;
- keep fusion agreement at 2 for first tests.

## 13.9 2DGS is too slow

Try:

- use `7000` iterations first;
- keep mesh extraction resolution moderate;
- use a separate optimized environment.

## 13.10 Gaussian video export fails

Most likely `gsplat` is missing or not installed correctly.

Immediate workaround:

- leave `Export native DA3 Gaussian Splat .ply` ON;
- turn `Also export DA3 Gaussian preview video` OFF.

---

# 14. Practical recommended starting points

## 14.1 Single-image printable relief starting point

- Crop enabled
- DA3Mono-Large
- process resolution around `1260`
- mesh grid `768`
- width `2.0`
- Z scale `0.45`
- base thickness `0.08`
- invert ON if that matches the desired relief direction
- depth curve OFF for baseline, then ON if background compression helps

## 14.2 Multi-image object reconstruction baseline

- input mode matching your data
- DA3-Giant-1.1
- process resolution `504`
- ray pose ON
- confidence percentile `40`
- max views `18` for an initial capped run, or `0` once you intentionally want more
- compare mode: TSDF + Poisson + Consistent Surfel
- smoothing `0`
- target faces `0`
- keep largest component OFF for diagnostics

## 14.3 Video-to-Gaussian baseline

- input mode: `Video file`
- sampling FPS `1 or 2`
- max views `0` or a modest cap if needed
- DA3-Giant-1.1
- native Gaussian `.ply` ON
- Gaussian preview video OFF unless `gsplat` is confirmed working

---

# 15. What each tab is really for

## `Depth`
Inspect and collect depth outputs.

## `3D / Print`
Inspect and collect the single-image printable relief solid.

## `Multi-Image 3D`
Run multi-view reconstruction and compare reconstruction methods.

## `About`
Workflow notes, general documentation summary, and project explanation.

---

# 16. Final usage guidance

If someone only remembers a few rules from this guide, the most useful are:

1. **Single-image and multi-image are different pipelines.**
2. **DA3 process resolution** is not the same thing as mesh resolution.
3. **Video sampling FPS** means how many frames to extract per second, not the video's original FPS.
4. **Maximum views** is the cap applied after sampling/extraction.
5. **Preview mirror** only changes the display GLB, not the exported OBJ/STL.
6. **Compare selected methods** is the best way to evaluate meshers fairly.
7. **Gaussian `.ply`** is not a mesh; it is a splat representation for a dedicated viewer.
8. If a control changes the neural prediction, you generally need a new DA3 inference. If it changes only downstream geometry, you can usually rebuild without rerunning DA3.

---

# 17. Suggested repo placement

This file is best added to the repo as something like:

- `MULTIVIEW_CONTROL_REFERENCE.md` if you want the emphasis on the multiview tab, or
- `GUI_REFERENCE_GUIDE.md` if you want one comprehensive operator manual for the whole application.

If you want, the next natural follow-up is to split this into:

1. a **GUI reference manual**;
2. a shorter **quick-start guide**;
3. a **troubleshooting guide**.
