# Multi-View and Gaussian Testing Guide

This guide is intended to make comparisons meaningful by changing one variable at a time.

---

# 1. Establish a fixed DA3 input baseline

Use one image set or one sampled video sequence that produces a good DA3 point cloud.

Do not change the source frames, DA3 model, process resolution, reference strategy, or pose mode while comparing meshers.

Recommended first baseline:

```text
Model:                  DA3-GIANT-1.1
Process resolution:     504
Ray pose:               ON
Confidence percentile:  40
```

For video, start at 1-2 sampling FPS.

---

# 2. Native mesh A/B/C comparison

Choose **Compare selected methods** and select:

```text
TSDF
Poisson
Consistent Surfel + Poisson
```

For a diagnostic comparison, temporarily use:

```text
Shared smoothing:          0
Target triangle count:     0
Keep largest component:    OFF
```

This prevents post-processing from hiding backend differences.

Judge:

- overall proportions,
- facial / object contours,
- thin structures,
- double surfaces,
- corrugation / terracing,
- inflated or melted regions,
- missing supported geometry.

---

# 3. Consistent Surfel tuning

Start with:

```text
Pixel stride:                    2
Neighbor cameras:                4
Minimum total supporting views:  2
Depth agreement tolerance:       2.0%
Normal disagreement:             55 degrees
Merge voxel size:                0 / auto
Maximum oriented surfels:        500000
Poisson depth:                   9
Density trim:                    0
```

Use the reported statistics rather than guessing.

Important outputs:

- sampled observations,
- accepted observations,
- acceptance percentage,
- merged surfels,
- median supporting views.

Tuning order:

1. Test depth tolerance 2% -> 3% -> 5% if too many holes appear.
2. Test 1% if the result is too permissive / doubled / rippled.
3. After finding a good tolerance, test minimum support = 3.
4. Compare normal tolerance 55 vs 90 vs 180 degrees.
5. Only then increase pixel density to stride 1.
6. Compare Poisson depth 9 vs 10.

After the raw geometry is satisfactory, add only light smoothing and then decimate to the desired print/export triangle count.

---

# 4. OpenMVS test

Use exactly the same DA3 source sequence and camera solution.

First run plain **OpenMVS**, then **OpenMVS + RefineMesh** with all other settings unchanged.

Suggested first settings:

```text
Initialization confidence:   95
Resolution level:             1
Neighbor views:               6
Minimum agreeing views:       2
Refine resolution level:      1
Smoothing:                    0
Target triangles:             0
```

The plain OpenMVS result tests geometric reconstruction. The RefineMesh result tests whether returning to the source photographs improves the initial surface.

---

# 5. 2DGS neural surface test

Do not begin with the longest run.

Use:

```text
Iterations:              7000
Mesh extraction:         1024
Depth statistic:         mean / 0
Unbounded:               OFF for compact subject
Smoothing:               0
Target triangles:        0
```

If 7000 iterations is promising, compare 15000. Only test 30000 if 15000 produces a meaningful quality increase.

---

# 6. Native DA3 Gaussian Splat test

Native Gaussian output is independent of the mesh method.

Supported models:

```text
DA3-GIANT-1.1
DA3NESTED-GIANT-LARGE-1.1
```

For a moving-camera scene / environment video:

```text
Video sampling FPS:         1 or 2
Maximum views:              0 for uncapped experiment
Process resolution:         504
Ray pose:                   ON
Native Gaussian PLY:        ON
Gaussian preview video:     OFF unless gsplat is installed
```

The Gaussian PLY should be inspected in a 3DGS-aware viewer, not a conventional mesh viewer.

The purpose of this test is visual novel-view reconstruction, not printable watertight geometry.

---

# 7. Understand video sampling before stress testing

For a 36-second clip:

```text
1 FPS -> about 36 sampled frames
2 FPS -> about 72 sampled frames
5 FPS -> about 180 sampled frames
30 FPS -> about 1080 sampled frames
```

If Maximum views is 18, only 18 evenly spaced samples are sent to DA3.
If Maximum views is 0, all sampled frames are sent to DA3.

A 30-FPS / Max-0 run is therefore an extreme stress test, not a normal quality preset.

---

# 8. Only test frame count after choosing a promising backend

Once a mesh or Gaussian path is working well, then compare input density.

Suggested sequence:

```text
8 views
12 views
18 views
36-72 well-spaced video frames
larger uncapped set if useful
```

More frames are not automatically better. Overlap, sharpness, viewpoint diversity, camera-pose consistency, and a mostly static scene matter more than raw frame count.

---

# 9. What to save when reporting a failure

Keep:

- console traceback,
- `multiview_metadata.json`,
- `multiview_input_order.txt`,
- `camera_pose_diagnostic.glb`,
- backend `*.log` files,
- DA3 point-cloud `scene.glb`,
- the failing mesh or Gaussian output,
- exact settings used.

Also collect:

```cmd
python --version
python -m pip freeze
nvidia-smi
```

When a mesh looks bad but the DA3 point cloud looks good, treat the mesher as the primary suspect. When the camera diagnostic is obviously wrong, fix input/frame/pose quality before spending time tuning the mesher.
