# Selectable DA3 / External Depth -> Blender/3D-Printable Solid
# Self-contained Gradio GUI
#
# Run from inside the Depth-Anything-3 repository / venv:
#   python da3_print_gui.py
#
# Recommended extra dependencies:
#   pip install --upgrade gradio trimesh open3d
#
# The DA3 package itself should already be installed with:
#   pip install -e .

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# OpenCV often gates EXR support behind this environment variable.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402

import gradio as gr
import trimesh

from depth_anything_3.api import DepthAnything3


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_MODEL_ID = "depth-anything/DA3MONO-LARGE"
MODEL_CHOICES = [
    ("DA3Mono-Large (recommended monocular)", "depth-anything/DA3MONO-LARGE"),
    ("DA3-Large-1.1 (any-view large)", "depth-anything/DA3-LARGE-1.1"),
    ("DA3-Giant-1.1 (any-view giant)", "depth-anything/DA3-GIANT-1.1"),
    ("DA3Nested-Giant-Large-1.1 (metric any-view)", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
]
MODEL_ID_TO_LABEL = {value: label for label, value in MODEL_CHOICES}
APP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = APP_ROOT / "gradio_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

_MODEL = None
_MODEL_DEVICE = None
_MODEL_ID_LOADED = None


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def get_model(model_id: str):
    global _MODEL, _MODEL_DEVICE, _MODEL_ID_LOADED

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This GUI is configured for GPU inference."
        )

    if _MODEL_DEVICE is None:
        _MODEL_DEVICE = torch.device("cuda")

    if _MODEL is None or _MODEL_ID_LOADED != model_id:
        if _MODEL is not None:
            print(f"Unloading {_MODEL_ID_LOADED} before loading {model_id}...")
            try:
                del _MODEL
            except Exception:
                pass
            _MODEL = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Loading {model_id} on {_MODEL_DEVICE}...")
        _MODEL = DepthAnything3.from_pretrained(model_id)
        _MODEL = _MODEL.to(device=_MODEL_DEVICE)
        _MODEL.eval()
        _MODEL_ID_LOADED = model_id

        print(
            f"Loaded {model_id} on "
            f"{torch.cuda.get_device_name(0)} "
            f"(PyTorch {torch.__version__}, CUDA {torch.version.cuda})"
        )

    return _MODEL, _MODEL_DEVICE


# ---------------------------------------------------------------------
# Image / depth helpers
# ---------------------------------------------------------------------

def image_value_to_pil(image_value) -> Image.Image:
    """Convert Gradio gr.Image input to a normal RGB PIL image."""
    if image_value is None:
        raise ValueError("Upload an image first.")

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, (str, Path)):
        return Image.open(image_value).convert("RGB")

    if isinstance(image_value, np.ndarray):
        arr = image_value

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                if arr.size and arr.max() <= 1.0:
                    arr = np.clip(arr * 255.0, 0, 255)
                arr = arr.astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")

        if arr.ndim == 3 and arr.shape[-1] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")

        if arr.ndim == 3:
            return Image.fromarray(arr[..., :3], mode="RGB")

    raise TypeError(f"Unsupported image type: {type(image_value)}")


def apply_crop(
    img: Image.Image,
    crop_enabled: bool,
    crop_x1,
    crop_y1,
    crop_x2,
    crop_y2,
) -> Image.Image:
    """Apply a validated pixel-coordinate crop if enabled."""
    if not crop_enabled:
        return img

    w, h = img.size

    x1 = max(0, min(w - 1, int(round(float(crop_x1)))))
    y1 = max(0, min(h - 1, int(round(float(crop_y1)))))
    x2 = max(1, min(w, int(round(float(crop_x2)))))
    y2 = max(1, min(h, int(round(float(crop_y2)))))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop: ({x1}, {y1}) to ({x2}, {y2}) for image {w}x{h}."
        )

    return img.crop((x1, y1, x2, y2))


def initialize_crop_ui(image_value):
    """Reset crop coordinates to the full uploaded image."""
    if image_value is None:
        return 0, 0, 1, 1, "Upload an image to initialize crop coordinates."

    img = image_value_to_pil(image_value)
    w, h = img.size
    return 0, 0, w, h, f"**Source resolution:** {w} × {h} pixels"


def preview_crop_ui(
    image_value,
    crop_enabled,
    crop_x1,
    crop_y1,
    crop_x2,
    crop_y2,
):
    img = image_value_to_pil(image_value)
    cropped = apply_crop(
        img,
        bool(crop_enabled),
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )
    w, h = cropped.size
    return cropped, f"**Image that will be sent to DA3:** {w} × {h} pixels"

def resize_float_map(float_map: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    img = Image.fromarray(float_map.astype(np.float32), mode="F")
    img = img.resize((out_w, out_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def normalize_depth(
    depth: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> np.ndarray:
    d = depth.astype(np.float32)
    finite = np.isfinite(d)

    if not finite.any():
        raise ValueError("Depth map contains no finite values.")

    valid = d[finite]
    lo = float(np.percentile(valid, lo_pct))
    hi = float(np.percentile(valid, hi_pct))

    if hi <= lo:
        hi = lo + 1e-6

    out = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def save_16bit_png(norm_map: np.ndarray, path: Path):
    arr16 = np.clip(norm_map * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(arr16, mode="I;16").save(path)


def save_8bit_png(norm_map: np.ndarray, path: Path):
    arr8 = np.clip(norm_map * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="L").save(path)


def save_exr_float32(float_map: np.ndarray, path: Path):
    arr = np.ascontiguousarray(float_map.astype(np.float32))
    ok = cv2.imwrite(str(path), arr)

    if not ok:
        raise RuntimeError(
            f"OpenCV could not write EXR: {path}\n"
            "If this is an OpenEXR support issue, verify that "
            "OPENCV_IO_ENABLE_OPENEXR=1 is set before importing cv2."
        )


def create_job_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    path = OUTPUT_ROOT / f"{stamp}-{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def copy_uploaded_file(src, dst: Path) -> Path | None:
    if not src:
        return None

    src_path = Path(src)
    shutil.copy2(src_path, dst)
    return dst


def human_vram() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return f"{allocated:.2f} GB allocated / {reserved:.2f} GB reserved"


# ---------------------------------------------------------------------
# DA3 depth generation
# ---------------------------------------------------------------------

def generate_depth_core(
    image_value,
    crop_enabled: bool,
    crop_x1,
    crop_y1,
    crop_x2,
    crop_y2,
    model_id: str,
    process_res: int,
    near_pct: float,
    far_pct: float,
    progress=None,
):
    if near_pct >= far_pct:
        raise ValueError("Near percentile must be lower than far percentile.")

    if progress:
        progress(0.02, desc="Preparing image")

    img = image_value_to_pil(image_value)
    img = apply_crop(
        img,
        bool(crop_enabled),
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )
    target_w, target_h = img.size

    job_dir = create_job_dir()

    input_path = job_dir / "input_used.png"
    img.save(input_path)

    if progress:
        progress(0.10, desc=f"Loading {MODEL_ID_TO_LABEL.get(model_id, model_id)}")

    model, device = get_model(model_id)

    if progress:
        progress(0.18, desc=f"Running DA3 at process_res={int(process_res)}")

    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model.inference(
            [img],
            process_res=int(process_res),
            process_res_method="upper_bound_resize",
        )

    infer_seconds = time.perf_counter() - start

    depth = np.asarray(prediction.depth[0], dtype=np.float32)

    conf_obj = getattr(prediction, "conf", None)
    conf = None
    if conf_obj is not None:
        conf = np.asarray(conf_obj[0], dtype=np.float32)

    if depth.shape != (target_h, target_w):
        depth = resize_float_map(depth, target_w, target_h)

    if conf is not None and conf.shape != (target_h, target_w):
        conf = resize_float_map(conf, target_w, target_h)

    if progress:
        progress(0.75, desc="Writing 32-bit EXR / PNG outputs")

    raw_npy = job_dir / "depth_raw_float32.npy"
    raw_exr = job_dir / "depth_raw_float32.exr"

    np.save(raw_npy, depth)
    save_exr_float32(depth, raw_exr)

    depth_far_white = normalize_depth(
        depth,
        lo_pct=float(near_pct),
        hi_pct=float(far_pct),
    )
    depth_near_white = 1.0 - depth_far_white

    far_exr = job_dir / "depth_far_white_norm32.exr"
    near_exr = job_dir / "depth_near_white_norm32.exr"
    far_png16 = job_dir / "depth_far_white_16bit.png"
    near_png16 = job_dir / "depth_near_white_16bit.png"
    far_preview = job_dir / "depth_far_white_preview_8bit.png"
    near_preview = job_dir / "depth_near_white_preview_8bit.png"

    save_exr_float32(depth_far_white, far_exr)
    save_exr_float32(depth_near_white, near_exr)
    save_16bit_png(depth_far_white, far_png16)
    save_16bit_png(depth_near_white, near_png16)
    save_8bit_png(depth_far_white, far_preview)
    save_8bit_png(depth_near_white, near_preview)

    depth_files = [
        input_path,
        raw_npy,
        raw_exr,
        near_exr,
        far_exr,
        near_png16,
        far_png16,
        near_preview,
        far_preview,
    ]

    if conf is not None:
        conf_npy = job_dir / "confidence_raw_float32.npy"
        conf_exr = job_dir / "confidence_raw_float32.exr"
        conf_norm_exr = job_dir / "confidence_norm32.exr"
        conf_png16 = job_dir / "confidence_16bit.png"
        conf_preview = job_dir / "confidence_preview_8bit.png"

        np.save(conf_npy, conf)
        save_exr_float32(conf, conf_exr)

        conf_norm = normalize_depth(conf, 1.0, 99.0)
        save_exr_float32(conf_norm, conf_norm_exr)
        save_16bit_png(conf_norm, conf_png16)
        save_8bit_png(conf_norm, conf_preview)

        depth_files.extend(
            [
                conf_npy,
                conf_exr,
                conf_norm_exr,
                conf_png16,
                conf_preview,
            ]
        )

    metadata = {
        "model": model_id,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "process_res": int(process_res),
        "input_width": target_w,
        "input_height": target_h,
        "near_percentile": float(near_pct),
        "far_percentile": float(far_pct),
        "inference_seconds": infer_seconds,
    }

    metadata_path = job_dir / "depth_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    depth_files.append(metadata_path)

    if progress:
        progress(1.0, desc="Depth complete")

    status = (
        "### Depth complete\n"
        f"- **Input used:** {target_w} × {target_h}\n"
        f"- **Model:** {MODEL_ID_TO_LABEL.get(model_id, model_id)} (`{model_id}`)\n"
        f"- **DA3 process resolution:** {int(process_res)}\n"
        f"- **Inference:** {infer_seconds:.2f} s\n"
        f"- **GPU:** {torch.cuda.get_device_name(0)}\n"
        f"- **VRAM now:** {human_vram()}\n"
        f"- **Output folder:** `{job_dir}`\n\n"
        "The **near-white 16-bit PNG** is convenient for Blender displacement; "
        "the **raw 32-bit EXR/NPY** preserves the DA3 depth values."
    )

    return {
        "job_dir": job_dir,
        "input_path": input_path,
        "raw_depth_path": raw_exr,
        "near_preview": near_preview,
        "depth_files": depth_files,
        "status": status,
    }


def generate_depth_ui(
    image_value,
    crop_enabled,
    crop_x1,
    crop_y1,
    crop_x2,
    crop_y2,
    model_id,
    process_res,
    near_pct,
    far_pct,
    progress=gr.Progress(),
):
    result = generate_depth_core(
        image_value,
        bool(crop_enabled),
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
        str(model_id),
        int(process_res),
        float(near_pct),
        float(far_pct),
        progress,
    )

    return (
        str(result["job_dir"]),
        str(result["input_path"]),
        str(result["near_preview"]),
        result["status"],
        [str(p) for p in result["depth_files"]],
    )


# ---------------------------------------------------------------------
# Solid mesh generation
# ---------------------------------------------------------------------

def load_depth_map(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(path).astype(np.float32)

    elif suffix == ".exr":
        depth = cv2.imread(
            str(path),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )

        if depth is None:
            raise RuntimeError(f"Could not read EXR: {path}")

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        depth = depth.astype(np.float32)

    else:
        arr = np.asarray(Image.open(path))

        if arr.ndim == 3:
            arr = arr[..., 0]

        if np.issubdtype(arr.dtype, np.integer):
            depth = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        else:
            depth = arr.astype(np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got {depth.shape}")

    return depth


def maybe_downsample_depth(depth: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = depth.shape

    if max(h, w) <= int(max_dim):
        return depth

    scale = float(max_dim) / max(h, w)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))

    return resize_float_map(depth, new_w, new_h)


def apply_depth_curve(
    depth_norm: np.ndarray,
    enabled: bool,
    foreground_percent: float,
    background_z_percent: float,
    threshold_mode: str,
):
    """
    Apply a smooth monotonic "Curves for depth" remap.

    depth_norm is expected to be normalized to [0, 1] *after* optional inversion,
    so 0 is the far/background end and 1 is the near/foreground end.

    The curve is constrained by three points:
        (0, 0), (threshold, background_z_fraction), (1, 1)

    A power curve y = x**gamma is chosen so it passes exactly through the
    threshold point. This gives a smooth transition with no artificial ridge.
    """
    x = np.clip(depth_norm.astype(np.float32), 0.0, 1.0)

    if not bool(enabled):
        return x, {
            "enabled": False,
            "threshold_mode": str(threshold_mode),
            "foreground_percent": float(foreground_percent),
            "background_z_percent": float(background_z_percent),
            "threshold": None,
            "gamma": 1.0,
        }

    fg_pct = float(np.clip(float(foreground_percent), 1.0, 99.0))
    bg_pct = float(np.clip(float(background_z_percent), 0.1, 99.9))

    if str(threshold_mode) == "Nearest pixels (percentile)":
        finite = np.isfinite(x)
        if not finite.any():
            raise ValueError("Depth map contains no finite values for depth-curve thresholding.")
        # Example: foreground_percent=25 -> threshold at the 75th percentile,
        # so roughly the nearest 25% of pixels lie above the threshold.
        threshold = float(np.percentile(x[finite], 100.0 - fg_pct))
    else:
        # Example: foreground_percent=25 -> top quarter of the normalized depth
        # range begins at 0.75.
        threshold = 1.0 - fg_pct / 100.0

    # Avoid log(0)/log(1) degeneracies. The clamp is tiny enough not to be
    # visually meaningful but keeps the mapping mathematically stable.
    threshold = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    background_fraction = float(np.clip(bg_pct / 100.0, 1e-4, 1.0 - 1e-4))

    gamma = float(np.log(background_fraction) / np.log(threshold))
    gamma = float(np.clip(gamma, 0.05, 100.0))

    curved = np.power(x, gamma).astype(np.float32)
    curved = np.clip(curved, 0.0, 1.0)

    return curved, {
        "enabled": True,
        "threshold_mode": str(threshold_mode),
        "foreground_percent": fg_pct,
        "background_z_percent": bg_pct,
        "threshold": threshold,
        "gamma": gamma,
    }


def build_solid_mesh(
    depth_norm: np.ndarray,
    mesh_width: float,
    z_scale: float,
    base_thickness: float,
):
    rows, cols = depth_norm.shape

    aspect = rows / cols
    mesh_height = mesh_width * aspect

    xs = np.linspace(
        -mesh_width / 2.0,
        mesh_width / 2.0,
        cols,
        dtype=np.float32,
    )

    ys = np.linspace(
        mesh_height / 2.0,
        -mesh_height / 2.0,
        rows,
        dtype=np.float32,
    )

    xx, yy = np.meshgrid(xs, ys)

    zz = (
        float(base_thickness)
        + depth_norm * float(z_scale)
    ).astype(np.float32)

    top_vertices = np.column_stack(
        (xx.ravel(), yy.ravel(), zz.ravel())
    ).astype(np.float32)

    n_top = len(top_vertices)

    r = np.arange(rows - 1)[:, None]
    c = np.arange(cols - 1)[None, :]

    v1 = (r * cols + c).ravel()
    v2 = (r * cols + c + 1).ravel()
    v4 = ((r + 1) * cols + c).ravel()
    v3 = ((r + 1) * cols + c + 1).ravel()

    top_faces_a = np.column_stack((v1, v3, v2))
    top_faces_b = np.column_stack((v1, v4, v3))

    top_faces = np.vstack(
        (top_faces_a, top_faces_b)
    ).astype(np.int32)

    # Clockwise perimeter when viewed from +Z.
    perimeter = []

    for col in range(cols):
        perimeter.append(col)

    for row in range(1, rows):
        perimeter.append(row * cols + (cols - 1))

    for col in range(cols - 2, -1, -1):
        perimeter.append((rows - 1) * cols + col)

    for row in range(rows - 2, 0, -1):
        perimeter.append(row * cols)

    perimeter = np.asarray(perimeter, dtype=np.int32)
    n_perimeter = len(perimeter)

    bottom_ring = top_vertices[perimeter].copy()
    bottom_ring[:, 2] = 0.0

    bottom_start = n_top

    bottom_center = np.array(
        [[0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    center_index = n_top + n_perimeter

    vertices = np.vstack(
        (top_vertices, bottom_ring, bottom_center)
    ).astype(np.float32)

    i = np.arange(n_perimeter, dtype=np.int32)
    j = (i + 1) % n_perimeter

    top_i = perimeter[i]
    top_j = perimeter[j]
    bottom_i = bottom_start + i
    bottom_j = bottom_start + j

    side_faces_a = np.column_stack(
        (top_i, top_j, bottom_j)
    )
    side_faces_b = np.column_stack(
        (top_i, bottom_j, bottom_i)
    )

    side_faces = np.vstack(
        (side_faces_a, side_faces_b)
    ).astype(np.int32)

    bottom_faces = np.column_stack(
        (
            np.full(n_perimeter, center_index, dtype=np.int32),
            bottom_i,
            bottom_j,
        )
    ).astype(np.int32)

    faces = np.vstack(
        (top_faces, side_faces, bottom_faces)
    ).astype(np.int32)

    return (
        vertices,
        faces,
        top_faces,
        side_faces,
        bottom_faces,
        mesh_height,
    )


def write_mtl(mtl_path: Path, texture_filename: str | None):
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl front_texture\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")

        if texture_filename:
            f.write(f"map_Kd {texture_filename}\n")

        f.write("\n")
        f.write("newmtl solid_base\n")
        f.write("Ka 0.500000 0.500000 0.500000\n")
        f.write("Kd 0.650000 0.650000 0.650000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")


def write_obj(
    obj_path: Path,
    vertices: np.ndarray,
    top_faces: np.ndarray,
    side_faces: np.ndarray,
    bottom_faces: np.ndarray,
    rows: int,
    cols: int,
    texture_filename: str | None,
):
    mtl_name = obj_path.with_suffix(".mtl").name

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# Closed DA3 heightfield solid\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("o da3_heightfield_solid\n")

        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

        for row in range(rows):
            v = 1.0 - row / (rows - 1) if rows > 1 else 0.0

            for col in range(cols):
                u = col / (cols - 1) if cols > 1 else 0.0
                f.write(f"vt {u:.8f} {v:.8f}\n")

        f.write("usemtl front_texture\n")

        for a, b, c in top_faces:
            a1, b1, c1 = int(a) + 1, int(b) + 1, int(c) + 1
            f.write(
                f"f {a1}/{a1} {b1}/{b1} {c1}/{c1}\n"
            )

        f.write("usemtl solid_base\n")

        for face_array in (side_faces, bottom_faces):
            for a, b, c in face_array:
                f.write(
                    f"f {int(a)+1} {int(b)+1} {int(c)+1}\n"
                )


def zip_job(job_dir: Path) -> Path:
    zip_path = job_dir / "DA3_depth_and_printable_mesh.zip"

    files = [
        p for p in job_dir.iterdir()
        if p.is_file() and p != zip_path
    ]

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for path in files:
            zf.write(path, arcname=path.name)

    return zip_path


# ---------------------------------------------------------------------
# Multi-view DA3 reconstruction
# ---------------------------------------------------------------------

MULTIVIEW_MODEL_IDS = {
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3-GIANT-1.1",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}


def collect_uploaded_files(file_list) -> list[Path]:
    """Resolve a Gradio multi-file upload into local Paths."""
    if not file_list:
        return []

    result = []
    for item in file_list:
        if item is None:
            continue

        if isinstance(item, (str, Path)):
            result.append(Path(item))
            continue

        if isinstance(item, dict):
            candidate = item.get("path") or item.get("name")
            if candidate:
                result.append(Path(candidate))
                continue

        candidate = getattr(item, "name", None)
        if candidate:
            result.append(Path(candidate))
            continue

        raise TypeError(f"Unsupported uploaded file type: {type(item)}")

    return result


def natural_sort_key(value) -> list:
    """Human/natural filename ordering: frame2 comes before frame10."""
    name = Path(value).name.casefold()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def evenly_spaced_indices(count: int, maximum: int) -> list[int]:
    """Choose deterministic, evenly spaced samples while retaining endpoints."""
    if maximum <= 0 or count <= maximum:
        return list(range(count))
    if maximum == 1:
        return [count // 2]

    raw = np.linspace(0, count - 1, maximum)
    indices = np.rint(raw).astype(int).tolist()

    # np.rint can theoretically duplicate indices for small arrays; preserve order.
    deduped = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)

    # Fill any lost slots deterministically.
    if len(deduped) < maximum:
        for idx in range(count):
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
                if len(deduped) >= maximum:
                    break
        deduped.sort()

    return deduped


def _copy_ordered_inputs(
    source_paths: list[Path],
    input_dir: Path,
) -> list[Path]:
    copied = []
    input_dir.mkdir(parents=True, exist_ok=True)

    for index, src in enumerate(source_paths):
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(f"Input image not found: {src}")

        suffix = src.suffix.lower() or ".png"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem)[:80]
        dst = input_dir / f"{index:03d}_{safe_name}{suffix}"
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def prepare_multiview_inputs(
    input_mode: str,
    multi_image_files,
    video_file,
    video_sample_fps: float,
    colmap_dir_path: str,
    colmap_sparse_subdir: str,
    max_views: int,
    job_dir: Path,
):
    """
    Prepare DA3 inputs in deterministic order.

    Returns:
        copied_images, input_extrinsics, input_intrinsics, source_kind, metadata
    """
    input_dir = job_dir / "multiview_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    input_extrinsics = None
    input_intrinsics = None
    source_meta = {"input_mode": str(input_mode)}

    if input_mode == "Video file":
        if not video_file:
            raise ValueError("Choose a video file first.")
        if float(video_sample_fps) <= 0:
            raise ValueError("Video sampling FPS must be greater than zero.")

        video_paths = collect_uploaded_files([video_file])
        if not video_paths:
            raise ValueError("Could not resolve the uploaded video file.")
        video_path = video_paths[0]

        # Use DA3's own VideoHandler so sampling behavior matches the pinned
        # upstream CLI: requested FPS -> frame interval -> sorted extracted files.
        try:
            from depth_anything_3.services.input_handlers import VideoHandler
        except Exception as exc:
            raise RuntimeError(
                "Could not import DA3's official VideoHandler from the installed "
                "Depth Anything 3 package."
            ) from exc

        extract_root = job_dir / "video_extract"
        extracted = [
            Path(p)
            for p in VideoHandler.process(
                str(video_path),
                str(extract_root),
                float(video_sample_fps),
            )
        ]
        extracted = sorted(extracted, key=natural_sort_key)

        indices = evenly_spaced_indices(len(extracted), int(max_views))
        selected = [extracted[i] for i in indices]
        copied_images = _copy_ordered_inputs(selected, input_dir)
        # Keep the job ZIP focused on the actual inference views rather than
        # every intermediate frame extracted from a long video.
        shutil.rmtree(extract_root, ignore_errors=True)

        source_meta.update(
            {
                "video_file": video_path.name,
                "requested_sample_fps": float(video_sample_fps),
                "frames_extracted": len(extracted),
                "frames_selected": len(selected),
                "selected_frame_names": [p.name for p in selected],
            }
        )
        source_kind = "video"

    elif input_mode == "COLMAP local dataset (known poses)":
        colmap_raw = str(colmap_dir_path or "").strip().strip('"')
        if not colmap_raw:
            raise ValueError(
                "Enter a local COLMAP dataset folder containing images/ and sparse/."
            )
        colmap_root = Path(colmap_raw)
        if not colmap_root.exists():
            raise ValueError(
                "Enter a valid local COLMAP dataset folder containing images/ and sparse/."
            )

        try:
            from depth_anything_3.services.input_handlers import ColmapHandler
        except Exception as exc:
            raise RuntimeError(
                "Could not import DA3's official ColmapHandler from the installed "
                "Depth Anything 3 package."
            ) from exc

        image_files, extrinsics, intrinsics = ColmapHandler.process(
            str(colmap_root),
            str(colmap_sparse_subdir or "").strip(),
        )

        records = list(zip(image_files, extrinsics, intrinsics))
        records.sort(key=lambda item: natural_sort_key(item[0]))

        indices = evenly_spaced_indices(len(records), int(max_views))
        selected = [records[i] for i in indices]

        source_paths = [Path(item[0]) for item in selected]
        input_extrinsics = np.asarray([item[1] for item in selected], dtype=np.float32)
        input_intrinsics = np.asarray([item[2] for item in selected], dtype=np.float32)
        copied_images = _copy_ordered_inputs(source_paths, input_dir)

        source_meta.update(
            {
                "colmap_dir": str(colmap_root),
                "sparse_subdir": str(colmap_sparse_subdir or ""),
                "colmap_views_available": len(records),
                "views_selected": len(selected),
                "known_poses_supplied": True,
            }
        )
        source_kind = "colmap"

    else:
        image_paths = collect_uploaded_files(multi_image_files)
        if len(image_paths) < 2:
            raise ValueError(
                "Upload at least 2 images of the same subject/scene for multi-view reconstruction."
            )

        # Deterministic natural ordering is important for ordered frame sequences,
        # and harmless/reproducible for unordered photo collections.
        image_paths = sorted(image_paths, key=natural_sort_key)
        indices = evenly_spaced_indices(len(image_paths), int(max_views))
        selected = [image_paths[i] for i in indices]
        copied_images = _copy_ordered_inputs(selected, input_dir)

        source_kind = (
            "ordered_frames"
            if input_mode == "Ordered uploaded frames"
            else "unordered_photos"
        )
        source_meta.update(
            {
                "uploaded_images": len(image_paths),
                "views_selected": len(selected),
                "selected_original_names": [p.name for p in selected],
            }
        )

    if len(copied_images) < 2:
        raise ValueError("At least 2 usable views are required after input selection.")

    return (
        copied_images,
        input_extrinsics,
        input_intrinsics,
        source_kind,
        source_meta,
    )


def extract_transformed_points_from_glb(glb_path: Path):
    """
    Extract the DA3 GLB point cloud into one world-space point array.

    DA3's GLB can contain scene-graph transforms, so we intentionally apply
    every node transform instead of reading raw geometry vertices directly.
    """
    scene = trimesh.load(str(glb_path), force="scene", process=False)

    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    points_parts = []
    color_parts = []

    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph.get(node_name)
        if geom_name is None or geom_name not in scene.geometry:
            continue

        geom = scene.geometry[geom_name]
        vertices = getattr(geom, "vertices", None)
        if vertices is None:
            continue

        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            continue

        world_vertices = trimesh.transform_points(vertices, transform)
        points_parts.append(world_vertices)

        colors = None
        if hasattr(geom, "colors") and getattr(geom, "colors") is not None:
            colors = np.asarray(geom.colors)
        elif hasattr(geom, "visual"):
            vertex_colors = getattr(geom.visual, "vertex_colors", None)
            if vertex_colors is not None:
                colors = np.asarray(vertex_colors)

        if colors is not None and len(colors) == len(vertices):
            colors = colors[:, :3].astype(np.float64)
            if colors.size and colors.max() > 1.0:
                colors /= 255.0
            color_parts.append(np.clip(colors, 0.0, 1.0))
        else:
            color_parts.append(
                np.full((len(vertices), 3), 0.75, dtype=np.float64)
            )

    if not points_parts:
        raise RuntimeError(
            f"No point-cloud geometry could be extracted from DA3 GLB: {glb_path}"
        )

    return (
        np.concatenate(points_parts, axis=0),
        np.concatenate(color_parts, axis=0),
    )


def _to_4x4_extrinsic(extrinsic) -> np.ndarray:
    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = ext
        return out
    raise ValueError(f"Expected 3x4 or 4x4 extrinsic, got {ext.shape}")


def create_camera_diagnostic_glb(
    prediction,
    output_dir: Path,
    conf_thresh_percentile: float,
    num_max_points: int,
):
    """
    Export an official DA3 diagnostic GLB with camera wireframes enabled.

    Using DA3's exporter here is important because its GLB path performs scene
    scale normalization; exporting the points and cameras together keeps them in
    the exact same display coordinate system.
    """
    diag_json = output_dir / "camera_poses.json"

    extrinsics = getattr(prediction, "extrinsics", None)
    intrinsics = getattr(prediction, "intrinsics", None)

    if extrinsics is not None:
        exts = np.asarray(extrinsics, dtype=np.float64)
        ixts = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        records = []
        for idx, ext in enumerate(exts):
            ext4 = _to_4x4_extrinsic(ext)
            c2w = np.linalg.inv(ext4)
            records.append(
                {
                    "index": idx,
                    "world_to_camera": ext4.tolist(),
                    "camera_to_world": c2w.tolist(),
                    "camera_center_world": c2w[:3, 3].tolist(),
                    "intrinsics": (
                        ixts[idx].tolist()
                        if ixts is not None and idx < len(ixts)
                        else None
                    ),
                }
            )
        diag_json.write_text(
            json.dumps({"cameras": records}, indent=2),
            encoding="utf-8",
        )
    else:
        diag_json = None

    diag_dir = output_dir / "da3_camera_diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_glb = None

    try:
        from depth_anything_3.utils.export import export as da3_export

        da3_export(
            prediction,
            "glb",
            str(diag_dir),
            glb={
                "conf_thresh_percentile": float(conf_thresh_percentile),
                "num_max_points": int(num_max_points),
                "show_cameras": True,
            },
        )

        candidates = list(diag_dir.rglob("*.glb"))
        if candidates:
            diag_glb = output_dir / "camera_pose_diagnostic.glb"
            shutil.copy2(candidates[0], diag_glb)
    except Exception as exc:
        print(f"WARNING: Could not create official DA3 camera diagnostic GLB: {exc}")

    return diag_glb, diag_json

def _cleanup_open3d_mesh(
    mesh,
    keep_largest_component: bool,
    smooth_iterations: int,
    target_faces: int,
):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if bool(keep_largest_component) and len(mesh.triangles) > 0:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) > 1:
            largest_cluster = int(np.argmax(cluster_n_triangles))
            mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster)
            mesh.remove_unreferenced_vertices()

    if int(smooth_iterations) > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=int(smooth_iterations)
        )

    if int(target_faces) > 0 and len(mesh.triangles) > int(target_faces):
        mesh = mesh.simplify_quadric_decimation(int(target_faces))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def _export_open3d_mesh_bundle(mesh, output_dir: Path, prefix: str):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for multi-view meshing.") from exc

    obj_path = output_dir / f"{prefix}.obj"
    stl_path = output_dir / f"{prefix}.stl"
    ply_path = output_dir / f"{prefix}.ply"
    preview_glb_path = output_dir / f"{prefix}_preview.glb"

    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(stl_path), mesh)
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)

    mesh_tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )
    try:
        mesh_tm.fix_normals(multibody=True)
    except Exception:
        pass

    preview_material = trimesh.visual.material.PBRMaterial(
        name="neutral_preview",
        baseColorFactor=[125, 125, 125, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=True,
    )
    mesh_tm.visual = trimesh.visual.TextureVisuals(material=preview_material)
    trimesh.Scene(mesh_tm).export(preview_glb_path)

    return {
        "obj_path": obj_path,
        "stl_path": stl_path,
        "ply_path": ply_path,
        "preview_glb_path": preview_glb_path,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "watertight": bool(mesh_tm.is_watertight),
        "euler_number": int(mesh_tm.euler_number),
    }


def reconstruct_poisson_mesh(
    glb_path: Path,
    output_dir: Path,
    voxel_size: float,
    poisson_depth: int,
    trim_percentile: float,
    smooth_iterations: int,
    target_faces: int,
    keep_largest_component: bool,
    prefix: str = "multiview_poisson",
):
    """Convert DA3's fused GLB point cloud into a Poisson surface."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Multi-image mesh reconstruction requires Open3D. "
            "Install it in this venv with: pip install open3d"
        ) from exc

    points, colors = extract_transformed_points_from_glb(glb_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    raw_ply_path = output_dir / f"{prefix}_point_cloud_raw.ply"
    o3d.io.write_point_cloud(str(raw_ply_path), pcd)

    if float(voxel_size) > 0.0:
        pcd = pcd.voxel_down_sample(float(voxel_size))

    if len(pcd.points) < 100:
        raise RuntimeError("Too few valid points remained for Poisson reconstruction.")

    if len(pcd.points) >= 1000:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )

    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except RuntimeError:
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(poisson_depth),
    )

    densities = np.asarray(densities)
    if len(densities) == 0 or len(mesh.vertices) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty mesh.")

    if float(trim_percentile) > 0.0:
        threshold = np.percentile(densities, float(trim_percentile))
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh = _cleanup_open3d_mesh(
        mesh,
        keep_largest_component=bool(keep_largest_component),
        smooth_iterations=int(smooth_iterations),
        target_faces=int(target_faces),
    )

    exported = _export_open3d_mesh_bundle(mesh, output_dir, prefix)
    exported.update(
        {
            "raw_ply_path": raw_ply_path,
            "input_points": int(len(points)),
            "filtered_points": int(len(pcd.points)),
            "method": "Poisson",
        }
    )
    return exported


def _prediction_confidence_mask(prediction, percentile: float):
    depth = np.asarray(prediction.depth, dtype=np.float32)
    base_valid = np.isfinite(depth) & (depth > 0)

    sky = getattr(prediction, "sky", None)
    if sky is not None:
        sky_arr = np.asarray(sky, dtype=bool)
        if sky_arr.shape == depth.shape:
            base_valid &= ~sky_arr

    conf = getattr(prediction, "conf", None)
    threshold = None
    if conf is not None:
        conf_arr = np.asarray(conf, dtype=np.float32)
        if conf_arr.shape == depth.shape:
            conf_values = conf_arr[base_valid & np.isfinite(conf_arr)]
            if len(conf_values):
                threshold = float(np.percentile(conf_values, float(percentile)))
                base_valid &= np.isfinite(conf_arr) & (conf_arr >= threshold)

    return base_valid, threshold


def reconstruct_tsdf_mesh(
    prediction,
    output_dir: Path,
    conf_thresh_percentile: float,
    tsdf_voxels_per_median_depth: int,
    tsdf_truncation_voxels: float,
    tsdf_depth_trunc_percentile: float,
    smooth_iterations: int,
    target_faces: int,
    keep_largest_component: bool,
    prefix: str = "multiview_tsdf",
):
    """
    Fuse DA3's organized depth maps + intrinsics + world-to-camera poses directly
    into an Open3D scalable TSDF volume, then extract a triangle mesh.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "TSDF reconstruction requires Open3D. Install it with: pip install open3d"
        ) from exc

    depth = np.asarray(prediction.depth, dtype=np.float32)
    colors = np.asarray(prediction.processed_images)
    extrinsics = getattr(prediction, "extrinsics", None)
    intrinsics = getattr(prediction, "intrinsics", None)

    if extrinsics is None or intrinsics is None:
        raise RuntimeError(
            "DA3 did not return camera intrinsics/extrinsics, so TSDF fusion cannot run."
        )

    exts = np.asarray(extrinsics, dtype=np.float64)
    ixts = np.asarray(intrinsics, dtype=np.float64)

    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape N,H,W; got {depth.shape}")
    if len(exts) != len(depth) or len(ixts) != len(depth):
        raise ValueError("Depth/camera count mismatch in DA3 prediction.")

    valid_mask, conf_threshold = _prediction_confidence_mask(
        prediction,
        float(conf_thresh_percentile),
    )
    valid_depths = depth[valid_mask]
    if len(valid_depths) < 100:
        raise RuntimeError("Too few valid DA3 depth samples remained for TSDF fusion.")

    median_depth = float(np.median(valid_depths))
    depth_trunc = float(
        np.percentile(valid_depths, float(tsdf_depth_trunc_percentile))
    )
    if not np.isfinite(median_depth) or median_depth <= 0:
        raise RuntimeError("Could not derive a positive median scene depth for TSDF.")
    if not np.isfinite(depth_trunc) or depth_trunc <= 0:
        depth_trunc = float(np.max(valid_depths))

    detail = max(32, int(tsdf_voxels_per_median_depth))
    voxel_length = median_depth / float(detail)
    sdf_trunc = voxel_length * max(1.0, float(tsdf_truncation_voxels))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated_views = 0
    per_view_valid_pixels = []

    for idx in range(len(depth)):
        d = depth[idx].copy()
        frame_valid = valid_mask[idx].copy()
        frame_valid &= d <= depth_trunc
        d[~frame_valid] = 0.0

        valid_count = int(np.count_nonzero(frame_valid))
        per_view_valid_pixels.append(valid_count)
        if valid_count < 50:
            continue

        color = colors[idx]
        h, w = d.shape
        if color.shape[:2] != (h, w):
            color = cv2.resize(
                color,
                (w, h),
                interpolation=cv2.INTER_AREA,
            )
        color = np.ascontiguousarray(np.clip(color, 0, 255).astype(np.uint8))
        d = np.ascontiguousarray(d.astype(np.float32))

        k = ixts[idx]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(w),
            int(h),
            float(k[0, 0]),
            float(k[1, 1]),
            float(k[0, 2]),
            float(k[1, 2]),
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(d),
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        # DA3 documents extrinsics as OpenCV/COLMAP world-to-camera matrices,
        # which is the convention Open3D's TSDF integrate() expects.
        volume.integrate(
            rgbd,
            intrinsic,
            _to_4x4_extrinsic(exts[idx]),
        )
        integrated_views += 1

    if integrated_views < 2:
        raise RuntimeError(
            f"Only {integrated_views} views had enough valid depth for TSDF fusion."
        )

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh.")

    mesh = _cleanup_open3d_mesh(
        mesh,
        keep_largest_component=bool(keep_largest_component),
        smooth_iterations=int(smooth_iterations),
        target_faces=int(target_faces),
    )

    extracted_pcd = volume.extract_point_cloud()
    tsdf_pcd_path = output_dir / f"{prefix}_surface_point_cloud.ply"
    o3d.io.write_point_cloud(str(tsdf_pcd_path), extracted_pcd)

    exported = _export_open3d_mesh_bundle(mesh, output_dir, prefix)
    exported.update(
        {
            "surface_point_cloud_path": tsdf_pcd_path,
            "method": "TSDF",
            "integrated_views": int(integrated_views),
            "per_view_valid_pixels": per_view_valid_pixels,
            "confidence_threshold": conf_threshold,
            "median_depth": median_depth,
            "depth_trunc": depth_trunc,
            "voxel_length": float(voxel_length),
            "sdf_trunc": float(sdf_trunc),
        }
    )
    return exported


def _mesh_result_files(result: dict) -> list[Path]:
    keys = [
        "raw_ply_path",
        "surface_point_cloud_path",
        "ply_path",
        "obj_path",
        "stl_path",
        "preview_glb_path",
    ]
    return [Path(result[k]) for k in keys if result.get(k)]


def _mesh_status_lines(label: str, result: dict) -> list[str]:
    lines = [
        f"#### {label}",
        f"- **Vertices:** {result['vertices']:,}",
        f"- **Triangles:** {result['triangles']:,}",
        f"- **Watertight:** {'YES ✅' if result['watertight'] else 'NO ⚠️'}",
        f"- **Euler number:** {result['euler_number']}",
    ]
    if result.get("method") == "Poisson":
        lines.extend(
            [
                f"- **DA3 GLB points:** {result['input_points']:,}",
                f"- **Points after cleanup:** {result['filtered_points']:,}",
            ]
        )
    if result.get("method") == "TSDF":
        lines.extend(
            [
                f"- **Views integrated:** {result['integrated_views']}",
                f"- **Auto voxel length:** {result['voxel_length']:.6g}",
                f"- **TSDF truncation:** {result['sdf_trunc']:.6g}",
                f"- **Depth truncation:** {result['depth_trunc']:.6g}",
            ]
        )
    return lines


def build_multiview_mesh_core(
    input_mode: str,
    multi_image_files,
    video_file,
    video_sample_fps: float,
    colmap_dir_path: str,
    colmap_sparse_subdir: str,
    align_to_input_ext_scale: bool,
    max_views: int,
    model_id: str,
    process_res: int,
    auto_ref_strategy: bool,
    ref_view_strategy: str,
    use_ray_pose: bool,
    conf_thresh_percentile: float,
    num_max_points: int,
    reconstruction_mode: str,
    tsdf_voxels_per_median_depth: int,
    tsdf_truncation_voxels: float,
    tsdf_depth_trunc_percentile: float,
    voxel_size: float,
    poisson_depth: int,
    trim_percentile: float,
    smooth_iterations: int,
    target_faces: int,
    keep_largest_component: bool,
    progress=None,
):
    if model_id not in MULTIVIEW_MODEL_IDS:
        raise ValueError(
            "Multi-view reconstruction requires DA3-Large-1.1, DA3-Giant-1.1, "
            "or DA3Nested-Giant-Large-1.1."
        )

    job_dir = create_job_dir()

    if progress:
        progress(0.02, desc="Preparing / ordering multi-view inputs")

    (
        copied_images,
        input_extrinsics,
        input_intrinsics,
        source_kind,
        source_meta,
    ) = prepare_multiview_inputs(
        input_mode=str(input_mode),
        multi_image_files=multi_image_files,
        video_file=video_file,
        video_sample_fps=float(video_sample_fps),
        colmap_dir_path=str(colmap_dir_path or ""),
        colmap_sparse_subdir=str(colmap_sparse_subdir or ""),
        max_views=int(max_views),
        job_dir=job_dir,
    )

    input_order_path = job_dir / "multiview_input_order.txt"
    input_order_path.write_text(
        "\n".join(f"{idx:03d}  {path.name}" for idx, path in enumerate(copied_images))
        + "\n",
        encoding="utf-8",
    )

    if bool(auto_ref_strategy):
        if source_kind in {"video", "ordered_frames"}:
            effective_ref_strategy = "middle"
        else:
            effective_ref_strategy = "saddle_balanced"
    else:
        effective_ref_strategy = str(ref_view_strategy)

    # Known COLMAP poses already condition the model; ray-pose estimation is for
    # pose-free input, so it is intentionally disabled in this mode.
    effective_use_ray_pose = bool(use_ray_pose) and input_extrinsics is None

    export_dir = job_dir / "da3_multiview"
    export_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.08, desc="Loading multi-view DA3 model")

    model, device = get_model(model_id)

    if progress:
        pose_desc = (
            "known COLMAP poses"
            if input_extrinsics is not None
            else ("ray pose" if effective_use_ray_pose else "camera decoder")
        )
        progress(
            0.14,
            desc=(
                f"Running {MODEL_ID_TO_LABEL.get(model_id, model_id)} "
                f"on {len(copied_images)} views ({pose_desc})"
            ),
        )

    start = time.perf_counter()

    # One DA3 prediction is reused for both TSDF and Poisson in A/B mode.
    prediction = model.inference(
        image=[str(p) for p in copied_images],
        extrinsics=input_extrinsics,
        intrinsics=input_intrinsics,
        align_to_input_ext_scale=bool(align_to_input_ext_scale),
        use_ray_pose=bool(effective_use_ray_pose),
        process_res=int(process_res),
        process_res_method="upper_bound_resize",
        export_dir=str(export_dir),
        export_format="mini_npz-glb",
        ref_view_strategy=str(effective_ref_strategy),
        conf_thresh_percentile=float(conf_thresh_percentile),
        num_max_points=int(num_max_points),
        show_cameras=False,
    )

    inference_seconds = time.perf_counter() - start

    glb_candidates = list(export_dir.rglob("*.glb"))
    if not glb_candidates:
        raise FileNotFoundError(
            "DA3 multi-view inference completed but no GLB point-cloud export was found."
        )
    scene_glb = glb_candidates[0]

    # Save explicit prediction arrays so an A/B run is reproducible without
    # repeating inference if someone wants to inspect the raw DA3 geometry.
    prediction_npz = job_dir / "da3_prediction_for_reconstruction.npz"
    np.savez_compressed(
        prediction_npz,
        depth=np.asarray(prediction.depth),
        conf=(
            np.asarray(prediction.conf)
            if getattr(prediction, "conf", None) is not None
            else np.array([])
        ),
        extrinsics=(
            np.asarray(prediction.extrinsics)
            if getattr(prediction, "extrinsics", None) is not None
            else np.array([])
        ),
        intrinsics=(
            np.asarray(prediction.intrinsics)
            if getattr(prediction, "intrinsics", None) is not None
            else np.array([])
        ),
        sky=(
            np.asarray(prediction.sky)
            if getattr(prediction, "sky", None) is not None
            else np.array([])
        ),
    )

    diag_glb, diag_json = create_camera_diagnostic_glb(
        prediction=prediction,
        output_dir=job_dir,
        conf_thresh_percentile=float(conf_thresh_percentile),
        num_max_points=int(num_max_points),
    )

    if progress:
        progress(0.55, desc="Reconstructing surface mesh")

    run_tsdf = reconstruction_mode in {
        "TSDF (recommended)",
        "A/B: TSDF + Poisson",
    }
    run_poisson = reconstruction_mode in {
        "Poisson",
        "A/B: TSDF + Poisson",
    }

    results = {}
    errors = {}

    if run_tsdf:
        try:
            results["tsdf"] = reconstruct_tsdf_mesh(
                prediction=prediction,
                output_dir=job_dir,
                conf_thresh_percentile=float(conf_thresh_percentile),
                tsdf_voxels_per_median_depth=int(tsdf_voxels_per_median_depth),
                tsdf_truncation_voxels=float(tsdf_truncation_voxels),
                tsdf_depth_trunc_percentile=float(tsdf_depth_trunc_percentile),
                smooth_iterations=int(smooth_iterations),
                target_faces=int(target_faces),
                keep_largest_component=bool(keep_largest_component),
                prefix=(
                    "multiview_mesh"
                    if reconstruction_mode == "TSDF (recommended)"
                    else "multiview_tsdf"
                ),
            )
        except Exception as exc:
            errors["tsdf"] = f"{type(exc).__name__}: {exc}"
            if reconstruction_mode == "TSDF (recommended)":
                raise

    if run_poisson:
        try:
            results["poisson"] = reconstruct_poisson_mesh(
                glb_path=scene_glb,
                output_dir=job_dir,
                voxel_size=float(voxel_size),
                poisson_depth=int(poisson_depth),
                trim_percentile=float(trim_percentile),
                smooth_iterations=int(smooth_iterations),
                target_faces=int(target_faces),
                keep_largest_component=bool(keep_largest_component),
                prefix=(
                    "multiview_mesh"
                    if reconstruction_mode == "Poisson"
                    else "multiview_poisson"
                ),
            )
        except Exception as exc:
            errors["poisson"] = f"{type(exc).__name__}: {exc}"
            if reconstruction_mode == "Poisson":
                raise

    if not results:
        raise RuntimeError(
            "All requested reconstruction methods failed: "
            + "; ".join(f"{k}: {v}" for k, v in errors.items())
        )

    # TSDF is primary whenever available because it directly fuses DA3 depth +
    # camera geometry. Poisson remains available for controlled A/B comparison.
    primary_key = "tsdf" if "tsdf" in results else "poisson"
    secondary_key = None
    if reconstruction_mode == "A/B: TSDF + Poisson" and len(results) > 1:
        secondary_key = "poisson" if primary_key == "tsdf" else "tsdf"

    metadata = {
        "model": model_id,
        "model_label": MODEL_ID_TO_LABEL.get(model_id, model_id),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "source": source_meta,
        "image_count": len(copied_images),
        "images_in_inference_order": [p.name for p in copied_images],
        "process_res": int(process_res),
        "process_res_method": "upper_bound_resize",
        "auto_ref_strategy": bool(auto_ref_strategy),
        "requested_ref_view_strategy": str(ref_view_strategy),
        "effective_ref_view_strategy": str(effective_ref_strategy),
        "requested_use_ray_pose": bool(use_ray_pose),
        "effective_use_ray_pose": bool(effective_use_ray_pose),
        "known_camera_poses": input_extrinsics is not None,
        "align_to_input_ext_scale": bool(align_to_input_ext_scale),
        "conf_thresh_percentile": float(conf_thresh_percentile),
        "num_max_points": int(num_max_points),
        "reconstruction_mode": str(reconstruction_mode),
        "tsdf": {
            "voxels_per_median_depth": int(tsdf_voxels_per_median_depth),
            "truncation_voxels": float(tsdf_truncation_voxels),
            "depth_trunc_percentile": float(tsdf_depth_trunc_percentile),
        },
        "poisson": {
            "voxel_downsample": float(voxel_size),
            "poisson_depth": int(poisson_depth),
            "trim_percentile": float(trim_percentile),
        },
        "postprocess": {
            "smooth_iterations": int(smooth_iterations),
            "target_faces": int(target_faces),
            "keep_largest_component": bool(keep_largest_component),
        },
        "inference_seconds": inference_seconds,
        "results": {
            key: {
                k: v
                for k, v in result.items()
                if not isinstance(v, Path)
            }
            for key, result in results.items()
        },
        "errors": errors,
    }

    metadata_path = job_dir / "multiview_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    output_files = [scene_glb, prediction_npz, input_order_path, metadata_path]
    if diag_glb is not None:
        output_files.append(diag_glb)
    if diag_json is not None:
        output_files.append(diag_json)
    for result in results.values():
        output_files.extend(_mesh_result_files(result))

    if progress:
        progress(0.92, desc="Creating multi-view ZIP package")

    zip_path = job_dir / "DA3_multiview_mesh.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in job_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                zf.write(path, arcname=path.relative_to(job_dir))

    if progress:
        progress(1.0, desc="Multi-view reconstruction complete")

    status_lines = [
        "### Multi-image reconstruction complete",
        f"- **Input mode:** {input_mode}",
        f"- **Model:** {MODEL_ID_TO_LABEL.get(model_id, model_id)} (`{model_id}`)",
        f"- **Views used:** {len(copied_images)}",
        f"- **DA3 process resolution:** {int(process_res)}",
        f"- **Reference strategy:** {effective_ref_strategy}",
        (
            "- **Pose source:** known COLMAP cameras (pose-conditioned depth)"
            if input_extrinsics is not None
            else (
                "- **Pose estimator:** ray head (quality mode)"
                if effective_use_ray_pose
                else "- **Pose estimator:** camera decoder (fast mode)"
            )
        ),
        f"- **Inference:** {inference_seconds:.2f} s",
        f"- **Reconstruction:** {reconstruction_mode}",
        f"- **GPU:** {torch.cuda.get_device_name(0)}",
        f"- **VRAM now:** {human_vram()}",
        f"- **Output folder:** `{job_dir}`",
        "",
    ]

    for key in ("tsdf", "poisson"):
        if key in results:
            status_lines.extend(_mesh_status_lines(results[key]["method"], results[key]))
            status_lines.append("")
        elif key in errors:
            status_lines.extend(
                [
                    f"#### {key.upper()} failed",
                    f"- `{errors[key]}`",
                    "",
                ]
            )

    status_lines.extend(
        [
            "**A/B note:** TSDF uses DA3's organized depth maps + confidence + intrinsics + "
            "extrinsics directly. Poisson uses the exact same DA3 prediction after its GLB "
            "point-cloud export, so `A/B: TSDF + Poisson` is a controlled mesher comparison.",
            "",
            "`camera_pose_diagnostic.glb` overlays camera centers/directions on DA3's point cloud "
            "so a bad camera path can be distinguished from a bad surface reconstructor.",
        ]
    )

    return {
        "primary_preview": results[primary_key]["preview_glb_path"],
        "secondary_preview": (
            results[secondary_key]["preview_glb_path"]
            if secondary_key is not None
            else None
        ),
        "diagnostic_glb": diag_glb,
        "status": "\n".join(status_lines),
        "files": output_files,
        "zip_path": zip_path,
    }


def build_multiview_mesh_ui(
    input_mode,
    multi_image_files,
    video_file,
    video_sample_fps,
    colmap_dir_path,
    colmap_sparse_subdir,
    align_to_input_ext_scale,
    max_views,
    model_id,
    process_res,
    auto_ref_strategy,
    ref_view_strategy,
    use_ray_pose,
    conf_thresh_percentile,
    num_max_points,
    reconstruction_mode,
    tsdf_voxels_per_median_depth,
    tsdf_truncation_voxels,
    tsdf_depth_trunc_percentile,
    voxel_size,
    poisson_depth,
    trim_percentile,
    smooth_iterations,
    target_faces,
    keep_largest_component,
    progress=gr.Progress(),
):
    result = build_multiview_mesh_core(
        input_mode=str(input_mode),
        multi_image_files=multi_image_files,
        video_file=video_file,
        video_sample_fps=float(video_sample_fps),
        colmap_dir_path=str(colmap_dir_path or ""),
        colmap_sparse_subdir=str(colmap_sparse_subdir or ""),
        align_to_input_ext_scale=bool(align_to_input_ext_scale),
        max_views=int(max_views),
        model_id=str(model_id),
        process_res=int(process_res),
        auto_ref_strategy=bool(auto_ref_strategy),
        ref_view_strategy=str(ref_view_strategy),
        use_ray_pose=bool(use_ray_pose),
        conf_thresh_percentile=float(conf_thresh_percentile),
        num_max_points=int(num_max_points),
        reconstruction_mode=str(reconstruction_mode),
        tsdf_voxels_per_median_depth=int(tsdf_voxels_per_median_depth),
        tsdf_truncation_voxels=float(tsdf_truncation_voxels),
        tsdf_depth_trunc_percentile=float(tsdf_depth_trunc_percentile),
        voxel_size=float(voxel_size),
        poisson_depth=int(poisson_depth),
        trim_percentile=float(trim_percentile),
        smooth_iterations=int(smooth_iterations),
        target_faces=int(target_faces),
        keep_largest_component=bool(keep_largest_component),
        progress=progress,
    )

    return (
        str(result["primary_preview"]),
        (
            str(result["secondary_preview"])
            if result["secondary_preview"] is not None
            else None
        ),
        (
            str(result["diagnostic_glb"])
            if result["diagnostic_glb"] is not None
            else None
        ),
        result["status"],
        [str(p) for p in result["files"]],
        str(result["zip_path"]),
    )

def build_mesh_core(
    job_dir_value,
    external_depth_path,
    external_texture_path,
    use_external_depth: bool,
    max_dim: int,
    mesh_width: float,
    z_scale: float,
    base_thickness: float,
    invert: bool,
    mirror_x: bool,
    near_pct: float,
    far_pct: float,
    depth_curve_enabled: bool,
    depth_curve_foreground_percent: float,
    depth_curve_background_z_percent: float,
    depth_curve_threshold_mode: str,
    progress=None,
):
    if not use_external_depth and not job_dir_value:
        raise ValueError(
            "Generate a depth map first, or enable external depth and upload a depth file."
        )

    if use_external_depth and not external_depth_path:
        raise ValueError("Enable external depth requires an uploaded depth file.")

    if float(mesh_width) <= 0:
        raise ValueError("Mesh width must be greater than zero.")

    if float(z_scale) <= 0:
        raise ValueError("Relief / Z scale must be greater than zero.")

    if float(base_thickness) <= 0:
        raise ValueError("Base thickness must be greater than zero.")

    if float(near_pct) >= float(far_pct):
        raise ValueError("Near percentile must be below far percentile.")

    if use_external_depth:
        job_dir = create_job_dir()
        src_depth = Path(external_depth_path)
        raw_depth_path = job_dir / f"external_depth{src_depth.suffix.lower()}"
        copy_uploaded_file(src_depth, raw_depth_path)

        texture_source = None
        if external_texture_path:
            src_tex = Path(external_texture_path)
            texture_source = job_dir / f"external_texture{src_tex.suffix.lower()}"
            copy_uploaded_file(src_tex, texture_source)

    else:
        job_dir = Path(job_dir_value)
        raw_depth_path = job_dir / "depth_raw_float32.exr"
        texture_source = job_dir / "input_used.png"

        if not raw_depth_path.exists():
            raise FileNotFoundError(
                f"Could not find {raw_depth_path}"
            )

    if progress:
        progress(0.05, desc="Loading raw 32-bit depth")

    depth = load_depth_map(raw_depth_path)

    if progress:
        progress(0.12, desc="Resampling mesh grid")

    depth = maybe_downsample_depth(depth, int(max_dim))
    depth_norm = normalize_depth(
        depth,
        float(near_pct),
        float(far_pct),
    )

    if bool(invert):
        depth_norm = 1.0 - depth_norm

    # Preserve the linear normalized depth so the output package can show
    # exactly what the depth curve changed without rerunning DA3.
    depth_norm_precurve = depth_norm.copy()

    depth_norm, depth_curve_info = apply_depth_curve(
        depth_norm,
        bool(depth_curve_enabled),
        float(depth_curve_foreground_percent),
        float(depth_curve_background_z_percent),
        str(depth_curve_threshold_mode),
    )

    rows, cols = depth_norm.shape

    if progress:
        if depth_curve_info["enabled"]:
            progress(
                0.22,
                desc=(
                    f"Applying depth curve: foreground {depth_curve_info['foreground_percent']:.0f}% / "
                    f"background Z {depth_curve_info['background_z_percent']:.0f}%"
                ),
            )
        progress(0.25, desc=f"Building closed {cols}×{rows} solid")

    (
        vertices,
        faces,
        top_faces,
        side_faces,
        bottom_faces,
        mesh_height,
    ) = build_solid_mesh(
        depth_norm,
        float(mesh_width),
        float(z_scale),
        float(base_thickness),
    )

    if progress:
        progress(0.52, desc="Writing textured OBJ")

    albedo_path = job_dir / "albedo.png"
    if texture_source is not None and Path(texture_source).exists():
        albedo_img = Image.open(texture_source).convert("RGB")
        albedo_img.save(albedo_path)
        texture_filename = albedo_path.name
    else:
        texture_filename = None

    mtl_path = job_dir / "heightfield_solid.mtl"
    obj_path = job_dir / "heightfield_solid.obj"

    write_mtl(mtl_path, texture_filename)

    write_obj(
        obj_path=obj_path,
        vertices=vertices,
        top_faces=top_faces,
        side_faces=side_faces,
        bottom_faces=bottom_faces,
        rows=rows,
        cols=cols,
        texture_filename=texture_filename,
    )

    if progress:
        progress(0.72, desc="Writing watertight STL")

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )

    is_watertight = bool(mesh.is_watertight)
    euler_number = int(mesh.euler_number)

    stl_path = job_dir / "heightfield_solid.stl"
    mesh.export(stl_path)

    # Gradio's Model3D camera/coordinate convention makes this single-image
    # relief appear horizontally mirrored even though the exported OBJ/STL are
    # correctly oriented. Build a DISPLAY-ONLY GLB for Gradio. The actual
    # printable/editable mesh files are never mirrored.
    preview_glb_path = job_dir / "heightfield_solid_preview.glb"
    preview_vertices = np.asarray(vertices, dtype=np.float64).copy()
    if bool(mirror_x):
        preview_vertices[:, 0] *= -1.0

    preview_mesh = trimesh.Trimesh(
        vertices=preview_vertices,
        faces=np.asarray(faces),
        process=False,
    )
    try:
        preview_mesh.fix_normals(multibody=True)
    except Exception:
        pass

    preview_material = trimesh.visual.material.PBRMaterial(
        name="neutral_preview",
        baseColorFactor=[125, 125, 125, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=True,
    )
    preview_mesh.visual = trimesh.visual.TextureVisuals(
        material=preview_material
    )
    trimesh.Scene(preview_mesh).export(preview_glb_path)

    # Pre-curve and final previews make the remap easy to inspect without
    # rerunning the neural network. The final preview is the exact heightfield
    # used to create the mesh.
    mesh_depth_precurve_preview = job_dir / "mesh_depth_precurve_preview_16bit.png"
    mesh_depth_preview = job_dir / "mesh_depth_preview_16bit.png"
    save_16bit_png(depth_norm_precurve, mesh_depth_precurve_preview)
    save_16bit_png(depth_norm, mesh_depth_preview)

    mesh_meta = {
        "grid_width": cols,
        "grid_height": rows,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "mesh_width": float(mesh_width),
        "mesh_height": float(mesh_height),
        "z_scale": float(z_scale),
        "base_thickness": float(base_thickness),
        "max_total_thickness": float(base_thickness) + float(z_scale),
        "invert": bool(invert),
        "preview_mirror_x": bool(mirror_x),
        "near_percentile": float(near_pct),
        "far_percentile": float(far_pct),
        "watertight": is_watertight,
        "euler_number": euler_number,
        "depth_source": str(raw_depth_path),
        "used_external_depth": bool(use_external_depth),
        "depth_curve": depth_curve_info,
    }

    mesh_meta_path = job_dir / "mesh_metadata.json"
    mesh_meta_path.write_text(
        json.dumps(mesh_meta, indent=2),
        encoding="utf-8",
    )

    if progress:
        progress(0.90, desc="Creating ZIP package")

    zip_path = zip_job(job_dir)

    mesh_files = [
        obj_path,
        mtl_path,
        stl_path,
        preview_glb_path,
        mesh_depth_precurve_preview,
        mesh_depth_preview,
        mesh_meta_path,
    ]

    if albedo_path.exists():
        mesh_files.append(albedo_path)

    if progress:
        progress(1.0, desc="Printable mesh complete")

    status = (
        "### Printable solid complete\n"
        f"- **Mesh grid:** {cols} × {rows}\n"
        f"- **Vertices:** {len(vertices):,}\n"
        f"- **Triangles:** {len(faces):,}\n"
        f"- **Width:** {float(mesh_width):g}\n"
        f"- **Height:** {float(mesh_height):.4g}\n"
        f"- **Minimum thickness:** {float(base_thickness):g}\n"
        f"- **Maximum thickness:** "
        f"{float(base_thickness) + float(z_scale):g}\n"
        f"- **Gradio preview orientation correction:** {'ON' if bool(mirror_x) else 'OFF'}\n"
        f"- **Depth curve:** {'ON' if depth_curve_info['enabled'] else 'OFF'}\n"
        + (
            f"- **Depth-curve foreground:** nearest {depth_curve_info['foreground_percent']:.1f}% "
            f"({depth_curve_info['threshold_mode']})\n"
            f"- **Background Z allocation:** {depth_curve_info['background_z_percent']:.1f}%\n"
            f"- **Curve threshold:** {depth_curve_info['threshold']:.4f}\n"
            f"- **Curve gamma:** {depth_curve_info['gamma']:.4f}\n"
            if depth_curve_info["enabled"]
            else ""
        )
        + f"- **Watertight:** {'YES ✅' if is_watertight else 'NO ⚠️'}\n"
        + f"- **Euler number:** {euler_number}\n\n"
        + "The OBJ keeps the source image as the front texture when available. "
        "The STL contains geometry only and is the simplest file to send to a slicer. "
        "The GLB is a display-only Gradio preview and may be mirrored for correct on-screen orientation; "
        "that preview correction never changes the OBJ/STL."
    )

    return {
        "stl_path": stl_path,
        "preview_glb_path": preview_glb_path,
        "mesh_files": mesh_files,
        "zip_path": zip_path,
        "status": status,
    }


def build_mesh_ui(
    job_dir_value,
    external_depth_path,
    external_texture_path,
    use_external_depth,
    max_dim,
    mesh_width,
    z_scale,
    base_thickness,
    invert,
    mirror_x,
    mesh_near_pct,
    mesh_far_pct,
    depth_curve_enabled,
    depth_curve_foreground_percent,
    depth_curve_background_z_percent,
    depth_curve_threshold_mode,
    progress=gr.Progress(),
):
    result = build_mesh_core(
        job_dir_value,
        external_depth_path,
        external_texture_path,
        bool(use_external_depth),
        int(max_dim),
        float(mesh_width),
        float(z_scale),
        float(base_thickness),
        bool(invert),
        bool(mirror_x),
        float(mesh_near_pct),
        float(mesh_far_pct),
        bool(depth_curve_enabled),
        float(depth_curve_foreground_percent),
        float(depth_curve_background_z_percent),
        str(depth_curve_threshold_mode),
        progress,
    )

    return (
        str(result["preview_glb_path"]),
        result["status"],
        [str(p) for p in result["mesh_files"]],
        str(result["zip_path"]),
    )


# ---------------------------------------------------------------------
# Complete one-click pipeline
# ---------------------------------------------------------------------

def full_pipeline_ui(
    image_value,
    crop_enabled,
    crop_x1,
    crop_y1,
    crop_x2,
    crop_y2,
    model_id,
    process_res,
    depth_near_pct,
    depth_far_pct,
    max_dim,
    mesh_width,
    z_scale,
    base_thickness,
    invert,
    mirror_x,
    mesh_near_pct,
    mesh_far_pct,
    depth_curve_enabled,
    depth_curve_foreground_percent,
    depth_curve_background_z_percent,
    depth_curve_threshold_mode,
    progress=gr.Progress(),
):
    progress(0.01, desc="Starting complete pipeline")

    depth_result = generate_depth_core(
        image_value,
        bool(crop_enabled),
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
        str(model_id),
        int(process_res),
        float(depth_near_pct),
        float(depth_far_pct),
        None,
    )

    progress(0.62, desc="Depth complete — building printable mesh")

    mesh_result = build_mesh_core(
        str(depth_result["job_dir"]),
        None,
        None,
        False,
        int(max_dim),
        float(mesh_width),
        float(z_scale),
        float(base_thickness),
        bool(invert),
        bool(mirror_x),
        float(mesh_near_pct),
        float(mesh_far_pct),
        bool(depth_curve_enabled),
        float(depth_curve_foreground_percent),
        float(depth_curve_background_z_percent),
        str(depth_curve_threshold_mode),
        None,
    )

    progress(1.0, desc="Complete")

    combined_status = (
        depth_result["status"]
        + "\n\n---\n\n"
        + mesh_result["status"]
    )

    return (
        str(depth_result["job_dir"]),
        str(depth_result["input_path"]),
        str(depth_result["near_preview"]),
        combined_status,
        [str(p) for p in depth_result["depth_files"]],
        str(mesh_result["preview_glb_path"]),
        mesh_result["status"],
        [str(p) for p in mesh_result["mesh_files"]],
        str(mesh_result["zip_path"]),
    )


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1500px !important;
}
#title {
    text-align: center;
    margin-bottom: 0.25rem;
}
#subtitle {
    text-align: center;
    opacity: 0.82;
    margin-bottom: 1rem;
}
"""

with gr.Blocks(css=CSS, title="DA3 Depth / Multi-View → Printable 3D") as demo:
    job_state = gr.State(value=None)

    gr.Markdown(
        "# DA3 Depth / Multi-View → Printable 3D",
        elem_id="title",
    )
    gr.Markdown(
        "Crop the source with reliable pixel controls, generate **selectable DA3** depth on the GPU, "
        "then create a **closed watertight OBJ/STL** with a flat printable back, or use the multi-image tab for a fused 3D reconstruction.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=6):
            source_image = gr.Image(
                label="1. Source image",
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                height=500,
            )

            source_dimensions = gr.Markdown(
                "Upload an image to initialize crop coordinates."
            )

            with gr.Accordion("Crop before DA3", open=True):
                crop_enabled = gr.Checkbox(
                    value=True,
                    label="Enable crop",
                    info=(
                        "Use pixel coordinates to remove the ultrasound UI, "
                        "rulers, logo, text, and excess border before inference."
                    ),
                )

                with gr.Row():
                    crop_x1 = gr.Number(value=0, precision=0, label="Left / X1")
                    crop_y1 = gr.Number(value=0, precision=0, label="Top / Y1")

                with gr.Row():
                    crop_x2 = gr.Number(value=1, precision=0, label="Right / X2")
                    crop_y2 = gr.Number(value=1, precision=0, label="Bottom / Y2")

                crop_preview_btn = gr.Button("Preview Crop")
                crop_preview_status = gr.Markdown()
                crop_preview_image = gr.Image(
                    label="Crop preview / exact DA3 input",
                    interactive=False,
                    height=380,
                )

        with gr.Column(scale=4):
            gr.Markdown("### Depth settings")

            model_id = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL_ID,
                label="DA3 model",
                info=(
                    "DA3Mono-Large is the best dedicated monocular choice. "
                    "The Large/Giant any-view models are available for comparison. "
                    "Switching models will unload the previous model and load the new one."
                ),
            )

            process_res = gr.Dropdown(
                choices=[504, 756, 1008, 1260, 1512, 1764, 2016],
                value=1260,
                label="DA3 internal process resolution",
                info=(
                    "1260 is a strong starting point on an RTX 4090. "
                    "Higher is slower and may not always add useful detail."
                ),
            )

            with gr.Accordion("Depth normalization", open=False):
                depth_near_pct = gr.Slider(
                    0.0, 10.0, value=1.0, step=0.1,
                    label="Low percentile",
                )
                depth_far_pct = gr.Slider(
                    90.0, 100.0, value=99.0, step=0.1,
                    label="High percentile",
                )

            generate_btn = gr.Button(
                "Generate DA3 Depth",
                variant="primary",
            )

            gr.Markdown("### Printable mesh settings")

            max_dim = gr.Dropdown(
                choices=[256, 384, 512, 640, 768, 1024],
                value=768,
                label="Mesh grid max dimension",
                info=(
                    "768 gives a very detailed mesh without making the OBJ/STL absurdly large."
                ),
            )

            mesh_width = gr.Number(
                value=2.0,
                label="Object width",
                info=(
                    "Arbitrary units in Blender; STL is unitless. "
                    "You can scale to final print size in Blender/slicer."
                ),
            )

            z_scale = gr.Number(
                value=0.45,
                label="Relief / Z scale",
            )

            base_thickness = gr.Number(
                value=0.08,
                label="Minimum backing thickness",
            )

            invert = gr.Checkbox(
                value=True,
                label="Invert DA3 raw depth (nearer = farther outward)",
            )

            mirror_x = gr.Checkbox(
                value=True,
                label="Correct horizontal mirroring in Gradio preview",
                info=(
                    "Display-only correction for Gradio's 3D preview. "
                    "The OBJ, STL, texture, and depth maps are never mirrored by this option."
                ),
            )

            with gr.Accordion("Mesh depth normalization", open=False):
                gr.Markdown(
                    "These can be adjusted to reshape the relief **without rerunning DA3**."
                )
                mesh_near_pct = gr.Slider(
                    0.0, 10.0, value=1.0, step=0.1,
                    label="Mesh low percentile",
                )
                mesh_far_pct = gr.Slider(
                    90.0, 100.0, value=99.0, step=0.1,
                    label="Mesh high percentile",
                )

            with gr.Accordion("Depth curve / foreground emphasis", open=True):
                gr.Markdown(
                    "Like **Curves for depth**: compress the far/background portion of the depth "
                    "range while reserving more of the physical Z range for the near/foreground "
                    "details. This only remaps depth; it does **not** rerun DA3."
                )

                depth_curve_enabled = gr.Checkbox(
                    value=False,
                    label="Enable depth curve",
                )

                depth_curve_foreground_percent = gr.Slider(
                    1.0, 75.0, value=25.0, step=1.0,
                    label="Foreground depth to emphasize (%)",
                    info=(
                        "25 means the nearest/top 25% is treated as foreground. "
                        "The remaining 75% becomes the compressible background region."
                    ),
                )

                depth_curve_background_z_percent = gr.Slider(
                    1.0, 75.0, value=15.0, step=1.0,
                    label="Z range allocated to background (%)",
                    info=(
                        "Example: 15 means the background region only occupies the bottom 15% "
                        "of mesh relief, leaving 85% of the Z range for foreground detail. "
                        "Lower values compress the background more strongly."
                    ),
                )

                depth_curve_threshold_mode = gr.Dropdown(
                    choices=[
                        "Depth range",
                        "Nearest pixels (percentile)",
                    ],
                    value="Depth range",
                    label="Foreground selection basis",
                    info=(
                        "Depth range: top 25% means normalized depth 0.75–1.0. "
                        "Nearest pixels: chooses the cutoff so about 25% of image pixels are foreground."
                    ),
                )

            with gr.Accordion("Optional: use an external depth map", open=False):
                use_external_depth = gr.Checkbox(
                    value=False,
                    label="Bypass DA3 and build the mesh from an uploaded depth file",
                    info="Useful for comparing DA2, other depth tools, or your own processed EXR/PNG/NPY/TIFF files.",
                )
                external_depth_file = gr.File(
                    label="External depth file (.exr, .png, .npy, .tif, .tiff)",
                    type="filepath",
                    file_types=[".exr", ".png", ".npy", ".tif", ".tiff"],
                )
                external_texture_file = gr.File(
                    label="Optional color/albedo image for OBJ texture (.png, .jpg, .jpeg, .webp, .tif, .tiff)",
                    type="filepath",
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"],
                )

            build_btn = gr.Button(
                "Build / Rebuild Printable Mesh",
                variant="secondary",
            )

            run_all_btn = gr.Button(
                "Run Entire Pipeline",
                variant="primary",
            )

    with gr.Tabs():
        with gr.Tab("Depth"):
            with gr.Row():
                input_used_view = gr.Image(
                    label="Exact image sent to DA3",
                    interactive=False,
                )
                depth_preview_view = gr.Image(
                    label="Near-white depth preview",
                    interactive=False,
                )

            depth_status = gr.Markdown()
            depth_files = gr.File(
                label="Depth output files",
                file_count="multiple",
            )

        with gr.Tab("3D / Print"):
            model_viewer = gr.Model3D(
                label="Printable mesh preview (display-only GLB)",
                display_mode="solid",
                height=650,
                interactive=False,
            )

            mesh_status = gr.Markdown()

            mesh_files = gr.File(
                label="Mesh output files",
                file_count="multiple",
            )

            zip_download = gr.DownloadButton(
                label="Download complete ZIP package",
            )

        with gr.Tab("Multi-Image 3D"):
            gr.Markdown(
                "Use DA3's **any-view** models for multiple overlapping views of the same "
                "subject/scene. For best-quality reconstruction this tab now supports "
                "**ray-based pose estimation**, direct **video sampling**, **known COLMAP "
                "camera poses**, direct **TSDF fusion**, the legacy **Poisson** path, and a "
                "controlled **TSDF vs Poisson A/B** run from the exact same DA3 prediction."
            )

            with gr.Row():
                with gr.Column(scale=5):
                    input_mode = gr.Dropdown(
                        choices=[
                            "Unordered uploaded photos",
                            "Ordered uploaded frames",
                            "Video file",
                            "COLMAP local dataset (known poses)",
                        ],
                        value="Unordered uploaded photos",
                        label="Multi-view input mode",
                        info=(
                            "Use Ordered uploaded frames for frames already extracted from a video. "
                            "Use Video file to let the app sample frames itself. COLMAP enables "
                            "pose-conditioned depth from known camera intrinsics/extrinsics."
                        ),
                    )

                    multi_image_files = gr.File(
                        label="Uploaded photos / ordered frames",
                        file_count="multiple",
                        type="filepath",
                        file_types=[
                            ".png", ".jpg", ".jpeg", ".webp",
                            ".bmp", ".tif", ".tiff",
                        ],
                    )

                    video_file = gr.File(
                        label="Optional source video",
                        file_count="single",
                        type="filepath",
                        file_types=[
                            ".mp4", ".avi", ".mov", ".mkv",
                            ".flv", ".wmv", ".webm", ".m4v",
                        ],
                    )

                    video_sample_fps = gr.Number(
                        value=1.0,
                        label="Video sampling FPS",
                        info=(
                            "1.0 FPS is DA3's documented CLI default. Increase if camera motion is "
                            "fast; decrease if adjacent samples have too little viewpoint change."
                        ),
                    )

                    with gr.Accordion("COLMAP / known-camera input", open=False):
                        colmap_dir_path = gr.Textbox(
                            value="",
                            label="Local COLMAP dataset folder",
                            placeholder=r"D:\path\to\dataset",
                            info=(
                                "Local path on this computer. The folder must contain images/ and "
                                "sparse/. This uses DA3's own COLMAP input handler."
                            ),
                        )
                        colmap_sparse_subdir = gr.Textbox(
                            value="",
                            label="COLMAP sparse subdirectory",
                            placeholder="0",
                            info=(
                                "Leave blank when cameras/images are directly under sparse/. "
                                "Enter 0 for the common sparse/0/ layout."
                            ),
                        )
                        align_to_input_ext_scale = gr.Checkbox(
                            value=True,
                            label="Align DA3 depth to COLMAP camera scale",
                            info=(
                                "Matches DA3's documented align_to_input_ext_scale=True behavior."
                            ),
                        )

                    max_views = gr.Dropdown(
                        choices=[0, 4, 6, 8, 12, 18],
                        value=18,
                        label="Maximum views (0 = no limit)",
                        info=(
                            "DA3 training at the 504 base resolution used 2–18 views. The default "
                            "caps long videos/large sets at 18 and samples them evenly."
                        ),
                    )

                    multiview_model_id = gr.Dropdown(
                        choices=[
                            (
                                "DA3-Giant-1.1 (quality any-view)",
                                "depth-anything/DA3-GIANT-1.1",
                            ),
                            (
                                "DA3Nested-Giant-Large-1.1 (metric-scale nested)",
                                "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                            ),
                            (
                                "DA3-Large-1.1 (lighter/faster)",
                                "depth-anything/DA3-LARGE-1.1",
                            ),
                        ],
                        value="depth-anything/DA3-GIANT-1.1",
                        label="Multi-view DA3 model",
                        info=(
                            "Giant is the default quality choice. Nested adds metric scaling. "
                            "Large is useful for lower VRAM use / faster iteration."
                        ),
                    )

                    multiview_process_res = gr.Dropdown(
                        choices=[504, 756, 1008],
                        value=504,
                        label="DA3 multi-view process resolution",
                        info=(
                            "504 is DA3's documented default/base regime and is the recommended "
                            "starting point for Giant/Large. Higher values remain experimental."
                        ),
                    )

                    with gr.Accordion("Pose estimation / reference view", open=True):
                        use_ray_pose = gr.Checkbox(
                            value=True,
                            label="Use high-quality ray-based camera pose estimation",
                            info=(
                                "DA3 documents use_ray_pose=True as slightly slower but generally "
                                "more accurate. Automatically ignored when known COLMAP poses are supplied."
                            ),
                        )

                        auto_ref_strategy = gr.Checkbox(
                            value=True,
                            label="Automatically choose reference strategy from input type",
                            info=(
                                "Ordered frames/video -> middle. Unordered photos -> saddle_balanced. "
                                "Uncheck to force the manual strategy below."
                            ),
                        )

                        ref_view_strategy = gr.Dropdown(
                            choices=[
                                "saddle_balanced",
                                "saddle_sim_range",
                                "middle",
                                "first",
                            ],
                            value="saddle_balanced",
                            label="Manual reference-view strategy",
                            info=(
                                "saddle_balanced: general photos; saddle_sim_range: wider-baseline sets; "
                                "middle: ordered/video sequence; first: supported by DA3 but generally not recommended."
                            ),
                        )

                    with gr.Accordion("DA3 point-cloud / confidence quality", open=False):
                        conf_thresh_percentile = gr.Slider(
                            0.0,
                            80.0,
                            value=40.0,
                            step=1.0,
                            label="Confidence filter percentile",
                            info=(
                                "40 is DA3's documented GLB default. TSDF uses the same percentile "
                                "to mask low-confidence depth for a fair A/B comparison."
                            ),
                        )

                        num_max_points = gr.Dropdown(
                            choices=[
                                200000,
                                400000,
                                600000,
                                800000,
                                1000000,
                            ],
                            value=1000000,
                            label="Maximum DA3 GLB points",
                            info=(
                                "1,000,000 is DA3's documented GLB default. This affects the Poisson "
                                "path/diagnostic cloud; TSDF fuses DA3 depth maps directly."
                            ),
                        )

                    reconstruction_mode = gr.Dropdown(
                        choices=[
                            "TSDF (recommended)",
                            "A/B: TSDF + Poisson",
                            "Poisson",
                        ],
                        value="TSDF (recommended)",
                        label="Surface reconstruction method",
                        info=(
                            "TSDF directly fuses DA3 depth + confidence + camera intrinsics/extrinsics. "
                            "A/B runs TSDF and Poisson from one inference so the comparison is controlled."
                        ),
                    )

                    with gr.Accordion("TSDF quality settings", open=False):
                        tsdf_voxels_per_median_depth = gr.Dropdown(
                            choices=[128, 192, 256, 384, 512],
                            value=256,
                            label="TSDF detail (voxels per median scene depth)",
                            info=(
                                "Higher = smaller voxels / more detail / more RAM and CPU time. "
                                "256 is the quality-oriented starting point."
                            ),
                        )

                        tsdf_truncation_voxels = gr.Slider(
                            2.0,
                            10.0,
                            value=5.0,
                            step=0.5,
                            label="TSDF truncation distance (voxels)",
                            info=(
                                "Controls how far around each observed surface the signed-distance "
                                "field is integrated. 4–6 voxels is a useful starting range."
                            ),
                        )

                        tsdf_depth_trunc_percentile = gr.Slider(
                            90.0,
                            100.0,
                            value=99.5,
                            step=0.1,
                            label="Ignore farthest depth outliers above percentile",
                            info=(
                                "Protects TSDF scale from extreme far-depth/sky outliers. Sky pixels "
                                "are also removed automatically when DA3 provides a sky mask."
                            ),
                        )

                    with gr.Accordion("Poisson A/B settings", open=False):
                        voxel_size = gr.Number(
                            value=0.0,
                            label="Poisson point-cloud voxel downsample (0 = off)",
                            info=(
                                "Leave at 0 for maximum detail. Increase slightly only for huge/noisy clouds."
                            ),
                        )

                        poisson_depth = gr.Dropdown(
                            choices=[8, 9, 10, 11],
                            value=9,
                            label="Poisson reconstruction depth",
                            info=(
                                "Higher can retain more point-cloud detail but can amplify noise and use more RAM."
                            ),
                        )

                        trim_percentile = gr.Slider(
                            0.0,
                            10.0,
                            value=0.0,
                            step=0.25,
                            label="Trim lowest-density Poisson vertices (%)",
                            info=(
                                "0 preserves the full Poisson result. Try 1–3 only for weak skirts/debris."
                            ),
                        )

                    with gr.Accordion("Shared mesh post-processing", open=False):
                        smooth_iterations = gr.Slider(
                            0,
                            20,
                            value=3,
                            step=1,
                            label="Taubin smoothing iterations",
                            info="Applied equally to whichever mesh reconstructor(s) you run.",
                        )

                        target_faces = gr.Dropdown(
                            choices=[0, 100000, 200000, 400000, 800000],
                            value=400000,
                            label="Target triangle count (0 = full mesh)",
                            info=(
                                "400k is a substantial-detail default. Use 0 for raw/full mesh comparison."
                            ),
                        )

                        keep_largest_component = gr.Checkbox(
                            value=True,
                            label="Keep only the largest connected mesh component",
                            info="Usually removes disconnected reconstruction debris.",
                        )

                    multiview_btn = gr.Button(
                        "Run Multi-View Reconstruction",
                        variant="primary",
                    )

                with gr.Column(scale=5):
                    multiview_model_viewer = gr.Model3D(
                        label="Primary mesh preview (TSDF when available)",
                        display_mode="solid",
                        height=520,
                        interactive=False,
                    )

                    multiview_compare_viewer = gr.Model3D(
                        label="A/B comparison preview (Poisson in A/B mode)",
                        display_mode="solid",
                        height=420,
                        interactive=False,
                    )

                    multiview_camera_viewer = gr.Model3D(
                        label="DA3 camera-pose diagnostic (point cloud + camera directions)",
                        display_mode="solid",
                        height=420,
                        interactive=False,
                    )

                    multiview_status = gr.Markdown()

                    multiview_files = gr.File(
                        label="Multi-view output files",
                        file_count="multiple",
                    )

                    multiview_zip = gr.DownloadButton(
                        label="Download multi-view ZIP package",
                    )

        with gr.Tab("About"):
            gr.Markdown(
                """
### Workflow

**Depth stage**
1. Upload or paste the ultrasound render.
2. Use the pixel-coordinate crop controls and **Preview Crop** to remove UI text, rulers, logos, and unnecessary black border.
3. Choose the DA3 model you want to test and run it.
4. The app writes raw **float32 NPY**, raw **32-bit EXR**, normalized **32-bit EXR**, **16-bit PNG**, and previews.

**Mesh stage**
1. The raw float32 DA3 depth is normalized.
2. The front surface is sampled into a heightfield.
3. A flat back and perimeter walls are generated.
4. The result is exported as:
   - textured **OBJ + MTL + albedo**
   - watertight **STL**
   - metadata and a ZIP containing the complete job

You can rebuild the mesh repeatedly with different Z scale, base thickness, grid density, inversion, horizontal orientation correction, normalization, or **Depth Curve / foreground emphasis** **without rerunning the neural network**. The depth curve smoothly compresses background depth while reserving more physical Z range for foreground detail. You can also bypass DA3 entirely and upload an external depth EXR/PNG/NPY/TIFF for comparison.

**Multi-image 3D stage**
1. Choose unordered photos, naturally sorted ordered frames, a source video, or a local COLMAP dataset with known poses.
2. Start with **DA3-Giant-1.1 at 504**; **DA3Nested-Giant-Large-1.1** is also available when metric scaling is useful.
3. For pose-free images, high-quality **ray-based pose estimation** is enabled by default. Ordered frames/video automatically use the **middle** reference view; unordered photos use **saddle_balanced** unless you override it.
4. Direct video input uses DA3's own video sampling behavior with **1 FPS** as the documented default. Inputs are capped at 18 views by default and sampled evenly when necessary.
5. **TSDF** is the recommended mesh path and directly fuses DA3 depth + confidence + intrinsics + extrinsics. **Poisson** remains available, and **A/B: TSDF + Poisson** runs both from the same DA3 prediction.
6. A diagnostic GLB overlays predicted camera centers/directions on DA3's point cloud so camera-pose problems can be separated from meshing problems.
7. COLMAP mode uses known camera intrinsics/extrinsics for DA3 pose-conditioned depth estimation.
8. The job exports DA3's GLB/NPZ geometry, camera diagnostics, reconstructed OBJ/PLY/STL/preview GLBs, metadata, and a ZIP package.

### Printing note

STL stores geometry but no physical unit. Most slicers interpret STL coordinates as millimeters. If you prefer to work in Blender units first, import the STL/OBJ and scale the final object there before printing.
                """
            )

    # Initialize crop coordinates whenever a new source image is loaded.
    source_image.change(
        fn=initialize_crop_ui,
        inputs=[source_image],
        outputs=[
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            source_dimensions,
        ],
        queue=False,
    )

    crop_preview_btn.click(
        fn=preview_crop_ui,
        inputs=[
            source_image,
            crop_enabled,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        ],
        outputs=[
            crop_preview_image,
            crop_preview_status,
        ],
        queue=False,
    )

    # Depth-only run.
    generate_btn.click(
        fn=generate_depth_ui,
        inputs=[
            source_image,
            crop_enabled,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            model_id,
            process_res,
            depth_near_pct,
            depth_far_pct,
        ],
        outputs=[
            job_state,
            input_used_view,
            depth_preview_view,
            depth_status,
            depth_files,
        ],
    )

    # Mesh-only / rebuild run.
    build_btn.click(
        fn=build_mesh_ui,
        inputs=[
            job_state,
            external_depth_file,
            external_texture_file,
            use_external_depth,
            max_dim,
            mesh_width,
            z_scale,
            base_thickness,
            invert,
            mirror_x,
            mesh_near_pct,
            mesh_far_pct,
            depth_curve_enabled,
            depth_curve_foreground_percent,
            depth_curve_background_z_percent,
            depth_curve_threshold_mode,
        ],
        outputs=[
            model_viewer,
            mesh_status,
            mesh_files,
            zip_download,
        ],
    )

    # One-click complete pipeline.
    run_all_btn.click(
        fn=full_pipeline_ui,
        inputs=[
            source_image,
            crop_enabled,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            model_id,
            process_res,
            depth_near_pct,
            depth_far_pct,
            max_dim,
            mesh_width,
            z_scale,
            base_thickness,
            invert,
            mirror_x,
            mesh_near_pct,
            mesh_far_pct,
            depth_curve_enabled,
            depth_curve_foreground_percent,
            depth_curve_background_z_percent,
            depth_curve_threshold_mode,
        ],
        outputs=[
            job_state,
            input_used_view,
            depth_preview_view,
            depth_status,
            depth_files,
            model_viewer,
            mesh_status,
            mesh_files,
            zip_download,
        ],
    )


    multiview_btn.click(
        fn=build_multiview_mesh_ui,
        inputs=[
            input_mode,
            multi_image_files,
            video_file,
            video_sample_fps,
            colmap_dir_path,
            colmap_sparse_subdir,
            align_to_input_ext_scale,
            max_views,
            multiview_model_id,
            multiview_process_res,
            auto_ref_strategy,
            ref_view_strategy,
            use_ray_pose,
            conf_thresh_percentile,
            num_max_points,
            reconstruction_mode,
            tsdf_voxels_per_median_depth,
            tsdf_truncation_voxels,
            tsdf_depth_trunc_percentile,
            voxel_size,
            poisson_depth,
            trim_percentile,
            smooth_iterations,
            target_faces,
            keep_largest_component,
        ],
        outputs=[
            multiview_model_viewer,
            multiview_compare_viewer,
            multiview_camera_viewer,
            multiview_status,
            multiview_files,
            multiview_zip,
        ],
    )


if __name__ == "__main__":
    print()
    print("DA3 Depth / Multi-View -> Printable 3D")
    print("------------------------------------------")
    print(f"Outputs will be stored in: {OUTPUT_ROOT}")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available.")

    print()

    demo.queue().launch(
        inbrowser=True,
        server_name="127.0.0.1",
        show_error=True,
    )
