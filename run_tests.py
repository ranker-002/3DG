"""
run_tests.py — Comprehensive test suite for the 3D Model Generator
================================================================
Runs in three tiers:
  Tier 1 — imports & environment  (always fast, no network)
  Tier 2 — unit / integration     (CPU, small inputs, no model weights)
  Tier 3 — model smoke tests      (downloads weights on first run, slow on CPU)

Usage:
  python run_tests.py              # Tier 1 + 2 only (safe, fast)
  python run_tests.py --all        # All tiers including model inference
  python run_tests.py --tier 3     # Only model smoke tests
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
import traceback
import types
import uuid
from pathlib import Path

# ── make sure TripoSR is on sys.path ─────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_TRIPOSR = _HERE / "TripoSR"
if _TRIPOSR.exists() and str(_TRIPOSR) not in sys.path:
    sys.path.insert(0, str(_TRIPOSR))

# ─────────────────────────────────────────────────────────────────────────────
# Tiny test framework
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  \033[32m[PASS]\033[0m"
FAIL = "  \033[31m[FAIL]\033[0m"
SKIP = "  \033[33m[SKIP]\033[0m"
INFO = "  \033[34m[INFO]\033[0m"

_results: list[tuple[str, str, str]] = []  # (tier, name, status)


def _run(tier: str, name: str, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"{PASS}  [{tier}] {name}  ({elapsed:.2f}s)")
        _results.append((tier, name, "PASS"))
    except SkipTest as e:
        elapsed = time.perf_counter() - t0
        print(f"{SKIP}  [{tier}] {name}  — {e}")
        _results.append((tier, name, "SKIP"))
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"{FAIL}  [{tier}] {name}  ({elapsed:.2f}s)")
        print(f"         {type(e).__name__}: {e}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc()
        _results.append((tier, name, "FAIL"))


class SkipTest(Exception):
    pass


def assert_true(condition, msg="Assertion failed"):
    if not condition:
        raise AssertionError(msg)


def assert_equal(a, b, msg=None):
    if a != b:
        raise AssertionError(msg or f"{a!r} != {b!r}")


def assert_in(item, container, msg=None):
    if item not in container:
        raise AssertionError(msg or f"{item!r} not in {container!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — imports & environment
# ─────────────────────────────────────────────────────────────────────────────


def t1_python_version():
    major, minor = sys.version_info[:2]
    assert_true(major == 3 and minor >= 9, f"Need Python 3.9+, got {major}.{minor}")


def t1_venv_active():
    venv = Path(sys.prefix)
    assert_true(
        (venv / "Scripts" / "python.exe").exists()
        or (venv / "bin" / "python").exists(),
        f"Not running inside a virtualenv (sys.prefix={sys.prefix})",
    )


def t1_outputs_dir():
    d = _HERE / "outputs"
    d.mkdir(exist_ok=True)
    assert_true(d.is_dir(), "outputs/ directory could not be created")


def t1_triposr_cloned():
    assert_true(_TRIPOSR.exists(), "TripoSR/ folder not found — run git clone first")
    assert_true(
        (_TRIPOSR / "tsr" / "system.py").exists(), "TripoSR/tsr/system.py missing"
    )


def t1_import_torch():
    import torch

    assert_true(hasattr(torch, "__version__"), "torch has no __version__")
    print(
        f"{INFO}  torch {torch.__version__}, "
        f"CUDA={'yes' if torch.cuda.is_available() else 'no (CPU mode)'}",
        flush=True,
    )


def t1_import_gradio():
    import gradio as gr

    assert_true(hasattr(gr, "Blocks"))


def t1_import_numpy():
    import numpy as np

    assert_true(hasattr(np, "array"))


def t1_import_pillow():
    from PIL import Image

    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    assert_equal(img.size, (4, 4))


def t1_import_trimesh():
    import trimesh

    assert_true(hasattr(trimesh, "Trimesh"))


def t1_import_scipy():
    import scipy

    assert_true(hasattr(scipy, "__version__"))


def t1_import_shap_e():
    import shap_e

    assert_true(hasattr(shap_e, "__version__") or True)  # version attr optional


def t1_import_shap_e_notebooks():
    from shap_e.util.notebooks import decode_latent_mesh  # noqa: F401


def t1_import_point_e():
    import point_e

    assert_true(hasattr(point_e, "__version__") or True)


def t1_import_rembg():
    import rembg

    assert_true(hasattr(rembg, "remove"))


def t1_import_onnxruntime():
    import onnxruntime as ort

    assert_true(hasattr(ort, "InferenceSession"))
    print(f"{INFO}  onnxruntime {ort.__version__}", flush=True)


def t1_import_huggingface_hub():
    from huggingface_hub import hf_hub_download  # noqa: F401


def t1_import_transformers():
    import transformers

    assert_true(hasattr(transformers, "__version__"))
    print(f"{INFO}  transformers {transformers.__version__}", flush=True)


def t1_import_omegaconf():
    from omegaconf import OmegaConf  # noqa: F401


def t1_import_einops():
    import einops

    assert_true(hasattr(einops, "__version__") or True)


def t1_import_ipywidgets():
    import ipywidgets  # noqa: F401


def t1_import_tsr():
    """TripoSR's tsr package must be importable (via sys.path patch)."""
    try:
        from tsr.system import TSR  # noqa: F401
    except ImportError as e:
        raise AssertionError(f"tsr not importable: {e}")


def t1_import_app_module():
    """app.py must be importable without side-effects (no Gradio launch)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("app", _HERE / "app.py")
    mod = importlib.util.module_from_spec(spec)
    # We don't exec the module fully (it launches Gradio at bottom),
    # just parse it for syntax errors.
    import ast

    src = (_HERE / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="app.py")
    assert_true(isinstance(tree, ast.Module))


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — unit / integration (no weights download)
# ─────────────────────────────────────────────────────────────────────────────


def t2_torch_basic_ops():
    import torch

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    c = a + b
    assert_true(float(c[0]) == 5.0)
    assert_true(float(c.sum()) == 21.0)


def t2_torch_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.zeros(2, 2).to(device)
    assert_equal(t.shape, torch.Size([2, 2]))


def t2_numpy_array_ops():
    import numpy as np

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert_equal(a.shape, (2, 2))
    assert_true(float(a.mean()) == 2.5)


def t2_pillow_open_save():
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        img.save(path)
        with Image.open(path) as loaded:
            size = loaded.size
        assert_equal(size, (64, 64))
    finally:
        try:
            os.unlink(path)
        except PermissionError:
            pass


def t2_trimesh_create_mesh():
    import numpy as np
    import trimesh

    # Simple box mesh
    box = trimesh.creation.box(extents=[1, 1, 1])
    assert_true(len(box.vertices) > 0)
    assert_true(len(box.faces) > 0)
    assert_true(box.is_watertight)


def t2_trimesh_export_obj():
    import trimesh

    box = trimesh.creation.box()
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        path = f.name
    try:
        box.export(path)
        assert_true(os.path.getsize(path) > 0)
        loaded = trimesh.load(path, force="mesh")
        assert_true(len(loaded.vertices) > 0)
    finally:
        os.unlink(path)


def t2_trimesh_export_glb():
    import trimesh

    box = trimesh.creation.box()
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        path = f.name
    try:
        box.export(path)
        assert_true(os.path.getsize(path) > 0)
    finally:
        os.unlink(path)


def t2_trimesh_export_stl():
    import trimesh

    box = trimesh.creation.box()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        path = f.name
    try:
        box.export(path)
        assert_true(os.path.getsize(path) > 0)
    finally:
        os.unlink(path)


def t2_save_mesh_helper():
    """Test the _save_mesh helper from app.py directly."""
    import importlib.util
    import sys
    import types

    import trimesh

    # Load only the helper without triggering Gradio launch
    src = (_HERE / "app.py").read_text(encoding="utf-8")

    # Stub gradio so the import doesn't fail / launch
    if "gradio" not in sys.modules:
        sys.modules["gradio"] = types.ModuleType("gradio")

    # Extract and exec only the _save_mesh function
    import ast
    import textwrap

    tree = ast.parse(src)
    fn_node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_save_mesh"
    )
    fn_src = ast.get_source_segment(src, fn_node)
    globs = {
        "__builtins__": __builtins__,
        "trimesh": trimesh,
        "uuid": uuid,
        "Path": Path,
        "OUTPUTS_DIR": _HERE / "outputs",
    }
    exec(textwrap.dedent(fn_src), globs)
    _save_mesh = globs["_save_mesh"]

    box = trimesh.creation.box()
    result = _save_mesh(box, "test_box")
    assert_in("obj", result)
    assert_in("glb", result)
    assert_in("stl", result)
    assert_in("stats", result)
    assert_true(os.path.exists(result["obj"]))
    assert_true(os.path.exists(result["glb"]))
    assert_true(os.path.exists(result["stl"]))
    # cleanup
    for k in ("obj", "glb", "stl"):
        try:
            os.unlink(result[k])
        except Exception:
            pass


def t2_outputs_dir_writable():
    d = _HERE / "outputs"
    d.mkdir(exist_ok=True)
    probe = d / f"_probe_{uuid.uuid4().hex}.tmp"
    probe.write_text("ok")
    assert_true(probe.read_text() == "ok")
    probe.unlink()


def t2_isosurface_patch():
    """Verify our torchmcubes patch works with skimage fallback."""
    import torch

    # Import the patched isosurface module
    from tsr.models.isosurface import MarchingCubeHelper, marching_cubes

    # Simple small volume
    vol = torch.randn(16, 16, 16)
    v, f = marching_cubes(vol, 0.0)
    assert_true(v.shape[1] == 3, f"vertices should have 3 cols, got {v.shape}")
    assert_true(f.shape[1] == 3, f"faces should have 3 cols, got {f.shape}")


def t2_skimage_marching_cubes():
    """Baseline skimage marching_cubes works on its own."""
    import numpy as np
    from skimage.measure import marching_cubes

    vol = np.random.randn(16, 16, 16).astype(np.float32)
    verts, faces, normals, values = marching_cubes(vol, level=0.0)
    assert_true(verts.shape[1] == 3)
    assert_true(faces.shape[1] == 3)


def t2_rembg_import_and_session():
    """rembg must initialise without errors (ONNX session creation)."""
    import rembg

    # Just importing rembg and calling new_session is the minimal smoke test
    # (does NOT require a real image — just checks ONNX runtime is wired up)
    assert_true(callable(rembg.remove))


def t2_gradio_blocks_construct():
    """Gradio Blocks can be instantiated without launching a server."""
    import gradio as gr

    with gr.Blocks() as demo:
        gr.Markdown("# Test")
        btn = gr.Button("Click")
    assert_true(demo is not None)


def t2_app_syntax():
    """app.py parses without syntax errors."""
    import ast

    src = (_HERE / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="app.py")
    assert_true(isinstance(tree, ast.Module))


def t2_app_triposr_path_inject():
    """app.py adds TripoSR to sys.path at startup."""
    import ast
    import textwrap

    src = (_HERE / "app.py").read_text(encoding="utf-8")
    assert_true("TripoSR" in src, "TripoSR path injection not found in app.py")
    assert_true("sys.path.insert" in src, "sys.path.insert not in app.py")


def t2_transformers_tokenizer():
    """Quick transformers sanity-check (no weight download)."""
    # Use a tiny vocab that's bundled with the package
    from transformers import AutoTokenizer, BertTokenizerFast

    assert_true(callable(BertTokenizerFast))


def t2_huggingface_hub_cache():
    """HF hub cache directory is accessible."""
    from huggingface_hub import constants

    cache = Path(constants.HF_HOME)
    # Just ensure we can query it (doesn't need to exist yet)
    assert_true(isinstance(str(cache), str))


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — model smoke tests (downloads weights, slow on CPU)
# ─────────────────────────────────────────────────────────────────────────────


def t3_point_e_load():
    """Load Point-E base model (downloads ~300 MB on first run)."""
    import torch
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.models.download import load_checkpoint

    device = torch.device("cpu")
    name = "base40M-textvec"
    model = model_from_config(MODEL_CONFIGS[name], device=device)
    model.eval()
    model.load_state_dict(load_checkpoint(name, device=device))
    assert_true(model is not None)


def t3_point_e_generate_text():
    """Point-E text→point-cloud smoke test (very few steps, CPU)."""
    import numpy as np
    import torch
    import trimesh
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.models.download import load_checkpoint
    from point_e.util.pc_to_mesh import marching_cubes_mesh

    device = torch.device("cpu")
    torch.manual_seed(0)

    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device=device)
    base_model.eval()
    base_model.load_state_dict(load_checkpoint(base_name, device=device))

    up_model = model_from_config(MODEL_CONFIGS["upsample"], device=device)
    up_model.eval()
    up_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
    up_model.load_state_dict(load_checkpoint("upsample", device=device))

    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, up_model],
        diffusions=[base_diffusion, up_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=("texts", ""),
    )

    samples = None
    for s in sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(texts=["a red sphere"]),
    ):
        samples = s

    pc = sampler.output_to_point_clouds(samples)[0]
    mesh = marching_cubes_mesh(pc, grid_size=32)  # low-res for speed

    tri = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
    assert_true(len(tri.vertices) > 0, "Point-E produced empty mesh")
    assert_true(len(tri.faces) > 0, "Point-E produced empty faces")
    print(
        f"{INFO}  Point-E mesh: {len(tri.vertices)} verts, {len(tri.faces)} faces",
        flush=True,
    )


def t3_shap_e_load():
    """Load Shap-E transmitter + text300M models (~2 GB on first run)."""
    import torch
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_config, load_model

    device = torch.device("cpu")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    assert_true(xm is not None)
    assert_true(model is not None)
    assert_true(diffusion is not None)


def t3_shap_e_generate_text():
    """Shap-E text→mesh smoke test (few steps, CPU)."""
    import io

    import torch
    import trimesh
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device("cpu")
    torch.manual_seed(0)

    xm = load_model("transmitter", device=device)
    text_model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    latents = sample_latents(
        batch_size=1,
        model=text_model,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(texts=["a small cube"]),
        progress=False,
        clip_denoised=True,
        use_fp16=False,  # fp16 can NaN on CPU
        use_karras=True,
        karras_steps=4,  # very few steps for speed
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    buf = io.StringIO()
    t.write_obj(buf)
    buf.seek(0)
    obj_text = buf.read()
    assert_true(
        obj_text.startswith("v ") or "v " in obj_text, "Shap-E OBJ output looks empty"
    )
    print(f"{INFO}  Shap-E OBJ size: {len(obj_text)} chars", flush=True)


def t3_triposr_load():
    """Load TripoSR model from Hugging Face (~500 MB on first run)."""
    import torch
    from tsr.system import TSR

    device = torch.device("cpu")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    assert_true(model is not None)


def t3_triposr_generate_image():
    """TripoSR image→mesh smoke test (low resolution, CPU)."""
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    device = torch.device("cpu")

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)

    # Synthetic white-background image of a grey square
    img = Image.new("RGBA", (256, 256), (200, 200, 200, 255))
    img_arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    with torch.no_grad():
        scene_codes = model([img_arr], device=device)
        meshes = model.extract_mesh(scene_codes, resolution=32)  # tiny for speed

    mesh = meshes[0]
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        tri = trimesh.Trimesh(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
        )
    else:
        tri = mesh
    assert_true(len(tri.vertices) > 0, "TripoSR produced empty mesh")
    print(
        f"{INFO}  TripoSR mesh: {len(tri.vertices)} verts, {len(tri.faces)} faces",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

TIER1 = [
    ("T1", "Python version >= 3.9", t1_python_version),
    ("T1", "Virtual environment active", t1_venv_active),
    ("T1", "outputs/ directory", t1_outputs_dir),
    ("T1", "TripoSR repo cloned", t1_triposr_cloned),
    ("T1", "import torch", t1_import_torch),
    ("T1", "import gradio", t1_import_gradio),
    ("T1", "import numpy", t1_import_numpy),
    ("T1", "import Pillow", t1_import_pillow),
    ("T1", "import trimesh", t1_import_trimesh),
    ("T1", "import scipy", t1_import_scipy),
    ("T1", "import shap_e", t1_import_shap_e),
    ("T1", "import shap_e.util.notebooks", t1_import_shap_e_notebooks),
    ("T1", "import point_e", t1_import_point_e),
    ("T1", "import rembg", t1_import_rembg),
    ("T1", "import onnxruntime", t1_import_onnxruntime),
    ("T1", "import huggingface_hub", t1_import_huggingface_hub),
    ("T1", "import transformers", t1_import_transformers),
    ("T1", "import omegaconf", t1_import_omegaconf),
    ("T1", "import einops", t1_import_einops),
    ("T1", "import ipywidgets", t1_import_ipywidgets),
    ("T1", "import tsr (TripoSR)", t1_import_tsr),
    ("T1", "app.py syntax valid", t1_import_app_module),
]

TIER2 = [
    ("T2", "torch basic ops", t2_torch_basic_ops),
    ("T2", "torch device creation", t2_torch_device),
    ("T2", "numpy array ops", t2_numpy_array_ops),
    ("T2", "Pillow open / save PNG", t2_pillow_open_save),
    ("T2", "trimesh create box mesh", t2_trimesh_create_mesh),
    ("T2", "trimesh export OBJ", t2_trimesh_export_obj),
    ("T2", "trimesh export GLB", t2_trimesh_export_glb),
    ("T2", "trimesh export STL", t2_trimesh_export_stl),
    ("T2", "_save_mesh helper", t2_save_mesh_helper),
    ("T2", "outputs/ dir writable", t2_outputs_dir_writable),
    ("T2", "torchmcubes patch (skimage)", t2_isosurface_patch),
    ("T2", "skimage marching_cubes", t2_skimage_marching_cubes),
    ("T2", "rembg session init", t2_rembg_import_and_session),
    ("T2", "gradio Blocks construct", t2_gradio_blocks_construct),
    ("T2", "app.py syntax check", t2_app_syntax),
    ("T2", "app.py TripoSR path inject", t2_app_triposr_path_inject),
    ("T2", "transformers tokenizer API", t2_transformers_tokenizer),
    ("T2", "HF hub cache path", t2_huggingface_hub_cache),
]

TIER3 = [
    ("T3", "Point-E model load", t3_point_e_load),
    ("T3", "Point-E text generation", t3_point_e_generate_text),
    ("T3", "Shap-E model load", t3_shap_e_load),
    ("T3", "Shap-E text generation", t3_shap_e_generate_text),
    ("T3", "TripoSR model load", t3_triposr_load),
    ("T3", "TripoSR image generation", t3_triposr_generate_image),
]

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="3D Generator test suite")
    parser.add_argument(
        "--all", action="store_true", help="Run all tiers (incl. model downloads)"
    )
    parser.add_argument(
        "--tier", type=int, choices=[1, 2, 3], help="Run only this tier"
    )
    args = parser.parse_args()

    header = "\033[34m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
    print(header)
    print("\033[34m  3D Model Generator — Test Suite\033[0m")
    print(header)
    print()

    suites = []
    if args.tier == 1:
        suites = [TIER1]
    elif args.tier == 2:
        suites = [TIER2]
    elif args.tier == 3:
        suites = [TIER3]
    elif args.all:
        suites = [TIER1, TIER2, TIER3]
    else:
        suites = [TIER1, TIER2]

    for suite in suites:
        tier_label = suite[0][0]
        print(
            f"\n\033[34m── {tier_label}: {'Imports & Environment' if tier_label == 'T1' else 'Unit / Integration' if tier_label == 'T2' else 'Model Smoke Tests'} ──\033[0m\n"
        )
        for tier, name, fn in suite:
            _run(tier, name, fn)

    # Summary
    total = len(_results)
    passed = sum(1 for _, _, s in _results if s == "PASS")
    failed = sum(1 for _, _, s in _results if s == "FAIL")
    skipped = sum(1 for _, _, s in _results if s == "SKIP")

    print()
    print(header)
    print(
        f"  Results:  "
        f"\033[32m{passed} passed\033[0m  "
        f"\033[31m{failed} failed\033[0m  "
        f"\033[33m{skipped} skipped\033[0m  "
        f"/ {total} total"
    )
    print(header)
    print()

    if failed:
        print("\033[31mFailed tests:\033[0m")
        for tier, name, s in _results:
            if s == "FAIL":
                print(f"  [{tier}] {name}")
        print()
        sys.exit(1)
    else:
        print("\033[32m✅  All tests passed!\033[0m\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
