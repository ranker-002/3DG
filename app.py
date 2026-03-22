"""
3D Model Generator — 100% Free & Open-Source
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Models (all MIT / Apache 2.0, no API keys, run fully locally):
  • Shap-E      (OpenAI, MIT)    — text→3D & image→3D
  • TripoSR     (Stability, MIT) — image→3D, highest fidelity
  • Point-E     (OpenAI, MIT)    — text→point-cloud→mesh (fastest)

Stack: Python · PyTorch · Trimesh · Gradio · Three.js
"""

from __future__ import annotations

import gc
import io
import os
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

# ── Add TripoSR to path so `tsr` package is importable ────────────────────────
_TRIPOSR_DIR = Path(__file__).parent / "TripoSR"
if _TRIPOSR_DIR.exists() and str(_TRIPOSR_DIR) not in sys.path:
    sys.path.insert(0, str(_TRIPOSR_DIR))

import gradio as gr
import numpy as np
from PIL import Image

# ── optional heavy deps ────────────────────────────────────────────────────────
try:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    DEVICE = None

try:
    import trimesh

    TRIMESH_OK = True
except ImportError:
    TRIMESH_OK = False

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# Model loaders (lazy-cached)
# ═════════════════════════════════════════════════════════════════════════════

_cache: dict = {}


# ── Shap-E ────────────────────────────────────────────────────────────────────


def _load_shap_e():
    if "shap_e" in _cache:
        return _cache["shap_e"]
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model

    xm = load_model("transmitter", device=DEVICE)
    text_model = load_model("text300M", device=DEVICE)
    image_model = load_model("image300M", device=DEVICE)
    diffusion = diffusion_from_config(load_config("diffusion"))

    _cache["shap_e"] = dict(
        xm=xm,
        text_model=text_model,
        image_model=image_model,
        diffusion=diffusion,
        sample_latents=sample_latents,
    )
    return _cache["shap_e"]


# ── Point-E ───────────────────────────────────────────────────────────────────


def _load_point_e():
    if "point_e" in _cache:
        return _cache["point_e"]
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.models.download import load_checkpoint
    from point_e.util.pc_to_mesh import marching_cubes_mesh

    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device=DEVICE)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device=DEVICE))

    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device=DEVICE)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
    upsampler_model.load_state_dict(load_checkpoint("upsample", device=DEVICE))

    _cache["point_e"] = dict(
        base_model=base_model,
        base_diffusion=base_diffusion,
        upsampler_model=upsampler_model,
        upsampler_diffusion=upsampler_diffusion,
        marching_cubes_mesh=marching_cubes_mesh,
        PointCloudSampler=PointCloudSampler,
    )
    return _cache["point_e"]


# ── TripoSR ───────────────────────────────────────────────────────────────────


def _load_triposr():
    if "triposr" in _cache:
        return _cache["triposr"]
    # tsr package installed via: pip install git+https://github.com/VAST-AI-Research/TripoSR.git
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(DEVICE)

    _cache["triposr"] = dict(
        model=model,
        remove_background=remove_background,
        resize_foreground=resize_foreground,
    )
    return _cache["triposr"]


# ═════════════════════════════════════════════════════════════════════════════
# Generation functions
# ═════════════════════════════════════════════════════════════════════════════


def _save_mesh(mesh_obj, stem: str) -> dict:
    """Trimesh → OBJ + GLB + STL, return paths & stats."""
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = trimesh.util.concatenate(list(mesh_obj.geometry.values()))

    # Normalise to unit cube centred at origin
    mesh_obj.apply_translation(-mesh_obj.centroid)
    if max(mesh_obj.extents) > 0:
        mesh_obj.apply_scale(1.0 / max(mesh_obj.extents))

    uid = uuid.uuid4().hex[:8]
    base = OUTPUTS_DIR / f"{stem}_{uid}"

    obj_path = str(base) + ".obj"
    glb_path = str(base) + ".glb"
    stl_path = str(base) + ".stl"

    mesh_obj.export(obj_path)
    mesh_obj.export(glb_path)
    mesh_obj.export(stl_path)

    return dict(
        obj=obj_path,
        glb=glb_path,
        stl=stl_path,
        stats=dict(
            vertices=len(mesh_obj.vertices),
            faces=len(mesh_obj.faces),
            watertight=bool(mesh_obj.is_watertight),
            volume=round(float(mesh_obj.volume), 5)
            if mesh_obj.is_watertight
            else "N/A",
        ),
    )


# ── Shap-E: text → 3D ────────────────────────────────────────────────────────


def shap_e_text(prompt: str, guidance: float, steps: int, seed: int) -> dict:
    from shap_e.util.notebooks import decode_latent_mesh

    ctx = _load_shap_e()
    torch.manual_seed(seed)

    latents = ctx["sample_latents"](
        batch_size=1,
        model=ctx["text_model"],
        diffusion=ctx["diffusion"],
        guidance_scale=guidance,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    t = decode_latent_mesh(ctx["xm"], latents[0]).tri_mesh()
    buf = io.StringIO()
    t.write_obj(buf)
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.write(buf.read().encode())
    tmp.close()
    mesh = trimesh.load(tmp.name, force="mesh")
    return _save_mesh(mesh, "shape_text")


# ── Shap-E: image → 3D ───────────────────────────────────────────────────────


def shap_e_image(pil: Image.Image, guidance: float, steps: int, seed: int) -> dict:
    from shap_e.util.notebooks import decode_latent_mesh

    ctx = _load_shap_e()
    torch.manual_seed(seed)

    img = pil.convert("RGBA").resize((256, 256))
    latents = ctx["sample_latents"](
        batch_size=1,
        model=ctx["image_model"],
        diffusion=ctx["diffusion"],
        guidance_scale=guidance,
        model_kwargs=dict(images=[img]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    t = decode_latent_mesh(ctx["xm"], latents[0]).tri_mesh()
    buf = io.StringIO()
    t.write_obj(buf)
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.write(buf.read().encode())
    tmp.close()
    mesh = trimesh.load(tmp.name, force="mesh")
    return _save_mesh(mesh, "shape_img")


# ── TripoSR: image → 3D (highest fidelity) ───────────────────────────────────


def triposr_image(pil: Image.Image, mc_resolution: int = 256) -> dict:
    ctx = _load_triposr()
    model = ctx["model"]

    # TripoSR preprocessing
    img = ctx["remove_background"](pil)
    img = ctx["resize_foreground"](img, ratio=0.85)
    img_arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    with torch.no_grad():
        scene_codes = model([img_arr], device=DEVICE)
        meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)

    mesh = meshes[0]
    # TripoSR returns a trimesh-like object; ensure it's a Trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
        )
    return _save_mesh(mesh, "triposr")


# ── Point-E: text → point-cloud → mesh ───────────────────────────────────────


def point_e_text(prompt: str, steps: int, seed: int) -> dict:
    ctx = _load_point_e()
    torch.manual_seed(seed)

    sampler = ctx["PointCloudSampler"](
        device=DEVICE,
        models=[ctx["base_model"], ctx["upsampler_model"]],
        diffusions=[ctx["base_diffusion"], ctx["upsampler_diffusion"]],
        num_points=[1024, 4096 - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=("texts", ""),
    )

    samples = None
    for s in sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(texts=[prompt]),
    ):
        samples = s

    pc = sampler.output_to_point_clouds(samples)[0]

    # Marching-cubes mesh from point cloud
    mesh = ctx["marching_cubes_mesh"](pc, grid_size=128)
    tri = trimesh.Trimesh(
        vertices=mesh.verts,
        faces=mesh.faces,
        vertex_colors=(np.array(mesh.vertex_channels["R", "G", "B"]).T * 255).astype(
            np.uint8
        )
        if hasattr(mesh, "vertex_channels")
        else None,
    )
    return _save_mesh(tri, "pointe_text")


# ═════════════════════════════════════════════════════════════════════════════
# Gradio callback
# ═════════════════════════════════════════════════════════════════════════════


def run_generation(
    model_choice,
    input_mode,
    text_prompt,
    image_input,
    guidance_scale,
    diffusion_steps,
    mc_resolution,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if not TORCH_OK:
        return None, None, None, "❌ PyTorch not installed.", ""
    if not TRIMESH_OK:
        return None, None, None, "❌ Trimesh not installed.", ""

    try:
        progress(0.05, desc=f"Loading {model_choice}…")
        pil = None
        if image_input is not None:
            pil = (
                image_input
                if isinstance(image_input, Image.Image)
                else Image.fromarray(image_input)
            )

        # ── Route to correct model ────────────────────────────────────────
        if model_choice == "⚡ Point-E  (fastest · text)":
            if not text_prompt.strip():
                return None, None, None, "⚠️  Please enter a text prompt.", ""
            progress(0.1, desc="Sampling point cloud…")
            result = point_e_text(text_prompt, diffusion_steps, int(seed))

        elif model_choice == "🔷 Shap-E  (text → 3D)":
            if not text_prompt.strip():
                return None, None, None, "⚠️  Please enter a text prompt.", ""
            progress(0.1, desc="Shap-E text diffusion…")
            result = shap_e_text(
                text_prompt, guidance_scale, diffusion_steps, int(seed)
            )

        elif model_choice == "🖼️  Shap-E  (image → 3D)":
            if pil is None:
                return None, None, None, "⚠️  Please upload an image.", ""
            progress(0.1, desc="Shap-E image diffusion…")
            result = shap_e_image(pil, guidance_scale, diffusion_steps, int(seed))

        elif model_choice == "🏆 TripoSR  (image → 3D · best quality)":
            if pil is None:
                return None, None, None, "⚠️  Please upload an image.", ""
            progress(0.1, desc="Running TripoSR reconstruction…")
            result = triposr_image(pil, mc_resolution=int(mc_resolution))

        else:
            return None, None, None, "❌  Unknown model selected.", ""

        progress(0.95, desc="Exporting mesh…")
        s = result["stats"]
        wt_badge = (
            '<span class="stat-badge stat-ok">WATERTIGHT</span>'
            if s["watertight"]
            else '<span class="stat-badge stat-warn">OPEN</span>'
        )
        stats_html = f"""
<div class="stat-grid">
  <div class="stat-cell">
    <div class="stat-label">VERTICES</div>
    <div class="stat-value">{s["vertices"]:,}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-label">FACES</div>
    <div class="stat-value">{s["faces"]:,}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-label">TOPOLOGY</div>
    <div class="stat-value">{wt_badge}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-label">VOLUME</div>
    <div class="stat-value stat-mono">{s["volume"]}</div>
  </div>
</div>"""
        return (
            result["obj"],
            result["glb"],
            result["stl"],
            "✅  Generation complete.",
            stats_html,
        )

    except Exception:
        tb = traceback.format_exc()
        err_html = f'<div class="stat-error"><span class="stat-label">ERROR</span><pre>{tb}</pre></div>'
        return None, None, None, f"❌  Generation failed — see details below.", err_html


# ═════════════════════════════════════════════════════════════════════════════
# Three.js inline viewer
# ═════════════════════════════════════════════════════════════════════════════

VIEWER_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

.vp-shell {
  position: relative;
  width: 100%;
  height: 520px;
  background: #000;
  border: 1px solid #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
  font-family: 'Inter', sans-serif;
}
#vcanvas { width:100%; height:100%; display:none; }

/* ── Empty state ── */
.vp-empty {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 28px;
}
.vp-scene { width: 72px; height: 72px; perspective: 280px; }
.vp-cube {
  width: 72px; height: 72px; position: relative;
  transform-style: preserve-3d;
  animation: vp-spin 14s linear infinite;
}
.vp-face {
  position: absolute; width: 72px; height: 72px;
  border: 1px solid #222; background: transparent;
}
.vp-face.fr { transform: translateZ(36px); }
.vp-face.bk { transform: rotateY(180deg) translateZ(36px); }
.vp-face.lt { transform: rotateY(-90deg) translateZ(36px); }
.vp-face.rt { transform: rotateY( 90deg) translateZ(36px); }
.vp-face.tp { transform: rotateX( 90deg) translateZ(36px); }
.vp-face.bt { transform: rotateX(-90deg) translateZ(36px); }
@keyframes vp-spin {
  from { transform: rotateX(-22deg) rotateY(  0deg); }
  to   { transform: rotateX(-22deg) rotateY(360deg); }
}
.vp-idle-label { font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700; letter-spacing:.22em; color:#282828; text-transform:uppercase; }
.vp-idle-sub   { font-size:11px; color:#2e2e2e; margin-top:4px; text-align:center; }

/* ── Scan-line overlay ── */
.vp-scan {
  position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(
    to bottom,
    transparent 0px, transparent 3px,
    rgba(255,255,255,.008) 3px, rgba(255,255,255,.008) 4px
  );
}

/* ── Corner marks ── */
.vp-corner {
  position: absolute; width:18px; height:18px;
  border-color: #1f1f1f; border-style: solid;
}
.vp-corner.tl { top:14px;  left:14px;  border-width:1px 0 0 1px; }
.vp-corner.tr { top:14px;  right:14px; border-width:1px 1px 0 0; }
.vp-corner.bl { bottom:14px; left:14px;  border-width:0 0 1px 1px; }
.vp-corner.br { bottom:14px; right:14px; border-width:0 1px 1px 0; }

/* ── Toolbar ── */
.vp-bar {
  position: absolute; bottom:16px; left:50%; transform:translateX(-50%);
  display: flex; gap:5px;
  opacity: 0; transition: opacity .2s;
  background: rgba(0,0,0,.85);
  border: 1px solid #1c1c1c;
  border-radius: 6px; padding:5px;
  backdrop-filter: blur(10px);
}
.vp-bar button {
  background: transparent;
  border: 1px solid #222; color: #555;
  border-radius: 4px; padding: 4px 11px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; font-weight:700; letter-spacing:.12em;
  cursor: pointer; transition: all .12s;
}
.vp-bar button:hover { border-color:#444; color:#ccc; background: #111; }
</style>

<div class="vp-shell" id="vshell">
  <div class="vp-scan"></div>
  <div class="vp-corner tl"></div>
  <div class="vp-corner tr"></div>
  <div class="vp-corner bl"></div>
  <div class="vp-corner br"></div>

  <div class="vp-empty" id="vempty">
    <div class="vp-scene">
      <div class="vp-cube">
        <div class="vp-face fr"></div><div class="vp-face bk"></div>
        <div class="vp-face lt"></div><div class="vp-face rt"></div>
        <div class="vp-face tp"></div><div class="vp-face bt"></div>
      </div>
    </div>
    <div>
      <div class="vp-idle-label">Viewport</div>
      <div class="vp-idle-sub">Generate a model to preview it here</div>
    </div>
  </div>

  <canvas id="vcanvas"></canvas>

  <div class="vp-bar" id="vbar">
    <button onclick="resetCam()">RESET</button>
    <button onclick="toggleWire()">WIRE</button>
    <button onclick="toggleSpin()">SPIN</button>
    <button onclick="cycleLight()">LIGHT</button>
  </div>
</div>

<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';
import { OBJLoader }     from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/loaders/OBJLoader.js';

const canvas   = document.getElementById('vcanvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
camera.position.set(0, 0.5, 2.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping  = true;
controls.dampingFactor  = 0.07;
controls.autoRotate     = true;
controls.autoRotateSpeed = 0.55;
controls.minDistance    = 0.4;
controls.maxDistance    = 12;

/* ── Studio lights (neutral white) ── */
scene.add(new THREE.AmbientLight(0xffffff, 0.25));

const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(3, 5, 3); key.castShadow = true;
scene.add(key);

const fill = new THREE.DirectionalLight(0xffffff, 0.55);
fill.position.set(-3, 2, -2); scene.add(fill);

const rim = new THREE.DirectionalLight(0xffffff, 0.35);
rim.position.set(0, -2, -3); scene.add(rim);

/* ── Grid / ground ── */
const grid = new THREE.GridHelper(8, 28, 0x0f0f0f, 0x0a0a0a);
grid.position.y = -0.72; scene.add(grid);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(14, 14),
  new THREE.ShadowMaterial({ opacity: 0.45 })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.72;
ground.receiveShadow = true;
scene.add(ground);

/* ── State ── */
let obj3d = null, wireMode = false, origMats = [], spinning = true, lightIdx = 0;

const LIGHTS = [
  { bg: 0x000000, ki: 2.1, fi: 0.55, ri: 0.35, col: 0xd6d6d6, metal: 0.05, rough: 0.55 },
  { bg: 0x050505, ki: 2.5, fi: 0.40, ri: 0.30, col: 0xe8d8b0, metal: 0.55, rough: 0.20 },
  { bg: 0x000000, ki: 1.8, fi: 0.80, ri: 0.50, col: 0xb0c8e8, metal: 0.20, rough: 0.65 },
  { bg: 0x030303, ki: 1.5, fi: 1.00, ri: 0.60, col: 0xd0d0d0, metal: 0.80, rough: 0.10 },
];
let activeMat = new THREE.MeshStandardMaterial({ color: 0xd6d6d6, metalness: 0.05, roughness: 0.55 });

window.resetCam   = () => { camera.position.set(0, 0.5, 2.8); controls.reset(); };
window.toggleSpin = () => { spinning = !spinning; controls.autoRotate = spinning; };
window.toggleWire = () => {
  if (!obj3d) return;
  wireMode = !wireMode;
  if (wireMode) {
    origMats = [];
    const wm = new THREE.MeshBasicMaterial({ color: 0x333333, wireframe: true });
    obj3d.traverse(n => { if (n.isMesh) { origMats.push(n.material); n.material = wm; } });
  } else {
    let i = 0;
    obj3d.traverse(n => { if (n.isMesh) { n.material = origMats[i++] || activeMat; } });
    origMats = [];
  }
};
window.cycleLight = () => {
  lightIdx = (lightIdx + 1) % LIGHTS.length;
  const p = LIGHTS[lightIdx];
  scene.background.set(p.bg);
  key.intensity  = p.ki; fill.intensity = p.fi; rim.intensity  = p.ri;
  if (obj3d) obj3d.traverse(n => {
    if (n.isMesh && !wireMode) {
      n.material.color.set(p.col);
      n.material.metalness = p.metal;
      n.material.roughness = p.rough;
    }
  });
};

window.loadOBJInViewer = (url) => {
  if (obj3d) { scene.remove(obj3d); obj3d = null; origMats = []; wireMode = false; }
  document.getElementById('vempty').style.display = 'none';
  canvas.style.display = 'block';

  new OBJLoader().load(url, loaded => {
    const box    = new THREE.Box3().setFromObject(loaded);
    const size   = box.getSize(new THREE.Vector3()).length();
    const center = box.getCenter(new THREE.Vector3());
    loaded.position.sub(center);
    loaded.scale.multiplyScalar(1.55 / size);

    loaded.traverse(n => {
      if (n.isMesh) {
        n.material = activeMat.clone();
        n.castShadow = true; n.receiveShadow = true;
      }
    });
    scene.add(loaded); obj3d = loaded;
    document.getElementById('vbar').style.opacity = '1';
  });
};

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas); resize();
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script>
"""

# ═════════════════════════════════════════════════════════════════════════════
# Header SVG illustration
# ═════════════════════════════════════════════════════════════════════════════

HEADER_SVG = """
<svg viewBox="0 0 960 170" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:960px;height:auto;display:block;margin:0 auto;">
  <defs>
    <radialGradient id="glow-c" cx="50%" cy="50%" r="45%">
      <stop offset="0%"   stop-color="#222" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#000" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glow-l" cx="25%" cy="55%" r="30%">
      <stop offset="0%"   stop-color="#1a1a1a" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#000"    stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glow-r" cx="78%" cy="50%" r="28%">
      <stop offset="0%"   stop-color="#1a1a1a" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#000"    stop-opacity="0"/>
    </radialGradient>
  </defs>

  <!-- ambient glows -->
  <rect width="960" height="170" fill="url(#glow-c)"/>
  <rect width="960" height="170" fill="url(#glow-l)"/>
  <rect width="960" height="170" fill="url(#glow-r)"/>

  <!-- ══ LEFT — point cloud / triangulated mesh ══ -->
  <!-- triangle fills -->
  <polygon points="62,82 98,52 138,68"  fill="#101010" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="98,52 138,68 162,40" fill="#0d0d0d" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="62,82 138,68 108,108" fill="#111"   stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="138,68 162,40 192,72" fill="#0f0f0f" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="138,68 192,72 108,108" fill="#121212" stroke="#1e1e1e" stroke-width="1"/>
  <!-- edges -->
  <g stroke="#282828" stroke-width="1" fill="none">
    <line x1="62"  y1="82"  x2="98"  y2="52"/>
    <line x1="98"  y1="52"  x2="138" y2="68"/>
    <line x1="62"  y1="82"  x2="138" y2="68"/>
    <line x1="62"  y1="82"  x2="108" y2="108"/>
    <line x1="138" y1="68"  x2="108" y2="108"/>
    <line x1="98"  y1="52"  x2="162" y2="40"/>
    <line x1="138" y1="68"  x2="162" y2="40"/>
    <line x1="162" y1="40"  x2="192" y2="72"/>
    <line x1="138" y1="68"  x2="192" y2="72"/>
    <line x1="192" y1="72"  x2="108" y2="108"/>
    <line x1="108" y1="108" x2="80"  y2="128"/>
    <line x1="62"  y1="82"  x2="80"  y2="128"/>
    <line x1="192" y1="72"  x2="215" y2="92"/>
    <line x1="108" y1="108" x2="215" y2="92"/>
  </g>
  <!-- vertices -->
  <g fill="#303030">
    <circle cx="62"  cy="82"  r="2.5"/>
    <circle cx="138" cy="68"  r="2.5"/>
    <circle cx="108" cy="108" r="2.5"/>
    <circle cx="192" cy="72"  r="2.5"/>
    <circle cx="80"  cy="128" r="2"/>
    <circle cx="215" cy="92"  r="2"/>
  </g>
  <!-- highlighted vertices -->
  <circle cx="98"  cy="52"  r="3.5" fill="#484848"/>
  <circle cx="162" cy="40"  r="4"   fill="#555"/>

  <!-- ══ CENTER — isometric wireframe cube  (cx=480, cy=88, s=52) ══
       top=(480,36) tr=(525,62) br=(525,114) bot=(480,140) bl=(435,114) tl=(435,62) mid=(480,88)
  -->
  <!-- faces -->
  <polygon points="480,36 525,62 480,88 435,62" fill="#0c0c0c" stroke="#2a2a2a" stroke-width="1.5"/>
  <polygon points="525,62 525,114 480,140 480,88" fill="#111"   stroke="#2a2a2a" stroke-width="1.5"/>
  <polygon points="435,62 480,88 480,140 435,114" fill="#0e0e0e" stroke="#2a2a2a" stroke-width="1.5"/>
  <!-- outer edges (bright) -->
  <g stroke="#3a3a3a" stroke-width="2" fill="none">
    <line x1="480" y1="36"  x2="525" y2="62"/>
    <line x1="525" y1="62"  x2="480" y2="88"/>
    <line x1="480" y1="88"  x2="435" y2="62"/>
    <line x1="435" y1="62"  x2="480" y2="36"/>
    <line x1="525" y1="62"  x2="525" y2="114"/>
    <line x1="480" y1="88"  x2="480" y2="140"/>
    <line x1="435" y1="62"  x2="435" y2="114"/>
    <line x1="525" y1="114" x2="480" y2="140"/>
    <line x1="480" y1="140" x2="435" y2="114"/>
  </g>
  <!-- inner center lines (dashed) -->
  <g stroke="#222" stroke-width="1" stroke-dasharray="3,5" fill="none">
    <line x1="480" y1="36"  x2="435" y2="62"/>
    <line x1="435" y1="62"  x2="435" y2="114"/>
  </g>
  <!-- vertices -->
  <g fill="#383838">
    <circle cx="525" cy="62"  r="3"/>
    <circle cx="525" cy="114" r="3"/>
    <circle cx="435" cy="114" r="3"/>
    <circle cx="480" cy="140" r="3"/>
    <circle cx="480" cy="88"  r="3"/>
    <circle cx="435" cy="62"  r="3"/>
  </g>
  <circle cx="480" cy="36" r="4.5" fill="#606060"/>

  <!-- ══ RIGHT — perspective mesh surface ══ -->
  <!-- triangle fills -->
  <polygon points="700,52 744,52 722,80"  fill="#0f0f0f" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="744,52 788,52 766,80"  fill="#0d0d0d" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="722,80 766,80 744,110" fill="#111"    stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="744,52 766,80 788,52"  fill="#101010" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="766,80 810,80 788,110" fill="#0e0e0e" stroke="#1e1e1e" stroke-width="1"/>
  <polygon points="678,80 722,80 700,110" fill="#0c0c0c" stroke="#1e1e1e" stroke-width="1"/>
  <!-- edges -->
  <g stroke="#252525" stroke-width="1" fill="none">
    <line x1="678" y1="52"  x2="700" y2="52"/>
    <line x1="700" y1="52"  x2="744" y2="52"/>
    <line x1="744" y1="52"  x2="788" y2="52"/>
    <line x1="788" y1="52"  x2="832" y2="52"/>
    <line x1="678" y1="80"  x2="722" y2="80"/>
    <line x1="722" y1="80"  x2="766" y2="80"/>
    <line x1="766" y1="80"  x2="810" y2="80"/>
    <line x1="810" y1="80"  x2="854" y2="80"/>
    <line x1="656" y1="110" x2="700" y2="110"/>
    <line x1="700" y1="110" x2="744" y2="110"/>
    <line x1="744" y1="110" x2="788" y2="110"/>
    <line x1="788" y1="110" x2="832" y2="110"/>
    <!-- diagonal triangulations -->
    <line x1="700" y1="52"  x2="678" y2="80"/>
    <line x1="700" y1="52"  x2="722" y2="80"/>
    <line x1="744" y1="52"  x2="722" y2="80"/>
    <line x1="744" y1="52"  x2="766" y2="80"/>
    <line x1="788" y1="52"  x2="766" y2="80"/>
    <line x1="788" y1="52"  x2="810" y2="80"/>
    <line x1="832" y1="52"  x2="810" y2="80"/>
    <line x1="678" y1="80"  x2="656" y2="110"/>
    <line x1="678" y1="80"  x2="700" y2="110"/>
    <line x1="722" y1="80"  x2="700" y2="110"/>
    <line x1="722" y1="80"  x2="744" y2="110"/>
    <line x1="766" y1="80"  x2="744" y2="110"/>
    <line x1="766" y1="80"  x2="788" y2="110"/>
    <line x1="810" y1="80"  x2="788" y2="110"/>
    <line x1="810" y1="80"  x2="832" y2="110"/>
    <line x1="854" y1="80"  x2="832" y2="110"/>
  </g>
  <!-- vertices -->
  <g fill="#2e2e2e">
    <circle cx="700" cy="52"  r="2.5"/>
    <circle cx="744" cy="52"  r="3"/>
    <circle cx="788" cy="52"  r="2.5"/>
    <circle cx="832" cy="52"  r="2"/>
    <circle cx="678" cy="80"  r="2.5"/>
    <circle cx="722" cy="80"  r="2.5"/>
    <circle cx="766" cy="80"  r="3"/>
    <circle cx="810" cy="80"  r="2.5"/>
    <circle cx="700" cy="110" r="2.5"/>
    <circle cx="744" cy="110" r="3"/>
    <circle cx="788" cy="110" r="2.5"/>
    <circle cx="832" cy="110" r="2"/>
  </g>
  <circle cx="744" cy="52" r="4"   fill="#555"/>
  <circle cx="766" cy="80" r="4.5" fill="#606060"/>

  <!-- ══ Ambient connector dots ══ -->
  <g fill="#181818">
    <circle cx="290" cy="44"  r="1.5"/>
    <circle cx="320" cy="125" r="1.5"/>
    <circle cx="258" cy="98"  r="1"/>
    <circle cx="345" cy="70"  r="1.5"/>
    <circle cx="375" cy="140" r="1"/>
    <circle cx="590" cy="28"  r="1.5"/>
    <circle cx="615" cy="135" r="1"/>
    <circle cx="632" cy="62"  r="1.5"/>
    <circle cx="870" cy="42"  r="1.5"/>
    <circle cx="895" cy="118" r="1"/>
    <circle cx="910" cy="72"  r="1.5"/>
  </g>

  <!-- ══ Connector dashes (left→cube, cube→right) ══ -->
  <line x1="228" y1="88" x2="432" y2="88" stroke="#171717" stroke-width="1" stroke-dasharray="3,9"/>
  <line x1="528" y1="88" x2="672" y2="80" stroke="#171717" stroke-width="1" stroke-dasharray="3,9"/>
</svg>
"""


# ═════════════════════════════════════════════════════════════════════════════
# CSS — Vercel / industrial minimal
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
:root {
  --bg:      #000000;
  --s900:    #0a0a0a;
  --s800:    #111111;
  --s700:    #1a1a1a;
  --s600:    #262626;
  --s500:    #3f3f46;
  --s400:    #71717a;
  --s300:    #a1a1aa;
  --s200:    #d4d4d8;
  --s100:    #f4f4f5;
  --white:   #ffffff;
  --font:    'Inter', -apple-system, sans-serif;
  --mono:    'JetBrains Mono', 'Fira Code', monospace;
}
body, .gradio-container {
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--s100) !important;
  -webkit-font-smoothing: antialiased;
}
.gradio-container { max-width: 1480px !important; margin: 0 auto !important; padding: 0 !important; }
footer, .built-with { display: none !important; }

/* ── Header ── */
.hdr {
  padding: 56px 40px 0;
  text-align: center;
  position: relative;
}
.hdr-eyebrow {
  display: inline-flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 10px; font-weight: 700;
  letter-spacing: .22em; text-transform: uppercase; color: var(--s500);
  margin-bottom: 22px;
}
.hdr-eyebrow::before, .hdr-eyebrow::after {
  content: ''; display: block; width: 28px; height: 1px; background: var(--s600);
}
.hdr-title {
  font-family: var(--font); font-weight: 700;
  font-size: clamp(2.8rem, 5.5vw, 5rem);
  letter-spacing: -.055em; line-height: 1;
  color: var(--white); margin-bottom: 14px;
}
.hdr-title .dim { color: var(--s600); }
.hdr-sub {
  font-size: 14px; font-weight: 400; color: var(--s400);
  max-width: 440px; line-height: 1.65; margin: 0 auto 32px;
}
.hdr-badges {
  display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 40px;
}
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: 999px;
  font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: .08em;
  border: 1px solid var(--s700); color: var(--s400); background: var(--s900);
}
.badge-dot { width: 5px; height: 5px; border-radius: 50%; background: #22c55e; animation: blink 2.4s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.hdr-rule { width: 100%; height: 1px; background: var(--s700); margin: 0 0 0; }

/* ── Layout ── */
.main-row { gap: 0 !important; }
.ctrl-col  { border-right: 1px solid var(--s700) !important; padding: 32px 28px !important; }
.view-col  { padding: 32px 28px !important; background: var(--bg) !important; }

/* ── Section label ── */
.sec {
  font-family: var(--mono); font-size: 9px; font-weight: 700;
  letter-spacing: .2em; text-transform: uppercase; color: var(--s500);
  margin-bottom: 12px; display: flex; align-items: center; gap: 10px;
}
.sec::after { content:''; flex: 1; height: 1px; background: var(--s700); }

/* ── Model radio cards ── */
.model-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-bottom: 24px; }
.mcard {
  position: relative; padding: 13px 14px;
  border: 1px solid var(--s700); border-radius: 6px;
  background: var(--bg); cursor: pointer; transition: border-color .15s;
}
.mcard:hover    { border-color: var(--s500); }
.mcard.selected { border-color: var(--white); }
.mcard-icon { font-size: 16px; margin-bottom: 7px; display: block; }
.mcard-name { font-size: 11px; font-weight: 600; color: var(--s200); margin-bottom: 2px; }
.mcard-meta { font-family: var(--mono); font-size: 9px; color: var(--s500); }
.mcard-tag  {
  position: absolute; top: 9px; right: 9px;
  font-family: var(--mono); font-size: 8px; font-weight: 700; letter-spacing: .08em;
  padding: 2px 6px; border-radius: 3px; text-transform: uppercase;
}
.tag-img  { background: #0d1a0d; color: #4ade80; border: 1px solid #1a3d1a; }
.tag-text { background: #0d1526; color: #60a5fa; border: 1px solid #1a3056; }
.tag-fast { background: #1c0e00; color: #fb923c; border: 1px solid #3d2000; }
.tag-best { background: #1a1a1a; color: #e5e5e5; border: 1px solid #333; }

/* ── Form controls ── */
.gradio-container textarea,
.gradio-container input[type=text],
.gradio-container input[type=number] {
  background: var(--bg) !important; border: 1px solid var(--s700) !important;
  border-radius: 6px !important; color: var(--s100) !important;
  font-family: var(--font) !important; font-size: 13px !important;
  transition: border-color .15s !important;
}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus,
.gradio-container input[type=number]:focus {
  border-color: var(--s500) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,.04) !important;
  outline: none !important;
}
.gradio-container label > span:first-child {
  font-family: var(--mono) !important; font-size: 9px !important;
  font-weight: 700 !important; letter-spacing: .16em !important;
  text-transform: uppercase !important; color: var(--s500) !important;
}
.gradio-container select {
  background: var(--bg) !important; border: 1px solid var(--s700) !important;
  border-radius: 6px !important; color: var(--s100) !important;
}
.gradio-container input[type=range] { accent-color: var(--white) !important; }

/* Accordion */
.gradio-container .accordion {
  background: var(--bg) !important; border: 1px solid var(--s700) !important;
  border-radius: 6px !important; margin-top: 12px !important;
}

/* ── Divider ── */
.ctrl-divider { width:100%; height:1px; background:var(--s700); margin:20px 0; }

/* ── Generate button ── */
#gen-btn {
  width: 100% !important; background: var(--white) !important;
  color: var(--bg) !important; border: none !important;
  border-radius: 6px !important; padding: 13px !important;
  font-family: var(--font) !important; font-size: 13px !important;
  font-weight: 600 !important; letter-spacing: .02em !important;
  cursor: pointer !important; transition: all .15s !important;
  margin-top: 18px !important;
}
#gen-btn:hover {
  background: var(--s200) !important; transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(255,255,255,.08) !important;
}
#gen-btn:active { transform: translateY(0) !important; }

/* ── Status ── */
.status-bar {
  margin-top: 12px; padding: 10px 14px;
  border: 1px solid var(--s700); border-radius: 6px;
  background: var(--bg);
  font-family: var(--mono); font-size: 11px; color: var(--s400);
  min-height: 40px;
}

/* ── Stats grid ── */
.stat-grid {
  display: grid; grid-template-columns: repeat(4,1fr);
  gap: 1px; background: var(--s700);
  border: 1px solid var(--s700); border-radius: 7px;
  overflow: hidden; margin-top: 14px;
}
.stat-cell { background: var(--bg); padding: 11px 13px; }
.stat-label {
  font-family: var(--mono); font-size: 8px; font-weight: 700;
  letter-spacing: .18em; text-transform: uppercase; color: var(--s500); margin-bottom: 5px;
}
.stat-value {
  font-family: var(--mono); font-size: 14px; font-weight: 500; color: var(--white);
}
.stat-mono { font-size: 11px; }
.stat-badge {
  display: inline-block; padding: 2px 7px; border-radius: 3px;
  font-family: var(--mono); font-size: 8px; font-weight: 700; letter-spacing: .1em;
}
.stat-ok   { background:#0d1a0d; color:#4ade80; border:1px solid #1a3d1a; }
.stat-warn { background:#1c1000; color:#fbbf24; border:1px solid #3d2800; }
.stat-error { padding:14px; border:1px solid #3d1a1a; border-radius:7px; background:#120808; }
.stat-error pre { font-family:var(--mono); font-size:10px; color:#888; margin-top:8px; white-space:pre-wrap; }

/* ── Download buttons ── */
.dl-row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:7px; margin-top:12px; }
.dl-btn {
  padding: 10px; border: 1px solid var(--s700); border-radius: 6px;
  background: var(--bg); text-align: center; cursor: pointer;
  transition: border-color .15s;
}
.dl-btn:hover { border-color: var(--s500); }
.dl-label  { font-family:var(--mono); font-size:9px; font-weight:700; letter-spacing:.15em; color:var(--s400); margin-bottom:3px; }
.dl-format { font-size:11px; color:var(--s200); }
.gradio-container .file-preview {
  background: var(--bg) !important; border: 1px solid var(--s700) !important;
  border-radius: 6px !important;
}

/* ── Tips ── */
.tips {
  margin-top:14px; padding:14px 16px;
  border:1px solid var(--s700); border-radius:7px; background:var(--bg);
}
.tips-title { font-family:var(--mono); font-size:9px; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color:var(--s500); margin-bottom:10px; }
.tip { display:flex; gap:10px; font-size:11px; color:var(--s400); margin-bottom:6px; line-height:1.55; }
.tip-arrow { color:var(--s600); flex-shrink:0; font-family:var(--mono); }
.tip strong { color:var(--s300); font-weight:500; }

/* ── Model info pill ── */
.model-info-pill {
  padding:10px 13px; border-radius:6px;
  background:var(--s900); border:1px solid var(--s700);
  font-size:12px; color:var(--s400); line-height:1.55;
  margin-bottom:16px;
}
.model-info-pill strong { color:var(--s200); }
"""


# ═════════════════════════════════════════════════════════════════════════════
# Gradio UI — Vercel / Industrial minimal
# ═════════════════════════════════════════════════════════════════════════════

MODEL_CHOICES = [
    "TripoSR  — Image → 3D  (best quality)",
    "Shap-E   — Text  → 3D",
    "Shap-E   — Image → 3D",
    "Point-E  — Text  → 3D  (fastest)",
]

MODEL_CARDS_HTML = """
<div class="model-cards">
  <div class="mcard selected" id="mc-0" onclick="selectModel(0)">
    <span class="mcard-tag tag-best">BEST</span>
    <span class="mcard-icon">◈</span>
    <div class="mcard-name">TripoSR</div>
    <div class="mcard-meta">Image → Mesh · ~5 s</div>
  </div>
  <div class="mcard" id="mc-1" onclick="selectModel(1)">
    <span class="mcard-tag tag-text">TEXT</span>
    <span class="mcard-icon">◇</span>
    <div class="mcard-name">Shap-E</div>
    <div class="mcard-meta">Text → Mesh · ~20 s</div>
  </div>
  <div class="mcard" id="mc-2" onclick="selectModel(2)">
    <span class="mcard-tag tag-img">IMG</span>
    <span class="mcard-icon">◻</span>
    <div class="mcard-name">Shap-E</div>
    <div class="mcard-meta">Image → Mesh · ~20 s</div>
  </div>
  <div class="mcard" id="mc-3" onclick="selectModel(3)">
    <span class="mcard-tag tag-fast">FAST</span>
    <span class="mcard-icon">△</span>
    <div class="mcard-name">Point-E</div>
    <div class="mcard-meta">Text → Mesh · ~8 s</div>
  </div>
</div>
<script>
function selectModel(idx) {
  [0,1,2,3].forEach(i => {
    const el = document.getElementById('mc-' + i);
    if (el) el.classList.toggle('selected', i === idx);
  });
  // Sync the hidden Gradio dropdown
  const choices = [
    "TripoSR  — Image → 3D  (best quality)",
    "Shap-E   — Text  → 3D",
    "Shap-E   — Image → 3D",
    "Point-E  — Text  → 3D  (fastest)"
  ];
  const dd = document.querySelector('.gradio-dropdown select, select#model_dd');
  if (dd) { dd.value = choices[idx]; dd.dispatchEvent(new Event('change', {bubbles:true})); }
  // also try the Gradio internal input
  const inp = document.querySelector('[data-testid="dropdown"] input');
  if (inp) { inp.value = choices[idx]; inp.dispatchEvent(new Event('input', {bubbles:true})); }
}
</script>
"""

MODEL_INFO = {
    "TripoSR  — Image → 3D  (best quality)": "<strong>TripoSR</strong> &mdash; Stability AI &times; Tripo AI &middot; MIT &middot; "
    "Single-image reconstruction via LRM transformer. Highest geometric fidelity.",
    "Shap-E   — Text  → 3D": "<strong>Shap-E</strong> &mdash; OpenAI &middot; MIT &middot; "
    "Latent diffusion conditioned on CLIP text embeddings. Best with detailed prompts.",
    "Shap-E   — Image → 3D": "<strong>Shap-E</strong> &mdash; OpenAI &middot; MIT &middot; "
    "Same backbone, conditioned on CLIP image embeddings instead of text.",
    "Point-E  — Text  → 3D  (fastest)": "<strong>Point-E</strong> &mdash; OpenAI &middot; MIT &middot; "
    "GLIDE diffusion &rarr; point cloud &rarr; marching-cubes mesh. Fastest option.",
}


def update_info(choice):
    html = MODEL_INFO.get(choice, "")
    return f'<div class="model-info-pill">{html}</div>'


def toggle_inputs(choice):
    needs_img = "Image" in choice and "→ 3D" in choice
    needs_text = "Text" in choice
    show_guid = "TripoSR" not in choice
    show_mc = "TripoSR" in choice
    return (
        gr.update(visible=needs_text),
        gr.update(visible=needs_img),
        gr.update(visible=show_guid),
        gr.update(visible=show_mc),
    )


with gr.Blocks(title="3D FORGE — Mesh Generator") as demo:
    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="hdr">
      <div class="hdr-eyebrow">Industrial 3D Generation Engine</div>
      <h1 class="hdr-title">Mesh<span class="dim">.</span>Forge</h1>
      <p class="hdr-sub">
        Generate production-ready 3D meshes from text or images.
        100&nbsp;% open-source &mdash; runs fully offline, no API keys.
      </p>
      <div class="hdr-badges">
        <span class="badge"><span class="badge-dot"></span>Running locally</span>
        <span class="badge">MIT licence</span>
        <span class="badge">CPU / CUDA</span>
        <span class="badge">OBJ · GLB · STL</span>
      </div>
      {HEADER_SVG}
    </div>
    <div class="hdr-rule"></div>
    """)

    with gr.Row(equal_height=False, elem_classes=["main-row"]):
        # ── Left — controls ──────────────────────────────────────────────────
        with gr.Column(scale=4, elem_classes=["ctrl-col"]):
            gr.HTML('<div class="sec">Model</div>')
            gr.HTML(MODEL_CARDS_HTML)

            model_dd = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Active model",
                elem_id="model_dd",
                visible=True,
            )
            model_info_html = gr.HTML(update_info(MODEL_CHOICES[0]))

            gr.HTML('<div class="ctrl-divider"></div>')
            gr.HTML('<div class="sec">Input</div>')

            text_prompt = gr.Textbox(
                label="Text prompt",
                placeholder='e.g. "a worn leather satchel with brass buckles and stitched seams"',
                lines=4,
                visible=False,
            )
            image_input = gr.Image(
                label="Reference image",
                type="pil",
                visible=True,
            )

            with gr.Accordion("Advanced parameters", open=False):
                guidance = gr.Slider(
                    2.0,
                    20.0,
                    value=15.0,
                    step=0.5,
                    label="Guidance scale  (Shap-E / Point-E)",
                )
                steps = gr.Slider(
                    16,
                    128,
                    value=64,
                    step=4,
                    label="Diffusion steps",
                )
                mc_res = gr.Slider(
                    64,
                    512,
                    value=256,
                    step=32,
                    label="Marching-cubes resolution  (TripoSR)",
                    visible=True,
                )
                seed_num = gr.Number(value=42, label="Seed", precision=0)

            gen_btn = gr.Button("Generate", elem_id="gen-btn", variant="primary")
            status_md = gr.Markdown("", elem_classes=["status-bar"])

        # ── Right — viewer + outputs ─────────────────────────────────────────
        with gr.Column(scale=8, elem_classes=["view-col"]):
            gr.HTML('<div class="sec">Viewport</div>')
            gr.HTML(VIEWER_HTML)

            stats_html = gr.HTML("")

            gr.HTML('<div class="sec" style="margin-top:20px">Export</div>')
            with gr.Row():
                out_obj = gr.File(label="OBJ", file_types=[".obj"], interactive=False)
                out_glb = gr.File(label="GLB", file_types=[".glb"], interactive=False)
                out_stl = gr.File(label="STL", file_types=[".stl"], interactive=False)

            gr.HTML("""
            <div class="tips">
              <div class="tips-title">Usage notes</div>
              <div class="tip"><span class="tip-arrow">→</span><span><strong>TripoSR</strong> — clean product photo, white/grey background, single centred object.</span></div>
              <div class="tip"><span class="tip-arrow">→</span><span><strong>Shap-E text</strong> — describe material, colour, shape and proportions in detail.</span></div>
              <div class="tip"><span class="tip-arrow">→</span><span><strong>Point-E</strong> — ideal for fast iteration; lower polygon count than Shap-E.</span></div>
              <div class="tip"><span class="tip-arrow">→</span><span>Use <strong>GLB</strong> for Unity / Unreal / web, <strong>STL</strong> for 3D printing, <strong>OBJ</strong> for Blender.</span></div>
            </div>
            """)

    # ── Wiring ───────────────────────────────────────────────────────────────
    model_dd.change(update_info, inputs=model_dd, outputs=model_info_html)
    model_dd.change(
        toggle_inputs,
        inputs=model_dd,
        outputs=[text_prompt, image_input, guidance, mc_res],
    )

    gen_btn.click(
        run_generation,
        inputs=[
            model_dd,
            model_dd,
            text_prompt,
            image_input,
            guidance,
            steps,
            mc_res,
            seed_num,
        ],
        outputs=[out_obj, out_glb, out_stl, status_md, stats_html],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, css=CSS)
