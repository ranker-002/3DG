<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0,0,0,20,40&height=220&section=header&text=3D%20Model%20Generator&fontSize=62&fontColor=ffffff&animation=fadeIn&fontAlignY=40&desc=100%25%20Free%20%E2%80%A2%20Open-Source%20%E2%80%A2%20No%20API%20Keys%20%E2%80%A2%20Runs%20Fully%20Locally&descAlignY=62&descSize=18&descColor=aaaaaa"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-8B5CF6?style=for-the-badge)](https://github.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-f59e0b?style=for-the-badge)](https://github.com/ranker-002/3DG/pulls)

<br/>

> **Generate production-ready OBJ / GLB / STL meshes from text prompts or images.**
> Four state-of-the-art models. Zero cost. Zero cloud dependency. Zero API keys.

<br/>

```

> ⚠️ **Tip:** Open this file on [GitHub](https://github.com/ranker-002/3DG) to see rendered Mermaid diagrams and badges.

```

---

## ✨ Highlights

|  | Feature |
|---|---|
| 🧠 | **4 AI models** — TripoSR, Shap-E (text & image), Point-E |
| 🔒 | **Fully offline** — no API keys, no subscriptions, no data leaves your machine |
| 🖥️ | **Interactive 3D viewer** — embedded Three.js viewport with orbit controls |
| 📦 | **3 export formats** — `.obj`, `.glb`, `.stl` for every workflow |
| ⚡ | **GPU & CPU** — CUDA-accelerated or pure CPU fallback |
| 🧪 | **Comprehensive tests** — 3-tier test suite covering imports → inference |
| 🪟 | **Cross-platform** — `install.sh` for Linux/macOS, `run.ps1` for Windows |

</div>

---

## 📑 Table of Contents

- [Pipeline Overview](#-pipeline-overview)
- [System Architecture](#-system-architecture)
- [Models](#-models)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
  - [Linux / macOS](#linux--macos)
  - [Windows](#windows)
  - [Manual Install](#manual-install)
- [Usage](#-usage)
- [UI Walkthrough](#-ui-walkthrough)
- [Model Reference](#-model-reference)
- [Parameters](#-parameters)
- [Output Formats](#-output-formats)
- [Hardware Requirements](#-hardware-requirements)
- [Test Suite](#-test-suite)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔄 Pipeline Overview

```mermaid
flowchart TD
    A(["👤 User Input"]) --> B{Input Mode}

    B -->|"📝 Text Prompt"| C["Text-Based Models"]
    B -->|"🖼️ Image Upload"| D["Image-Based Models"]

    C --> E["⚡ Point-E\nbase40M-textvec"]
    C --> F["🔷 Shap-E Text\ntext300M"]

    D --> G["🔷 Shap-E Image\nimage300M"]
    D --> H["🏆 TripoSR\nstabilityai/TripoSR"]

    E -->|"Point Cloud\n+ Marching Cubes\n(grid 128³)"| I[/"🔺 Raw Triangle Mesh"/]
    F -->|"Latent Diffusion\n+ Karras Sampler\n+ Decode"| I
    G -->|"CLIP Embedding\n+ Latent Diffusion\n+ Decode"| I
    H -->|"BG Removal → rembg\nLRM Encoder\nNeRF Decoder"| I

    I --> J["⚙️ Mesh Processing\n(Trimesh)"]
    J --> K["📐 Normalize\nCentroid → Origin\nScale → Unit Cube"]

    K --> L{"📤 Export"}
    L --> M["📄 .OBJ\nWavefront"]
    L --> N["📦 .GLB\nglTF Binary"]
    L --> O["🖨️ .STL\nStereoLithography"]

    M & N & O --> P["🖥️ Three.js Viewer\n+ Download Buttons"]
    M & N & O --> Q[("💾 outputs/\nLocal Disk")]

    style A fill:#111,stroke:#444,color:#fff
    style B fill:#1a1a1a,stroke:#555,color:#eee
    style I fill:#0d1117,stroke:#30363d,color:#ccc
    style J fill:#0d1117,stroke:#30363d,color:#ccc
    style K fill:#0d1117,stroke:#30363d,color:#ccc
    style L fill:#1a1a1a,stroke:#555,color:#eee
    style P fill:#111,stroke:#444,color:#fff
    style Q fill:#111,stroke:#444,color:#fff
```

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph Browser["🌐 Browser  —  localhost:7860"]
        direction LR
        UI_T["📝 Text\nInput"]
        UI_I["🖼️ Image\nUpload"]
        UI_P["🎛️ Parameter\nControls"]
        UI_V["🧊 Three.js\n3D Viewer"]
        UI_S["📊 Mesh\nStats"]
        UI_D["⬇️ Download\nOBJ/GLB/STL"]
    end

    subgraph Gradio["⚙️  Gradio Backend  —  app.py"]
        direction TB
        RG["run_generation()"]
        subgraph Cache["🗄️  Lazy Model Cache  (_cache {})"]
            CM1["TripoSR\nModel Instance"]
            CM2["Shap-E\nxm + text300M + image300M"]
            CM3["Point-E\nbase + upsampler"]
        end
        SAVE["_save_mesh()\nNormalize → Export"]
    end

    subgraph Models["🧠  AI Models"]
        M1["🏆 TripoSR\nstabilityai/TripoSR\n~500 MB"]
        M2["🔷 Shap-E\nOpenAI CDN\n~2 GB"]
        M3["⚡ Point-E\nOpenAI CDN\n~300 MB"]
    end

    subgraph Processing["🔬  Processing Layer"]
        P1["rembg\nBackground Removal"]
        P2["Marching Cubes\nPC → Mesh"]
        P3["Trimesh\nMesh I/O & Repair"]
    end

    subgraph HF["☁️  HuggingFace Hub  (first run only)"]
        HF1["stabilityai/TripoSR\nWeights Download"]
    end

    UI_T & UI_I & UI_P --> RG
    RG <--> Cache
    Cache <--> Models
    RG --> P1 --> M1
    RG --> P2 --> M3
    Models --> P3
    P3 --> SAVE
    SAVE --> UI_V
    SAVE --> UI_S
    SAVE --> UI_D
    M1 <-.->|"auto-download"| HF1

    style Browser fill:#0d1117,stroke:#30363d,color:#ccc
    style Gradio fill:#111827,stroke:#374151,color:#ccc
    style Models fill:#0f172a,stroke:#1e3a5f,color:#ccc
    style Processing fill:#0a0a0a,stroke:#222,color:#ccc
    style HF fill:#0d1117,stroke:#30363d,color:#888
    style Cache fill:#1a1a2e,stroke:#333,color:#ccc
```

---

## 🧠 Models

| | Model | Licence | Input | GPU Speed | Quality | Best For |
|---|---|---|---|---|---|---|
| 🏆 | **TripoSR** | MIT | Image | ~5 s | ⭐⭐⭐⭐⭐ | Highest fidelity single-image reconstruction |
| 🔷 | **Shap-E Text** | MIT | Text | ~20 s | ⭐⭐⭐⭐ | Rich descriptive text prompts |
| 🖼️ | **Shap-E Image** | MIT | Image | ~20 s | ⭐⭐⭐⭐ | Image input with text-level control |
| ⚡ | **Point-E** | MIT | Text | ~8 s | ⭐⭐⭐ | Fast prototyping, coloured point clouds |

> All model weights are **downloaded automatically** on first run. No manual setup required.

---

## 🚀 Quick Start

```bash
git clone https://github.com/ranker-002/3DG.git
cd 3DG

# Linux / macOS — one command does everything
chmod +x install.sh && ./install.sh

# Activate and launch
source .venv/bin/activate
python app.py
# → Open http://localhost:7860
```

```powershell
# Windows PowerShell — one command launch (after manual install)
.\run.ps1
```

---

## 📦 Installation

### Linux / macOS

```bash
# Clone
git clone https://github.com/ranker-002/3DG.git
cd 3DG

# Auto-detect CUDA and install everything
chmod +x install.sh
./install.sh          # auto-detect GPU

# — or — force CPU-only
./install.sh cpu
```

The script will:

1. Verify Python 3.10+
2. Create a `.venv` virtual environment
3. Install PyTorch with the correct CUDA index URL
4. Install Shap-E, TripoSR, Point-E from source
5. Install all remaining dependencies from `requirements.txt`

---

### Windows

```powershell
git clone https://github.com/ranker-002/3DG.git
cd 3DG

# 1 — Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2 — PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3 — AI models from source
pip install git+https://github.com/openai/shap-e.git
pip install git+https://github.com/VAST-AI-Research/TripoSR.git
pip install git+https://github.com/openai/point-e.git

# 4 — Remaining deps
pip install -r requirements.txt

# 5 — Launch
python app.py
```

Or use the provided launcher after the steps above:

```powershell
.\run.ps1
```

---

### Manual Install

<details>
<summary>Click to expand — step-by-step with all options</summary>

```bash
# PyTorch variants
# CUDA 12.1 (RTX 30xx / 40xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (GTX 10xx / 20xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Models (MIT licensed, installed from source)
pip install git+https://github.com/openai/shap-e.git
pip install git+https://github.com/VAST-AI-Research/TripoSR.git
pip install git+https://github.com/openai/point-e.git

# Core dependencies
pip install -r requirements.txt

# Optional (performance improvements)
pip install pyfqmr       # faster quadric mesh decimation
pip install xatlas       # GPU-accelerated UV unwrapping
pip install open3d       # advanced point-cloud operations
```

</details>

---

## 🎮 Usage

```bash
# Start the Gradio web server
python app.py

# The UI is available at:
http://localhost:7860
```

---

## 🖥️ UI Walkthrough

```mermaid
sequenceDiagram
    actor User
    participant UI as Gradio UI
    participant BE as Backend (app.py)
    participant Cache as Model Cache
    participant FS as outputs/

    User->>UI: Select model & input mode
    User->>UI: Enter prompt or upload image
    User->>UI: Adjust parameters (optional)
    User->>UI: Click "Generate"

    UI->>BE: run_generation(model, prompt/image, params)

    BE->>Cache: Check _cache for loaded model
    alt Model not cached
        Cache-->>BE: miss
        BE->>BE: Load model weights (HuggingFace / CDN)
        BE->>Cache: Store loaded model
    else Model already cached
        Cache-->>BE: Return cached model
    end

    BE->>BE: Run inference
    BE->>BE: _save_mesh() → normalize → export
    BE->>FS: Write .obj / .glb / .stl

    BE-->>UI: Return file paths + mesh stats HTML
    UI-->>User: Show 3D viewer, stats, download buttons
    User->>UI: Interact with viewer (RESET / WIRE / SPIN / LIGHT)
    User->>UI: Download preferred format
```

### Viewer Controls

| Button | Action |
|---|---|
| **RESET** | Return camera to default position |
| **WIRE** | Toggle wireframe / solid rendering |
| **SPIN** | Toggle auto-rotation |
| **LIGHT** | Cycle through 4 studio lighting presets |

> **Orbit** — Left-click drag · **Zoom** — Scroll wheel · **Pan** — Right-click drag

---

## 📖 Model Reference

### 🏆 TripoSR

> *Stability AI × Tripo AI · MIT License*

- **Paper:** [TripoSR: Fast 3D Object Reconstruction from a Single Image (2024)](https://arxiv.org/abs/2403.02156)
- **Weights:** `stabilityai/TripoSR` on HuggingFace (~500 MB, auto-downloaded)
- **Architecture:** Large Reconstruction Model (LRM) — transformer encoder + triplane NeRF decoder
- **Pipeline:** `rembg` background removal → foreground resize (0.85 ratio) → LRM encoding → marching-cubes mesh extraction

**Image Tips:**

```
✅ Clean white or transparent background
✅ Single object, centred and well-lit
✅ Unambiguous silhouette
✅ Front-facing or 3/4 view
❌ Cluttered scenes
❌ Multiple overlapping objects
❌ Motion blur or heavy bokeh
```

---

### 🔷 Shap-E Text / Image

> *OpenAI · MIT License*

- **Paper:** [Shap-E: Generating Conditional 3D Implicit Functions (2023)](https://arxiv.org/abs/2305.02463)
- **Weights:** OpenAI CDN (~2 GB total, auto-downloaded on first run)
- **Architecture:** Latent diffusion conditioned on CLIP text/image embeddings → implicit 3D function → triangle mesh via `decode_latent_mesh`
- **Sampler:** Karras noise schedule with configurable steps and guidance scale

**Text Prompt Tips:**

```
✅ "a dark oak rocking chair with spindle back and curved runners"
✅ "matte black sci-fi helmet with blue visor and angular vents"
✅ "low-poly cartoon cactus in a small terracotta pot"
✅ "smooth white ceramic coffee mug with a C-shaped handle"
❌ "chair"           (too vague, no style or material)
❌ "a thing"         (no semantic content)
❌ "landscape"       (scenes not single objects — use images instead)
```

---

### ⚡ Point-E

> *OpenAI · MIT License*

- **Paper:** [Point-E: A System for Generating 3D Point Clouds from Complex Prompts (2022)](https://arxiv.org/abs/2212.08751)
- **Weights:** OpenAI CDN (~300 MB, auto-downloaded)
- **Architecture:** GLIDE-style image diffusion → 4,096-point coloured cloud via upsampler → marching-cubes mesh (128³ grid)
- **Fastest option** — best for rapid iteration and prototyping

---

## 🎛️ Parameters

| Parameter | Models | Range | Default | Description |
|---|---|---|---|---|
| **Guidance Scale** | Shap-E | 3 – 20 | 15 | Classifier-free guidance strength — higher = more prompt-adherent, less diverse |
| **Diffusion Steps** | Shap-E, Point-E | 16 – 128 | 64 | Karras denoising steps — more steps = higher quality, slower |
| **MC Resolution** | TripoSR | 128 – 512 | 256 | Marching-cubes grid resolution — higher = more detail, more VRAM |
| **Seed** | All | 0 – 2³² | 0 | Random seed for reproducibility — same seed + prompt = same mesh |

---

## 📤 Output Formats

```mermaid
graph LR
    M["🔺 Generated Mesh\n(Trimesh)"]

    M -->|"mesh.export(path.obj)"| OBJ["📄 .OBJ\nWavefront Object"]
    M -->|"mesh.export(path.glb)"| GLB["📦 .GLB\nglTF 2.0 Binary"]
    M -->|"mesh.export(path.stl)"| STL["🖨️ .STL\nStereoLithography"]

    OBJ --> T1["Blender\nMeshLab\nMaya\nAutoCAD"]
    GLB --> T2["Unity\nUnreal Engine\nThree.js\nAR / VR\nSketchfab"]
    STL --> T3["FDM 3D Printing\nResin Printing\nCNC Machining"]

    style M fill:#111,stroke:#333,color:#ccc
    style OBJ fill:#0d1117,stroke:#30363d,color:#ccc
    style GLB fill:#0d1117,stroke:#30363d,color:#ccc
    style STL fill:#0d1117,stroke:#30363d,color:#ccc
```

All meshes are automatically:
- **Centred** at the world origin (`centroid → [0, 0, 0]`)
- **Normalized** to a unit cube (`max(extents) = 1.0`)
- **Stamped** with a unique 8-char hex ID to prevent overwrites

---

## 💻 Hardware Requirements

| | Minimum | Recommended | Ideal |
|---|---|---|---|
| **GPU VRAM** | 4 GB | 8 GB | 16 GB+ |
| **RAM** | 8 GB | 16 GB | 32 GB |
| **Disk (models)** | 3 GB | 6 GB | 12 GB |
| **GPU** | GTX 1060 6 GB | RTX 3070 | RTX 4090 |
| **CPU** | Any 4-core | 8-core | 12-core+ |

### Expected Inference Times

```mermaid
gantt
    title Approximate Generation Time per Model (RTX 3080)
    dateFormat  X
    axisFormat  %ss

    section TripoSR
    Image → Mesh       :0, 5

    section Point-E
    Text → Mesh        :0, 8

    section Shap-E Text
    Text → Mesh        :0, 20

    section Shap-E Image
    Image → Mesh       :0, 22
```

> **CPU inference works** but expect 2–10 minutes per model. Weights are cached after the first load, so subsequent generations in the same session are faster.

---

## 🧪 Test Suite

The project ships with a comprehensive 3-tier test suite in `run_tests.py`.

```mermaid
flowchart LR
    T1["🟢 Tier 1\nImports & Environment\n─────────────────\n• Python version\n• venv active\n• outputs/ dir\n• TripoSR cloned\n• All imports\n~5 seconds"]

    T2["🔵 Tier 2\nUnit / Integration\n─────────────────\n• Torch basic ops\n• CUDA device\n• NumPy ops\n• Pillow I/O\n• Trimesh OBJ/GLB/STL\n• _save_mesh() helper\n• Gradio blocks\n• App syntax check\n~30 seconds"]

    T3["🟡 Tier 3\nModel Smoke Tests\n─────────────────\n• Point-E load\n• Point-E text gen\n• Shap-E load\n• Shap-E text gen\n• TripoSR load\n• TripoSR image gen\n⚠️ Downloads weights\n5–10 minutes"]

    T1 -->|always| T2
    T2 -->|opt-in| T3

    style T1 fill:#052e16,stroke:#16a34a,color:#bbf7d0
    style T2 fill:#0c1a4b,stroke:#3b82f6,color:#bfdbfe
    style T3 fill:#2d1b00,stroke:#d97706,color:#fde68a
```

### Running Tests

```bash
# Tier 1 + 2 only (fast, no model downloads)  — DEFAULT
python run_tests.py

# All tiers including model inference (downloads weights on first run)
python run_tests.py --all

# A specific tier only
python run_tests.py --tier 1
python run_tests.py --tier 2
python run_tests.py --tier 3

# Verbose error output
VERBOSE=1 python run_tests.py --all
```

### Test Coverage Summary

| Tier | Tests | Scope |
|---|---|---|
| T1 | 17 | Python env, venv, directory structure, all package imports |
| T2 | 16 | PyTorch ops, NumPy, Pillow I/O, Trimesh export, Gradio blocks, app syntax |
| T3 | 6 | Full model load + 1-step inference for each model |

---

## 🗂️ Project Structure

```
3DG/
├── 📄 app.py                  ← Main application: model logic + Gradio UI
│   ├── _load_shap_e()         │  Lazy loader with _cache{} dict
│   ├── _load_point_e()        │
│   ├── _load_triposr()        │
│   ├── shap_e_text()          │  Generation functions
│   ├── shap_e_image()         │
│   ├── triposr_image()        │
│   ├── point_e_text()         │
│   ├── run_generation()       │  Main Gradio callback
│   ├── VIEWER_HTML            │  Embedded Three.js viewport
│   └── HEADER_SVG             │  Decorative SVG banner
│
├── 🧪 run_tests.py            ← 3-tier test suite (52 test functions)
├── 🔬 test_imports.py         ← Quick import sanity check
│
├── 🛠️ install.sh              ← One-command installer (Linux/macOS)
├── 🪟 run.ps1                 ← Windows PowerShell launcher
├── 📋 requirements.txt        ← Python dependencies
│
├── 📁 TripoSR/                ← TripoSR source (git submodule)
│   └── tsr/                   │  tsr.system.TSR, tsr.utils.*
│
├── 📁 outputs/                ← Generated meshes (gitignored)
│   └── <model>_<uid>.{obj,glb,stl}
│
└── 📁 .venv/                  ← Virtual environment (gitignored)
```

---

## 🔮 Roadmap

```mermaid
timeline
    title Feature Roadmap
    section v1.x  (current)
        TripoSR image-to-3D      : Highest fidelity reconstruction
        Shap-E text & image      : OpenAI latent diffusion
        Point-E text             : Fast point-cloud pipeline
        Three.js viewer          : Interactive in-browser preview
        OBJ / GLB / STL export   : Production-ready formats
    section v2.x
        Texture baking           : UV + diffuse/normal maps
        Batch generation         : Multiple meshes in one run
        Mesh repair pipeline     : Hole filling, manifold fixing
        REST API mode            : Headless generation endpoint
    section v3.x
        Zero-1-to-3              : Novel-view synthesis model
        InstantMesh              : Multi-view reconstruction
        Video-to-3D              : Frame extraction pipeline
```

---

## 🤝 Contributing

Contributions are very welcome! Here's how to get started:

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/3DG.git
cd 3DG

# 2. Create a feature branch
git checkout -b feature/my-awesome-feature

# 3. Set up environment
./install.sh

# 4. Make your changes, then verify
python run_tests.py        # must pass T1 + T2
python run_tests.py --all  # run before opening a PR

# 5. Commit and push
git commit -m "feat: add my awesome feature"
git push origin feature/my-awesome-feature

# 6. Open a Pull Request on GitHub
```

### Contribution Guidelines

| Area | Notes |
|---|---|
| **New model** | Add a `_load_<model>()` + generation function, wire into `run_generation()`, add T3 smoke test |
| **Bug fix** | Include a failing T2 test that your fix makes pass |
| **UI change** | Screenshot or screen-recording in the PR description |
| **Deps** | Pin to a minimum version range, keep all licences MIT / Apache 2.0 |

---

## 📄 License

```
MIT License

Copyright (c) 2025 ranker-002

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### Dependency Licences

| Package | Licence | Link |
|---|---|---|
| TripoSR | MIT | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) |
| Shap-E | MIT | [openai/shap-e](https://github.com/openai/shap-e) |
| Point-E | MIT | [openai/point-e](https://github.com/openai/point-e) |
| Trimesh | MIT | [mikedh/trimesh](https://github.com/mikedh/trimesh) |
| Gradio | Apache 2.0 | [gradio-app/gradio](https://github.com/gradio-app/gradio) |
| PyTorch | BSD-3 | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| rembg | MIT | [danielgatis/rembg](https://github.com/danielgatis/rembg) |
| Three.js | MIT | [mrdoob/three.js](https://github.com/mrdoob/three.js) |

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0,0,0,20,40&height=100&section=footer"/>

**Built with ❤️ using only open-source, MIT-licensed AI models.**

*No cloud. No keys. No cost.*

[![Star on GitHub](https://img.shields.io/github/stars/ranker-002/3DG?style=social)](https://github.com/ranker-002/3DG)

</div>