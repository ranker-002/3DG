# ══════════════════════════════════════════════════════════════════════════════
# run.ps1 — Windows launcher for the 3D Model Generator
# Usage:  .\run.ps1
# ══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$BLUE   = "`e[34m"
$GREEN  = "`e[32m"
$YELLOW = "`e[33m"
$RED    = "`e[31m"
$NC     = "`e[0m"

Write-Host "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
Write-Host "${BLUE}  3D Model Generator — Launcher                      ${NC}"
Write-Host "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Locate script directory ───────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ── Check virtual environment ─────────────────────────────────────────────────
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-Not (Test-Path $VenvPython)) {
    Write-Host "${RED}✗  Virtual environment not found at .venv\${NC}"
    Write-Host "${YELLOW}   Run the install steps first (see README.md)${NC}"
    exit 1
}

Write-Host "${GREEN}✓  Virtual environment found${NC}"

# ── Check that gradio is installed ────────────────────────────────────────────
$gradioCheck = & $VenvPython -c "import gradio" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "${RED}✗  gradio not installed in .venv${NC}"
    Write-Host "${YELLOW}   Run:  .venv\Scripts\python.exe -m pip install gradio${NC}"
    exit 1
}

Write-Host "${GREEN}✓  Dependencies OK${NC}"

# ── Ensure outputs directory exists ──────────────────────────────────────────
$OutputsDir = Join-Path $ScriptDir "outputs"
if (-Not (Test-Path $OutputsDir)) {
    New-Item -ItemType Directory -Path $OutputsDir | Out-Null
    Write-Host "${GREEN}✓  Created outputs\ directory${NC}"
}

# ── Launch ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
Write-Host "${GREEN}  🚀  Starting app…  http://localhost:7860            ${NC}"
Write-Host "${GREEN}  Press Ctrl+C to stop.                               ${NC}"
Write-Host "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
Write-Host ""

& $VenvPython app.py
