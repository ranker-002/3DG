#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# install.sh — One-command setup for the 3D Model Generator
# Usage:
#   chmod +x install.sh
#   ./install.sh          # auto-detect CUDA
#   ./install.sh cpu      # force CPU-only
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'

echo -e "${BLU}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLU}  3D Model Generator — OSS Install Script            ${NC}"
echo -e "${BLU}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Python check ─────────────────────────────────────────────────────────────
python3 --version >/dev/null 2>&1 || { echo -e "${RED}Python 3 not found${NC}"; exit 1; }
PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GRN}✓ Python $PY found${NC}"

# ── Virtual env ──────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo -e "${YLW}Creating virtual environment…${NC}"
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q

# ── PyTorch ──────────────────────────────────────────────────────────────────
MODE=${1:-auto}
if [ "$MODE" == "cpu" ]; then
  echo -e "${YLW}Installing PyTorch (CPU)…${NC}"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
else
  # Try to detect CUDA version
  if command -v nvcc &>/dev/null; then
    CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.')
    case "$CUDA" in
      121|122|123|124) IDX="cu121" ;;
      118|119|120)     IDX="cu118" ;;
      *)               IDX="cu121" ;;
    esac
  else
    IDX="cu121"
  fi
  echo -e "${YLW}Installing PyTorch (CUDA index: $IDX)…${NC}"
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/$IDX" -q
fi

# ── Model packages (from source) ─────────────────────────────────────────────
echo -e "${YLW}Installing Shap-E (OpenAI, MIT)…${NC}"
pip install "git+https://github.com/openai/shap-e.git" -q

echo -e "${YLW}Installing TripoSR (Stability AI, MIT)…${NC}"
pip install "git+https://github.com/VAST-AI-Research/TripoSR.git" -q

echo -e "${YLW}Installing Point-E (OpenAI, MIT)…${NC}"
pip install "git+https://github.com/openai/point-e.git" -q

# ── Remaining deps ────────────────────────────────────────────────────────────
echo -e "${YLW}Installing remaining dependencies…${NC}"
pip install -r requirements.txt -q

echo ""
echo -e "${GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GRN}  ✅  Installation complete!                          ${NC}"
echo -e "${GRN}  Run:  source .venv/bin/activate && python app.py   ${NC}"
echo -e "${GRN}  Then open: http://localhost:7860                    ${NC}"
echo -e "${GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
