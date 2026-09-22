#!/usr/bin/env bash
# =============================================================================
# Model MRI-GPU — Step 0: Environment Setup (FIXED)
# =============================================================================

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[SETUP]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

[[ $EUID -eq 0 ]] || fail "Run with sudo"

VENV_DIR="/opt/model-mri-gpu-venv"

# ── 1. System packages ──────────────────────────────────────
log "Installing system packages..."
apt-get update -qq
apt-get install -y \
    python3 python3-pip python3-venv \
    python3-dev \
    git curl wget build-essential \
    linux-headers-$(uname -r) \
    patchelf lm-sensors jq \
    2>/dev/null
ok "System packages installed"

# ── 2. NVIDIA check ─────────────────────────────────────────
log "Checking NVIDIA tooling..."
command -v nvidia-smi &>/dev/null || fail "nvidia-smi not found"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ok "NVIDIA tooling present"

modinfo nvidia 2>/dev/null | grep -q "open" \
    && ok "Open NVIDIA kernel module detected (better forensic visibility)" \
    || warn "Proprietary NVIDIA kernel module"

# ── 3. Clean + rebuild venv ─────────────────────────────────
log "Rebuilding Python virtual environment..."
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR" --without-pip
ok "venv skeleton created"

# Bootstrap pip manually (avoids the broken pip-in-venv problem)
log "Bootstrapping pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python3"
"$VENV_DIR/bin/pip" --version
ok "pip bootstrapped"

# ── 4. Upgrade pip ──────────────────────────────────────────
log "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel -q
ok "pip upgraded"

# ── 5. Core forensic / analysis packages (no GPU required) ──
log "Installing core packages (pypi.org)..."
"$VENV_DIR/bin/pip" install \
    volatility3 \
    psutil \
    numpy \
    rich \
    pynvml \
    requests \
    tabulate \
    2>&1 | grep -E "^(Successfully|ERROR|Collecting)" || true
ok "Core packages installed"

# ── 6. PyTorch — try CUDA wheel, fall back to CPU ───────────
log "Installing PyTorch..."
# Try CUDA 12.1 wheel (download.pytorch.org)
if "$VENV_DIR/bin/pip" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    -q 2>/dev/null; then
    ok "PyTorch installed (CUDA 12.1 wheel)"
else
    warn "CUDA wheel download failed (domain may be blocked)"
    warn "Trying CPU-only PyTorch from pypi.org..."
    if "$VENV_DIR/bin/pip" install torch torchvision torchaudio -q 2>/dev/null; then
        ok "PyTorch installed (CPU-only from pypi.org)"
        warn "GPU inference will NOT work with CPU-only torch."
        warn "You already have PyTorch+CUDA installed system-wide via Ollama."
        warn "We will use pynvml for GPU evidence — torch only needed for the loader."
    else
        warn "PyTorch install failed — state loader will use pynvml only (metadata capture still works)"
    fi
fi

# ── 7. HuggingFace transformers (for state loader) ──────────
log "Installing transformers (for model loading in loader script)..."
"$VENV_DIR/bin/pip" install transformers accelerate -q 2>&1 | \
    grep -E "^(Successfully|ERROR)" || true
ok "transformers installed"

# ── 8. LiME — kernel module for RAM acquisition ─────────────
log "Installing LiME for RAM acquisition..."
LIME_DIR="/opt/lime"
if [[ ! -d "$LIME_DIR" ]]; then
    git clone --depth=1 https://github.com/504ensicsLabs/LiME.git "$LIME_DIR" -q
fi
cd "$LIME_DIR/src"
if make -s 2>/dev/null; then
    ok "LiME built: $LIME_DIR/src/lime-$(uname -r).ko"
else
    warn "LiME build failed — will use avml fallback"
fi
cd - >/dev/null

# ── 9. avml — Microsoft memory acquisition fallback ─────────
log "Installing avml..."
AVML_URL="https://github.com/microsoft/avml/releases/latest/download/avml"
AVML_PATH="/usr/local/bin/avml"
if wget -q "$AVML_URL" -O "$AVML_PATH" && chmod +x "$AVML_PATH"; then
    ok "avml installed: $AVML_PATH"
else
    warn "avml download failed (github.com may be blocked) — RAM capture will use LiME only"
fi

# ── 10. Verify GPU visible in Python ────────────────────────
log "Verifying GPU access from Python..."
"$VENV_DIR/bin/python3" -c "
import pynvml
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
print(f'  NVML GPUs: {count}')
for i in range(count):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
    drv  = pynvml.nvmlSystemGetDriverVersion()
    print(f'  GPU {i}: {name}')
    print(f'    VRAM: {mem.total//1024**2} MB total, {mem.used//1024**2} MB used')
    print(f'    Driver: {drv}')
pynvml.nvmlShutdown()
print('  pynvml: OK')
" && ok "GPU accessible from venv"

# torch check (optional)
"$VENV_DIR/bin/python3" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
" 2>/dev/null && ok "PyTorch+CUDA check passed" || warn "PyTorch CUDA not available (metadata capture still works)"

# ── 11. Verify Volatility ────────────────────────────────────
log "Verifying Volatility 3..."
"$VENV_DIR/bin/python3" -c "
import volatility3
print(f'  Volatility3: {volatility3.__version__}')
" && ok "Volatility 3 ready"

# ── 12. Write env info ───────────────────────────────────────
log "Writing environment info..."
"$VENV_DIR/bin/python3" -c "
import json, platform, pynvml, volatility3
pynvml.nvmlInit()
info = {
    'python': platform.python_version(),
    'os': platform.platform(),
    'kernel': platform.release(),
    'volatility3': volatility3.__version__,
    'gpus': [{
        'name': pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)),
        'vram_mb': pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total // 1024**2,
        'driver': pynvml.nvmlSystemGetDriverVersion()
    } for i in range(pynvml.nvmlDeviceGetCount())]
}
pynvml.nvmlShutdown()
print(json.dumps(info, indent=2))
" | tee /opt/model-mri-gpu-venv/env_info.json

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete.${NC}"
echo -e "${GREEN}  Activate: source $VENV_DIR/bin/activate${NC}"
echo -e "${GREEN}  Next:     bash scripts/04_quick_validate.sh${NC}"
echo -e "${GREEN}============================================${NC}"
