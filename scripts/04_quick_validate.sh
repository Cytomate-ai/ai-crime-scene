#!/usr/bin/env bash
# =============================================================================
# Model MRI-GPU — Quick Validation (NO root, NO RAM capture)
# Run this first to confirm the metadata pipeline works end-to-end
# on your research machine before doing the full RAM capture.
# =============================================================================

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[VALIDATE]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

VENV="/opt/model-mri-gpu-venv"
SIGNAL_DIR="/tmp/model-mri-gpu-signals"
CAPTURE_DIR="/tmp/model-mri-captures"

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Model MRI-GPU — Quick Validation${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# ── 1. Python environment ──
log "Checking Python environment..."
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
    ok "Virtual environment: $VENV"
else
    warn "Virtual environment not found. Trying system Python."
fi

python3 -c "import torch, pynvml, volatility3, rich; print('  All imports OK')" && ok "Python dependencies" \
    || fail "Python dependencies missing. Run 00_setup_environment.sh first."

# ── 2. GPU check ──
log "Checking GPU..."
python3 -c "
import torch, pynvml
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
print(f'  GPUs: {count}')
for i in range(count):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f'  GPU {i}: {pynvml.nvmlDeviceGetName(h)} — {mem.total//1024**2}MB total, {mem.used//1024**2}MB used')
pynvml.nvmlShutdown()
" && ok "GPU accessible via pynvml"

# ── 3. Simulate a quick S0→S1→S2 metadata-only run ──
log "Running metadata-only state simulation..."
mkdir -p "$SIGNAL_DIR" "$CAPTURE_DIR"

# Clear old signals
rm -f "$SIGNAL_DIR"/*.json "$SIGNAL_DIR"/*.done 2>/dev/null || true

# Run loader with --no-wait (auto-advances without waiting for capture)
log "Starting GPU state loader (metadata only, no RAM capture)..."
python3 "$(dirname "$0")/01_gpu_state_loader.py" \
    --no-wait \
    --model "Qwen/Qwen2.5-0.5B" \
    2>/dev/null &
LOADER_PID=$!

log "Loader PID: $LOADER_PID"

# Simulate metadata capture for each state
for state in S0 S1 S2 S3 S5; do
    sig_file="$SIGNAL_DIR/state_${state}.json"
    log "Waiting for $state signal..."
    for i in $(seq 1 60); do
        [[ -f "$sig_file" ]] && break
        sleep 2
    done

    if [[ ! -f "$sig_file" ]]; then
        warn "Timeout on $state — skipping"
        continue
    fi

    # Capture GPU metadata only (no RAM)
    outdir="$CAPTURE_DIR/${state}_metadata"
    mkdir -p "$outdir"
    cp "$sig_file" "$outdir/state_signal.json" 2>/dev/null || true

    # nvidia-smi captures
    nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,memory.total,\
memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,\
power.draw,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits \
        > "$outdir/nvidia_smi_gpu.csv" 2>/dev/null || true

    nvidia-smi --query-compute-apps=pid,used_memory,name \
        --format=csv,noheader \
        > "$outdir/nvidia_smi_compute_procs.csv" 2>/dev/null || true

    nvidia-smi --query-gpu=pci.bus_id,bar1.memory.total,bar1.memory.used,bar1.memory.free \
        --format=csv,noheader \
        > "$outdir/nvidia_smi_bar1.csv" 2>/dev/null || true

    # /proc NVIDIA PIDs
    for pid_dir in /proc/[0-9]*/fd; do
        pid=$(echo "$pid_dir" | grep -o '[0-9]*' | head -1)
        ls -la "$pid_dir" 2>/dev/null | grep -q nvidia 2>/dev/null && \
            echo "PID $pid uses NVIDIA device" >> "$outdir/nvidia_proc_pids.txt" || true
    done 2>/dev/null || true

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$outdir/capture_timestamp.txt"
    ok "  $state metadata captured"

    # Signal loader we're done
    touch "$SIGNAL_DIR/captured_${state}.done"
done

# Wait for loader to finish
wait $LOADER_PID 2>/dev/null || true
ok "Loader complete"

# ── 4. Run analyzer ──
log "Running capture analyzer..."
python3 "$(dirname "$0")/03_analyze_captures.py" --captures-dir "$CAPTURE_DIR"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Quick validation complete.${NC}"
echo -e "${GREEN}  Full RAM capture: sudo bash 02_capture_states.sh${NC}"
echo -e "${GREEN}================================================${NC}"
