#!/usr/bin/env bash
# =============================================================================
# Model MRI-GPU — Step 2: State Capture
# Watches for state signals from 01_gpu_state_loader.py and
# captures RAM image + GPU metadata at each state.
#
# Run as: sudo bash 02_capture_states.sh
# Requires: avml or LiME built in step 0
# =============================================================================

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[CAPTURE]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
SIGNAL_DIR="/tmp/model-mri-gpu-signals"
CAPTURE_DIR="/tmp/model-mri-captures"
STATES=("S0" "S1" "S2" "S3" "S4" "S5")
RAM_CAPTURE=true         # Set false if you only want metadata (no LiME/avml)
COMPRESS=true            # gzip RAM images

mkdir -p "$CAPTURE_DIR"
mkdir -p "$SIGNAL_DIR"

# Detect acquisition method
if command -v avml &>/dev/null; then
    ACQUIRE_METHOD="avml"
elif ls /opt/lime/src/lime-*.ko 2>/dev/null | head -1 &>/dev/null; then
    ACQUIRE_METHOD="lime"
    LIME_KO=$(ls /opt/lime/src/lime-*.ko | head -1)
else
    ACQUIRE_METHOD="none"
    warn "No memory acquisition tool found (avml/LiME). Will capture metadata only."
    RAM_CAPTURE=false
fi

log "Acquisition method: $ACQUIRE_METHOD"
log "Capture directory: $CAPTURE_DIR"
log "Watching signal dir: $SIGNAL_DIR"

# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

acquire_ram() {
    local state="$1"
    local output_raw="$CAPTURE_DIR/${state}_ram.lime"

    if [[ "$RAM_CAPTURE" == false ]]; then
        warn "Skipping RAM capture (no acquisition tool)"
        return
    fi

    log "  Acquiring RAM image → $output_raw"
    local start_ts=$(date +%s)

    case "$ACQUIRE_METHOD" in
        avml)
            avml "$output_raw" 2>/dev/null
            ;;
        lime)
            insmod "$LIME_KO" "path=$output_raw format=lime" 2>/dev/null || true
            sleep 2
            rmmod lime 2>/dev/null || true
            ;;
    esac

    local end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))
    local size=$(du -sh "$output_raw" 2>/dev/null | cut -f1 || echo "?")
    ok "  RAM acquired in ${elapsed}s — size: $size"

    if [[ "$COMPRESS" == true ]]; then
        log "  Compressing..."
        gzip -f "$output_raw"
        ok "  Compressed: ${output_raw}.gz"
    fi
}

capture_gpu_metadata() {
    local state="$1"
    local outdir="$CAPTURE_DIR/${state}_metadata"
    mkdir -p "$outdir"

    log "  Capturing GPU metadata..."

    # nvidia-smi full dump
    nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,memory.total,\
memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,\
power.draw,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits \
        > "$outdir/nvidia_smi_gpu.csv" 2>/dev/null || true

    # Running compute processes
    nvidia-smi --query-compute-apps=pid,used_memory,name \
        --format=csv,noheader \
        > "$outdir/nvidia_smi_compute_procs.csv" 2>/dev/null || true

    # Full nvidia-smi output
    nvidia-smi > "$outdir/nvidia_smi_full.txt" 2>/dev/null || true

    # BAR1 memory (mapped host memory)
    nvidia-smi --query-gpu=pci.bus_id,bar1.memory.total,bar1.memory.used,bar1.memory.free \
        --format=csv,noheader \
        > "$outdir/nvidia_smi_bar1.csv" 2>/dev/null || true

    # /proc entries for GPU processes
    for pid_dir in /proc/[0-9]*/fd; do
        pid=$(echo "$pid_dir" | grep -o '[0-9]*' | head -1)
        if ls -la "$pid_dir" 2>/dev/null | grep -q nvidia 2>/dev/null; then
            echo "PID $pid uses NVIDIA device" >> "$outdir/nvidia_proc_pids.txt"
            cat "/proc/$pid/maps" > "$outdir/proc_${pid}_maps.txt" 2>/dev/null || true
            cat "/proc/$pid/status" > "$outdir/proc_${pid}_status.txt" 2>/dev/null || true
            cat "/proc/$pid/smaps_rollup" > "$outdir/proc_${pid}_smaps.txt" 2>/dev/null || true
            cat "/proc/$pid/cmdline" | tr '\0' ' ' > "$outdir/proc_${pid}_cmdline.txt" 2>/dev/null || true
        fi
    done 2>/dev/null || true

    # Kernel modules
    lsmod | grep -i nvidia > "$outdir/nvidia_modules.txt" 2>/dev/null || true

    # /proc/kallsyms — NVIDIA symbols
    grep -i nvidia /proc/kallsyms > "$outdir/kallsyms_nvidia.txt" 2>/dev/null || \
        echo "Permission denied (run as root)" > "$outdir/kallsyms_nvidia.txt"

    # Sysfs GPU info
    find /sys/bus/pci/drivers/nvidia/ -maxdepth 2 2>/dev/null | head -50 > "$outdir/sysfs_nvidia.txt" || true

    # dmesg NVIDIA entries
    dmesg | grep -i nvidia | tail -50 > "$outdir/dmesg_nvidia.txt" 2>/dev/null || true

    # Signal file (from loader)
    SIG_FILE="$SIGNAL_DIR/state_${state}.json"
    if [[ -f "$SIG_FILE" ]]; then
        cp "$SIG_FILE" "$outdir/state_signal.json"
    fi

    # Timestamp
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$outdir/capture_timestamp.txt"

    ok "  Metadata captured → $outdir"
}

capture_state() {
    local state="$1"
    local sig_file="$SIGNAL_DIR/state_${state}.json"
    local done_file="$SIGNAL_DIR/captured_${state}.done"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Capturing state: $state"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    acquire_ram "$state"
    capture_gpu_metadata "$state"

    # Signal back to loader
    touch "$done_file"
    ok "State $state captured. Signal sent to loader."
}

wait_for_state() {
    local state="$1"
    local sig_file="$SIGNAL_DIR/state_${state}.json"
    log "Waiting for state $state signal..."
    while [[ ! -f "$sig_file" ]]; do
        sleep 1
        echo -n "."
    done
    echo ""
    ok "State $state signal received."
}

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Model MRI-GPU — State Capture Script${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
log "Start 01_gpu_state_loader.py in another terminal now."
echo ""

for state in "${STATES[@]}"; do
    wait_for_state "$state"
    capture_state "$state"
    echo ""
done

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  All states captured.${NC}"
echo -e "${GREEN}  Captures: $CAPTURE_DIR${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
log "Contents of $CAPTURE_DIR:"
ls -lh "$CAPTURE_DIR"

echo ""
log "Next step:"
echo "  python3 03_volatility_plugin.py --captures-dir $CAPTURE_DIR"
