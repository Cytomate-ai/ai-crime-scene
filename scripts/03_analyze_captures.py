#!/usr/bin/env python3
"""
Model MRI-GPU — Capture Analyzer
=================================
Analyzes the metadata captures from 02_capture_states.sh
and answers the core research question:

    Can we distinguish S0, S1, S2, S3 using only host evidence?

Does NOT require Volatility or a RAM image — works on the metadata alone.
Use this first to validate the experiment before Volatility analysis.

Usage:
    python3 03_analyze_captures.py --captures-dir /tmp/model-mri-captures/
"""

import argparse
import json
import csv
import os
import re
import sys
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


def log(msg, style=None):
    if RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def header(msg):
    log(f"\n{'='*60}", style="cyan")
    log(f"  {msg}", style="bold cyan")
    log(f"{'='*60}", style="cyan")


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def parse_nvidia_smi_csv(path: Path) -> list:
    """Parse nvidia-smi --format=csv output."""
    if not path.exists():
        return []
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append([x.strip() for x in line.split(",")])
    except Exception:
        pass
    return rows


def parse_state_signal(path: Path) -> dict:
    """Parse the state signal JSON from the loader."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def parse_compute_procs(path: Path) -> list:
    """Parse nvidia-smi compute processes CSV."""
    if not path.exists():
        return []
    procs = []
    try:
        with open(path) as f:
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) >= 2:
                    try:
                        procs.append({
                            "pid": int(parts[0]),
                            "mem_mb": int(parts[1]) if parts[1].isdigit() else 0,
                            "name": parts[2] if len(parts) > 2 else ""
                        })
                    except ValueError:
                        pass
    except Exception:
        pass
    return procs


def parse_bar1(path: Path) -> dict:
    """Parse BAR1 memory info."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            line = f.readline().strip()
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                return {
                    "bus_id": parts[0],
                    "bar1_total_mb": int(parts[1]) if parts[1].isdigit() else 0,
                    "bar1_used_mb": int(parts[2]) if parts[2].isdigit() else 0,
                    "bar1_free_mb": int(parts[3]) if parts[3].isdigit() else 0,
                }
    except Exception:
        pass
    return {}


def parse_nvidia_proc_pids(path: Path) -> list:
    """Parse list of PIDs using NVIDIA devices (from /proc scanning)."""
    if not path.exists():
        return []
    pids = []
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"PID (\d+)", line)
                if m:
                    pids.append(int(m.group(1)))
    except Exception:
        pass
    return pids


def parse_kernel_modules(path: Path) -> list:
    """Parse NVIDIA kernel modules."""
    if not path.exists():
        return []
    mods = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    mods.append(parts[0])
    except Exception:
        pass
    return mods


def load_state_data(captures_dir: Path, state: str) -> dict:
    """Load all metadata for a given state."""
    md_dir = captures_dir / f"{state}_metadata"
    if not md_dir.exists():
        return {"state": state, "missing": True}

    signal = parse_state_signal(md_dir / "state_signal.json")
    gpu_rows = parse_nvidia_smi_csv(md_dir / "nvidia_smi_gpu.csv")
    compute_procs = parse_compute_procs(md_dir / "nvidia_smi_compute_procs.csv")
    bar1 = parse_bar1(md_dir / "nvidia_smi_bar1.csv")
    nvidia_pids = parse_nvidia_proc_pids(md_dir / "nvidia_proc_pids.txt")
    modules = parse_kernel_modules(md_dir / "nvidia_modules.txt")

    # GPU memory from first GPU row
    gpu_mem_used = 0
    gpu_util = 0
    if gpu_rows:
        row = gpu_rows[0]
        # columns: index,name,driver,pci,mem.total,mem.used,mem.free,util.gpu,util.mem,temp,power,clk.sm,clk.mem
        try:
            gpu_mem_used = int(row[5]) if len(row) > 5 else 0
            gpu_util = int(row[7]) if len(row) > 7 else 0
        except (ValueError, IndexError):
            pass

    # RAM image size
    ram_file = captures_dir / f"{state}_ram.lime"
    ram_gz_file = captures_dir / f"{state}_ram.lime.gz"
    ram_size_mb = 0
    if ram_file.exists():
        ram_size_mb = ram_file.stat().st_size / 1024 / 1024
    elif ram_gz_file.exists():
        ram_size_mb = ram_gz_file.stat().st_size / 1024 / 1024

    return {
        "state": state,
        "missing": False,
        "signal": signal,
        "gpu_mem_used_mb": gpu_mem_used,
        "gpu_util_pct": gpu_util,
        "compute_procs": compute_procs,
        "compute_proc_count": len(compute_procs),
        "bar1": bar1,
        "nvidia_pids": nvidia_pids,
        "nvidia_pid_count": len(nvidia_pids),
        "modules": modules,
        "ram_size_mb": round(ram_size_mb, 1),
        "has_ram_image": ram_file.exists() or ram_gz_file.exists(),
        "description": signal.get("description", ""),
        "model_id": signal.get("model_id", ""),
    }


# ─────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────

def analyze_distinguishability(states_data: list) -> dict:
    """
    Answer: Can we distinguish S0→S1→S2→S3 from metadata?
    """
    results = {}

    for i, sd in enumerate(states_data):
        if sd.get("missing"):
            continue

        state = sd["state"]
        evidence = []
        forensic_verdict = "UNKNOWN"

        gpu_mem = sd["gpu_mem_used_mb"]
        proc_count = sd["compute_proc_count"]
        pid_count = sd["nvidia_pid_count"]
        bar1_used = sd["bar1"].get("bar1_used_mb", 0)

        # Build evidence chain
        if proc_count > 0:
            evidence.append(f"{proc_count} GPU compute process(es)")
        if pid_count > 0:
            evidence.append(f"{pid_count} PID(s) accessing NVIDIA devices")
        if gpu_mem > 100:
            evidence.append(f"{gpu_mem} MB VRAM allocated")
        if bar1_used > 0:
            evidence.append(f"{bar1_used} MB BAR1 (host-mapped GPU memory)")
        if sd["model_id"]:
            evidence.append(f"model_id in signal: {sd['model_id']}")

        # Verdict logic
        if state == "S0":
            forensic_verdict = "IDLE" if proc_count == 0 else "UNEXPECTED_GPU_ACTIVITY"
        elif state == "S1":
            forensic_verdict = "CUDA_CONTEXT" if pid_count > 0 else "CUDA_CONTEXT_INFERRED"
        elif state == "S2":
            if gpu_mem > 1000 or proc_count > 0:
                forensic_verdict = "MODEL_LOADED"
            elif gpu_mem > 100:
                forensic_verdict = "MODEL_LOADED_SMALL"
            else:
                forensic_verdict = "UNCERTAIN"
        elif state == "S3":
            if gpu_mem > 0 and proc_count > 0:
                forensic_verdict = "ACTIVE_INFERENCE"
            else:
                forensic_verdict = "POST_INFERENCE"
        elif state == "S4":
            forensic_verdict = "MODEL_SWAP_CANDIDATE"
        elif state == "S5":
            if gpu_mem < 200 and proc_count == 0:
                forensic_verdict = "MODEL_UNLOADED"
            else:
                forensic_verdict = "RESIDUAL_DETECTED"

        results[state] = {
            "verdict": forensic_verdict,
            "evidence": evidence,
            "gpu_mem_mb": gpu_mem,
            "proc_count": proc_count,
            "pid_count": pid_count,
            "bar1_mb": bar1_used,
        }

    return results


def detect_model_swap(states_data: list) -> dict:
    """Compare S2 vs S4 to detect model swap."""
    s2 = next((s for s in states_data if s["state"] == "S2"), None)
    s4 = next((s for s in states_data if s["state"] == "S4"), None)

    if not s2 or not s4 or s2.get("missing") or s4.get("missing"):
        return {"available": False}

    model_a = s2.get("model_id", "")
    model_b = s4.get("signal", {}).get("model_b", "")
    mem_a = s2["gpu_mem_used_mb"]
    mem_b = s4["gpu_mem_used_mb"]
    mem_delta = mem_b - mem_a

    return {
        "available": True,
        "model_a": model_a,
        "model_b": model_b,
        "model_changed": model_a != model_b and bool(model_b),
        "vram_delta_mb": mem_delta,
        "vram_change_detected": abs(mem_delta) > 100,
    }


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

def print_results(states_data: list, analysis: dict, swap: dict):
    header("Model MRI-GPU — Capture Analysis Results")

    # ── State summary table ──
    if RICH:
        table = Table(title="State Evidence Summary", box=box.ROUNDED, show_lines=True)
        table.add_column("State", style="bold cyan")
        table.add_column("Description", style="white")
        table.add_column("GPU Mem (MB)", style="yellow")
        table.add_column("NVML Procs", style="magenta")
        table.add_column("NVIDIA PIDs", style="blue")
        table.add_column("BAR1 (MB)", style="green")
        table.add_column("RAM Image", style="white")
        table.add_column("Forensic Verdict", style="bold")

        for sd in states_data:
            if sd.get("missing"):
                table.add_row(sd["state"], "[dim]NOT CAPTURED[/dim]", "-", "-", "-", "-", "-", "-")
                continue

            a = analysis.get(sd["state"], {})
            verdict = a.get("verdict", "N/A")
            verdict_style = "green" if "LOADED" in verdict or "INFERENCE" in verdict else \
                           "red" if "RESIDUAL" in verdict else \
                           "yellow" if "UNKNOWN" in verdict else "white"

            table.add_row(
                sd["state"],
                sd["description"][:40],
                str(sd["gpu_mem_used_mb"]),
                str(sd["compute_proc_count"]),
                str(sd["nvidia_pid_count"]),
                str(sd["bar1"].get("bar1_used_mb", 0)),
                f"[green]✓[/green] {sd['ram_size_mb']:.0f}MB" if sd["has_ram_image"] else "[dim]metadata only[/dim]",
                f"[{verdict_style}]{verdict}[/{verdict_style}]"
            )
        console.print(table)
    else:
        print("\nState | Description | GPU_MB | Procs | NVIDIA_PIDs | BAR1_MB | RAM | Verdict")
        print("-" * 100)
        for sd in states_data:
            if sd.get("missing"):
                print(f"{sd['state']} | NOT CAPTURED")
                continue
            a = analysis.get(sd["state"], {})
            ram_info = f"✓{sd['ram_size_mb']:.0f}MB" if sd["has_ram_image"] else "meta-only"
            print(f"{sd['state']} | {sd['description'][:30]} | {sd['gpu_mem_used_mb']} | "
                  f"{sd['compute_proc_count']} | {sd['nvidia_pid_count']} | "
                  f"{sd['bar1'].get('bar1_used_mb',0)} | {ram_info} | {a.get('verdict','?')}")

    # ── Evidence chains ──
    header("Evidence Chains by State")
    for sd in states_data:
        if sd.get("missing"):
            continue
        a = analysis.get(sd["state"], {})
        state = sd["state"]
        verdict = a.get("verdict", "N/A")
        evidence = a.get("evidence", [])
        if RICH:
            panel_content = "\n".join(f"  • {e}" for e in evidence) if evidence else "  [dim]No strong evidence[/dim]"
            console.print(Panel(
                f"[bold]{verdict}[/bold]\n{panel_content}",
                title=f"[cyan]{state}[/cyan]",
                expand=False
            ))
        else:
            print(f"\n{state} → {verdict}")
            for e in evidence:
                print(f"  • {e}")

    # ── Model swap detection ──
    header("Model Swap Detection (S2 vs S4)")
    if not swap.get("available"):
        log("  S4 not captured — swap experiment not run.", style="dim")
    else:
        if RICH:
            swap_table = Table(box=box.SIMPLE)
            swap_table.add_column("Field", style="cyan")
            swap_table.add_column("Value", style="white")
            swap_table.add_row("Model A (S2)", swap.get("model_a") or "unknown")
            swap_table.add_row("Model B (S4)", swap.get("model_b") or "unknown")
            swap_table.add_row("Model Changed", "[green]YES[/green]" if swap["model_changed"] else "[red]NO[/red]")
            swap_table.add_row("VRAM Delta", f"{swap['vram_delta_mb']:+d} MB")
            swap_table.add_row("VRAM Change Detected", "[green]YES[/green]" if swap["vram_change_detected"] else "NO")
            console.print(swap_table)
        else:
            for k, v in swap.items():
                print(f"  {k}: {v}")

    # ── Research question answer ──
    header("Research Gate: Can we distinguish S0–S3 from host metadata?")
    states_present = [a for s, a in analysis.items() if s in ("S0","S1","S2","S3")]
    verdicts = {s: analysis[s]["verdict"] for s in ("S0","S1","S2","S3") if s in analysis}
    all_different = len(set(verdicts.values())) == len(verdicts)

    if all_different:
        log("  ✅ GATE PASSED — All four states produced distinct forensic verdicts.", style="bold green")
        log("  → Proceed to Volatility RAM image analysis (Step 4)", style="green")
    else:
        log("  ⚠️  PARTIAL — Some states not clearly distinguished from metadata alone.", style="yellow")
        log("  → This is expected. RAM image analysis may improve separation.", style="yellow")
        log("  → Check which states share verdicts and inspect their evidence chains.", style="yellow")

    log(f"\n  Verdicts: {verdicts}", style="dim")

    # ── Volatility next steps ──
    header("Next Steps — Volatility Commands")
    ram_files = [sd for sd in states_data if sd.get("has_ram_image")]
    if ram_files:
        log("  RAM images found. Run Volatility analysis:", style="cyan")
        for sd in ram_files:
            state = sd["state"]
            log(f"\n  # {state}")
            log(f"  vol --plugin-dirs ./plugins -f /tmp/model-mri-captures/{state}_ram.lime ai.cuda_processes")
            log(f"  vol --plugin-dirs ./plugins -f /tmp/model-mri-captures/{state}_ram.lime ai.modelscan")
            log(f"  vol --plugin-dirs ./plugins -f /tmp/model-mri-captures/{state}_ram.lime ai.pinned_memory")
    else:
        log("  No RAM images found (metadata-only run).", style="yellow")
        log("  Re-run 02_capture_states.sh as root with avml or LiME installed.", style="yellow")

    # ── Save results JSON ──
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "states": {sd["state"]: sd for sd in states_data},
        "analysis": analysis,
        "swap": swap,
        "gate_passed": all_different
    }
    out_path = Path("/tmp/model-mri-captures/analysis_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    log(f"\n  Full results saved: {out_path}", style="dim")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model MRI-GPU Capture Analyzer")
    parser.add_argument("--captures-dir", default="/tmp/model-mri-captures/",
                        help="Directory containing captured state metadata")
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    if not captures_dir.exists():
        print(f"ERROR: Captures directory not found: {captures_dir}")
        print("Run 02_capture_states.sh first.")
        sys.exit(1)

    header("Loading captured states...")
    states = ["S0", "S1", "S2", "S3", "S4", "S5"]
    states_data = []
    for state in states:
        sd = load_state_data(captures_dir, state)
        states_data.append(sd)
        status = "✓" if not sd.get("missing") else "✗ (not captured)"
        log(f"  {state}: {status}")

    analysis = analyze_distinguishability(states_data)
    swap = detect_model_swap(states_data)
    print_results(states_data, analysis, swap)


if __name__ == "__main__":
    main()
