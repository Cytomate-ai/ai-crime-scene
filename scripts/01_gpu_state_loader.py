#!/usr/bin/env python3
"""
Model MRI-GPU — GPU State Loader
Creates S1 (CUDA init), S2 (model loaded), S3 (inference run)
and signals the capture script via lockfiles.

Usage:
    python3 01_gpu_state_loader.py --model <huggingface_model_id>
    python3 01_gpu_state_loader.py --model Qwen/Qwen2.5-0.5B  # smallest test
    python3 01_gpu_state_loader.py --model meta-llama/Llama-3.2-1B

Run ALONGSIDE 02_capture_states.sh (in a separate terminal).
"""

import argparse
import os
import sys
import time
import json
import signal
import hashlib
from pathlib import Path
from datetime import datetime

import torch
import pynvml

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
SIGNAL_DIR = Path("/tmp/model-mri-gpu-signals")
SIGNAL_DIR.mkdir(exist_ok=True)

PROMPT = "The security implications of AI model replacement in production are"


# ------------------------------------------------------------
# Signal helpers (communicate state to capture script)
# ------------------------------------------------------------
def signal_state(state: str, metadata: dict = None):
    """Write a state signal file that the capture script watches."""
    sig_file = SIGNAL_DIR / f"state_{state}.json"
    payload = {
        "state": state,
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid(),
        **(metadata or {})
    }
    sig_file.write_text(json.dumps(payload, indent=2))
    print(f"\n[STATE] → {state} (PID {os.getpid()})")
    if metadata:
        for k, v in metadata.items():
            print(f"         {k}: {v}")


def wait_for_capture(state: str, timeout: int = 120):
    """Wait until the capture script signals it has captured this state."""
    done_file = SIGNAL_DIR / f"captured_{state}.done"
    print(f"[WAIT]  Waiting for capture of {state}...", end="", flush=True)
    start = time.time()
    while not done_file.exists():
        if time.time() - start > timeout:
            print(f"\n[WARN]  Timeout waiting for {state} capture — continuing anyway")
            return False
        print(".", end="", flush=True)
        time.sleep(2)
    print(" captured.")
    return True


def gpu_snapshot() -> dict:
    """Snapshot current GPU memory state via pynvml."""
    pynvml.nvmlInit()
    gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            proc_list = [{"pid": p.pid, "used_mb": p.usedGpuMemory // 1024**2} for p in procs]
        except Exception:
            proc_list = []
        gpus.append({
            "index": i,
            "name": pynvml.nvmlDeviceGetName(h),
            "used_mb": mem.used // 1024**2,
            "free_mb": mem.free // 1024**2,
            "total_mb": mem.total // 1024**2,
            "processes": proc_list
        })
    pynvml.nvmlShutdown()
    return {"gpus": gpus, "timestamp": datetime.utcnow().isoformat()}


def model_fingerprint(model) -> dict:
    """Compute a lightweight fingerprint of model weights (first+last layer)."""
    params = list(model.parameters())
    if not params:
        return {"fingerprint": "no_params"}

    def tensor_hash(t):
        data = t.detach().cpu().float().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    return {
        "first_layer_hash": tensor_hash(params[0]),
        "last_layer_hash": tensor_hash(params[-1]),
        "param_count": sum(p.numel() for p in params),
        "dtype": str(params[0].dtype)
    }


# ------------------------------------------------------------
# Main state machine
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model MRI-GPU State Loader")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model ID to load")
    parser.add_argument("--second-model", default=None,
                        help="Optional second model for swap experiment")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for capture signals (auto-advance)")
    args = parser.parse_args()

    # Clear old signals
    for f in SIGNAL_DIR.glob("*.json"):
        f.unlink()
    for f in SIGNAL_DIR.glob("*.done"):
        f.unlink()

    print("=" * 60)
    print("  Model MRI-GPU — State Loader")
    print(f"  Model: {args.model}")
    print(f"  PID:   {os.getpid()}")
    print("=" * 60)
    print(f"\nSignal dir: {SIGNAL_DIR}")
    print("Run capture script in another terminal:")
    print("  sudo bash 02_capture_states.sh\n")

    # --------------------------------------------------------
    # S0 — Baseline (GPU idle, no Python CUDA, no model)
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("STATE S0 — GPU Idle Baseline")
    print("=" * 40)
    snap = gpu_snapshot()
    signal_state("S0", {
        "description": "GPU idle — no CUDA context, no model",
        "gpu_used_mb": snap["gpus"][0]["used_mb"] if snap["gpus"] else 0
    })
    if not args.no_wait:
        wait_for_capture("S0")

    # --------------------------------------------------------
    # S1 — CUDA initialized, no model loaded
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("STATE S1 — CUDA Context Initialized")
    print("=" * 40)
    print("Initializing CUDA context...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Force CUDA context creation
        dummy = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        del dummy
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  WARNING: No CUDA GPU available. Running in CPU mode.")

    snap = gpu_snapshot()
    signal_state("S1", {
        "description": "CUDA context created, no model",
        "gpu_used_mb": snap["gpus"][0]["used_mb"] if snap["gpus"] else 0,
        "cuda_version": torch.version.cuda,
        "device": str(device)
    })
    if not args.no_wait:
        wait_for_capture("S1")

    # --------------------------------------------------------
    # S2 — Model loaded on GPU
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("STATE S2 — Model Loaded on GPU")
    print("=" * 40)
    print(f"Loading model: {args.model}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        fp = model_fingerprint(model)
        print(f"  Model loaded.")
        print(f"  Params: {fp['param_count']:,}")
        print(f"  Dtype: {fp['dtype']}")
        print(f"  First layer hash: {fp['first_layer_hash']}")
    except ImportError:
        print("  transformers not installed — using synthetic model")
        model = torch.nn.Linear(4096, 4096).to(device).half()
        tokenizer = None
        fp = model_fingerprint(model)

    snap = gpu_snapshot()
    signal_state("S2", {
        "description": "Model loaded on GPU",
        "model_id": args.model,
        "gpu_used_mb": snap["gpus"][0]["used_mb"] if snap["gpus"] else 0,
        "model_fingerprint": fp,
        "gpu_processes": snap["gpus"][0]["processes"] if snap["gpus"] else []
    })
    if not args.no_wait:
        wait_for_capture("S2")

    # --------------------------------------------------------
    # S3 — Inference executed
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("STATE S3 — Inference Executed")
    print("=" * 40)
    print(f"Running inference with prompt: '{PROMPT[:50]}...'")

    output_text = None
    if tokenizer is not None:
        try:
            inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Output (first 100 chars): {output_text[:100]}")
        except Exception as e:
            print(f"  Inference error: {e}")
    else:
        # Synthetic inference
        with torch.no_grad():
            x = torch.randn(1, 4096, device=device).half()
            out = model(x)
            output_text = f"synthetic_output_norm={float(out.norm()):.4f}"

    snap = gpu_snapshot()
    signal_state("S3", {
        "description": "Inference complete, KV cache active",
        "model_id": args.model,
        "prompt": PROMPT,
        "output_preview": (output_text or "")[:200],
        "gpu_used_mb": snap["gpus"][0]["used_mb"] if snap["gpus"] else 0,
        "gpu_processes": snap["gpus"][0]["processes"] if snap["gpus"] else []
    })
    if not args.no_wait:
        wait_for_capture("S3")

    # --------------------------------------------------------
    # S4 (optional) — Model swap
    # --------------------------------------------------------
    if args.second_model:
        print("\n" + "=" * 40)
        print("STATE S4 — Model Swap (A → B)")
        print("=" * 40)
        print(f"Unloading {args.model}...")
        del model
        if tokenizer:
            del tokenizer
        torch.cuda.empty_cache()

        print(f"Loading {args.second_model}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer2 = AutoTokenizer.from_pretrained(args.second_model, trust_remote_code=True)
            model2 = AutoModelForCausalLM.from_pretrained(
                args.second_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            model2.eval()
            fp2 = model_fingerprint(model2)
        except Exception as e:
            print(f"  Second model load error: {e}")
            fp2 = {}
            model2 = None

        snap = gpu_snapshot()
        signal_state("S4", {
            "description": "Model swapped from A to B",
            "model_a": args.model,
            "model_b": args.second_model,
            "gpu_used_mb": snap["gpus"][0]["used_mb"] if snap["gpus"] else 0,
            "model_b_fingerprint": fp2
        })
        if not args.no_wait:
            wait_for_capture("S4")

    # --------------------------------------------------------
    # S5 — Unloaded (residual check)
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("STATE S5 — Model Unloaded (Residual Check)")
    print("=" * 40)
    try:
        del model
    except NameError:
        pass
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(3)

    snap = gpu_snapshot()
    signal_state("S5", {
        "description": "Model unloaded — testing for residual artifacts",
        "gpu_used_mb": snap["gpus"][0]["used_mb"] if snap["gpus"] else 0,
    })
    if not args.no_wait:
        wait_for_capture("S5")

    print("\n[DONE] State machine complete.")
    print(f"Signal files in: {SIGNAL_DIR}")
    print("Now run the Volatility analysis:")
    print("  python3 03_volatility_plugin.py --captures-dir /tmp/model-mri-captures/")


if __name__ == "__main__":
    main()
