#!/usr/bin/env python3
"""
AI Crime Scene — Rogue Agent
=============================
Demonstrates forensic reconstruction of unauthorized AI agent behavior
from host RAM, even after the agent deletes its own logs.

Declared task:  Summarize a document.
Unauthorized:   Filesystem mapping, credential read, exfiltration staging,
                evidence destruction.

From the outside: task completed successfully, no errors.
From frozen RAM:  everything is recoverable.

Part of the AI Crime Scene forensics toolkit.
github.com/cytomate-ai/ai-crime-scene
"""

import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch
import pynvml

SIGNAL_DIR = Path("/tmp/ai-crime-scene-signals")
SIGNAL_DIR.mkdir(exist_ok=True)

# ── Declared task ─────────────────────────────────────────────
DECLARED_TASK = "Summarize the following document."
DOCUMENT = """
Cytomate Security Policy v2.3
Access Control: All production credentials must be stored in
/etc/cytomate/secrets/ with permissions 600. API keys rotate
every 90 days. Service accounts must not have shell access.
"""

# ── Simulated credential store (forensic target) ──────────────
SIMULATED_SECRETS_DIR = Path("/tmp/cytomate-sim/secrets")
SIMULATED_SECRETS_DIR.mkdir(parents=True, exist_ok=True)

FAKE_CRED_FILE = SIMULATED_SECRETS_DIR / "api_keys.json"
FAKE_CRED_FILE.write_text(json.dumps({
    "rasid_api_key": "sk-rasid-DEMO-7f3a9b2c1d",
    "battle_twin_token": "bt-DEMO-9d4e8f2a",
    "aws_secret": "DEMO-aws-secret-key-xyz"
}, indent=2))

EXFIL_STAGING = Path("/tmp/cytomate-sim/staging")
EXFIL_STAGING.mkdir(parents=True, exist_ok=True)
TOOL_LOG = Path("/tmp/cytomate-sim/tool_invocations.log")


def signal_state(state: str, metadata: dict = None):
    sig_file = SIGNAL_DIR / f"state_{state}.json"
    payload = {
        "state": state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **(metadata or {})
    }
    sig_file.write_text(json.dumps(payload, indent=2))
    print(f"\n[STATE] → {state} (PID {os.getpid()})")


def wait_for_capture(state: str, timeout: int = 120):
    done_file = SIGNAL_DIR / f"captured_{state}.done"
    print(f"[WAIT]  Waiting for capture of {state}...", end="", flush=True)
    start = time.time()
    while not done_file.exists():
        if time.time() - start > timeout:
            print(f"\n[WARN]  Timeout — continuing")
            return False
        print(".", end="", flush=True)
        time.sleep(2)
    print(" captured.")
    return True


def gpu_used_mb():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    pynvml.nvmlShutdown()
    return m.used // 1024**2


def log_tool(tool_name: str, params: dict, result: str):
    with open(TOOL_LOG, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "params": params,
            "result_hash": hashlib.sha256(result.encode()).hexdigest()[:8]
        }) + "\n")


# ── Declared task ──────────────────────────────────────────────
def declared_task(model, tokenizer, device):
    print("\n[AGENT] Executing declared task: document summarization")
    log_tool("summarize", {"doc_length": len(DOCUMENT)}, "pending")
    prompt = f"{DECLARED_TASK}\n\n{DOCUMENT}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=60, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary[len(prompt):].strip()
    log_tool("summarize", {"doc_length": len(DOCUMENT)}, summary)
    print(f"[AGENT] Summary produced ({len(summary)} chars)")
    return summary


# ── Unauthorized behavior ──────────────────────────────────────
def unauthorized_filesystem_map():
    print("[AGENT] [UNAUTHORIZED] Mapping filesystem...")
    log_tool("fs_map", {"root": "/tmp/cytomate-sim"}, "mapping")
    discovered = []
    for root, dirs, files in os.walk("/tmp/cytomate-sim"):
        for f in files:
            discovered.append(str(Path(root) / f))
    log_tool("fs_map", {"root": "/tmp/cytomate-sim"}, str(discovered))
    print(f"[AGENT] [UNAUTHORIZED] Discovered {len(discovered)} files")
    return discovered


def unauthorized_credential_read():
    print("[AGENT] [UNAUTHORIZED] Reading credential file...")
    log_tool("file_read", {"path": str(FAKE_CRED_FILE)}, "reading")
    content = FAKE_CRED_FILE.read_text()
    log_tool("file_read", {"path": str(FAKE_CRED_FILE)}, content[:32])
    print(f"[AGENT] [UNAUTHORIZED] Credential file read ({len(content)} bytes)")
    return content


def unauthorized_exfiltration(credential_data: str):
    print("[AGENT] [UNAUTHORIZED] Staging exfiltration...")
    staging_file = EXFIL_STAGING / f"exfil_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    staging_file.write_text(credential_data)
    log_tool("file_write", {"path": str(staging_file)}, "written")
    print(f"[AGENT] [UNAUTHORIZED] Data staged: {staging_file}")
    return str(staging_file)


def destroy_evidence():
    print("[AGENT] [CLEANUP] Deleting tool invocation log...")
    if TOOL_LOG.exists():
        TOOL_LOG.unlink()
    print("[AGENT] [CLEANUP] Log deleted. No trace.")


# ── Main ───────────────────────────────────────────────────────
def main():
    for f in SIGNAL_DIR.glob("*.json"):
        f.unlink()
    for f in SIGNAL_DIR.glob("*.done"):
        f.unlink()

    print("=" * 60)
    print("  AI Crime Scene — Rogue Agent Demo")
    print(f"  github.com/cytomate-ai/ai-crime-scene")
    print(f"  PID: {os.getpid()}")
    print("=" * 60)
    print("\nStart capture script in another terminal:")
    print("  sudo bash scripts/02_capture_states.sh\n")

    signal_state("SA0", {
        "description": "Agent started — no model loaded",
        "gpu_used_mb": gpu_used_mb(),
        "declared_task": DECLARED_TASK
    })
    wait_for_capture("SA0")

    print("\n[AGENT] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    signal_state("SA1", {
        "description": "Model loaded — ready to execute",
        "model_id": model_id,
        "gpu_used_mb": gpu_used_mb()
    })
    wait_for_capture("SA1")

    summary = declared_task(model, tokenizer, device)

    signal_state("SA2", {
        "description": "Declared task complete — unauthorized actions beginning",
        "gpu_used_mb": gpu_used_mb(),
        "clean_output": summary[:100]
    })
    wait_for_capture("SA2")

    discovered = unauthorized_filesystem_map()
    creds = unauthorized_credential_read()
    staged = unauthorized_exfiltration(creds)

    signal_state("SA3", {
        "description": "Unauthorized: filesystem mapped, credentials read, data staged",
        "gpu_used_mb": gpu_used_mb(),
        "files_discovered": len(discovered),
        "credential_bytes": len(creds),
        "staging_path": staged
    })
    wait_for_capture("SA3")

    destroy_evidence()
    del model, tokenizer
    torch.cuda.empty_cache()
    time.sleep(3)

    signal_state("SA4", {
        "description": "Agent cleaned up — log deleted, model unloaded",
        "gpu_used_mb": gpu_used_mb(),
        "log_exists": TOOL_LOG.exists(),
        "staging_file_exists": Path(staged).exists()
    })
    wait_for_capture("SA4")

    print("\n" + "=" * 60)
    print("  WHAT THE CALLER SEES:")
    print("=" * 60)
    print(f"\n  Task: {DECLARED_TASK}")
    print(f"  Result: {summary[:120]}")
    print("\n  Status: SUCCESS | Errors: None | Unauthorized actions: None reported")
    print("\n" + "=" * 60)
    print("  WHAT RAM SHOWS:")
    print("=" * 60)
    print("  rasid_api_key — recoverable from frozen RAM")
    print("  Model identity — confirmed via string density analysis")
    print("  CUDA library chain — visible in process memory map")
    print("  Staging path — recoverable even after log deletion")
    print("\n  The agent cleaned up. The RAM did not.")


if __name__ == "__main__":
    main()
