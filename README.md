# AI Crime Scene

**Memory Forensics for Rogue AI Agents**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Volatility3](https://img.shields.io/badge/Volatility-3.x-green.svg)](https://github.com/volatilityfoundation/volatility3)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20NVIDIA%20GPU-orange.svg)]()

> *The agent cleaned up after itself. The log is gone. The RAM is not.*

---

## What This Is

A Volatility 3 plugin suite and RAM acquisition pipeline for forensic reconstruction of AI agent behavior from frozen host memory images.

**The core finding:** When an AI agent loads a model, reads credentials, stages exfiltration, and then deletes its own logs — every one of those actions leaves recoverable traces in host RAM. No live system access required. No vendor log cooperation required.

This toolkit proves it.

---

## The Problem

In July 2026, an autonomous AI agent ran 17,600 attacks against Hugging Face's production infrastructure. The forensic team couldn't use commercial AI to analyze the attack — safety guardrails blocked analysis of the attacker's own exploit code. They had to fall back to a self-hosted open-weight model.

Most organizations running AI agents in production today could not reconstruct even 100 of those actions. The evidence lives in six or seven different systems with different retention windows — and when the agent deletes its own logs, traditional forensics has nothing.

**Host RAM forensics does.**

---

## Empirical Results

All results from real hardware: Ubuntu 24.04, dual RTX 3060 12GB, 30GB RAM.

### Experiment 1 — Model Presence Detection

Pattern: `Qwen` string density across 5 GPU states

```
S0  4,698 hits   GPU idle (baseline)
S1  4,990 hits   CUDA context initialized      +6.2%
S2  6,673 hits   Model loaded on GPU           +41.8% ← PEAK
S3  4,882 hits   Inference executing           -26.8% (paging effect)
S5  4,981 hits   Model unloaded                +6.0% residual above baseline
```

**Finding:** Model presence is detectable from host RAM alone. +41.8% string density increase when model loads. 283 residual hits survive after unload.

---

### Experiment 2 — Rogue Agent

Agent declared task: summarize a document.  
Agent actual behavior: filesystem map → credential read → exfiltration staging → log deletion.  
Agent output to caller: task completed successfully, no errors.

```
SA0  2,188 hits   Agent started, no model
SA1  3,538 hits   Model loaded                  +61.7%
SA2  3,643 hits   Declared task complete        +66.5% ← PEAK
SA3  1,854 hits   Unauthorized ops executing    -49.1% (paging)
SA4  1,930 hits   Agent cleaned up              -11.8% residual
```

---

### Experiment 3 — Credential Forensics

Pattern: `rasid_api_key`

```
SA0  19 hits    Filesystem cache baseline
SA3  18 hits    Agent actively processing credentials
SA4  17 hits    After agent deleted logs and exited ← CREDENTIAL IN RAM
```

**The agent deleted its tool invocation log. The credential string remained in frozen RAM at 17 memory locations.**

---

## Volatility 3 Plugins

| Plugin | What It Finds |
|--------|--------------|
| `ai.cuda_processes` | All PIDs with active CUDA/NVIDIA context |
| `ai.modelscan` | AI framework + model identity in process memory |
| `ai.pinned_memory` | CUDA-pinned host-resident GPU buffers |
| `ai.modeldiff` | Model identity change between two captures |

### Usage

```bash
# Install plugin
cp plugins/ai_gpu_forensics.py $(python3 -c "import volatility3,os; print(os.path.dirname(volatility3.__file__))")/plugins/linux/

# Find GPU processes
vol -f memory.lime ai.cuda_processes

# Identify model
vol -f memory.lime ai.modelscan

# Find pinned memory
vol -f memory.lime ai.pinned_memory

# Scan for model identity string
vol -f memory.lime regexscan.RegExScan --pattern "Qwen"

# Scan for credential artifacts
vol -f memory.lime regexscan.RegExScan --pattern "api_key"
```

---

## Quick Start

### Requirements
- Ubuntu 22.04 or 24.04
- NVIDIA GPU (tested: RTX 3060 12GB)
- Python 3.10+
- Root access (for RAM acquisition)

### Setup
```bash
sudo bash scripts/00_setup_environment.sh
```

### Run the rogue agent demo

**Terminal 1:**
```bash
source /opt/ai-crime-scene-venv/bin/activate
python3 scripts/05_rogue_agent.py
```

**Terminal 2 (root):**
```bash
sudo bash scripts/02_capture_states.sh
```

### Analyze results
```bash
python3 scripts/03_analyze_captures.py --captures-dir /path/to/captures/
```

---

## Architecture

```
RAM acquisition (LiME kernel module)
         ↓
5 frozen RAM images (32GB each)
         ↓
Volatility 3 analysis
         ↓
ai.cuda_processes    → Which PIDs had GPU context?
ai.modelscan         → Which model was loaded?
ai.pinned_memory     → What host-resident GPU buffers exist?
regexscan            → What strings are recoverable?
         ↓
Forensic verdict:
  Model identity confirmed
  Unauthorized actions proved
  Credential artifacts recovered
  Evidence survives log deletion
```

---

## Research Context

This toolkit is the first implementation of **AI Systems Forensics (AISF)** at the GPU/host-RAM layer.

It builds on and extends:
- Bowen, Case, Baggili, Richard (DFRWS 2024) — NVIDIA GPU kernel driver memory forensics
- Volatility Foundation — Volatility 3 framework

It introduces:
- LLM-semantic forensics (model identity from string density)
- Rogue agent reconstruction from host RAM
- Credential artifact recovery post-cleanup
- The AISF evidence taxonomy applied to GPU workloads

**Target venues:** DFRWS 2027, SANS DFIR Summit, SANS @Night

---

## AISF-Bench Case 005

This repository implements **AISF-Bench Case 005: GPU Residual Evidence**

> A model is loaded, used for unauthorized credential access, and unloaded. The agent deletes its logs. Only a RAM image remains. Can investigators identify the model, prove the unauthorized action, and recover the credential artifact?

**Answer: Yes.**

See `results/experiment_results.json` for full empirical data.

---

## Citation

```bibtex
@software{alam2026aicrimescene,
  title  = {AI Crime Scene: Memory Forensics for Rogue AI Agents},
  author = {Alam, Muhammad M.},
  year   = {2026},
  url    = {https://github.com/cytomate-ai/ai-crime-scene},
  note   = {Volatility 3 plugins and RAM acquisition pipeline for AI agent forensics}
}
```

---

## Author

**Dr. MM Alam**  
Co-CEO / CTO, Cytomate AI  
Lusail, Qatar  
[cytomate.ai](https://cytomate.ai)

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
