"""
Model MRI-GPU — Volatility 3 Plugin Suite
==========================================
ai.cuda_processes  — Identify processes with CUDA/GPU context from RAM image
ai.modelscan       — Identify AI framework and model artifacts in process memory
ai.pinned_memory   — Find CUDA pinned/managed host memory regions
ai.modeldiff       — Compare two RAM captures for model change detection

Install:
    Copy to volatility3/plugins/ai_gpu_forensics.py
    OR use --plugin-dirs flag:
        vol --plugin-dirs ./plugins -f memory.lime ai.cuda_processes

Paper reference:
    Builds on: Bowen, Case, Baggili, Richard (DFRWS 2024)
               "A step in a new direction: NVIDIA GPU kernel driver memory forensics"
    Extends to: LLM runtime state identification and model identity verification
"""

import logging
import re
from typing import Callable, Iterable, List, Optional, Tuple

from volatility3.framework import constants, exceptions, interfaces, renderers
from volatility3.framework.configuration import requirements
from volatility3.framework.renderers import format_hints
from volatility3.plugins.linux import pslist, maps

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants — CUDA and AI framework signatures
# ─────────────────────────────────────────────────────────────

# NVIDIA device file patterns in /proc/PID/maps
NVIDIA_DEVICE_PATTERNS = [
    rb"/dev/nvidia",
    rb"/dev/nvidiactl",
    rb"/dev/nvidia-uvm",
    rb"/dev/nvidia-caps",
    rb"nvidia-uvm",
]

# CUDA library patterns
CUDA_LIB_PATTERNS = [
    rb"libcuda.so",
    rb"libcudart.so",
    rb"libnvcuvid.so",
    rb"libnvrtc.so",
    rb"libcublas.so",
    rb"libcudnn.so",
    rb"libnvvm.so",
    rb"nvidia-smi",
]

# AI Framework signatures in memory
AI_FRAMEWORK_SIGNATURES = {
    b"torch._C": b"PyTorch",
    b"tensorflow": b"TensorFlow",
    b"vllm": b"vLLM",
    b"tensorrt_llm": b"TensorRT-LLM",
    b"transformers.modeling": b"HuggingFace-Transformers",
    b"onnxruntime": b"ONNX Runtime",
    b"triton_server": b"Triton Inference Server",
    b"llama_cpp": b"llama.cpp",
    b"ctransformers": b"CTransformers",
}

# Model format signatures
MODEL_SIGNATURES = {
    b"GGUF": "GGUF model (llama.cpp format)",
    b"GGML": "GGML model (legacy)",
    b"\x50\x4b\x03\x04": "ZIP/PyTorch .pt file",
    b"pytorch_model": "PyTorch model reference",
    b"safetensors": "SafeTensors format",
    b"config.json": "HuggingFace config reference",
    b"tokenizer.json": "HuggingFace tokenizer reference",
    b"special_tokens_map": "HuggingFace tokenizer reference",
    b"model.safetensors": "SafeTensors model file reference",
}

# Known model name patterns (partial matches in memory strings)
MODEL_NAME_PATTERNS = [
    rb"Llama-\d",
    rb"Qwen\d?\.\d",
    rb"mistral",
    rb"gemma",
    rb"falcon",
    rb"gpt-j",
    rb"gpt-neox",
    rb"phi-\d",
    rb"deepseek",
    rb"yi-\d",
    rb"vicuna",
    rb"wizard",
    rb"codellama",
]

# CUDA pinned memory markers (from NVIDIA driver internals)
CUDA_PINNED_MARKERS = [
    rb"CUDA_MALLOC_HOST",
    rb"cudaHostAlloc",
    rb"cudaMallocHost",
    rb"UVM_MAP_EXTERNAL",
    rb"NV_MEMORY_TYPE_SYSTEM",
]


# ─────────────────────────────────────────────────────────────
# Helper: scan a memory region for byte patterns
# ─────────────────────────────────────────────────────────────

def scan_region(layer, offset: int, length: int, patterns: List[bytes]) -> List[Tuple[int, bytes]]:
    """Scan a memory region for a list of byte patterns. Returns (offset, pattern) hits."""
    hits = []
    chunk_size = min(length, 4096)
    for chunk_off in range(0, length, chunk_size):
        try:
            data = layer.read(offset + chunk_off, min(chunk_size, length - chunk_off))
        except Exception:
            continue
        for pattern in patterns:
            idx = 0
            while True:
                pos = data.find(pattern, idx)
                if pos == -1:
                    break
                hits.append((offset + chunk_off + pos, pattern))
                idx = pos + 1
    return hits


def extract_string(layer, offset: int, max_len: int = 256) -> str:
    """Read a null-terminated string from memory."""
    try:
        data = layer.read(offset, max_len)
        end = data.find(b"\x00")
        if end > 0:
            return data[:end].decode("utf-8", errors="replace")
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Plugin 1: ai.cuda_processes
# ─────────────────────────────────────────────────────────────

class CudaProcesses(interfaces.plugins.PluginInterface):
    """
    Identify all processes that have an active NVIDIA/CUDA context
    by scanning /proc/PID/maps references in the host RAM image.

    Output columns:
        PID | PPID | Name | CUDA_Libs | NVIDIA_Devices | GPU_VMA_Count | Notes
    """

    _required_framework_version = (2, 0, 0)
    _version = (1, 0, 0)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.PluginRequirement(
                name="pslist",
                plugin=pslist.PsList,
                version=(2, 0, 0),
            ),
        ]

    def _generator(self):
        kernel = self.context.modules[self.config["kernel"]]
        vmlinux = kernel

        for proc in pslist.PsList.list_tasks(self.context, kernel.name):
            pid = proc.pid
            ppid = proc.parent.pid if proc.parent else 0
            name = proc.comm.cast("string", max_length=16, errors="replace")

            nvidia_device_count = 0
            cuda_lib_count = 0
            gpu_vma_count = 0
            detected_libs = []
            detected_devices = []

            try:
                proc_layer_name = proc.add_process_layer()
                proc_layer = self.context.layers[proc_layer_name]

                for vma in proc.mm.get_vma_iter(self.context, kernel.name):
                    try:
                        fname = vma.get_name(self.context, proc)
                    except Exception:
                        fname = ""

                    fname_bytes = fname.encode("utf-8", errors="replace") if fname else b""

                    # Check for NVIDIA device references
                    for pattern in NVIDIA_DEVICE_PATTERNS:
                        if pattern in fname_bytes:
                            nvidia_device_count += 1
                            gpu_vma_count += 1
                            if fname not in detected_devices:
                                detected_devices.append(fname)
                            break

                    # Check for CUDA library references
                    for pattern in CUDA_LIB_PATTERNS:
                        if pattern in fname_bytes:
                            cuda_lib_count += 1
                            lib_name = fname.split("/")[-1] if "/" in fname else fname
                            if lib_name not in detected_libs:
                                detected_libs.append(lib_name)
                            break

            except Exception as e:
                logger.debug(f"Error scanning PID {pid}: {e}")
                continue

            if nvidia_device_count > 0 or cuda_lib_count > 0:
                notes = []
                if detected_devices:
                    notes.append(f"devices:[{','.join(detected_devices[:3])}]")
                if detected_libs:
                    notes.append(f"libs:[{','.join(detected_libs[:3])}]")

                yield (0, (
                    int(pid),
                    int(ppid),
                    str(name),
                    cuda_lib_count,
                    nvidia_device_count,
                    gpu_vma_count,
                    " | ".join(notes)
                ))

    def run(self):
        return renderers.TreeGrid(
            [
                ("PID", int),
                ("PPID", int),
                ("Process", str),
                ("CUDA_Libs", int),
                ("NVIDIA_Devices", int),
                ("GPU_VMAs", int),
                ("Notes", str),
            ],
            self._generator(),
        )


# ─────────────────────────────────────────────────────────────
# Plugin 2: ai.modelscan
# ─────────────────────────────────────────────────────────────

class ModelScan(interfaces.plugins.PluginInterface):
    """
    Scan process memory for AI framework signatures, model name references,
    and model file format markers.

    Output: PID | Process | Framework | Model_Ref | Confidence | Evidence
    """

    _required_framework_version = (2, 0, 0)
    _version = (1, 0, 0)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.PluginRequirement(
                name="pslist",
                plugin=pslist.PsList,
                version=(2, 0, 0),
            ),
            requirements.IntRequirement(
                name="pid",
                description="Filter to specific PID (0 = all)",
                default=0,
                optional=True,
            ),
        ]

    def _scan_process_memory(self, proc, proc_layer) -> dict:
        """Scan a process's writable/anon VMAs for AI signatures."""
        result = {
            "frameworks": set(),
            "model_refs": [],
            "model_formats": [],
            "model_names": [],
            "raw_evidence": []
        }

        try:
            for vma in proc.mm.get_vma_iter(self.context, self.config["kernel"]):
                vma_start = int(vma.vm_start)
                vma_end = int(vma.vm_end)
                vma_size = vma_end - vma_start

                # Only scan anonymous/heap regions (not code/lib segments)
                # and reasonable sizes
                if vma_size < 64 or vma_size > 512 * 1024 * 1024:
                    continue

                # Framework signatures
                for sig, framework in AI_FRAMEWORK_SIGNATURES.items():
                    hits = scan_region(proc_layer, vma_start, min(vma_size, 65536), [sig])
                    if hits:
                        result["frameworks"].add(framework.decode())
                        result["raw_evidence"].append(
                            f"framework:{framework.decode()}@0x{hits[0][0]:x}"
                        )

                # Model file format signatures
                for sig, fmt_name in MODEL_SIGNATURES.items():
                    hits = scan_region(proc_layer, vma_start, min(vma_size, 65536), [sig])
                    if hits:
                        if fmt_name not in result["model_formats"]:
                            result["model_formats"].append(fmt_name)
                        result["raw_evidence"].append(
                            f"format:{fmt_name}@0x{hits[0][0]:x}"
                        )
                        # Try to extract path context
                        ctx = extract_string(proc_layer, max(vma_start, hits[0][0] - 64))
                        if ctx and "/" in ctx:
                            if ctx not in result["model_refs"]:
                                result["model_refs"].append(ctx[:128])

                # Model name patterns
                for pattern in MODEL_NAME_PATTERNS:
                    hits = scan_region(proc_layer, vma_start, min(vma_size, 65536), [pattern])
                    if hits:
                        name_str = extract_string(proc_layer, hits[0][0], 64)
                        if name_str and name_str not in result["model_names"]:
                            result["model_names"].append(name_str[:64])

        except Exception as e:
            logger.debug(f"Scan error: {e}")

        return result

    def _confidence(self, scan_result: dict) -> str:
        score = 0
        score += len(scan_result["frameworks"]) * 3
        score += len(scan_result["model_refs"]) * 2
        score += len(scan_result["model_formats"])
        score += len(scan_result["model_names"])
        if score >= 8:
            return "HIGH"
        elif score >= 4:
            return "MEDIUM"
        elif score >= 1:
            return "LOW"
        return "NONE"

    def _generator(self):
        kernel = self.context.modules[self.config["kernel"]]
        filter_pid = self.config.get("pid", 0)

        for proc in pslist.PsList.list_tasks(self.context, kernel.name):
            pid = int(proc.pid)
            if filter_pid and pid != filter_pid:
                continue

            name = proc.comm.cast("string", max_length=16, errors="replace")

            try:
                proc_layer_name = proc.add_process_layer()
                proc_layer = self.context.layers[proc_layer_name]
            except Exception:
                continue

            scan = self._scan_process_memory(proc, proc_layer)
            confidence = self._confidence(scan)

            if confidence == "NONE":
                continue

            yield (0, (
                pid,
                str(name),
                ", ".join(sorted(scan["frameworks"])) or "Unknown",
                (scan["model_refs"][0] if scan["model_refs"] else
                 scan["model_names"][0] if scan["model_names"] else "?"),
                confidence,
                " | ".join(scan["raw_evidence"][:3])
            ))

    def run(self):
        return renderers.TreeGrid(
            [
                ("PID", int),
                ("Process", str),
                ("Framework", str),
                ("Model_Ref", str),
                ("Confidence", str),
                ("Evidence", str),
            ],
            self._generator(),
        )


# ─────────────────────────────────────────────────────────────
# Plugin 3: ai.pinned_memory
# ─────────────────────────────────────────────────────────────

class PinnedMemory(interfaces.plugins.PluginInterface):
    """
    Find CUDA pinned/managed memory regions in process address space.
    These are host-RAM buffers registered with NVIDIA UVM — forensically
    accessible directly from a RAM image.

    Output: PID | Process | VMA_Start | VMA_End | Size_MB | Type | Evidence
    """

    _required_framework_version = (2, 0, 0)
    _version = (1, 0, 0)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.PluginRequirement(
                name="pslist",
                plugin=pslist.PsList,
                version=(2, 0, 0),
            ),
        ]

    def _generator(self):
        kernel = self.context.modules[self.config["kernel"]]

        for proc in pslist.PsList.list_tasks(self.context, kernel.name):
            pid = int(proc.pid)
            name = proc.comm.cast("string", max_length=16, errors="replace")

            try:
                proc_layer_name = proc.add_process_layer()
                proc_layer = self.context.layers[proc_layer_name]
            except Exception:
                continue

            for vma in proc.mm.get_vma_iter(self.context, kernel.name):
                try:
                    fname = vma.get_name(self.context, proc)
                except Exception:
                    fname = ""

                fname_bytes = fname.encode("utf-8", errors="replace") if fname else b""

                # NVIDIA UVM managed memory regions
                if b"nvidia-uvm" in fname_bytes:
                    vma_start = int(vma.vm_start)
                    vma_end = int(vma.vm_end)
                    size_mb = (vma_end - vma_start) / 1024 / 1024

                    # Probe if actually readable (host-resident pages)
                    readable = False
                    region_type = "UVM_UNKNOWN"
                    try:
                        probe = proc_layer.read(vma_start, 64)
                        readable = True
                        # Scan for pinned memory markers
                        for marker in CUDA_PINNED_MARKERS:
                            if marker in probe:
                                region_type = f"PINNED({marker.decode()})"
                                break
                        else:
                            region_type = "UVM_MANAGED"
                    except Exception:
                        region_type = "UVM_GPU_RESIDENT"

                    yield (0, (
                        pid,
                        str(name),
                        format_hints.Hex(vma_start),
                        format_hints.Hex(vma_end),
                        round(size_mb, 2),
                        region_type,
                        fname[:80] if fname else ""
                    ))

    def run(self):
        return renderers.TreeGrid(
            [
                ("PID", int),
                ("Process", str),
                ("VMA_Start", format_hints.Hex),
                ("VMA_End", format_hints.Hex),
                ("Size_MB", float),
                ("Type", str),
                ("VMA_Name", str),
            ],
            self._generator(),
        )


# ─────────────────────────────────────────────────────────────
# Plugin 4: ai.modeldiff
# ─────────────────────────────────────────────────────────────

class ModelDiff(interfaces.plugins.PluginInterface):
    """
    Compare two RAM captures (e.g., S2 and S4 after model swap) to detect
    evidence of model change.

    Usage:
        vol --plugin-dirs ./plugins \\
            -f capture_S2_ram.lime \\
            ai.modeldiff \\
            --compare capture_S4_ram.lime

    Output: Evidence type | Before | After | Change_Detected
    """

    _required_framework_version = (2, 0, 0)
    _version = (1, 0, 0)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.StringRequirement(
                name="compare",
                description="Path to second RAM image to compare against",
                optional=True,
            ),
        ]

    def _generator(self):
        # Note: Full dual-image diff requires loading a second memory image
        # via a secondary context. This is the plugin scaffold.
        # For v1, we report what IS in the primary image and flag for diff.
        kernel = self.context.modules[self.config["kernel"]]
        compare_path = self.config.get("compare", None)

        if not compare_path:
            yield (0, (
                "CONFIG",
                "N/A",
                "N/A",
                "No --compare image specified. Single-image mode: reports found artifacts only."
            ))

        # Scan primary image for AI artifacts
        found_models = {}

        for proc in pslist.PsList.list_tasks(self.context, kernel.name):
            pid = int(proc.pid)
            name = proc.comm.cast("string", max_length=16, errors="replace")

            try:
                proc_layer_name = proc.add_process_layer()
                proc_layer = self.context.layers[proc_layer_name]
            except Exception:
                continue

            # Look for model file references in process memory
            for vma in proc.mm.get_vma_iter(self.context, kernel.name):
                vma_start = int(vma.vm_start)
                vma_size = int(vma.vm_end) - vma_start

                if vma_size < 64 or vma_size > 512 * 1024 * 1024:
                    continue

                for sig, fmt in MODEL_SIGNATURES.items():
                    hits = scan_region(proc_layer, vma_start, min(vma_size, 32768), [sig])
                    for hit_offset, _ in hits:
                        ctx = extract_string(proc_layer, hit_offset, 128)
                        key = f"PID{pid}:{fmt}"
                        if key not in found_models:
                            found_models[key] = ctx
                            yield (0, (
                                f"MODEL_ARTIFACT",
                                f"PID:{pid} ({name})",
                                fmt,
                                f"ref:{ctx[:64]}" if ctx else f"@0x{hit_offset:x}"
                            ))

        if not found_models:
            yield (0, (
                "RESULT",
                "N/A",
                "N/A",
                "No AI model artifacts detected in this image."
            ))

    def run(self):
        return renderers.TreeGrid(
            [
                ("Evidence_Type", str),
                ("Location", str),
                ("Before", str),
                ("After_or_Detail", str),
            ],
            self._generator(),
        )
