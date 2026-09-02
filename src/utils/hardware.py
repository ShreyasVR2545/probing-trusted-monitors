"""Hardware detection and plan selection.

Probes the machine once, writes HARDWARE.md and configs/hardware.yaml, and
returns the plan every other script reads. The single hard rule enforced here:
bf16 is only ever selected when torch.cuda.is_bf16_supported() is true.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class HardwareInfo:
    cuda_available: bool = False
    device_name: str | None = None
    vram_gb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    bf16_supported: bool = False
    torch_version: str | None = None
    cuda_version: str | None = None
    cpu_count: int = 0
    ram_gb: float = 0.0
    free_disk_gb: float = 0.0
    platform: str = ""
    mps_available: bool = False


@dataclass
class Plan:
    """The resolved experiment plan implied by the detected hardware."""

    tier: str
    primary_model: str
    secondary_model: str | None
    quantization: str  # "4bit" | "8bit" | "none"
    dtype: str  # "bfloat16" | "float16" | "float32"
    device: str
    max_context_tokens: int
    layer_stride: int
    run_location: str  # "local" | "kaggle"
    notes: list[str] = field(default_factory=list)


def _ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=60,
            )
            if out.returncode == 0 and out.stdout.strip():
                return round(int(out.stdout.strip()) / 1024**3, 1)
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * size / 1024**3, 1)
    except Exception as exc:  # noqa: BLE001 -- diagnostic only, reported not swallowed
        print(f"[hardware] RAM probe failed: {exc!r}")
    return 0.0


def detect() -> HardwareInfo:
    info = HardwareInfo(
        cpu_count=os.cpu_count() or 0,
        ram_gb=_ram_gb(),
        free_disk_gb=round(shutil.disk_usage(REPO_ROOT).free / 1024**3, 1),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )
    try:
        import torch
    except ImportError as exc:
        print(f"[hardware] torch not importable: {exc!r}")
        return info

    info.torch_version = str(torch.__version__)
    info.cuda_version = str(torch.version.cuda) if torch.version.cuda else None
    info.cuda_available = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    info.mps_available = bool(mps and mps.is_available())
    if info.cuda_available:
        props = torch.cuda.get_device_properties(0)
        info.device_name = props.name
        info.vram_gb = round(props.total_memory / 1024**3, 2)
        info.compute_capability = (props.major, props.minor)
        info.bf16_supported = bool(torch.cuda.is_bf16_supported())
    return info


# Model ids. Qwen2.5-Instruct family throughout: 32k native context, which is
# the binding constraint for agent trajectories.
M7B = "Qwen/Qwen2.5-7B-Instruct"
M3B = "Qwen/Qwen2.5-3B-Instruct"
M1_5B = "Qwen/Qwen2.5-1.5B-Instruct"


def choose_plan(info: HardwareInfo) -> Plan:
    notes: list[str] = []
    if not info.cuda_available:
        notes.append(
            "No CUDA device. Local runs would be prohibitively slow -- Kaggle path selected."
        )
        return Plan(
            tier="cpu-or-mps", primary_model=M3B, secondary_model=M1_5B,
            quantization="none", dtype="float32", device="cpu",
            max_context_tokens=8192, layer_stride=2, run_location="kaggle", notes=notes,
        )

    cc = info.compute_capability or (0, 0)
    modern = cc[0] >= 8
    dtype = "bfloat16" if info.bf16_supported else "float16"
    if not info.bf16_supported:
        notes.append("bf16 unsupported on this device -- fp16 selected.")

    vram = info.vram_gb
    if vram >= 16 and modern:
        tier, primary, secondary, quant, ctx, stride = (
            "cuda-large", M7B, M3B, "none", 32768, 1)
    elif vram >= 10 and modern:
        tier, primary, secondary, quant, ctx, stride = (
            "cuda-medium", M3B, M1_5B, "none", 24576, 1)
    elif vram >= 7.5 and modern:
        tier, primary, secondary, quant, ctx, stride = (
            "cuda-small", M3B, M1_5B, "4bit", 16384, 2)
        notes.append(
            "~8GB tier: 3B in 4-bit is primary, 1.5B secondary. Context capped at "
            "16k so the KV cache plus per-layer activation capture fits alongside weights."
        )
    elif vram >= 10:  # cc 7.x, e.g. T4
        tier, primary, secondary, quant, ctx, stride = (
            "cuda-turing", M3B, M1_5B, "4bit", 16384, 2)
        notes.append("Compute capability 7.x -- fp16 only, no bf16.")
    else:
        notes.append(f"Only {vram}GB VRAM -- below the local floor, Kaggle path selected.")
        return Plan(
            tier="cuda-tiny", primary_model=M3B, secondary_model=M1_5B,
            quantization="4bit", dtype=dtype, device="cuda",
            max_context_tokens=8192, layer_stride=2, run_location="kaggle", notes=notes,
        )

    if cc[0] >= 12:
        notes.append(
            "Blackwell (cc 12.x): requires torch built against CUDA 12.8+ and a "
            "bitsandbytes release carrying sm_120 kernels."
        )
    return Plan(
        tier=tier, primary_model=primary, secondary_model=secondary,
        quantization=quant, dtype=dtype, device="cuda",
        max_context_tokens=ctx, layer_stride=stride, run_location="local", notes=notes,
    )


def assert_dtype_supported(dtype: str) -> None:
    """Guard called at the entry of every script that loads a model."""
    if dtype != "bfloat16":
        return
    import torch

    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            "Config requests bfloat16 but torch.cuda.is_bf16_supported() is False. "
            "Re-run scripts/run_all.py --stage -1 to regenerate configs/hardware.yaml."
        )


def load_plan(path: str | None = None) -> Plan:
    path = path or os.path.join(REPO_ROOT, "configs", "hardware.yaml")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Plan(**raw["plan"])


def _fmt_cc(cc: tuple[int, int] | None) -> str:
    return f"{cc[0]}.{cc[1]}" if cc else "n/a"


def write_outputs(info: HardwareInfo, plan: Plan) -> dict[str, Any]:
    os.makedirs(os.path.join(REPO_ROOT, "configs"), exist_ok=True)
    payload = {"hardware": asdict(info), "plan": asdict(plan)}
    payload["hardware"]["compute_capability"] = list(info.compute_capability or [])
    with open(os.path.join(REPO_ROOT, "configs", "hardware.yaml"), "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)

    notes_md = "\n".join("- " + n for n in plan.notes) or "- none"
    md = f"""# Hardware

Auto-generated by `src/utils/hardware.py`. Do not edit by hand -- re-run
`python scripts/run_all.py --stage -1` to refresh.

## Detected

| Field | Value |
|---|---|
| Platform | {info.platform} |
| CUDA available | {info.cuda_available} |
| Device | {info.device_name or "n/a"} |
| VRAM | {info.vram_gb} GB |
| Compute capability | {_fmt_cc(info.compute_capability)} |
| bf16 supported | {info.bf16_supported} |
| torch | {info.torch_version} (CUDA {info.cuda_version}) |
| CPU cores | {info.cpu_count} |
| RAM | {info.ram_gb} GB |
| Free disk | {info.free_disk_gb} GB |

## Plan

| Field | Value |
|---|---|
| Tier | `{plan.tier}` |
| Run location | **{plan.run_location}** |
| Primary monitor | `{plan.primary_model}` |
| Secondary monitor | `{plan.secondary_model}` |
| Quantization | {plan.quantization} |
| Compute dtype | {plan.dtype} |
| Max context | {plan.max_context_tokens} tokens |
| Layer stride (activation capture) | every {plan.layer_stride} layer(s) |

### Notes

{notes_md}

## Local vs Kaggle

The tier table in the project brief maps VRAM and compute capability onto a
plan. This machine resolves to `{plan.tier}`, so the experiment runs
**{plan.run_location}**. Stages 4-6 (transfer, analysis, figures) are CPU-cheap
and run locally in either case -- only Stages 1-3, which need model forward
passes, are affected by this choice. `notebooks/kaggle_runner.ipynb` and
`KAGGLE.md` exist so the GPU stages can be moved to a Kaggle session either as
a fallback or to run the 7B monitor as a robustness check.
"""
    with open(os.path.join(REPO_ROOT, "HARDWARE.md"), "w") as fh:
        fh.write(md)
    return payload


def main() -> None:
    info = detect()
    plan = choose_plan(info)
    write_outputs(info, plan)
    print(f"Detected: {info.device_name} | {info.vram_gb}GB | cc {_fmt_cc(info.compute_capability)} "
          f"| bf16={info.bf16_supported}")
    print(f"Plan: tier={plan.tier} model={plan.primary_model} quant={plan.quantization} "
          f"dtype={plan.dtype} ctx={plan.max_context_tokens} run={plan.run_location}")
    for n in plan.notes:
        print(f"  note: {n}")
    print("Wrote HARDWARE.md and configs/hardware.yaml")


if __name__ == "__main__":
    main()
