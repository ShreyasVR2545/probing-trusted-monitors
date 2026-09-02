"""Power and thermal safety for sustained GPU inference on a laptop.

## Why this exists

This machine hard-reset twice under sustained GPU inference (2026-08-24 and
2026-09-02), both times with bugcheck `0x00020001` (HYPERVISOR_ERROR) and
byte-for-byte identical parameters 1-3. No WHEA error, no display-driver TDR,
and nothing at all in the System log for the three minutes before the reset --
an instantaneous, silent death under load. See CRASH_MITIGATION.md for the full
evidence.

The GPU is an RTX 5070 Laptop under NVIDIA Dynamic Boost: a 50 W default limit
against a 115 W maximum, with the budget shifting between CPU and GPU. The
monitor workload oscillates hard between a very heavy prefill (one large matmul
burst over a long context) and a long run of light, memory-bound decode steps,
once per trajectory. That is close to a worst case for supply transients.

Nothing here can prove causation without a kernel dump, and none was retained.
What it does is remove the workload's sharpest load steps, which is the part
this project controls:

* **Chunked prefill** -- the long-context prefill is fed through the KV cache in
  segments instead of one burst, so peak draw steps up gradually. Also lowers
  peak VRAM.
* **Cooldown and thermal guard** -- a pause between trajectories, extended while
  the GPU is above a temperature threshold, so the duty cycle is not pinned at
  100%.
* **CPU headroom cap** -- caps the maximum processor state while a run is in
  flight. Under Dynamic Boost the CPU and GPU share a power budget, so leaving
  CPU headroom lowers the combined transient. Reversible, and restored on exit
  including on Ctrl-C.
* **Heartbeat** -- the trajectory in flight is written to disk before work
  starts, so if the machine does die we know exactly where, and the run resumes
  from there rather than restarting.

Capping the GPU power limit directly (`nvidia-smi -pl`) is the most effective
mitigation available but needs an elevated shell; CRASH_MITIGATION.md gives the
command for the user to run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_COOLDOWN_S = 1.5
DEFAULT_MAX_TEMP_C = 78
DEFAULT_CPU_MAX_PCT = 70
DEFAULT_PREFILL_CHUNK = 2048


def _nvidia_smi(args: list[str], timeout: int = 30) -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[power] nvidia-smi failed: {exc!r}")
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


@dataclass
class GpuTelemetry:
    temperature_c: float | None
    power_w: float | None
    power_limit_w: float | None
    sm_clock_mhz: float | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def gpu_telemetry() -> GpuTelemetry:
    raw = _nvidia_smi([
        "--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return GpuTelemetry(None, None, None, None)
    parts = [p.strip() for p in raw.splitlines()[0].split(",")]

    def num(x: str) -> float | None:
        try:
            return float(x)
        except ValueError:
            return None

    while len(parts) < 4:
        parts.append("")
    return GpuTelemetry(num(parts[0]), num(parts[1]), num(parts[2]), num(parts[3]))


# --------------------------------------------------------------------------
# CPU headroom cap (reversible)
# --------------------------------------------------------------------------

def _active_scheme_guid() -> str | None:
    out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                         text=True, timeout=60)
    if out.returncode != 0:
        return None
    m = re.search(r"([0-9a-fA-F-]{36})", out.stdout)
    return m.group(1) if m else None


def _read_cpu_max_pct(guid: str) -> int | None:
    out = subprocess.run(["powercfg", "/query", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return None
    m = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", out.stdout)
    return int(m.group(1), 16) if m else None


def _set_cpu_max_pct(guid: str, pct: int) -> bool:
    for cmd in (["powercfg", "/setacvalueindex", guid, "SUB_PROCESSOR",
                 "PROCTHROTTLEMAX", str(pct)],
                ["powercfg", "/setactive", guid]):
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            print(f"[power] {' '.join(cmd[:2])} failed: {out.stderr.strip()}")
            return False
    return True


# --------------------------------------------------------------------------
# Heartbeat
# --------------------------------------------------------------------------

class Heartbeat:
    """Records the unit of work in flight, so a hard reset is diagnosable.

    Written before the work starts and flushed to disk. After a crash the file
    names the exact trajectory that was running when the machine died.
    """

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(REPO_ROOT, "results", "heartbeat.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def beat(self, stage: str, item: str, index: int, total: int,
             extra: dict[str, Any] | None = None) -> None:
        payload = {
            "stage": stage, "item": item, "index": index, "total": total,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": gpu_telemetry().to_dict(), **(extra or {}),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    def last(self) -> dict[str, Any] | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path) as fh:
            return json.load(fh)


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

class PowerGuard:
    """Context manager applying the reversible mitigations around a GPU run."""

    def __init__(self, cooldown_s: float = DEFAULT_COOLDOWN_S,
                 max_temp_c: float = DEFAULT_MAX_TEMP_C,
                 cpu_max_pct: int | None = DEFAULT_CPU_MAX_PCT,
                 torch_threads: int = 4, enabled: bool = True, logger: Any = None):
        self.cooldown_s = cooldown_s
        self.max_temp_c = max_temp_c
        self.cpu_max_pct = cpu_max_pct
        self.torch_threads = torch_threads
        self.enabled = enabled
        self.log = logger
        self._guid: str | None = None
        self._prev_cpu_pct: int | None = None
        self.waits = 0
        self.total_wait_s = 0.0

    def _say(self, msg: str) -> None:
        if self.log:
            self.log.info(f"[power] {msg}")
        else:
            print(f"[power] {msg}")

    def __enter__(self) -> "PowerGuard":
        if not self.enabled:
            return self
        try:
            import torch

            torch.set_num_threads(self.torch_threads)
        except ImportError:
            pass
        if self.cpu_max_pct is not None and os.name == "nt":
            self._guid = _active_scheme_guid()
            if self._guid:
                self._prev_cpu_pct = _read_cpu_max_pct(self._guid)
                if _set_cpu_max_pct(self._guid, self.cpu_max_pct):
                    self._say(f"CPU max processor state {self._prev_cpu_pct}% "
                              f"-> {self.cpu_max_pct}% for this run")
                else:
                    self._prev_cpu_pct = None
        t = gpu_telemetry()
        self._say(f"start: GPU {t.temperature_c}C, {t.power_w}W "
                  f"(limit {t.power_limit_w}W), cooldown {self.cooldown_s}s, "
                  f"thermal ceiling {self.max_temp_c}C")
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        if self._guid and self._prev_cpu_pct is not None:
            if _set_cpu_max_pct(self._guid, self._prev_cpu_pct):
                self._say(f"restored CPU max processor state to {self._prev_cpu_pct}%")
            else:
                self._say(f"WARNING: could not restore CPU max processor state; "
                          f"run: powercfg /setacvalueindex {self._guid} SUB_PROCESSOR "
                          f"PROCTHROTTLEMAX {self._prev_cpu_pct} && powercfg /setactive {self._guid}")
        self._say(f"paused {self.waits} times for cooling, "
                  f"{self.total_wait_s:.0f}s total")

    def cooldown(self) -> None:
        """Pause between units of work; wait longer while the GPU is hot."""
        if not self.enabled:
            return
        time.sleep(self.cooldown_s)
        deadline = time.time() + 120
        waited = 0.0
        while time.time() < deadline:
            t = gpu_telemetry()
            if t.temperature_c is None or t.temperature_c < self.max_temp_c:
                break
            if waited == 0.0:
                self._say(f"GPU at {t.temperature_c}C >= {self.max_temp_c}C -- pausing")
                self.waits += 1
            time.sleep(3.0)
            waited += 3.0
        self.total_wait_s += waited + self.cooldown_s
