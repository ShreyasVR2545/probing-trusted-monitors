# Host crash under sustained GPU load — evidence and mitigations

This machine hard-reset twice while running sustained GPU inference. This file
records what the logs actually say, what was changed in response, and what is
still recommended but needs an elevated shell.

## What happened

| | Crash 1 | Crash 2 |
|---|---|---|
| Time | 2026-08-24 05:01:28 | 2026-09-02 23:36:39 |
| Bugcheck | `0x00020001` | `0x00020001` |
| Param 1 | `0x28` | `0x28` |
| Param 2 | `0x1` | `0x1` |
| Param 3 | `0x29b92701` | `0x29b92701` |
| Param 4 | `0xfc810000` | `0xfc800000` |
| Referenced dump | `082426-14984-01.dmp` | `090226-16203-01.dmp` |

Bugcheck `0x00020001` is `HYPERVISOR_ERROR`.

## What the logs rule in and out

**Parameters 1–3 are byte-identical across two crashes nine days apart.** Random
memory corruption does not reproduce a bugcheck to the bit; this is a
deterministic code path being hit under a specific condition.

- **No WHEA events at all** (`Microsoft-Windows-WHEA-Logger`, 30-day window).
  No correctable machine-check errors were logged, which argues against gradual
  RAM or CPU degradation — though a fault that kills the box instantly need not
  be logged.
- **No display-driver TDR.** No `nvlddmkm` event and no Event 4101 at either
  crash. The GPU driver did not reset — the machine died outright.
- **Silent for three minutes before death.** The last System-log entry before
  crash 2 is at 23:33:25 (a routine `IsolatedUserMode` trustlet start); the
  reset is at 23:36:39. Nothing in between. Instantaneous, not a cascade.
- **Memory Integrity (HVCI) is running.** `Win32_DeviceGuard` reports
  `VirtualizationBasedSecurityStatus = 2` and `SecurityServicesRunning = {2}`.
  The hypervisor is always resident, which is why a low-level fault surfaces as
  `HYPERVISOR_ERROR` rather than as a conventional bugcheck.
- **No dump was retained.** `C:\Windows\Minidump` is empty despite the event log
  naming files in it, so neither crash can be analysed post hoc. `CrashDumpEnabled`
  is `3` (minidump) and the page file is ample (18 GB against 32 GB RAM), so the
  dumps were most likely removed by Disk Cleanup or Storage Sense.

## The hardware context

`nvidia-smi` reports an **RTX 5070 Laptop GPU** (Blackwell, cc 12.0, 7.96 GB)
on driver 592.07, under **NVIDIA Dynamic Boost**:

```
power.default_limit = 50 W      power.max_limit = 115 W      power.min_limit = 5 W
```

Dynamic Boost shifts a shared budget between CPU and GPU, so a GPU that idles
near 13 W can be pulled to 115 W within a frame while the CPU is also boosting.

The monitor workload was close to a worst case for that. Per trajectory it did:
a very heavy prefill over a long context (one large matmul burst), then ~100
light, memory-bound decode steps, then **a second full prefill** for activation
capture. That is a large, repeated load oscillation at roughly 0.03 Hz,
sustained for as long as the run lasts.

**This is a correlation, not a proven cause.** Both crashes occurred under this
workload and no dump survives to confirm the mechanism. What follows removes
the workload's sharpest load steps, which is the part this project controls.

## Mitigations applied in code

All are in `src/utils/power.py` and are on by default (`power.enabled` in
`configs/base.yaml`).

1. **Single chunked prefill instead of two bursts.** The prompt is fed through
   the KV cache in 1024-token segments, and `generate()` then continues from
   that warm cache; the capture pass reuses the same cache cropped back to the
   prompt. The long-context prefill now happens **once** per trajectory instead
   of twice, and never as one burst.
   Measured: **33 s → 7 s** per trajectory, **peak VRAM 4.7 GB → 2.8 GB**.
2. **`logits_to_keep=1`.** The LM head is only evaluated at the scoring
   position. Computing logits at every position would materialise
   `seq_len × 152k` floats — about 5 GB at a 16k context.
3. **Cooldown and thermal guard.** 1.5 s between trajectories, extended while
   the GPU is at or above 78 °C, so the duty cycle is never pinned at 100%.
4. **CPU headroom cap.** Maximum processor state is set to 70% for the duration
   of a run and restored afterwards (including on Ctrl-C). Under Dynamic Boost
   the CPU and GPU share a budget, so CPU headroom lowers the combined
   transient. Reversible and applied only while a run is in flight.
5. **Per-trajectory checkpointing and a heartbeat.** `results/heartbeat.json`
   names the trajectory in flight, fsynced before work starts. A hard reset now
   costs one trajectory (~7 s) rather than a whole stage, and the run resumes
   from exactly where it died.

Verified after rewiring: the cache-reuse path is bit-identical to
crop-and-replay and agrees with a fresh full forward on the argmax. The residual
0.59 raw-logit difference is bf16 chunked-versus-single matmul variation; it is
identical for every trajectory in a run, so channel comparisons are unaffected.

## Recommended, but needs an elevated shell

These cannot be applied unelevated and are **not** applied automatically. Run
them in an **Administrator** PowerShell.

**1. Cap the GPU power limit — the single most effective mitigation.** It caps
the top of the Dynamic Boost range, so the 50 → 115 W excursion cannot happen:

```powershell
nvidia-smi -pl 65
```

Not persistent across reboots. To check it took: `nvidia-smi -q -d POWER`.

**2. Keep crash dumps, so a recurrence is diagnosable.** Currently dumps are
being deleted before they can be read:

```powershell
# keep a full kernel dump instead of a minidump
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' CrashDumpEnabled 2
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' AlwaysKeepMemoryDump 1
# stop Storage Sense from removing them
Set-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy' 01 0
```

**3. If it recurs even with the power cap**, the next things to test, in order:

- Run `mdsched.exe` (Windows Memory Diagnostic), or MemTest86 for a real pass.
- Update the NVIDIA driver, or roll back from 592.07 to the previous branch —
  a driver-level fault under HVCI produces exactly this signature.
- Temporarily disable Memory Integrity (Windows Security → Device security →
  Core isolation). This is a **security downgrade**, so treat it as a
  diagnostic step, not a fix: if the crashes stop, the fault is a driver
  interacting with HVCI and the driver is what needs changing.
- Check the Lenovo power adapter is the rated one and fully seated; an
  under-rated adapter shows up exactly as instant resets under combined
  CPU+GPU load.

## If it crashes again

The run resumes automatically — `python scripts/run_all.py --stage 2` picks up
from the manifest and skips every trajectory already cached. Before resuming,
read `results/heartbeat.json` to see which trajectory was in flight, and check
whether a dump was written this time.
