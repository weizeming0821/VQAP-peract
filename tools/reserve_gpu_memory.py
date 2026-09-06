#!/usr/bin/env python3
"""Safely reserve currently-free memory on one NVIDIA GPU.

The process allocates CUDA memory once and then blocks without running a compute
loop.  Run one instance per GPU so that every reservation has an independent
PID and can be released independently.

Examples:

    python tools/reserve_gpu_memory.py --gpu 1 --memory-mib 40000
    python tools/reserve_gpu_memory.py \
        --gpu GPU-8a55f26c-da29-2480-0bbe-37389420e325 \
        --memory-mib 32000

Release a reservation with the PID printed by this program:

    kill -TERM <pid>

CUDA also releases the allocation if the process is killed or crashes.  An
existing compute process is reported as a warning but does not block a reserve
unless ``--require-idle`` is used.  The requested allocation must always leave
the configured safety margin free at preflight time.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


MIB = 1024 * 1024
MIN_SAFETY_MARGIN_MIB = 512
DEFAULT_SAFETY_MARGIN_MIB = 2048
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]+$")


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    total_mib: int
    free_mib: int


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    used_memory_mib: int | None
    process_name: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve currently-free memory on one NVIDIA GPU, then sleep without doing "
            "GPU compute. Run one process per GPU."
        )
    )
    parser.add_argument(
        "--gpu",
        required=True,
        help="physical nvidia-smi GPU index or full GPU UUID",
    )
    parser.add_argument(
        "--memory-mib",
        required=True,
        type=positive_int,
        help="tensor memory to hold, in MiB (nvidia-smi may show a little more)",
    )
    parser.add_argument(
        "--chunk-mib",
        type=positive_int,
        default=512,
        help="allocation chunk size in MiB (default: 512)",
    )
    parser.add_argument(
        "--safety-margin-mib",
        type=positive_int,
        default=DEFAULT_SAFETY_MARGIN_MIB,
        help=(
            "free memory that must remain after the requested allocation; "
            f"minimum {MIN_SAFETY_MARGIN_MIB} MiB "
            f"(default: {DEFAULT_SAFETY_MARGIN_MIB})"
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=nonnegative_float,
        default=0.0,
        help="release automatically after this many seconds; 0 means indefinitely",
    )
    parser.add_argument(
        "--require-idle",
        action="store_true",
        help="refuse to start if the target GPU already has a compute process",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="run live safety checks but do not import torch or allocate memory",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments only; do not query or touch any GPU",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    selector = args.gpu.strip()
    if selector.isdigit():
        if int(selector) < 0:
            parser.error("--gpu index must be non-negative")
    elif not GPU_UUID_RE.fullmatch(selector):
        parser.error("--gpu must be a non-negative nvidia-smi index or full GPU UUID")
    args.gpu = selector

    if args.safety_margin_mib < MIN_SAFETY_MARGIN_MIB:
        parser.error(
            f"--safety-margin-mib must be at least {MIN_SAFETY_MARGIN_MIB}"
        )


def run_nvidia_smi(query: str) -> str:
    command = [
        "nvidia-smi",
        query,
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("nvidia-smi timed out after 20 seconds") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed: {detail or 'unknown error'}")
    return result.stdout


def parse_mib(value: str) -> int:
    value = value.strip()
    if value in {"", "N/A", "[N/A]"}:
        raise ValueError(f"memory value is unavailable: {value!r}")
    return int(float(value))


def query_gpus() -> list[GpuInfo]:
    output = run_nvidia_smi(
        "--query-gpu=index,uuid,name,memory.total,memory.free"
    )
    gpus: list[GpuInfo] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", maxsplit=4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {raw_line!r}")
        index, uuid, name, total_mib, free_mib = fields
        gpus.append(
            GpuInfo(
                index=int(index),
                uuid=uuid,
                name=name,
                total_mib=parse_mib(total_mib),
                free_mib=parse_mib(free_mib),
            )
        )
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return gpus


def resolve_gpu(selector: str, gpus: list[GpuInfo]) -> GpuInfo:
    if selector.isdigit():
        index = int(selector)
        matches = [gpu for gpu in gpus if gpu.index == index]
    else:
        matches = [gpu for gpu in gpus if gpu.uuid.lower() == selector.lower()]

    if len(matches) != 1:
        available = ", ".join(f"{gpu.index}={gpu.uuid}" for gpu in gpus)
        raise RuntimeError(
            f"GPU selector {selector!r} did not match exactly one GPU; "
            f"available GPUs: {available}"
        )
    return matches[0]


def query_compute_processes() -> list[ComputeProcess]:
    output = run_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name"
    )
    processes: list[ComputeProcess] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi process row: {raw_line!r}")
        gpu_uuid, pid, used_memory, process_name = fields
        used_mib = None if used_memory in {"N/A", "[N/A]"} else parse_mib(used_memory)
        processes.append(
            ComputeProcess(
                gpu_uuid=gpu_uuid,
                pid=int(pid),
                used_memory_mib=used_mib,
                process_name=process_name,
            )
        )
    return processes


def describe_processes(processes: list[ComputeProcess]) -> str:
    return "; ".join(
        f"PID {process.pid} ({process.process_name}, "
        f"{process.used_memory_mib if process.used_memory_mib is not None else 'N/A'} MiB)"
        for process in processes
    )


def run_preflight(
    args: argparse.Namespace,
) -> tuple[GpuInfo, list[ComputeProcess]]:
    gpu = resolve_gpu(args.gpu, query_gpus())
    active = [
        process
        for process in query_compute_processes()
        if process.gpu_uuid.lower() == gpu.uuid.lower()
    ]
    if active and args.require_idle:
        raise RuntimeError(
            f"GPU {gpu.index} already has compute work: "
            f"{describe_processes(active)}. --require-idle was requested."
        )

    required_mib = args.memory_mib + args.safety_margin_mib
    if required_mib > gpu.free_mib:
        raise RuntimeError(
            f"GPU {gpu.index} has {gpu.free_mib} MiB free, but the request needs "
            f"{args.memory_mib} + {args.safety_margin_mib} MiB safety margin."
        )
    return gpu, active


def set_process_name(gpu_index: int) -> None:
    """Set a recognizable Linux comm name; failure is harmless."""
    try:
        libc = ctypes.CDLL(None)
        name = f"gpu-reserve-{gpu_index}".encode()[:15]
        libc.prctl(15, ctypes.c_char_p(name), 0, 0, 0)  # PR_SET_NAME = 15
    except Exception:
        pass


def release_allocations(torch_module, allocations: list[object]) -> None:
    allocations.clear()
    gc.collect()
    try:
        torch_module.cuda.empty_cache()
    except Exception as exc:
        print(f"[WARN] CUDA cache cleanup reported: {exc}", file=sys.stderr)


def reserve(args: argparse.Namespace, gpu: GpuInfo) -> int:
    # Use the stable UUID so this process always exposes exactly the GPU that
    # was checked above, regardless of an inherited CUDA_VISIBLE_DEVICES value.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    set_process_name(gpu.index)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for allocation. Run this script in a CUDA-enabled "
            "environment that provides torch."
        ) from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "CUDA is unavailable after selecting the target GPU. Check the CUDA "
            "driver, PyTorch build, and CUDA_VISIBLE_DEVICES permissions."
        )

    stop_event = threading.Event()
    stop_reason = {"value": "requested"}

    def handle_stop(signum, _frame) -> None:
        stop_reason["value"] = signal.Signals(signum).name
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    allocations: list[object] = []
    remaining_mib = args.memory_mib
    try:
        torch.cuda.set_device(0)
        while remaining_mib > 0:
            if stop_event.is_set():
                print("[INFO] Stop requested during allocation; releasing partial reserve.")
                return 0
            current_mib = min(args.chunk_mib, remaining_mib)
            allocations.append(
                torch.empty(current_mib * MIB, dtype=torch.uint8, device="cuda:0")
            )
            remaining_mib -= current_mib

        torch.cuda.synchronize(0)
        allocated_mib = torch.cuda.memory_allocated(0) // MIB
        reserved_mib = torch.cuda.memory_reserved(0) // MIB
        free_mib, total_mib = (value // MIB for value in torch.cuda.mem_get_info(0))
        pid = os.getpid()
        print(
            f"[READY] PID {pid} reserved GPU {gpu.index} ({gpu.uuid})\n"
            f"        tensor allocated: {allocated_mib} MiB\n"
            f"        PyTorch reserved: {reserved_mib} MiB\n"
            f"        device free/total: {free_mib}/{total_mib} MiB\n"
            f"        release command: kill -TERM {pid}",
            flush=True,
        )

        if args.duration_seconds == 0:
            stop_event.wait()
        else:
            deadline = time.monotonic() + args.duration_seconds
            while not stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_reason["value"] = "duration elapsed"
                    break
                stop_event.wait(remaining)

        print(f"[INFO] Releasing GPU {gpu.index}: {stop_reason['value']}", flush=True)
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory while creating the reservation. All partial "
            "allocations will be released; no holder will remain running."
        ) from exc
    finally:
        release_allocations(torch, allocations)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    if args.dry_run:
        print(
            "[DRY-RUN] Arguments are valid. No GPU was queried or allocated.\n"
            f"          GPU selector: {args.gpu}\n"
            f"          reserve: {args.memory_mib} MiB\n"
            f"          chunk: {args.chunk_mib} MiB\n"
            f"          safety margin: {args.safety_margin_mib} MiB\n"
            f"          require idle: {args.require_idle}\n"
            f"          duration: "
            f"{'indefinite' if args.duration_seconds == 0 else f'{args.duration_seconds:g} s'}"
        )
        return 0

    gpu, active = run_preflight(args)
    print(
        f"[CHECK] GPU {gpu.index}: {gpu.name}, UUID={gpu.uuid}, "
        f"free/total={gpu.free_mib}/{gpu.total_mib} MiB"
    )
    if active:
        print(
            f"[WARN] GPU {gpu.index} already has compute work: "
            f"{describe_processes(active)}"
        )
        print(
            "[WARN] The free-memory check is only a snapshot. Existing jobs may "
            "request more memory later."
        )
    print(
        f"[CHECK] Safe to reserve {args.memory_mib} MiB while retaining "
        f"{args.safety_margin_mib} MiB of pre-allocation headroom."
    )
    if args.check_only:
        print("[CHECK] Check-only mode: no CUDA context or allocation was created.")
        return 0
    return reserve(args, gpu)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
