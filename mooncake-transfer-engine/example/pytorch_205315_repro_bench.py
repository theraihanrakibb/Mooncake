#!/usr/bin/env python3
# Copyright 2026 KVCache.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone reproducer and comparison benchmark for pytorch/pytorch#205315.

Reproduces the exact two-thread poller + GIL contention scenario in PyTorch's
`torch.distributed._transport.mooncake.MooncakeTransport` during RL weight sync
(e.g. 32,768 x 16 KiB one-sided READs in flight) and compares three designs:

  1. `before_205315` (PyTorch MooncakeTransport before PR #205315):
     - Main thread submits N single-entry `batch_transfer_async` requests under
       an RLock and calls `_poll_cv.notify()` on every submit.
     - Background poller thread copies `list(self._batches.items())` every sweep
       and calls `transfer_check_status(batch_id)` on ALL pending batches,
       releasing/reacquiring the GIL N times per sweep and starving `_submit`.

  2. `pytorch_pr_205315` (PyTorch PR #205315 `_POLL_WINDOW = 256` fix):
     - Wakes poller only when `_batches` was empty.
     - Checks at most `_POLL_WINDOW` (256) batches per sweep and rotates pending
       ones via `OrderedDict.move_to_end` (trades up to ~N/256 ms tail detection
       delay, and still pays N individual Python/pybind submit calls).

  3. `batched_arrays_check_status` (Coalesced `batch_transfer_async_*_arrays` +
     fixed whole-batch `transfer_check_status`):
     - Coalesces per-peer descriptors into `batch_transfer_async_{read,write}_arrays`
       (1 batch per peer by default, or `--sub_batches K`) and polls completion
       on the same background poller thread via whole-batch `transfer_check_status`.

Usage:
  # 1. Real Mooncake RDMA loopback (single command, spawns target subprocess):
  python3 mooncake-transfer-engine/example/pytorch_205315_repro_bench.py \
      --mode loopback --protocol rdma --device_name mlx5_0 --hugepages

  # 2. Reproduce PR #205315's congested 2s in-flight regime on a single host
  #    (still executes real pybind + C++ GIL release/acquire on every check):
  python3 mooncake-transfer-engine/example/pytorch_205315_repro_bench.py \
      --mode loopback --protocol rdma --device_name mlx5_0 \
      --emulate_latency_ms 2000 --rounds 1

  # 3. Two-node RDMA benchmark:
  #   Node A (target):
  python3 mooncake-transfer-engine/example/pytorch_205315_repro_bench.py \
      --mode target --local_server_name <target_ip>:12345 \
      --protocol rdma --device_name mlx5_0 --pool_size_mb 1024 --hugepages
  #   Node B (initiator):
  python3 mooncake-transfer-engine/example/pytorch_205315_repro_bench.py \
      --mode initiator --local_server_name <initiator_ip>:12346 \
      --target_server_name <target_ip>:<actual_port> \
      --protocol rdma --device_name mlx5_0 --pool_size_mb 1024 --hugepages

  # 4. Pure-Python + ctypes C-GIL fake engine (for environments without Mooncake built):
  python3 mooncake-transfer-engine/example/pytorch_205315_repro_bench.py \
      --fake_engine --emulate_latency_ms 2000 --rounds 1
"""

from __future__ import annotations

import argparse
from array import array
from collections import OrderedDict
from concurrent.futures import Future
import ctypes
import ctypes.util
import itertools
import json
import os
import signal
import statistics
import subprocess
import sys
from threading import Condition, RLock, Thread
import time
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def _import_mooncake_engine() -> tuple[Any, Any]:
    try:
        from mooncake.engine import TransferEngine, TransferOpcode

        return TransferEngine, TransferOpcode
    except ImportError:
        from engine import TransferEngine, TransferOpcode  # type: ignore[import-not-found]

        return TransferEngine, TransferOpcode


def make_u64_buffer(values: list[int]) -> Any:
    if np is not None:
        return np.asarray(values, dtype=np.uint64)
    return array("Q", values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce pytorch/pytorch#205315 and compare batched arrays + transfer_check_status"
    )
    parser.add_argument(
        "--mode",
        choices=["loopback", "target", "initiator"],
        default="loopback",
        help="Execution mode (default: loopback)",
    )
    parser.add_argument(
        "--local_server_name",
        default="127.0.0.1:12345",
        help="Local server name (default: 127.0.0.1:12345)",
    )
    parser.add_argument(
        "--target_server_name",
        default="",
        help="Target server address (required in initiator mode)",
    )
    parser.add_argument(
        "--metadata_server",
        default="P2PHANDSHAKE",
        help="Metadata server connection string (default: P2PHANDSHAKE)",
    )
    parser.add_argument(
        "--protocol",
        default="rdma",
        help="Transport protocol, e.g. rdma, tcp, efa (default: rdma)",
    )
    parser.add_argument(
        "--device_name",
        default="",
        help="Initiator NIC/device filter, e.g. mlx5_0 (default: auto)",
    )
    parser.add_argument(
        "--target_device_name",
        default="",
        help="Target NIC/device filter in loopback mode (defaults to --device_name)",
    )
    parser.add_argument(
        "--opcode",
        choices=["read", "write"],
        default="read",
        help="Transfer opcode matching MooncakeTransport (default: read)",
    )
    parser.add_argument(
        "--num_transfers",
        type=int,
        default=32768,
        help="Number of transfers per weight-sync step (PR #205315 default: 32768)",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=int,
        default=16384,
        help="Bytes per transfer (PR #205315 default: 16384 = 16 KiB)",
    )
    parser.add_argument(
        "--pool_size_mb",
        type=int,
        default=1024,
        help="Registered memory pool size in MiB (default: 1024)",
    )
    parser.add_argument(
        "--poll_window",
        type=int,
        default=256,
        help="Window size for PR #205315 poller (default: 256)",
    )
    parser.add_argument(
        "--sub_batches",
        type=int,
        default=1,
        help="Number of coalesced array batches to split --num_transfers into "
        "in Case 3 (default: 1 = one multi-entry batch per peer)",
    )
    parser.add_argument(
        "--emulate_latency_ms",
        type=float,
        default=0.0,
        help="Optional minimum in-flight completion latency in ms (e.g. 2000 to "
        "reproduce PR #205315's 2s congested network test on a single host while "
        "still executing real C++/GIL transitions on every check; default: 0)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Measured rounds per case (default: 3)",
    )
    parser.add_argument(
        "--cases",
        default="before,pr205315,batched",
        help="Comma-separated cases to run: before,pr205315,batched",
    )
    parser.add_argument(
        "--hugepages",
        action="store_true",
        help="Try MAP_HUGETLB when allocating host memory",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Allocate registered memory on CUDA GPU via PyTorch",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device ID when --use_gpu is set (default: 0)",
    )
    parser.add_argument(
        "--fake_engine",
        action="store_true",
        help="Run with a C-GIL-releasing ctypes fake engine (no Mooncake build needed)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save JSON results",
    )
    return parser.parse_args()


class MemoryPool:
    """Page-aligned mmap or CUDA tensor memory pool."""

    def __init__(
        self, size_bytes: int, *, use_gpu: bool, gpu_id: int, hugepages: bool
    ) -> None:
        self.size_bytes = size_bytes
        self._libc: Any = None
        self._tensor: Any = None
        self.addr = 0

        if use_gpu:
            import torch

            torch.cuda.set_device(gpu_id)
            self._tensor = torch.zeros(
                size_bytes, dtype=torch.uint8, device=f"cuda:{gpu_id}"
            )
            self.addr = int(self._tensor.data_ptr())
            return

        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_name, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        libc.munmap.restype = ctypes.c_int
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._libc = libc

        prot = 0x1 | 0x2
        flags = 0x02 | 0x20
        map_hugetlb = 0x40000
        map_failed = ctypes.c_void_p(-1).value

        ptr = None
        if hugepages:
            ptr = libc.mmap(None, size_bytes, prot, flags | map_hugetlb, -1, 0)
            if not ptr or ptr == map_failed:
                ptr = None
        if ptr is None:
            ptr = libc.mmap(None, size_bytes, prot, flags, -1, 0)
        if not ptr or ptr == map_failed:
            raise RuntimeError(
                f"mmap({size_bytes}) failed with errno={ctypes.get_errno()}"
            )

        ctypes.memset(ptr, 0, size_bytes)
        self.addr = int(ptr)

    def close(self) -> None:
        if self.addr and self._libc is not None:
            self._libc.munmap(ctypes.c_void_p(self.addr), self.size_bytes)
            self.addr = 0
        self._tensor = None


class _FakeOpcode:
    Read = 0
    Write = 1


class _CtypesGilFakeEngine:
    """Fake TransferEngine that releases/reacquires the Python GIL via libc C calls."""

    def __init__(self, pool_bytes: int) -> None:
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        # Standard ctypes.CDLL calls release & reacquire the GIL around the C call,
        # matching pybind11::gil_scoped_release contention behavior.
        self._libc = ctypes.CDLL(libc_name)
        self._libc.getpid.restype = ctypes.c_int
        self._remote_buf = MemoryPool(
            pool_bytes, use_gpu=False, gpu_id=0, hugepages=False
        )
        self._next_bid = 1
        self._live: set[int] = set()

    def close(self) -> None:
        self._remote_buf.close()

    def get_first_buffer_address(self, _target: str) -> int:
        return self._remote_buf.addr

    def _alloc_bid(self) -> int:
        bid = self._next_bid
        self._next_bid += 1
        self._live.add(bid)
        return bid

    def batch_transfer_async(
        self,
        _target: str,
        buffers: list[int],
        peer_buffers: list[int],
        lengths: list[int],
        _opcode: Any,
    ) -> int:
        _ = (len(buffers), len(peer_buffers), len(lengths))
        self._libc.getpid()
        return self._alloc_bid()

    def batch_transfer_async_read_arrays(
        self, _target: str, buffers: Any, peer_buffers: Any, lengths: Any
    ) -> int:
        _ = (memoryview(buffers), memoryview(peer_buffers), memoryview(lengths))
        self._libc.getpid()
        return self._alloc_bid()

    def batch_transfer_async_write_arrays(
        self, _target: str, buffers: Any, peer_buffers: Any, lengths: Any
    ) -> int:
        _ = (memoryview(buffers), memoryview(peer_buffers), memoryview(lengths))
        self._libc.getpid()
        return self._alloc_bid()

    def transfer_check_status(self, batch_id: int) -> int:
        self._libc.getpid()
        if batch_id == 0 or batch_id not in self._live:
            return -1
        self._live.remove(batch_id)
        return 1


class _CountingEngineProxy:
    """Counts `transfer_check_status` calls and optionally holds in-flight completion.

    When `emulate_latency_s > 0`, premature checks still invoke
    `engine.transfer_check_status(0)` so every poll crosses pybind11 and executes
    `pybind11::gil_scoped_release` without freeing the live `BatchDesc` early.
    """

    def __init__(self, engine: Any, emulate_latency_s: float) -> None:
        self._engine = engine
        self._emulate_s = emulate_latency_s
        self._ready_at: dict[int, float] = {}
        self.check_calls = 0

    def reset_counters(self) -> None:
        self.check_calls = 0
        self._ready_at.clear()

    def batch_transfer_async(
        self,
        target: str,
        buffers: list[int],
        peer_buffers: list[int],
        lengths: list[int],
        opcode: Any,
    ) -> int:
        bid = self._engine.batch_transfer_async(
            target, buffers, peer_buffers, lengths, opcode
        )
        if bid != 0 and self._emulate_s > 0.0:
            self._ready_at[bid] = time.perf_counter() + self._emulate_s
        return bid

    def batch_transfer_async_read_arrays(
        self, target: str, buffers: Any, peer_buffers: Any, lengths: Any
    ) -> int:
        bid = self._engine.batch_transfer_async_read_arrays(
            target, buffers, peer_buffers, lengths
        )
        if bid != 0 and self._emulate_s > 0.0:
            self._ready_at[bid] = time.perf_counter() + self._emulate_s
        return bid

    def batch_transfer_async_write_arrays(
        self, target: str, buffers: Any, peer_buffers: Any, lengths: Any
    ) -> int:
        bid = self._engine.batch_transfer_async_write_arrays(
            target, buffers, peer_buffers, lengths
        )
        if bid != 0 and self._emulate_s > 0.0:
            self._ready_at[bid] = time.perf_counter() + self._emulate_s
        return bid

    def transfer_check_status(self, batch_id: int) -> int:
        self.check_calls += 1
        if self._emulate_s > 0.0:
            ready = self._ready_at.get(batch_id, 0.0)
            if time.perf_counter() < ready:
                # Cross real C++ pybind + gil_scoped_release without freeing batch_id.
                self._engine.transfer_check_status(0)
                return 0
            self._ready_at.pop(batch_id, None)
        return self._engine.transfer_check_status(batch_id)


class MooncakePollerHarness:
    """Faithful replica of PyTorch `MooncakeTransport` locking & poller thread."""

    def __init__(
        self,
        engine: _CountingEngineProxy,
        *,
        window_size: int | None,
        wake_only_when_idle: bool,
    ) -> None:
        self._engine = engine
        self._window_size = window_size
        self._wake_only_when_idle = wake_only_when_idle
        self._operation_lock = RLock()
        self._poll_cv = Condition()
        self._pending: dict[int, Future[None]] = {}
        self._batches: OrderedDict[int, Future[None]] = OrderedDict()
        self._poller: Thread | None = None
        self._poller_stopped = False
        self._next_work_id = 1
        self.sweeps = 0
        self.peak_inflight = 0

    def _prune(self, key: int) -> None:
        with self._operation_lock:
            self._pending.pop(key, None)

    def submit_single(
        self,
        peer: str,
        local_addr: int,
        remote_addr: int,
        size: int,
        opcode: Any,
    ) -> Future[None]:
        """Exact `MooncakeTransport._start` + `_submit` single-entry path."""
        with self._operation_lock:
            batch_id = self._engine.batch_transfer_async(
                peer, [local_addr], [remote_addr], [size], opcode
            )
            if batch_id == 0:
                raise RuntimeError("Mooncake batch_transfer_async returned 0")
            fut: Future[None] = Future()
            work_id = self._next_work_id
            self._next_work_id += 1
            self._pending[work_id] = fut
            fut.add_done_callback(lambda _, wid=work_id: self._prune(wid))
            with self._poll_cv:
                if not self._wake_only_when_idle or not self._batches:
                    self._poll_cv.notify()
                self._batches[batch_id] = fut
                if len(self._batches) > self.peak_inflight:
                    self.peak_inflight = len(self._batches)
                if self._poller is None:
                    self._poller = Thread(
                        target=self._poll, name="mooncake-repro", daemon=True
                    )
                    self._poller.start()
            return fut

    def submit_arrays(
        self,
        peer: str,
        local_arr: Any,
        remote_arr: Any,
        len_arr: Any,
        is_read: bool,
    ) -> Future[None]:
        """Coalesced array-batch submission settled by the same poller thread."""
        with self._operation_lock:
            fn = (
                self._engine.batch_transfer_async_read_arrays
                if is_read
                else self._engine.batch_transfer_async_write_arrays
            )
            batch_id = fn(peer, local_arr, remote_arr, len_arr)
            if batch_id == 0:
                raise RuntimeError("Mooncake batch_transfer_async_*_arrays returned 0")
            fut: Future[None] = Future()
            work_id = self._next_work_id
            self._next_work_id += 1
            self._pending[work_id] = fut
            fut.add_done_callback(lambda _, wid=work_id: self._prune(wid))
            with self._poll_cv:
                if not self._batches:
                    self._poll_cv.notify()
                self._batches[batch_id] = fut
                if len(self._batches) > self.peak_inflight:
                    self.peak_inflight = len(self._batches)
                if self._poller is None:
                    self._poller = Thread(
                        target=self._poll, name="mooncake-repro", daemon=True
                    )
                    self._poller.start()
            return fut

    def _poll(self) -> None:
        engine = self._engine
        window = self._window_size
        delay = 0.0
        while True:
            with self._poll_cv:
                if delay:
                    self._poll_cv.wait(delay)
                while not self._batches and not self._poller_stopped:
                    self._poll_cv.wait()
                    delay = 0.0
                if self._poller_stopped:
                    return
                if window is None:
                    batches = list(self._batches.items())
                else:
                    batches = list(itertools.islice(self._batches.items(), window))

            self.sweeps += 1
            settled: list[tuple[int, Future[None], int]] = []
            pending: list[int] = []
            for batch_id, fut in batches:
                status = engine.transfer_check_status(batch_id)
                if status == 0:
                    if window is not None:
                        pending.append(batch_id)
                else:
                    settled.append((batch_id, fut, status))

            with self._poll_cv:
                for batch_id in pending:
                    self._batches.move_to_end(batch_id)
                for batch_id, fut, _ in settled:
                    if self._batches.get(batch_id) is fut:
                        del self._batches[batch_id]

            for _, fut, status in settled:
                if status == 1:
                    fut.set_result(None)
                else:
                    fut.set_exception(
                        RuntimeError(f"Mooncake transfer failed with status {status}")
                    )
            delay = 0.0 if settled else min(max(delay * 2, 1e-6), 1e-3)

    def close(self) -> None:
        with self._poll_cv:
            self._poller_stopped = True
            self._poll_cv.notify()
        if self._poller is not None:
            self._poller.join(timeout=5.0)


def actual_server_name(
    local_server_name: str, metadata_server: str, engine: Any
) -> str:
    if metadata_server == "P2PHANDSHAKE":
        host = local_server_name.rpartition(":")[0] or local_server_name
        return f"{host}:{engine.get_rpc_port()}"
    return local_server_name


def run_one_round(
    case_name: str,
    proxy: _CountingEngineProxy,
    peer: str,
    local_addrs: list[int],
    remote_addrs: list[int],
    lengths: list[int],
    sub_arrays: list[tuple[Any, Any, Any]],
    opcode_enum: Any,
    is_read: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    proxy.reset_counters()
    if case_name == "before":
        harness = MooncakePollerHarness(
            proxy, window_size=None, wake_only_when_idle=False
        )
    elif case_name == "pr205315":
        harness = MooncakePollerHarness(
            proxy, window_size=args.poll_window, wake_only_when_idle=True
        )
    elif case_name == "batched":
        harness = MooncakePollerHarness(
            proxy, window_size=args.poll_window, wake_only_when_idle=True
        )
    else:
        raise ValueError(f"Unknown case {case_name!r}")

    try:
        t0 = time.perf_counter()
        if case_name in ("before", "pr205315"):
            futures = [
                harness.submit_single(
                    peer, local_addrs[i], remote_addrs[i], lengths[i], opcode_enum
                )
                for i in range(len(local_addrs))
            ]
        else:
            futures = [
                harness.submit_arrays(peer, la, ra, lna, is_read)
                for la, ra, lna in sub_arrays
            ]
        t_submit = time.perf_counter()
        for fut in futures:
            fut.result(timeout=120.0)
        t_done = time.perf_counter()
    finally:
        harness.close()

    submit_ms = (t_submit - t0) * 1000.0
    total_ms = (t_done - t0) * 1000.0
    overhead_ms = max(0.0, total_ms - args.emulate_latency_ms)
    total_bytes = args.num_transfers * args.chunk_bytes
    gbps = (total_bytes / 1e9) / (total_ms / 1000.0) if total_ms > 0 else 0.0
    return {
        "submit_ms": submit_ms,
        "total_ms": total_ms,
        "overhead_ms": overhead_ms,
        "checks": float(proxy.check_calls),
        "sweeps": float(harness.sweeps),
        "peak_inflight": float(harness.peak_inflight),
        "gbps": gbps,
    }


def run_benchmark(
    raw_engine: Any,
    opcode_cls: Any,
    target_name: str,
    local_base: int,
    remote_base: int,
    pool_bytes: int,
    args: argparse.Namespace,
) -> None:
    is_read = args.opcode == "read"
    opcode_enum = opcode_cls.Read if is_read else opcode_cls.Write
    usable = pool_bytes - args.chunk_bytes
    if usable <= 0:
        raise ValueError("pool_size_mb is smaller than chunk_bytes")

    n = args.num_transfers
    chunk = args.chunk_bytes
    local_addrs = [local_base + ((i * chunk) % usable) for i in range(n)]
    remote_addrs = [remote_base + ((i * chunk) % usable) for i in range(n)]
    lengths = [chunk] * n

    num_subs = max(1, min(args.sub_batches, n))
    sub_arrays: list[tuple[Any, Any, Any]] = []
    step = (n + num_subs - 1) // num_subs
    for start in range(0, n, step):
        stop = min(n, start + step)
        sub_arrays.append(
            (
                make_u64_buffer(local_addrs[start:stop]),
                make_u64_buffer(remote_addrs[start:stop]),
                make_u64_buffer(lengths[start:stop]),
            )
        )

    # Connection warmup (without latency emulation)
    warm_proxy = _CountingEngineProxy(raw_engine, 0.0)
    warm_h = MooncakePollerHarness(
        warm_proxy, window_size=args.poll_window, wake_only_when_idle=True
    )
    try:
        la0 = make_u64_buffer(local_addrs[: min(16, n)])
        ra0 = make_u64_buffer(remote_addrs[: min(16, n)])
        ln0 = make_u64_buffer(lengths[: min(16, n)])
        warm_h.submit_arrays(target_name, la0, ra0, ln0, is_read).result(timeout=30.0)
    finally:
        warm_h.close()

    selected_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    labels = {
        "before": "1. Before #205315 (unbounded sweep)",
        "pr205315": f"2. PyTorch #205315 (window={args.poll_window})",
        "batched": f"3. Batched arrays + check_status (batches={len(sub_arrays)})",
    }

    total_mib = (n * chunk) / (1024 * 1024)
    print("\n" + "=" * 116)
    print(
        f"[PyTorch #205315 Reproducer] {n} x {chunk // 1024} KiB {args.opcode.upper()} "
        f"({total_mib:.1f} MiB) | engine={'fake(ctypes-GIL)' if args.fake_engine else args.protocol} "
        f"| emulate_latency={args.emulate_latency_ms:.0f} ms | rounds={args.rounds}"
    )
    print("=" * 116)
    print(
        f"{'Case':<48}  {'Submit(ms)':>10}  {'Total(ms)':>10}  "
        f"{'Ovhd(ms)':>9}  {'Checks':>11}  {'Sweeps':>8}  {'PeakPend':>8}  {'GB/s':>6}"
    )
    print("-" * 116)

    proxy = _CountingEngineProxy(raw_engine, args.emulate_latency_ms / 1000.0)
    summary: dict[str, Any] = {
        "num_transfers": n,
        "chunk_bytes": chunk,
        "opcode": args.opcode,
        "protocol": "fake" if args.fake_engine else args.protocol,
        "emulate_latency_ms": args.emulate_latency_ms,
        "rounds": args.rounds,
        "cases": {},
    }

    baseline_total: float | None = None
    for case_key in selected_cases:
        runs = [
            run_one_round(
                case_key,
                proxy,
                target_name,
                local_addrs,
                remote_addrs,
                lengths,
                sub_arrays,
                opcode_enum,
                is_read,
                args,
            )
            for _ in range(args.rounds)
        ]
        med = {
            k: statistics.median([r[k] for r in runs])
            for k in (
                "submit_ms",
                "total_ms",
                "overhead_ms",
                "checks",
                "sweeps",
                "peak_inflight",
                "gbps",
            )
        }
        if baseline_total is None:
            baseline_total = med["total_ms"]
        summary["cases"][case_key] = {
            "label": labels[case_key],
            "median": {k: round(v, 3) for k, v in med.items()},
            "speedup_vs_first": round(baseline_total / med["total_ms"], 3)
            if med["total_ms"] > 0
            else 0.0,
        }
        print(
            f"{labels[case_key]:<48}  {med['submit_ms']:>10.1f}  "
            f"{med['total_ms']:>10.1f}  {med['overhead_ms']:>9.1f}  "
            f"{int(round(med['checks'])):>11,d}  {int(round(med['sweeps'])):>8,d}  "
            f"{int(round(med['peak_inflight'])):>8,d}  {med['gbps']:>6.2f}"
        )

    print("=" * 116)
    if "before" in summary["cases"] and "batched" in summary["cases"]:
        b_tot = summary["cases"]["before"]["median"]["total_ms"]
        c_tot = summary["cases"]["batched"]["median"]["total_ms"]
        b_chk = summary["cases"]["before"]["median"]["checks"]
        c_chk = max(1.0, summary["cases"]["batched"]["median"]["checks"])
        print(
            f"Summary vs Before #205315: total wall time {b_tot:.1f} ms -> {c_tot:.1f} ms "
            f"({b_tot / c_tot:.2f}x faster), status checks {int(b_chk):,d} -> {int(c_chk):,d} "
            f"({b_chk / c_chk:,.0f}x fewer)."
        )
    if "pr205315" in summary["cases"] and "batched" in summary["cases"]:
        p_tot = summary["cases"]["pr205315"]["median"]["total_ms"]
        c_tot = summary["cases"]["batched"]["median"]["total_ms"]
        p_sub = summary["cases"]["pr205315"]["median"]["submit_ms"]
        c_sub = max(0.01, summary["cases"]["batched"]["median"]["submit_ms"])
        print(
            f"Summary vs PyTorch #205315: total wall time {p_tot:.1f} ms -> {c_tot:.1f} ms "
            f"({p_tot / c_tot:.2f}x faster), submit time {p_sub:.1f} ms -> {c_sub:.2f} ms "
            f"({p_sub / c_sub:,.0f}x faster)."
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON report to {args.output}")


def run_target(args: argparse.Namespace) -> None:
    TransferEngine, _ = _import_mooncake_engine()
    pool_bytes = args.pool_size_mb * 1024 * 1024
    pool = MemoryPool(
        pool_bytes,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
        hugepages=args.hugepages,
    )
    engine = TransferEngine()
    ret = engine.initialize(
        args.local_server_name,
        args.metadata_server,
        args.protocol,
        args.device_name,
    )
    if ret != 0:
        pool.close()
        raise RuntimeError(f"Target initialize failed with {ret}")
    ret = engine.register_memory(pool.addr, pool_bytes)
    if ret != 0:
        pool.close()
        raise RuntimeError(f"Target register_memory failed with {ret}")

    server_name = actual_server_name(
        args.local_server_name, args.metadata_server, engine
    )
    print(f"TARGET_READY:{json.dumps({'server_name': server_name})}", flush=True)

    stop = False

    def _on_stop(_sig: int, _frm: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)
    try:
        while not stop:
            time.sleep(0.2)
    finally:
        engine.unregister_memory(pool.addr)
        pool.close()


def run_initiator(args: argparse.Namespace, target_server_name: str) -> None:
    TransferEngine, TransferOpcode = _import_mooncake_engine()
    pool_bytes = args.pool_size_mb * 1024 * 1024
    pool = MemoryPool(
        pool_bytes,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
        hugepages=args.hugepages,
    )
    engine = TransferEngine()
    ret = engine.initialize(
        args.local_server_name,
        args.metadata_server,
        args.protocol,
        args.device_name,
    )
    if ret != 0:
        pool.close()
        raise RuntimeError(f"Initiator initialize failed with {ret}")
    ret = engine.register_memory(pool.addr, pool_bytes)
    if ret != 0:
        pool.close()
        raise RuntimeError(f"Initiator register_memory failed with {ret}")

    try:
        remote_base = engine.get_first_buffer_address(target_server_name)
        if remote_base == 0:
            raise RuntimeError(
                f"Failed to get first buffer address from {target_server_name}"
            )
        run_benchmark(
            engine,
            TransferOpcode,
            target_server_name,
            pool.addr,
            remote_base,
            pool_bytes,
            args,
        )
    finally:
        engine.unregister_memory(pool.addr)
        pool.close()


def run_loopback(args: argparse.Namespace) -> None:
    target_dev = args.target_device_name or args.device_name
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        "target",
        "--local_server_name",
        "127.0.0.1",
        "--metadata_server",
        "P2PHANDSHAKE",
        "--protocol",
        args.protocol,
        "--pool_size_mb",
        str(args.pool_size_mb),
    ]
    if target_dev:
        cmd.extend(["--device_name", target_dev])
    if args.hugepages:
        cmd.append("--hugepages")
    if args.use_gpu:
        cmd.extend(["--use_gpu", "--gpu_id", str(args.gpu_id)])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        target_info: dict[str, Any] | None = None
        deadline = time.monotonic() + 30.0
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(
                    f"Target subprocess exited early ({proc.returncode}):\n{err}"
                )
            line = proc.stdout.readline()
            if line.startswith("TARGET_READY:"):
                target_info = json.loads(line.partition(":")[2].strip())
                break
        if target_info is None:
            raise RuntimeError("Timed out waiting for target subprocess")

        init_args = argparse.Namespace(**vars(args))
        init_args.local_server_name = "127.0.0.1"
        init_args.metadata_server = "P2PHANDSHAKE"
        run_initiator(init_args, target_info["server_name"])
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def run_fake(args: argparse.Namespace) -> None:
    pool_bytes = args.pool_size_mb * 1024 * 1024
    local_pool = MemoryPool(
        pool_bytes, use_gpu=False, gpu_id=0, hugepages=args.hugepages
    )
    fake = _CtypesGilFakeEngine(pool_bytes)
    try:
        remote_base = fake.get_first_buffer_address("127.0.0.1:20001")
        run_benchmark(
            fake,
            _FakeOpcode,
            "127.0.0.1:20001",
            local_pool.addr,
            remote_base,
            pool_bytes,
            args,
        )
    finally:
        fake.close()
        local_pool.close()


def main() -> None:
    args = parse_args()
    if args.fake_engine:
        run_fake(args)
    elif args.mode == "target":
        run_target(args)
    elif args.mode == "initiator":
        if not args.target_server_name:
            raise SystemExit("--target_server_name is required in initiator mode")
        run_initiator(args, args.target_server_name)
    else:
        run_loopback(args)


if __name__ == "__main__":
    main()
