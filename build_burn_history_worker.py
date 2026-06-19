#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone worker to build prediction burn_history H5. No torch/CUDA.
Uses multiprocessing.Pool (safe fork) + SharedMemory for parallel read of 21 years.
Progress bar: total = 21 × keys_per_year, updated via shared counter.
"""

import os
import sys
import json
import argparse
import time
import h5py
import numpy as np
from multiprocessing import Pool, Value
from multiprocessing.shared_memory import SharedMemory
from tqdm import tqdm

# Burn history time range (must match dataload_h5.py)
BURN_HISTORY_START_YEAR = 2000
BURN_HISTORY_END_YEAR = 2020


def _t_days_burn_history():
    total = 0
    for y in range(BURN_HISTORY_START_YEAR, BURN_HISTORY_END_YEAR + 1):
        total += 366 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365
    return total


T_DAYS_BURN_HISTORY = _t_days_burn_history()

# Set by main before Pool; workers read via fork (copy-on-write)
_KEY_TO_IDX = {}
_SHM_NAME = ""
_BUF_SHAPE = (0, 0)
_YEAR_OFFSETS = {}
_COUNTER = None  # Value('L', 0)


def _worker_one_year(args):
    year_file, year, g0 = args
    n_read = 0
    try:
        shm = SharedMemory(name=_SHM_NAME)
        buf = np.ndarray(_BUF_SHAPE, dtype=np.uint8, buffer=shm.buf)
        # 增大 chunk 缓存到 512MB
        rdcc = 512 * 1024 * 1024
        with h5py.File(year_file, "r", libver="latest", rdcc_nbytes=rdcc) as yf:
            # 预分配一个临时数组用于 read_direct，避免每次创建新数组
            # 假设大部分时间序列长度不超过 366
            max_len = 366
            temp_buf = np.zeros(max_len, dtype=np.float64)
            
            local_count = 0
            for key in yf.keys():
                local_count += 1
                if local_count % 1000 == 0:
                    with _COUNTER.get_lock():
                        _COUNTER.value += 1000
                
                idx = _KEY_TO_IDX.get(key)
                if idx is None:
                    continue
                try:
                    ds = yf[key]
                    ty = ds.shape[1]
                    
                    # 直接读入预分配的 buffer
                    ds.read_direct(temp_buf, np.s_[0, :ty], np.s_[:ty])
                    raw = temp_buf[:ty]
                    
                    # in-place clip 和转换
                    np.clip(raw, 0, 10, out=raw)
                    buf[idx, g0 : g0 + ty] = raw.astype(np.uint8)
                    n_read += 1
                except Exception:
                    pass
            with _COUNTER.get_lock():
                _COUNTER.value += local_count % 1000
        shm.close()
    except Exception:
        pass
    return year, n_read


def main():
    parser = argparse.ArgumentParser(description="Build prediction burn_history H5 (parallel)")
    parser.add_argument("--h5-dir", required=True, help="Directory with {year}_year_dataset.h5")
    parser.add_argument("--output", required=True, help="Output H5 path")
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--pixel-keys-file", required=True, help="JSON file: list of 'row_col' strings")
    args = parser.parse_args()

    # Load pixel keys
    with open(args.pixel_keys_file, "r", encoding="utf-8") as f:
        pixel_keys = json.load(f)
    if not pixel_keys:
        print("No pixel keys", file=sys.stderr)
        sys.exit(1)

    # Expand with patch neighbourhood
    half = args.patch_size // 2
    pixel_coords = set()
    for k in pixel_keys:
        parts = str(k).split("_", 1)
        if len(parts) == 2:
            r, c = int(parts[0]), int(parts[1])
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    pixel_coords.add((r + dr, c + dc))

    sorted_coords = sorted(pixel_coords)
    key_to_idx = {f"{r}_{c}": i for i, (r, c) in enumerate(sorted_coords)}
    N = len(sorted_coords)
    T = T_DAYS_BURN_HISTORY

    # Year offsets
    year_offsets = {}
    off = 0
    for y in range(BURN_HISTORY_START_YEAR, BURN_HISTORY_END_YEAR + 1):
        year_offsets[y] = off
        off += 366 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365

    # Keys per year (for progress total): sample first available year
    keys_per_year = N
    for y in range(BURN_HISTORY_START_YEAR, BURN_HISTORY_END_YEAR + 1):
        path = os.path.join(args.h5_dir, f"{y}_year_dataset.h5")
        if os.path.isfile(path):
            with h5py.File(path, "r") as f:
                keys_per_year = len(f)
            break
    total_steps = 21 * keys_per_year

    # Allocate shared buffer
    global _KEY_TO_IDX, _SHM_NAME, _BUF_SHAPE, _YEAR_OFFSETS, _COUNTER
    _KEY_TO_IDX = key_to_idx
    _YEAR_OFFSETS = year_offsets
    _COUNTER = Value("L", 0)

    shm = SharedMemory(create=True, size=N * T)
    _SHM_NAME = shm.name
    _BUF_SHAPE = (N, T)
    buf = np.ndarray(_BUF_SHAPE, dtype=np.uint8, buffer=shm.buf)
    buf[:] = 0

    # Build task list
    tasks = []
    for year in range(BURN_HISTORY_START_YEAR, BURN_HISTORY_END_YEAR + 1):
        year_file = os.path.join(args.h5_dir, f"{year}_year_dataset.h5")
        if os.path.isfile(year_file):
            tasks.append((year_file, year, year_offsets[year]))

    # 限制 worker 数量，避免 I/O 风暴
    # 4-6 个 worker 通常对 HDF5 读取最友好，再多反而增加随机寻道
    max_io_workers = 12 
    n_workers = min(len(tasks), max(1, os.cpu_count() or 4), max_io_workers)
    print(f"Building burn_history: {N} pixels × {T} days, {len(tasks)} years, {n_workers} I/O workers")
    print(f"Progress total: {total_steps} (21 × {keys_per_year})")
    t0 = time.time()

    # Run pool; main thread updates progress bar from shared counter
    pbar = tqdm(total=total_steps, desc="burn_history", unit=" keys", ncols=120, mininterval=0.5)
    with Pool(processes=n_workers) as pool:
        async_res = pool.map_async(_worker_one_year, tasks)
        while not async_res.ready():
            pbar.n = min(_COUNTER.value, total_steps)
            pbar.refresh()
            time.sleep(0.15)
        pbar.n = _COUNTER.value
        pbar.refresh()
    pbar.close()

    # Copy buffer out of shared memory for writing (worker may still be closing)
    buf_out = np.empty((N, T), dtype=np.uint8)
    buf_out[:] = buf

    shm.close()
    shm.unlink()

    elapsed = time.time() - t0
    print(f"Read done in {elapsed:.0f}s, writing H5 ...")

    # Write H5
    tmp_path = args.output + ".tmp"
    try:
        keys_arr = np.array([f"{r}_{c}" for r, c in sorted_coords], dtype=h5py.special_dtype(vlen=str))
        with h5py.File(tmp_path, "w", libver="latest") as f:
            f.attrs["format"] = "indexed"
            f.attrs["start_year"] = BURN_HISTORY_START_YEAR
            f.attrs["end_year"] = BURN_HISTORY_END_YEAR
            f.attrs["t_days"] = T
            f.attrs["patch_size"] = args.patch_size
            f.create_dataset(
                "data",
                data=buf_out,
                dtype="u1",
                chunks=(min(1024, N), min(365, T)),
                compression="lzf",
                shuffle=True,
            )
            f.create_dataset("keys", data=keys_arr)
        os.rename(tmp_path, args.output)
        total_elapsed = time.time() - t0
        sz_mb = os.path.getsize(args.output) / 1e6
        print(f"Done in {total_elapsed:.0f}s: {os.path.basename(args.output)} ({sz_mb:.0f} MB)")
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    main()
