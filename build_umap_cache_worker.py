#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone worker to build UMAP cache H5 in indexed format.

目标：
- 复用 OptimizedDataLoader 取得和测试阶段完全一致的窗口与元数据（date_int, row, col, window_idx）
- 复用 SSM_learnable_umap_2.UMAPAnchorLaplacian._umap_transform 计算 UMAP 嵌入
- 仅对 patch 中心像素计算 UMAP：key = f"{date_int}_{window_idx}_{row}_{col}"

生成的 H5 结构：
- attrs:
    format          = "umap_indexed"
    model_name      = "SSM_learnable_umap_2"
    seq_len         = <seq_len>
    patch_size      = <patch_size>
    umap_model_path = <umap_model_path>
- datasets:
    data: (N, D) float32
    keys: (N,)  variable-length utf-8 string

性能优化（vs 原版）：
  1. patch_size=1 建数据集 → preload 正常启用，避免每 batch 重读 H5
  2. 直接切片 preloaded_array，完全绕过 get_batch_data 的 patch 组装开销
  3. 每个 window 一次性处理全部像素（原版每 256 像素一次，约 97 批/窗口）
  4. UMAP n_jobs=-1，使用全部 CPU 核心
  5. 矢量化 key 生成，无 Python 级 for 循环写 key
"""

import os
import sys
import argparse
import multiprocessing

import h5py
import numpy as np
from tqdm import tqdm

import torch

from test_vis_opt import OptimizedDataLoader
from model_zoo.SSM_learnable_umap_2 import UMAPAnchorLaplacian


def _build_dataset(h5_files, seq_len: int) -> OptimizedDataLoader:
    """
    构建仅用于 UMAP cache 构建的数据集（patch_size=1，启用 preload）。

    使用 patch_size=1 而非 13，避免组装 13×13 空间 patch，
    preloaded_array 形状为 [N_pixels, C, T_total]，
    可直接通过时间切片取中心像素的时序数据。
    """
    h5_files = list(h5_files)
    if not h5_files:
        raise ValueError("No H5 files provided for UMAP cache building")

    ds = OptimizedDataLoader(
        h5_file_path=h5_files[0],
        lookback_length=seq_len,
        forecast_horizon=1,
        start_offset=365,
        max_windows=None,
        preload_data=True,   # patch_size=1 才能启用 preload
        num_workers=multiprocessing.cpu_count(),
        chunk_size=2048,
        h5_file_paths=h5_files,
        patch_size=1,        # 只取中心像素，无需空间 patch
        burn_history_path=None,
        umap_cache_path=None,
    )
    return ds


def _build_umap_layer(
    umap_model_path: str,
    library_path: str,
    n_channels: int = 38,
    n_jobs: int = -1,
) -> UMAPAnchorLaplacian:
    """构建仅用于 _umap_transform 的 UMAPAnchorLaplacian，设置 n_jobs=-1。"""
    layer = UMAPAnchorLaplacian(
        umap_model_path=umap_model_path,
        library_path=library_path,
        n_input_channels=n_channels,
        hidden_dim=32,
        d_model=380,
        adapt_dim=32,
        temperature=1.0,
        lambda_anchor=0.0,
        device="cpu",
    )
    layer.eval()
    if hasattr(layer, "umap_model") and hasattr(layer.umap_model, "n_jobs"):
        layer.umap_model.n_jobs = n_jobs
        print(f"[UMAP cache worker] Set umap_model.n_jobs={n_jobs} (CPUs: {multiprocessing.cpu_count()})")
    return layer


def _scale_umap_input(x_nct: np.ndarray) -> None:
    """
    UMAP 专用缩放（in-place）：ch0/4, ch37/17，仅当 max>1.25 时缩放。
    x_nct: [N, C, T]
    """
    if float(np.nanmax(x_nct[:, 0, :])) > 1.25:
        x_nct[:, 0, :] /= 4.0
    if x_nct.shape[1] > 37 and float(np.nanmax(x_nct[:, 37, :])) > 1.25:
        x_nct[:, 37, :] /= 17.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UMAP cache (indexed) for testing")
    parser.add_argument("--h5-files", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v7/2024_year_dataset.h5',
                        help="Comma-separated list of yearly H5 files")
    parser.add_argument("--cache-path", default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v7/cache',
                        help="Target UMAP cache H5 path (will be created/overwritten)")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Lookback length (must match Tester.lookback_length)")
    parser.add_argument("--patch-size", type=int, default=13,
                        help="Stored in cache attrs for compatibility (default 13)")
    parser.add_argument("--umap-model-path", type=str,
                        default="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10.joblib")
    parser.add_argument("--umap-library-path", type=str,
                        default="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10_libraries_by_label.npz")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Pixels per UMAP call. 0=all at once (default, fastest).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="UMAP n_jobs (-1=all CPUs, default)")

    args = parser.parse_args()

    h5_files = [x.strip() for x in args.h5_files.split(",") if x.strip()]
    if not h5_files:
        print("No valid --h5-files specified", file=sys.stderr)
        sys.exit(1)

    print(f"[UMAP cache worker] H5 files : {h5_files}")
    print(f"[UMAP cache worker] Cache    : {args.cache_path}")
    print(f"[UMAP cache worker] seq_len  : {args.seq_len}")
    print(f"[UMAP cache worker] n_jobs   : {args.n_jobs}")

    # ── 1. Build dataset (patch_size=1, preload enabled) ──────────────────
    print("[UMAP cache worker] Building dataset (patch_size=1, full preload)...")
    dataset = _build_dataset(h5_files, args.seq_len)

    if dataset.preloaded_array is None:
        print("[UMAP cache worker] ERROR: preloaded_array is None.", file=sys.stderr)
        sys.exit(1)

    preloaded   = dataset.preloaded_array   # [N, C, T_total]
    time_offset = dataset.time_offset
    N           = dataset.num_pixels
    C           = int(preloaded.shape[1])
    num_windows = dataset.num_windows
    seq_len     = args.seq_len

    print(f"[UMAP cache worker] Preloaded: shape={preloaded.shape}  N={N} C={C}")
    print(f"[UMAP cache worker] Windows  : {num_windows}")

    # ── 2. Build UMAP layer ───────────────────────────────────────────────
    umap_layer = _build_umap_layer(
        args.umap_model_path, args.umap_library_path,
        n_channels=C, n_jobs=args.n_jobs,
    )

    # ── 3. Probe embedding dim D ──────────────────────────────────────────
    probe_n = min(8, N)
    probe_x_nct = preloaded[:probe_n, :, :seq_len].copy()
    _scale_umap_input(probe_x_nct)
    probe_x = torch.from_numpy(
        probe_x_nct.transpose(0, 2, 1).reshape(probe_n, seq_len * C)
    )
    with torch.no_grad():
        D = int(umap_layer._umap_transform(probe_x).shape[1])
    print(f"[UMAP cache worker] UMAP embedding dim D={D}")

    # ── 4. Per-window UMAP (all pixels at once) ───────────────────────────
    pixel_batch_size = args.batch_size if args.batch_size > 0 else N
    pixel_coords     = dataset.pixel_coords  # [N, 2]

    all_keys = []
    all_zs   = []

    for win_idx in tqdm(range(num_windows), desc="Building UMAP cache", unit="win"):
        window  = dataset.valid_time_windows[win_idx]
        t_start = int(window["lookback_start"]) - time_offset
        t_end   = t_start + seq_len

        if t_start < 0 or t_end > preloaded.shape[2]:
            # Skip cross-year boundary windows not covered by preloaded_array
            continue

        # date_int for key
        date_int = int(dataset._calculate_window_target_date(win_idx).strftime("%Y%m%d"))

        # [N, C, T_lb] → scale → [N, T_lb*C]
        x_nct = preloaded[:, :, t_start:t_end].copy()
        _scale_umap_input(x_nct)
        x_flat_np = x_nct.transpose(0, 2, 1).reshape(N, seq_len * C)  # [N, T*C]

        # Process in chunks to control peak memory; default: one chunk = all N
        z_parts = []
        for pstart in range(0, N, pixel_batch_size):
            pend    = min(pstart + pixel_batch_size, N)
            x_chunk = torch.from_numpy(x_flat_np[pstart:pend])
            with torch.no_grad():
                z_parts.append(umap_layer._umap_transform(x_chunk).cpu().numpy())
        z_win = np.concatenate(z_parts, axis=0)  # [N, D]

        # Vectorised key generation
        rows     = pixel_coords[:, 0].astype(int)
        cols     = pixel_coords[:, 1].astype(int)
        win_keys = [f"{date_int}_{win_idx}_{r}_{c}" for r, c in zip(rows, cols)]

        all_keys.extend(win_keys)
        all_zs.append(z_win)

    if not all_keys:
        print("[UMAP cache worker] No keys generated, abort.", file=sys.stderr)
        sys.exit(1)

    # ── 5. Write H5 cache ────────────────────────────────────────────────
    data_arr  = np.concatenate(all_zs, axis=0).astype("float32")
    keys_arr  = np.array(all_keys, dtype=h5py.string_dtype(encoding="utf-8"))
    Total, D_ = data_arr.shape
    print(f"[UMAP cache worker] Writing cache: N={Total}, D={D_}")

    cache_dir = os.path.dirname(os.path.abspath(args.cache_path))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp_path = args.cache_path + ".tmp"

    try:
        with h5py.File(tmp_path, "w") as f:
            f.attrs["format"]          = "umap_indexed"
            f.attrs["model_name"]      = "SSM_learnable_umap_2"
            f.attrs["seq_len"]         = int(args.seq_len)
            f.attrs["patch_size"]      = int(args.patch_size)
            f.attrs["umap_model_path"] = str(args.umap_model_path)

            f.create_dataset(
                "data",
                data=data_arr,
                dtype="float32",
                chunks=(min(1024, Total), min(D_, 128)),
                compression="lzf",
                shuffle=True,
            )
            f.create_dataset("keys", data=keys_arr)

        os.replace(tmp_path, args.cache_path)
        sz_mb = os.path.getsize(args.cache_path) / 1e6
        print(f"[UMAP cache worker] Done: {os.path.basename(args.cache_path)} ({sz_mb:.1f} MB)")
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    main()
