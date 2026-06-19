#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 UMAP + DPS 光谱库的脚本。

从「无空间结构」的 H5 文件加载数据（1x1 单点像素多变量时间序列，非 13x13 patch），
按指定训练年份和使用的时序长度训练 SpectralLibraryBuilder，并保存训练参数与模型。

数据约定：
- H5 数据生成与解析方式与 dataload_h5.py 一致（见该文件中 WindowCachedDataset、_build_window_cache 及
  原始年数据格式：每年一个文件 {year}_year_dataset.h5，key 为 {row}_{col}，形状 (C, T)）。
- 本脚本针对「单文件含多年」的场景，--h5 指向一个 H5，--year 指定使用哪些年份。支持的格式（按优先级）：
  1) dataload_h5 窗口缓存（windows_*.h5）：含 "windows" (N,C,L) 或 (N,C,H,W,L) 与 "meta" (N,3)。
     meta[i,0]=date_int(yyyymmdd)，年份=date_int//10000；单点 1x1 时取中心像素，得到 (N,T,C)。
  2) 顶层以年份为名的 group（"2000","2001" 等），或整文件 "data"+"year"/"years" 数组按年筛选。
  3) 单一大数组 "data"/"X"/"timeseries" 或多 key 每样本 (C,T)。
- --seq-len：只使用最后 x 天参与训练。
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import joblib
from tqdm import tqdm

# 保证可导入 spectral_lib（从项目根或 spectral_lib 目录运行均可）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from spectral_lib.umap_dps import SpectralLibraryBuilder  # noqa: E402


def preprocess_X_for_umap(
    X,
    divide_first_by=4,
    divide_last_by=17,
):
    """
    输入 UMAP 前的预处理：
    1) 第一个通道（第 0 维特征）除以 divide_first_by；
    2) 最后一通道使用原始值，除以 divide_last_by。
    输入: X (N, T, C)，dtype float32
    返回: X_out (N, T, C), new_n_channels
    """
    X = np.asarray(X, dtype=np.float32).copy()
    N, T, C = X.shape
    X[:, :, 0] /= float(divide_first_by)
    X[:, :, -1] /= float(divide_last_by)
    return X, int(C)


def _load_3d_from_file(f, data_name=None):
    """从已打开的 h5py.File 中加载 3 维数组。优先 data_name，否则尝试 data/X/timeseries/samples。"""
    for name in (data_name,) if data_name else ("data", "X", "timeseries", "samples"):
        if name is None:
            continue
        if name in f and isinstance(f[name], h5py.Dataset):
            arr = np.asarray(f[name], dtype=np.float32)
            if arr.ndim == 3:
                return arr
    return None


def _to_NTC(data):
    """统一为 (N, T, C)。"""
    if data.ndim != 3:
        return None
    N, d1, d2 = data.shape
    if d1 < d2:
        data = np.transpose(data, (0, 2, 1))
    return data


def _load_window_cache_chunk(args):
    """
    子进程内加载窗口缓存的一块索引（供多进程并行调用，须为模块级可 pickle）。
    args: (h5_path, indices, seq_len, ndim, shape_5d)
      shape_5d: 仅 ndim==5 时使用 (N, C, H, W, L)。
    返回: (M, T, C) float32
    """
    h5_path, indices, seq_len, ndim, shape_5d = args
    indices = np.asarray(indices, dtype=np.intp)
    with h5py.File(h5_path, "r") as f:
        win = f["windows"]
        if ndim == 3:
            chunk = np.asarray(win[indices], dtype=np.float32)  # (M, C, L)
            chunk = np.transpose(chunk, (0, 2, 1))  # (M, L, C)
        else:
            chunk = np.asarray(win[indices], dtype=np.float32)  # (M, C, H, W, L)
            _, C, H, W, L = shape_5d
            if H == 1 and W == 1:
                chunk = chunk[:, :, 0, 0, :]
            else:
                ch, cw = H // 2, W // 2
                chunk = chunk[:, :, ch, cw, :]
            chunk = np.transpose(chunk, (0, 2, 1))  # (M, L, C)
        T_full = chunk.shape[1]
        if seq_len is not None and seq_len < T_full:
            chunk = chunk[:, -seq_len:, :]
    return chunk


def load_1x1_timeseries_from_h5(h5_path, seq_len=None, years=None, n_workers=0):
    """
    从无空间结构的 H5（单文件，可含多年）中加载单点多变量时间序列。

    years: 若给定列表，仅加载这些年份的数据（见文档）。
    n_workers: 并行加载的进程数（多进程，非多线程；HDF5 读多用多进程更稳）。0=单进程，>=1 时对窗口缓存格式分块多进程加载。

    返回:
        X: np.ndarray, shape (N, T, C)，dtype float32。
        meta: dict，含 n_samples, n_time, n_channels, source_path 等。
    """
    years_set = set(years) if years else None
    abspath = os.path.abspath(h5_path)

    with h5py.File(h5_path, "r") as f:
        # ----- 0) dataload_h5 窗口缓存格式：windows + meta（支持多进程分块加载）-----
        if "windows" in f and "meta" in f:
            win_ds = f["windows"]
            meta_ds = f["meta"]
            meta_arr = np.asarray(meta_ds)
            if meta_arr.ndim != 2 or meta_arr.shape[1] < 1:
                pass
            else:
                N_file = win_ds.shape[0]
                if meta_arr.shape[0] != N_file:
                    pass
                else:
                    year_arr = (meta_arr[:, 0].astype(np.int64) // 10000).astype(int)
                    ndim = win_ds.ndim
                    if ndim == 3:
                        _, C, T_full = win_ds.shape
                        shape_5d = None
                    elif ndim == 5:
                        _, C, H, W, T_full = win_ds.shape
                        shape_5d = tuple(win_ds.shape)
                    else:
                        shape_5d = None
                        T_full = 0

                    if ndim in (3, 5) and (shape_5d is not None or ndim == 3):
                        if years_set is not None:
                            indices = np.where(np.isin(year_arr, list(years_set)))[0]
                            if len(indices) == 0:
                                raise ValueError(
                                    f"窗口中未找到 --year {sorted(years_set)} 中任一年的样本。"
                                )
                        else:
                            indices = np.arange(N_file, dtype=np.intp)

                        T_used = min(seq_len, T_full) if seq_len is not None else T_full
                        if seq_len is not None and seq_len > T_full:
                            raise ValueError(
                                f"seq_len={seq_len} > total time steps {T_full}"
                            )

                        n_workers = int(n_workers) if n_workers else 0
                        n_load = len(indices)
                        use_parallel = n_workers >= 1 and n_load > 0

                        if use_parallel:
                            n_chunks = min(n_workers * 4, n_load)
                            n_chunks = max(1, n_chunks)
                            chunk_size = (n_load + n_chunks - 1) // n_chunks
                            index_chunks = [
                                indices[i : i + chunk_size]
                                for i in range(0, n_load, chunk_size)
                            ]
                            chunk_args = [
                                (
                                    abspath,
                                    chunk_idx,
                                    seq_len,
                                    ndim,
                                    shape_5d if ndim == 5 else (0, int(C), 0, 0, int(T_full)),
                                )
                                for chunk_idx in index_chunks
                            ]
                            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                                futures = [ex.submit(_load_window_cache_chunk, a) for a in chunk_args]
                                chunks = [
                                    f.result()
                                    for f in tqdm(
                                        futures,
                                        desc=f"加载分块({n_workers}进程)",
                                        unit="chunk",
                                        leave=True,
                                    )
                                ]
                            data = np.concatenate(chunks, axis=0)
                        else:
                            # 按数据点分块读取，进度条显示已加载样本数 / 总样本数
                            chunk_size = max(1, min(50000, n_load // 50))
                            parts_list = []
                            with tqdm(
                                total=n_load,
                                desc="加载数据点",
                                unit="sample",
                                leave=True,
                            ) as pbar:
                                for start in range(0, n_load, chunk_size):
                                    end = min(start + chunk_size, n_load)
                                    idx_slice = indices[start:end]
                                    win = np.asarray(win_ds[idx_slice], dtype=np.float32)
                                    if ndim == 3:
                                        part = np.transpose(win, (0, 2, 1))
                                    else:
                                        N_, C_, H, W, L = win.shape
                                        if H == 1 and W == 1:
                                            part = np.transpose(win[:, :, 0, 0, :], (0, 2, 1))
                                        else:
                                            ch, cw = H // 2, W // 2
                                            part = np.transpose(win[:, :, ch, cw, :], (0, 2, 1))
                                    if seq_len is not None and part.shape[1] > seq_len:
                                        part = part[:, -seq_len:, :]
                                    parts_list.append(part)
                                    pbar.update(len(idx_slice))
                            data = np.concatenate(parts_list, axis=0)

                        labels_arr = None
                        if "labels" in f:
                            labels_arr = np.asarray(f["labels"][indices], dtype=np.int8)
                        year_per_sample = (meta_arr[indices, 0].astype(np.int64) // 10000)
                        landcover_arr = None
                        try:
                            C_base = int(f.attrs.get("channels_base", C))
                            t_idx = min(T_full - 1, max(0, T_full // 2)) if T_full else 0
                            if ndim == 5:
                                cy, cx = win_ds.shape[2] // 2, win_ds.shape[3] // 2
                                lc = np.asarray(
                                    win_ds[indices, C_base - 1, cy, cx, t_idx],
                                    dtype=np.float32,
                                )
                            else:
                                ps = int(f.attrs.get("patch_size", 1))
                                if ps <= 1:
                                    lc_flat_idx = C_base - 1
                                else:
                                    lc_flat_idx = (C_base - 1) * (ps * ps) + (ps // 2) * ps + (ps // 2)
                                lc = np.asarray(win_ds[indices, lc_flat_idx, t_idx], dtype=np.float32)
                            landcover_arr = np.clip(np.round(lc).astype(np.int32), 0, 255)
                        except Exception:
                            pass
                        meta = {
                            "n_samples": data.shape[0],
                            "n_time": data.shape[1],
                            "n_channels": C,
                            "source_path": abspath,
                            "format": "dataload_h5_window_cache",
                            "labels": labels_arr,
                            "year": year_per_sample,
                            "landcover": landcover_arr,
                        }
                        return data, meta

        # ----- 1) 按年份 group 加载（顶层 group 名为 4 位数字年份）-----
        year_groups = [k for k in f.keys() if isinstance(f[k], h5py.Group) and str(k).isdigit() and len(str(k)) == 4]
        if year_groups and years_set:
            requested = [str(y) for y in years_set if str(y) in year_groups]
            if requested:
                parts = []
                for yg in tqdm(
                    sorted(requested),
                    desc="加载年份 group",
                    unit="year",
                    leave=True,
                ):
                    g = f[yg]
                    arr = _load_3d_from_file(g)
                    if arr is None:
                        keys = [k for k in g.keys() if isinstance(g[k], h5py.Dataset)]
                        if not keys:
                            continue
                        chunks = []
                        for k in sorted(keys, key=lambda x: (int(x) if str(x).isdigit() else x)):
                            a = np.asarray(g[k], dtype=np.float32)
                            if a.ndim != 2:
                                continue
                            if a.shape[0] < a.shape[1]:
                                a = a.T
                            chunks.append(a)
                        if chunks:
                            arr = np.stack(chunks, axis=0)
                        else:
                            continue
                    arr = _to_NTC(arr)
                    if arr is not None:
                        parts.append(arr)
                if parts:
                    T, C = parts[0].shape[1], parts[0].shape[2]
                    for i, p in enumerate(parts[1:], 1):
                        if p.shape[1] != T or p.shape[2] != C:
                            raise ValueError(f"Year group {requested[i]} 的 T/C 与其它不一致。")
                    data = np.concatenate(parts, axis=0)
                    T_total, C = data.shape[1], data.shape[2]
                    if seq_len is not None:
                        if seq_len > T_total:
                            raise ValueError(f"seq_len={seq_len} > total time steps {T_total}")
                        data = data[:, -seq_len:, :]
                        T_used = seq_len
                    else:
                        T_used = T_total
                    meta = {
                        "n_samples": data.shape[0],
                        "n_time": T_used,
                        "n_channels": C,
                        "source_path": abspath,
                    }
                    return data, meta

        # ----- 2) 整文件一大块 + year 数组筛选 -----
        data = _load_3d_from_file(f)
        if data is not None:
            data = _to_NTC(data)
            year_arr = None
            for name in ("year", "years"):
                if name in f and isinstance(f[name], h5py.Dataset):
                    y = np.asarray(f[name]).ravel()
                    if y.size == data.shape[0]:
                        year_arr = y.astype(int)
                        break
            if year_arr is not None and years_set is not None:
                mask = np.isin(year_arr, list(years_set))
                if not np.any(mask):
                    raise ValueError(f"文件中未找到 --year {sorted(years_set)} 中任一年的样本。")
                data = data[mask]
            T_total = data.shape[1]
            C = data.shape[2]
            if seq_len is not None:
                if seq_len > T_total:
                    raise ValueError(f"seq_len={seq_len} > total time steps {T_total}")
                data = data[:, -seq_len:, :]
                T_used = seq_len
            else:
                T_used = T_total
            meta = {
                "n_samples": data.shape[0],
                "n_time": T_used,
                "n_channels": C,
                "source_path": abspath,
            }
            return data, meta

        # ----- 3) 多 key 样本（每个 key 一个 2D）-----
        keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        if not keys:
            if "samples" in f and isinstance(f["samples"], h5py.Group):
                keys = list(f["samples"].keys())
                parent = f["samples"]
            else:
                raise ValueError(f"No 3D dataset or multiple 2D datasets found in {h5_path}")
        else:
            parent = f
        get_ds = parent.__getitem__

        chunks = []
        key_list = sorted(keys, key=lambda x: (int(x) if str(x).isdigit() else x))
        for k in tqdm(key_list, desc="加载样本", unit="sample", leave=True):
            arr = np.asarray(get_ds(k), dtype=np.float32)
            if arr.ndim != 2:
                continue
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            chunks.append(arr)
        if not chunks:
            raise ValueError(f"No valid 2D datasets in {h5_path}")
        data = np.stack(chunks, axis=0)
        data = _to_NTC(data)
        T_total = data.shape[1]
        C = data.shape[2]
        if seq_len is not None:
            if seq_len > T_total:
                raise ValueError(f"seq_len={seq_len} > total time steps {T_total}")
            data = data[:, -seq_len:, :]
            T_used = seq_len
        else:
            T_used = T_total
    meta = {
        "n_samples": data.shape[0],
        "n_time": T_used,
        "n_channels": C,
        "source_path": abspath,
    }
    return data, meta


# 每个 stratum 最少样本数（不足则全部纳入）
MIN_SAMPLES_PER_STRATUM = 100


def sample_stratified_pos_neg_1to1(labels, year, landcover, max_samples=None, random_state=42):
    """
    按 (year, landcover) 分层抽样，层内正负 1:1。
    - 每个 stratum 至少贡献 MIN_SAMPLES_PER_STRATUM 个样本（不足则全部纳入）。
    - 若指定 max_samples N：先保证上述最少覆盖，再将剩余配额 (N - reserved) 在各 stratum 间均匀分配，
      使年份与 landcover 尽量均匀；不做随机截断。
    labels: (N,) 0=负 1=正
    year: (N,) 年份
    landcover: (N,) 土地覆盖类型；若为 None 则仅按 year 分层。
    返回: 选中的样本索引 (ndarray int)。
    """
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels).ravel()
    year = np.asarray(year).ravel()
    N = len(labels)
    if landcover is not None:
        landcover = np.asarray(landcover).ravel()
        if len(landcover) != N:
            landcover = None
    if landcover is None:
        landcover = np.zeros(N, dtype=np.int32)

    strata = list(set(zip(year.tolist(), landcover.tolist())))
    # 每层: (pos_idx, neg_idx, cap=2*min(n_pos,n_neg))
    stratum_data = []
    for (y, lc) in strata:
        mask = (year == y) & (landcover == lc)
        pos_idx = np.where(mask & (labels == 1))[0]
        neg_idx = np.where(mask & (labels == 0))[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        cap = 2 * min(n_pos, n_neg)
        if cap == 0:
            continue
        stratum_data.append((pos_idx, neg_idx, cap))
    K = len(stratum_data)

    if not stratum_data:
        return np.array([], dtype=np.intp)

    # 阶段 1：每层至少 min(100, cap)，不足 100 则全部纳入
    min_per = MIN_SAMPLES_PER_STRATUM
    min_takes = [min(min_per, cap) for (_, _, cap) in stratum_data]
    reserved = sum(min_takes)

    if max_samples is None or max_samples <= 0:
        quota_per_stratum = [min_takes[i] for i in range(len(stratum_data))]
    else:
        if reserved > max_samples:
            # 总预留超过 N：按比例缩减每层，但每层至少 2（1 正 1 负）保证覆盖
            scale = max_samples / reserved
            quota_per_stratum = []
            for i, (_, _, cap) in enumerate(stratum_data):
                q = max(2, int(round(min_takes[i] * scale)))
                q = (q // 2) * 2
                q = min(cap, max(2, q))
                quota_per_stratum.append(q)
            while sum(quota_per_stratum) > max_samples:
                for i in range(len(quota_per_stratum)):
                    if quota_per_stratum[i] > 2 and sum(quota_per_stratum) > max_samples:
                        quota_per_stratum[i] -= 2
                        break
        else:
            budget = max_samples - reserved
            base_extra = budget // K
            remainder = budget - K * base_extra
            quota_per_stratum = []
            for i, (_, _, cap) in enumerate(stratum_data):
                extra = base_extra + (1 if i < remainder else 0)
                q = min(cap, min_takes[i] + extra)
                q = (q // 2) * 2
                quota_per_stratum.append(min(cap, max(min_takes[i], q)))
            total = sum(quota_per_stratum)
            if total < max_samples:
                for _ in range(max_samples - total):
                    best_i = None
                    for i in range(len(stratum_data)):
                        if quota_per_stratum[i] < stratum_data[i][2]:
                            if best_i is None or quota_per_stratum[i] < quota_per_stratum[best_i]:
                                best_i = i
                    if best_i is None:
                        break
                    quota_per_stratum[best_i] = min(stratum_data[best_i][2], quota_per_stratum[best_i] + 2)
            elif total > max_samples:
                for _ in range(total - max_samples):
                    reduced = False
                    for i in range(len(quota_per_stratum)):
                        if quota_per_stratum[i] > min_takes[i]:
                            quota_per_stratum[i] -= 2
                            reduced = True
                            break
                    if not reduced:
                        break

    selected = []
    for i, (pos_idx, neg_idx, cap) in enumerate(stratum_data):
        q = quota_per_stratum[i]
        q = min(q, cap)
        q = (q // 2) * 2
        if q <= 0:
            continue
        half = q // 2
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        take_pos = min(half, n_pos)
        take_neg = min(half, n_neg)
        take = min(take_pos, take_neg) * 2
        if take <= 0:
            continue
        sel_pos = rng.choice(pos_idx, size=take // 2, replace=False)
        sel_neg = rng.choice(neg_idx, size=take // 2, replace=False)
        selected.extend(sel_pos.tolist())
        selected.extend(sel_neg.tolist())
    return np.array(selected, dtype=np.intp)


def parse_args():
    p = argparse.ArgumentParser(
        description="从无空间结构 H5 加载单点时序数据，训练 UMAP+DPS 光谱库并保存。"
    )
    p.add_argument(
        "--h5",
        type=str,
        default='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v7/cache_correct_single_points/windows_f7a30b073176f46c08f907935ea30977.h5',
        help="H5 文件路径（或目录；若为目录则需配合 --year 使用，加载 {dir}/{year}_year_dataset.h5）",
    )
    p.add_argument(
        "--year",
        type=int,
        nargs="+",
        default=[2000,2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                      2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        help="训练年份列表，如 --year 2000 2001 2002。若 --h5 为目录则加载 {h5}/{year}_year_dataset.h5 并合并；否则仅用于保存命名。",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="参与训练的时间序列长度（天）。总长 365 时可用 10、20 等表示仅用最后 x 天。默认 365。",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./spectral_lib_checkpoints_spatial_temporal_reduction",
        help="保存模型与参数的目录。",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="umap_dps",
        help="保存文件名前缀，例如 umap_dps -> umap_dps_year2023_seq20.json / .joblib。",
    )
    # SpectralLibraryBuilder 参数
    p.add_argument("--n-components", type=int, default=5, help="UMAP 降维维度 c。")
    p.add_argument("--n-library-size", type=int, default=50, help="DPS 光谱库样本数 K。")
    p.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors。")
    p.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist。")
    p.add_argument("--dc-percentile", type=float, default=2.0, help="DPS 截断距离百分位。")
    p.add_argument("--seed", type=int, default=42, help="随机种子。")
    p.add_argument("--divide-last-by", type=float, default=17.0, help="最后一通道（原始值）除以该值再送入 UMAP。")
    p.add_argument("--divide-first-by", type=float, default=4.0, help="第一个通道（第 0 维特征）除以该值再送入 UMAP。")
    _ncpu = os.cpu_count() or 8
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, _ncpu),
        help="H5 多进程并行加载数（仅窗口缓存格式）。0=单进程，默认 min(8, CPU数)。",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="训练使用的最大样本数。先保证每 stratum 至少 100（不足则全取），再将剩余配额在 year×landcover 间均匀分配；不随机截断。",
    )
    p.add_argument(
        "--library-max-per-class",
        type=int,
        default=20000,
        help="构建正/负光谱库时，正类、负类各自最多参与 DPS 的样本数，避免 N² 距离矩阵 OOM。超出则随机子采样。",
    )
    return p.parse_args()


def main():
    args = parse_args()

    h5_path = args.h5
    if not os.path.isfile(h5_path):
        raise SystemExit(f"H5 文件不存在: {h5_path}")

    years = args.year  # list or None
    if years is not None:
        years = sorted(set(years))
    n_workers = getattr(args, "workers", 0) or 0
    if n_workers >= 1:
        print(f"加载 H5: {h5_path}，使用最后 {args.seq_len} 天时序。【多进程并行】workers={n_workers}")
    else:
        print(f"加载 H5: {h5_path}，使用最后 {args.seq_len} 天时序。（单进程，可加 --workers 8 加速）")
    if years:
        print(f"指定年份: {years}")
    X, meta = load_1x1_timeseries_from_h5(
        h5_path, seq_len=args.seq_len, years=years, n_workers=n_workers
    )

    # 输入预处理：第一通道除以 divide_first_by，最后一通道除以 divide_last_by（原始值）
    X, meta["n_channels"] = preprocess_X_for_umap(
        X,
        divide_first_by=getattr(args, "divide_first_by", 4.0),
        divide_last_by=getattr(args, "divide_last_by", 17.0),
    )

    # 按 (year, landcover) 分层、正负 1:1 抽样，可选 --max-samples 截断
    if (
        meta.get("format") == "dataload_h5_window_cache"
        and meta.get("labels") is not None
        and meta.get("year") is not None
    ):
        labels = np.asarray(meta["labels"]).ravel()
        year_arr = np.asarray(meta["year"]).ravel()
        lc_arr = meta.get("landcover")
        n_before = len(labels)
        sel = sample_stratified_pos_neg_1to1(
            labels, year_arr, lc_arr,
            max_samples=getattr(args, "max_samples", None),
            random_state=args.seed,
        )
        if len(sel) < n_before:
            X = X[sel]
            meta["labels"] = meta["labels"][sel]
            meta["year"] = meta["year"][sel]
            if meta.get("landcover") is not None:
                meta["landcover"] = meta["landcover"][sel]
            meta["n_samples"] = len(sel)
            n_pos = int(np.sum(np.asarray(meta["labels"]) == 1))
            n_neg = len(sel) - n_pos
            print(f"抽样后: N={len(sel)} (正={n_pos}, 负={n_neg}, 1:1 按 year×landcover；每层≥{MIN_SAMPLES_PER_STRATUM}，max_samples={getattr(args, 'max_samples', None)})")
        else:
            print(f"未做 1:1 抽样（或抽样后未截断），N={n_before}")
    else:
        max_s = getattr(args, "max_samples", None)
        if max_s is not None and max_s > 0 and meta["n_samples"] > max_s:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(meta["n_samples"], size=max_s, replace=False)
            X = X[sel]
            for k in ("labels", "year", "landcover"):
                if meta.get(k) is not None:
                    meta[k] = np.asarray(meta[k])[sel]
            meta["n_samples"] = len(sel)
            print(f"非窗口缓存格式，截断至 --max-samples={max_s}")
    print(f"样本数 N={meta['n_samples']}, 时间步 T={meta['n_time']}, 通道数 C={meta['n_channels']}")

    # 训练
    builder = SpectralLibraryBuilder(
        n_components=args.n_components,
        n_library_size=args.n_library_size,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        dc_percentile=args.dc_percentile,
        random_state=args.seed,
    )
    builder.fit(X)

    # 保存目录
    os.makedirs(args.out_dir, exist_ok=True)
    base = f"{args.out_prefix}_seq{args.seq_len}"

    # 1) 保存训练参数（便于复现与查看）
    params = {
        "year": years,
        "seq_len": args.seq_len,
        "h5_path": meta["source_path"],
        "n_samples": meta["n_samples"],
        "n_time": meta["n_time"],
        "n_channels": meta["n_channels"],
        "divide_last_by": getattr(args, "divide_last_by", 17.0),
        "divide_first_by": getattr(args, "divide_first_by", 4.0),
        "n_components": args.n_components,
        "n_library_size": args.n_library_size,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "dc_percentile": args.dc_percentile,
        "random_state": args.seed,
    }
    params_path = os.path.join(args.out_dir, f"{base}_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"已保存参数: {params_path}")

    # 2) 保存完整 builder（含 UMAP 与 DPS 结果，可 load 后 get_library / transform）
    model_path = os.path.join(args.out_dir, f"{base}.joblib")
    joblib.dump(builder, model_path)
    print(f"已保存模型: {model_path}")

    # 3) 可选：保存库索引与嵌入，便于离线分析
    npz_path = os.path.join(args.out_dir, f"{base}_library.npz")
    np.savez(
        npz_path,
        library_indices=builder.library_indices_,
        library_embeddings=builder.library_embeddings_,
    )
    print(f"已保存库索引与嵌入: {npz_path}")

    # 4) 若为窗口缓存且含 labels：用全量数据分正负做 UMAP 映射，并分别 DPS 得到正/负样本光谱库
    if meta.get("format") == "dataload_h5_window_cache" and meta.get("labels") is not None:
        T, C = meta["n_time"], meta["n_channels"]
        c = args.n_components
        n_workers = getattr(args, "workers", 0) or 0
        print("[全量数据] 重新加载全量数据（未抽样）用于正/负光谱库构建...")
        X_full, meta_full = load_1x1_timeseries_from_h5(
            h5_path, seq_len=args.seq_len, years=years, n_workers=n_workers
        )
        # 与训练阶段相同的预处理（原始值缩放，无随机）
        X_full, meta_full["n_channels"] = preprocess_X_for_umap(
            X_full,
            divide_first_by=getattr(args, "divide_first_by", 4.0),
            divide_last_by=getattr(args, "divide_last_by", 17.0),
        )
        labels_full = np.asarray(meta_full["labels"]).ravel()
        pos_mask = labels_full == 1
        neg_mask = labels_full == 0
        n_pos, n_neg = int(np.sum(pos_mask)), int(np.sum(neg_mask))
        print(f"[全量] 正样本 {n_pos}，负样本 {n_neg}，分别映射并选库...")
        # 全量阶段为 transform（投影），临时改进度条文案避免与「训练」混淆
        _old_tqdm = getattr(builder.umap_model, "tqdm_kwds", None)
        builder.umap_model.tqdm_kwds = dict(desc="UMAP 投影", leave=True)
        if n_pos > 0:
            X_pos = X_full[pos_mask]
            pos_flat = X_pos.reshape(-1, C)
            pos_emb_flat = builder.umap_model.transform(pos_flat)
            pos_emb = pos_emb_flat.reshape(n_pos, T, c).astype(np.float32)
        else:
            pos_emb = np.zeros((0, T, c), dtype=np.float32)
        if n_neg > 0:
            X_neg = X_full[neg_mask]
            neg_flat = X_neg.reshape(-1, C)
            neg_emb_flat = builder.umap_model.transform(neg_flat)
            neg_emb = neg_emb_flat.reshape(n_neg, T, c).astype(np.float32)
        else:
            neg_emb = np.zeros((0, T, c), dtype=np.float32)
        if _old_tqdm is not None:
            builder.umap_model.tqdm_kwds = _old_tqdm
        by_label_path = os.path.join(args.out_dir, f"{base}_by_label.npz")
        np.savez(
            by_label_path,
            pos_embeddings=pos_emb,
            neg_embeddings=neg_emb,
            n_pos=n_pos,
            n_neg=n_neg,
            n_components=c,
            seq_len=T,
        )
        print(f"已保存全量正/负样本 UMAP 嵌入: {by_label_path}")

        # 在正、负嵌入上分别做 DPS 选库 → 正样本光谱库、负样本光谱库（每类超 library_max 则子采样，避免 OOM）
        library_max = getattr(args, "library_max_per_class", 20000) or 20000
        rng_lib = np.random.default_rng(args.seed)
        if pos_emb.shape[0] > 0:
            n_pos_in = pos_emb.shape[0]
            if n_pos_in > library_max:
                sub_idx = rng_lib.choice(n_pos_in, size=library_max, replace=False)
                pos_emb_dps = pos_emb[sub_idx]
                print(f"[DPS 正] 全量 {n_pos_in} → 子采样 {library_max} 以控制内存")
            else:
                pos_emb_dps = pos_emb
            pos_lib_idx, pos_lib_emb = builder.select_library_from_embeddings(pos_emb_dps)
            n_pos_lib = len(pos_lib_idx)
        else:
            pos_lib_idx = np.array([], dtype=np.intp)
            pos_lib_emb = np.zeros((0, T, c), dtype=np.float32)
            n_pos_lib = 0
        if neg_emb.shape[0] > 0:
            n_neg_in = neg_emb.shape[0]
            if n_neg_in > library_max:
                sub_idx = rng_lib.choice(n_neg_in, size=library_max, replace=False)
                neg_emb_dps = neg_emb[sub_idx]
                print(f"[DPS 负] 全量 {n_neg_in} → 子采样 {library_max} 以控制内存")
            else:
                neg_emb_dps = neg_emb
            neg_lib_idx, neg_lib_emb = builder.select_library_from_embeddings(neg_emb_dps)
            n_neg_lib = len(neg_lib_idx)
        else:
            neg_lib_idx = np.array([], dtype=np.intp)
            neg_lib_emb = np.zeros((0, T, c), dtype=np.float32)
            n_neg_lib = 0
        libraries_path = os.path.join(args.out_dir, f"{base}_libraries_by_label.npz")
        np.savez(
            libraries_path,
            pos_library_indices=pos_lib_idx,
            pos_library_embeddings=pos_lib_emb,
            neg_library_indices=neg_lib_idx,
            neg_library_embeddings=neg_lib_emb,
            n_pos=n_pos,
            n_neg=n_neg,
            n_pos_library=n_pos_lib,
            n_neg_library=n_neg_lib,
            n_components=c,
            seq_len=T,
        )
        print(f"已保存正样本光谱库 ({n_pos_lib})、负样本光谱库 ({n_neg_lib}): {libraries_path}")
    else:
        if meta.get("format") != "dataload_h5_window_cache":
            print("[映射] 非窗口缓存格式，跳过按标签映射与光谱库构建。")
        elif meta.get("labels") is None:
            print("[映射] 无 labels，跳过按标签映射与光谱库构建。")

    print("完成。")


if __name__ == "__main__":
    main()
