# Lightweight module for ProcessPoolExecutor workers.
# Only imports os, h5py, numpy - avoids slow model loading when spawning processes.
"""预加载 worker 独立模块，避免子进程导入主脚本时触发模型加载等重初始化。"""
import os
import h5py
import numpy as np
from typing import Optional


def _read_patch_from_h5(f, all_dataset_names: set,
                        center_row: int, center_col: int, patch_size: int,
                        t_start: int = None, t_end: int = None) -> Optional[np.ndarray]:
    """Read (C, H, W, T) patch centered at (center_row, center_col)."""
    r = patch_size // 2
    base_shape = None
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            name = f"{center_row + dr}_{center_col + dc}"
            if name in all_dataset_names:
                ds = f[name]
                base_shape = ds.shape
                break
        if base_shape is not None:
            break
    if base_shape is None:
        return None
    C = int(base_shape[0])
    if t_start is not None and t_end is not None:
        T_out = t_end - t_start
    else:
        T_out = int(base_shape[1])
        t_start, t_end = 0, T_out

    patch = np.zeros((C, patch_size, patch_size, T_out), dtype=np.float32)
    for i, dr in enumerate(range(-r, r + 1)):
        for j, dc in enumerate(range(-r, r + 1)):
            name = f"{center_row + dr}_{center_col + dc}"
            if name in all_dataset_names:
                try:
                    arr = f[name][:, t_start:t_end]
                    if arr.shape[0] == C and arr.shape[1] == T_out:
                        patch[:, i, j, :] = arr.astype(np.float32, copy=False)
                except Exception:
                    pass
    return patch


def read_datasets_chunk(args):
    """进程池 worker：批量读取一组 dataset 的时间切片。
    
    Args: (file_path, dataset_names_list, t_start, t_end)
    Returns: dict { name: np.ndarray(C, T_slice) }
    """
    file_path, dataset_names, t_start, t_end = args
    result = {}
    f = h5py.File(file_path, 'r')
    try:
        for name in dataset_names:
            try:
                arr = f[name][:, t_start:t_end]
                result[name] = arr.astype(np.float32, copy=False)
            except Exception:
                pass
    finally:
        f.close()
    return result


def load_patch_chunk_worker(args):
    """进程池 worker：加载一块像素的 patch 数据。独立模块，spawn 时只需导入此文件。"""
    (year_files, pixel_chunk, lookback_start, forecast_end, target_year,
     total_channels, lookback_length, forecast_horizon, patch_size, total_time_steps) = args
    ps = patch_size
    past_list = []
    future_list = []
    local_indices = []

    opens = {}
    for y, p in year_files.items():
        if os.path.exists(p):
            opens[y] = h5py.File(p, 'r')

    def get_names(y):
        return set(opens[y].keys()) if y in opens else set()

    for local_idx, row, col in pixel_chunk:
        if row < 0 or col < 0:
            continue
        past_patch = np.zeros((total_channels, ps, ps, lookback_length), dtype=np.float32)
        future_patch = np.zeros((total_channels, ps, ps, forecast_horizon), dtype=np.float32)

        if lookback_start >= 0:
            f = opens.get(target_year)
            if f is not None:
                names = get_names(target_year)
                t_start = lookback_start
                t_end = lookback_start + lookback_length + forecast_horizon
                patch_slice = _read_patch_from_h5(f, names, row, col, ps, t_start=t_start, t_end=t_end)
                if patch_slice is not None and patch_slice.shape[-1] >= lookback_length + forecast_horizon:
                    past_patch[:] = patch_slice[:, :, :, :lookback_length]
                    future_patch[:] = patch_slice[:, :, :, lookback_length:lookback_length + forecast_horizon]
        else:
            prev_year = target_year - 1
            f_prev, f_curr = opens.get(prev_year), opens.get(target_year)
            prev_data, curr_data = None, None
            if f_prev:
                prev_t_start = max(0, total_time_steps + lookback_start)
                prev_t_end = total_time_steps
                patch_prev = _read_patch_from_h5(f_prev, get_names(prev_year), row, col, ps, t_start=prev_t_start, t_end=prev_t_end)
                if patch_prev is not None:
                    prev_data = patch_prev
            if f_curr:
                patch_curr = _read_patch_from_h5(f_curr, get_names(target_year), row, col, ps, t_start=0, t_end=forecast_end)
                if patch_curr is not None:
                    curr_data = patch_curr
            if prev_data is not None and curr_data is not None:
                full = np.concatenate([prev_data, curr_data], axis=-1)
                if full.shape[-1] >= lookback_length + forecast_horizon:
                    past_patch[:] = full[:, :, :, :lookback_length]
                    future_patch[:] = full[:, :, :, lookback_length:lookback_length + forecast_horizon]

        past_list.append(past_patch)
        future_list.append(future_patch)
        local_indices.append(local_idx)

    for f in opens.values():
        f.close()

    return local_indices, past_list, future_list
