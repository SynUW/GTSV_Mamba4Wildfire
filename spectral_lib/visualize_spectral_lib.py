#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光谱库与 UMAP 流形可视化。

图1：光谱库可视化。两行分别为 positive / negative 光谱库；
     每行内多个子图，每个子图为一条 T×c 的光谱（c 条曲线，每条长度 T）。
图2：UMAP 流形空间可视化（轨迹 + 起终点）。
图3（新）：t-SNE 二维散点，仅散点、无轨迹/起终点；正/负样本各自估计局部点密度，用不同色系的颜色深浅表示。
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image



def plot_spectral_libraries(libraries_path, out_path, n_show_per_row=6, figsize_per_subplot=(3, 2)):
    """
    图1：光谱库可视化。每行 n_show_per_row 个子图，显示全部正/负样本；每个子图为一条 T×c 光谱。
    整图共用一个图例（嵌入维度 c0, c1, ...），放在图右侧。
    """
    data = np.load(libraries_path, allow_pickle=True)
    pos_emb = data["pos_library_embeddings"]
    neg_emb = data["neg_library_embeddings"]
    c = int(data["n_components"])
    T = int(data["seq_len"])

    n_pos, n_neg = pos_emb.shape[0], neg_emb.shape[0]
    n_cols = max(1, n_show_per_row)
    n_rows_pos = (n_pos + n_cols - 1) // n_cols
    n_rows_neg = (n_neg + n_cols - 1) // n_cols
    n_rows = n_rows_pos + n_rows_neg

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False,
    )

    x = np.arange(T)
    legend_handles, legend_labels = None, None

    def draw_one(ax, emb_one):
        """在 ax 上画一条 (T, c) 光谱；带 label 供整图图例。"""
        nonlocal legend_handles, legend_labels
        for ch in range(c):
            ax.plot(x, emb_one[:, ch], label=f"c{ch}", alpha=0.8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_xlim(0, T - 1)
        ax.set_ylabel("embedding")
        ax.grid(True, alpha=0.3)

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            if row < n_rows_pos:
                idx = row * n_cols + col
                if idx < n_pos:
                    draw_one(ax, pos_emb[idx])
                    if row == 0 and col == 0:
                        ax.set_title("Positive library", fontsize=10)
                else:
                    ax.axis("off")
            else:
                r_neg = row - n_rows_pos
                idx = r_neg * n_cols + col
                if idx < n_neg:
                    draw_one(ax, neg_emb[idx])
                    if r_neg == 0 and col == 0:
                        ax.set_title("Negative library", fontsize=10)
                else:
                    ax.axis("off")
            if col > 0:
                ax.set_yticklabels([])
            ax.set_xlabel("time")

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图1 已保存: {out_path}")


def plot_umap_manifold(
    by_label_path,
    out_path,
    n_trajectories_per_class=15,
    random_state=42,
):
    """
    图2：UMAP 时序轨迹图（Trajectory/Comet Plot）。嵌入形状 (N, T, c)，保留时间结构：
    - 背景：所有样本展平为 (N*T, 2)，极低透明度 + 小点绘制（正=橙，负=蓝）。
    - 轨迹：正/负各随机抽 10–20 条，画时序连线；负样本半透明蓝线+起终点小圆点；
      正样本红色实线，起点 t=0 绿色圆点，终点红色 X。
    """
    data = np.load(by_label_path, allow_pickle=True)
    pos_emb = np.asarray(data["pos_embeddings"])  # (N_pos, T, c)
    neg_emb = np.asarray(data["neg_embeddings"])   # (N_neg, T, c)
    c = pos_emb.shape[2]
    if c < 2:
        print("UMAP 维度 c < 2，无法绘制 2D 流形，跳过图2。")
        return

    rng = np.random.default_rng(random_state)
    n_pos, n_neg = pos_emb.shape[0], neg_emb.shape[0]

    # 前两维用于 2D 平面
    pos_2d_all = pos_emb[:, :, :2]   # (N_pos, T, 2)
    neg_2d_all = neg_emb[:, :, :2]   # (N_neg, T, 2)

    # 背景：所有时间步展平，极低透明度 + 小点
    pos_flat = pos_2d_all.reshape(-1, 2)
    neg_flat = neg_2d_all.reshape(-1, 2)

    # Publication-style settings for Fig2
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    # 2cm x 2cm
    fig, ax = plt.subplots(figsize=(2.0 / 2.54, 2.0 / 2.54))
    neg_color = (0.6, 0.6, 0.6)  # gray
    pos_color = "tab:orange"

    ax.scatter(
        neg_flat[:, 0], neg_flat[:, 1],
        c=[neg_color], alpha=0.05, s=0.01, label="Negative (all points)", rasterized=True,
    )
    ax.scatter(
        pos_flat[:, 0], pos_flat[:, 1],
        c=pos_color, alpha=0.05, s=0.01, label="Positive (all points)", rasterized=True,
    )

    # 采样轨迹条数
    n_traj = min(n_trajectories_per_class, n_neg, n_pos)
    n_traj = max(1, n_traj)
    idx_neg = rng.choice(n_neg, size=n_traj, replace=False)
    idx_pos = rng.choice(n_pos, size=n_traj, replace=False)

    # 负样本轨迹：半透明蓝线，起终点小圆点
    for i in idx_neg:
        traj = neg_2d_all[i]  # (T, 2)
        t = np.arange(traj.shape[0])
        x = traj[:, 0].astype(np.float64, copy=False)
        y = traj[:, 1].astype(np.float64, copy=False)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            continue
        # Interpolate missing coordinates to keep trajectories continuous.
        xi = np.interp(t, t[finite], x[finite])
        yi = np.interp(t, t[finite], y[finite])
        ax.plot(
            xi,
            yi,
            color=neg_color,
            alpha=0.45,
            linewidth=0.9,
            zorder=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
        ax.scatter(xi[0], yi[0], c=[neg_color], s=2.5, alpha=0.7, zorder=3)
        ax.scatter(xi[-1], yi[-1], c=[neg_color], s=2.5, alpha=0.7, zorder=3)

    # 正样本轨迹：红色实线，起点绿色圆点，终点红色 X
    for i in idx_pos:
        traj = pos_2d_all[i]  # (T, 2)
        t = np.arange(traj.shape[0])
        x = traj[:, 0].astype(np.float64, copy=False)
        y = traj[:, 1].astype(np.float64, copy=False)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            continue
        xi = np.interp(t, t[finite], x[finite])
        yi = np.interp(t, t[finite], y[finite])
        ax.plot(
            xi,
            yi,
            color="black",
            alpha=0.95,
            linewidth=0.6,
            zorder=4,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
        ax.scatter(xi[0], yi[0], c="lime", s=6, edgecolors="green", linewidths=0.6, zorder=5)
        ax.scatter(xi[-1], yi[-1], marker="x", c="red", s=10, linewidths=1.6, zorder=5)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    # No title per request
    ax.set_title("")
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # 图例：背景点 + 轨迹线 + 起终点（用 Line2D 做代理，不污染主图）
    # Omit legend for 2cm canvas (too dense)

    # Remove all text (no fonts) for Fig2
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove figure margins and then do pixel-level crop to remove any leftover whitespace.
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()

    tmp_path = out_path + ".tmp.png"
    plt.savefig(tmp_path, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close()

    img = Image.open(tmp_path).convert("RGB")
    arr = np.asarray(img)
    bg = arr[0, 0].astype(np.int16)
    tol = 3  # tolerance for anti-aliased background
    mask = np.any(np.abs(arr.astype(np.int16) - bg) > tol, axis=-1)
    if np.any(mask):
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad_px = 2
        x0 = max(0, x0 - pad_px)
        y0 = max(0, y0 - pad_px)
        x1 = min(arr.shape[1] - 1, x1 + pad_px)
        y1 = min(arr.shape[0] - 1, y1 + pad_px)
        img_cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
        img_cropped.save(out_path)
    else:
        img.save(out_path)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    print(f"图2 已保存: {out_path} (轨迹: 正/负各 {n_traj} 条)")


def plot_tsne_density_scatter(
    by_label_path,
    out_path,
    *,
    max_samples: int = 20000,
    perplexity: int = 30,
    density_neighbors: int = 30,
    random_state: int = 42,
    point_size: float = 1.0,
    figsize: tuple = (4.0 / 2.54, 4.0 / 2.54),
    dpi: int = 150,
):
    """
    图3：对 by_label 中的 pos/neg 嵌入做 t-SNE（联合降维），再分别估计各类内部的局部点密度，
    用散点颜色深浅绘制（正样本偏橙、负样本灰阶）；无轨迹、无 beginning/end 标记；
    无 colorbar、无标题；坐标轴固定为 [-120,120]（以 0 为中心），x/y 各 3 个刻度 -120, 0, 120。
    """
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors

    data = np.load(by_label_path, allow_pickle=True)
    pos_emb = np.asarray(data["pos_embeddings"], dtype=np.float64)  # (N_pos, T, c)
    neg_emb = np.asarray(data["neg_embeddings"], dtype=np.float64)  # (N_neg, T, c)
    if pos_emb.size == 0 and neg_emb.size == 0:
        print("图3：无正负嵌入，跳过 t-SNE 密度图。")
        return
    if pos_emb.size == 0 or neg_emb.size == 0:
        print("图3：仅有一类样本，仍做 t-SNE，但密度仅在非空类上计算。")

    def _flat(X):
        if X.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return X.reshape(X.shape[0], -1)

    X_pos = _flat(pos_emb)
    X_neg = _flat(neg_emb)
    parts = []
    labels = []
    if X_pos.shape[0] > 0:
        parts.append(X_pos)
        labels.append(np.ones(X_pos.shape[0], dtype=np.int8))
    if X_neg.shape[0] > 0:
        parts.append(X_neg)
        labels.append(np.zeros(X_neg.shape[0], dtype=np.int8))
    X_all = np.vstack(parts)
    y_all = np.concatenate(labels)
    N = X_all.shape[0]

    rng = np.random.default_rng(random_state)
    if N > max_samples:
        sub = rng.choice(N, size=max_samples, replace=False)
        X_all = X_all[sub]
        y_all = y_all[sub]
        N = max_samples
        print(f"图3：子采样至 N={N} 以控制 t-SNE 耗时。")

    perp = int(min(max(5, perplexity), N - 1))
    tsne_kw = dict(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
    )
    try:
        tsne = TSNE(**tsne_kw, learning_rate="auto", n_jobs=-1)
    except TypeError:
        try:
            tsne = TSNE(**tsne_kw, n_jobs=-1)
        except TypeError:
            tsne = TSNE(**tsne_kw)
    emb = tsne.fit_transform(X_all)

    pos_m = y_all == 1
    neg_m = y_all == 0
    pos_xy = emb[pos_m]
    neg_xy = emb[neg_m]

    def _local_density(xy: np.ndarray) -> np.ndarray:
        n = xy.shape[0]
        if n == 0:
            return np.array([])
        k = int(min(max(2, density_neighbors), n))
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(xy)
        dists, _ = nn.kneighbors(xy)
        # 用第 k-1 邻距的倒数作局部密度（距离越小密度越大）
        rho = 1.0 / (dists[:, -1] + 1e-9)
        rmin, rmax = float(rho.min()), float(rho.max())
        if rmax - rmin < 1e-12:
            return np.ones_like(rho)
        return (rho - rmin) / (rmax - rmin)

    dens_pos = _local_density(pos_xy)
    dens_neg = _local_density(neg_xy)

    with plt.rc_context(
        {
            "axes.labelsize": 7,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.titlesize": 7,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    if neg_xy.shape[0] > 0:
        ax.scatter(
            neg_xy[:, 0],
            neg_xy[:, 1],
            c=dens_neg,
            cmap="Greys",
            s=point_size,
            alpha=0.85,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            vmin=0.0,
            vmax=1.0,
            label="Negative",
        )
    if pos_xy.shape[0] > 0:
        ax.scatter(
            pos_xy[:, 0],
            pos_xy[:, 1],
            c=dens_pos,
            cmap="Oranges",
            s=point_size,
            alpha=0.85,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            vmin=0.0,
            vmax=1.0,
            label="Positive",
            zorder=2,
        )
    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=5)
    ax.grid(False)
    plt.tight_layout()
    # 固定视窗：以 0 为中心 [-120,120]；须在 tight_layout 之后再设，否则会被自动缩放覆盖
    lim = 120.0
    tick3 = np.linspace(-lim, lim, 3)
    ax.set_autoscale_on(False)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(tick3)
    ax.set_yticks(tick3)
    ax.set_aspect("equal", adjustable="box")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"图3 已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="光谱库与 t-SNE 可视化")
    parser.add_argument(
        "--libraries_npz",
        type=str,
        default='./spectral_lib_checkpoints/umap_dps_seq10_libraries_by_label.npz',
        help="光谱库 npz 路径（*_libraries_by_label.npz）",
    )
    parser.add_argument(
        "--by_label_npz",
        type=str,
        nargs="?",
        default='./spectral_lib_checkpoints/umap_dps_seq10_by_label.npz',
        help="全量嵌入 npz 路径（*_by_label.npz），用于 UMAP 流形图；不传则与 libraries 同目录同前缀推断",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default='./visualize_spectral_lib_results',
        help="输出目录，默认与 libraries_npz 同目录",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=6,
        help="图1 每行显示的光谱库样本数",
    )
    parser.add_argument(
        "--umap-n-trajectories",
        type=int,
        default=15,
        help="图2 时序轨迹图每类绘制的轨迹条数（正/负各随机抽 10–20 条）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--tsne-max-samples",
        type=int,
        default=20000,
        help="图3 t-SNE 最大样本数（超过则随机子采样）",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="图3 t-SNE perplexity",
    )
    parser.add_argument(
        "--tsne-density-k",
        type=int,
        default=30,
        help="图3 各类内部局部密度：K 近邻（用于颜色深浅）",
    )
    args = parser.parse_args()

    libraries_path = os.path.abspath(args.libraries_npz)
    if not os.path.isfile(libraries_path):
        print(f"文件不存在: {libraries_path}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(libraries_path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(libraries_path))[0]
    if base.endswith("_libraries_by_label"):
        base = base.replace("_libraries_by_label", "")

    by_label_path = args.by_label_npz
    if by_label_path is None:
        by_label_path = os.path.join(os.path.dirname(libraries_path), base + "_by_label.npz")
    if not os.path.isfile(by_label_path):
        by_label_path = None

    # 图1：光谱库
    fig1_path = os.path.join(out_dir, f"{base}_fig1_spectral_libraries.png")
    plot_spectral_libraries(libraries_path, fig1_path, n_show_per_row=args.n_show)

    # 图2：UMAP 流形空间
    if by_label_path:
        fig2_path = os.path.join(out_dir, f"{base}_fig2_umap_manifold.png")
        plot_umap_manifold(
            by_label_path,
            fig2_path,
            n_trajectories_per_class=args.umap_n_trajectories,
            random_state=args.seed,
        )
    else:
        print("未找到 by_label npz，跳过 UMAP 流形图。")

    # 图3：t-SNE 密度散点（新图，不影响图2）
    if by_label_path:
        fig3_path = os.path.join(out_dir, f"{base}_fig3_tsne_density_scatter.png")
        plot_tsne_density_scatter(
            by_label_path,
            fig3_path,
            max_samples=args.tsne_max_samples,
            perplexity=args.tsne_perplexity,
            density_neighbors=args.tsne_density_k,
            random_state=args.seed,
        )
    else:
        print("未找到 by_label npz，跳过图3 t-SNE 密度散点。")

    print("完成。")


if __name__ == "__main__":
    main()
