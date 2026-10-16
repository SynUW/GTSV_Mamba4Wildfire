"""GTSV-Mamba (final model of the paper), self-contained.

Three independent branches, one local and two graph-context branches ("lgg"):
  local branch          Variable-Temporal-Spatial Mamba backbone + seasonal date encoding + head
  graph-context branch  the same backbone, plus an ecoclimatic and a fire-inertia retrieval graph
                        over a fixed, labelled library of training samples
Training routes each sample's loss to one branch drawn uniformly at random (independent
subnetworks within one model, Havasi et al., ICLR 2021); evaluation averages the branch logits.
Each branch keeps an exponential moving average of its weights (decay 0.998), used at evaluation.

Backbone (per pixel, then per patch)
  variable stage  per-day feature tokens (FT-Transformer tokenizer), self-attention across variables
                  within each day, pooled per day and added to the day tokens
  temporal stage  day tokens with a learned step embedding, bidirectional Mamba (Vim bimamba-v2)
  spatial stage   row and column bidirectional Mamba scans over the patch (VMamba cross scan)

Retrieval graphs (TabR-style values, Gorishniy et al., ICLR 2024)
  eco   neighbours by the squared distance of learned keys of the central-cell trajectory
  hist  neighbours by the generalized Tanimoto coefficient of causal fire-history descriptors,
        ties broken by day-of-year proximity, softmax with a learned temperature
  For both, library samples from the target's year are masked, the n = 16 most similar are kept,
  and the message of neighbour j is E(y_j) + phi(k_i - k_j).

Inputs
  x_enc         (B, T=10, V=38, 13, 13)  z-scored inputs from model_adapter_unified (FIRMS channel 0 zeroed)
  x_mark_enc    (B, T, 5)                time marks, [..., 0] = (year - 2000) / 24, [..., -1] = (doy - 1) / 365
  burn_history  (B, 1, 13, 13, 7671)     daily FIRMS detections 2000-01-01 .. 2020-12-31 of the patch
Output
  logits        (B, 1)

The library is an .npz with arrays env (N, 380), regime (N, 30), label (N,), year (N,), doy (N,)
computed on the 2000-2020 training rows with the same preprocessing as the model inputs
(see `library_descriptors`). Pass its path as `memory_path` or `configs.gtsv_memory`.
Requires mamba_ssm.
"""
import datetime as dt
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mamba_ssm import Mamba

HISTORY_START = dt.date(2000, 1, 1)
HISTORY_DAYS = 7671          # 2000-01-01 .. 2020-12-31
HISTORY_YEARS = 21
CENTER = 6

FINAL_CONFIG = dict(branches="lgg", d=64, depth=2, k=16, drop_path=0.1, ema=0.998,
                    steps=10, channels=38, height=13, width=13)


# --------------------------------------------------------------------------- building blocks

def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return drop_path(x, self.p, self.training)


class BiMamba(nn.Module):
    """Vim bimamba-v2: forward and reversed scans with separate selective SSMs, averaged."""

    def __init__(self, d, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd = Mamba(d_model=d, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return 0.5 * (self.fwd(x) + self.bwd(x.flip(1)).flip(1))


class CrossScan2D(nn.Module):
    """VMamba scans=0: row-major, column-major and both reversals, merged by averaging."""

    def __init__(self, d, height, width):
        super().__init__()
        self.height, self.width = height, width
        self.row = BiMamba(d)
        self.col = BiMamba(d)

    def forward(self, x):
        n, length, d = x.shape
        row = self.row(x)
        grid = x.reshape(n, self.height, self.width, d)
        col = self.col(grid.transpose(1, 2).reshape(n, length, d).contiguous())
        col = col.reshape(n, self.width, self.height, d).transpose(1, 2).reshape(n, length, d)
        return 0.5 * (row + col)


class SetAttention(nn.Module):
    """Self-attention over an unordered token set (variables), as in iTransformer."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


class MixerBlock(nn.Module):
    """Pre-norm residual block: x + DropPath(mixer(LN x)), then x + DropPath(MLP(LN x))."""

    def __init__(self, d, mixer, drop=0.1, dropout=0.1, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_ratio * d, d))
        self.drop = DropPath(drop)

    def forward(self, x):
        x = x + self.drop(self.mixer(self.norm1(x).contiguous()))
        return x + self.drop(self.mlp(self.norm2(x)))


# --------------------------------------------------------------------------- backbone

class Backbone(nn.Module):
    """Variable-Temporal-Spatial Mamba backbone (paper Sec. 4.1.1)."""

    def __init__(self, d, depth, drop, steps=10, channels=38, height=13, width=13, act_ckpt=True):
        super().__init__()
        self.act_ckpt = act_ckpt
        self.pixel_embed = nn.Linear(steps * channels, d)
        # temporal stage
        self.day_embed = nn.Linear(channels, d)
        self.day_pos = nn.Parameter(torch.zeros(steps, d))
        nn.init.trunc_normal_(self.day_pos, std=0.02)
        self.t_blocks = nn.ModuleList([MixerBlock(d, BiMamba(d), drop) for _ in range(depth)])
        self.t_norm = nn.LayerNorm(d)
        self.t_out = nn.Linear(2 * d, d)
        # variable stage (per-day feature tokens)
        self.feat_weight = nn.Parameter(torch.empty(channels, d))
        self.feat_bias = nn.Parameter(torch.zeros(channels, d))
        nn.init.trunc_normal_(self.feat_weight, std=0.02)
        self.v_blocks = nn.ModuleList([MixerBlock(d, SetAttention(d), drop) for _ in range(depth)])
        self.v_norm = nn.LayerNorm(d)
        self.v_out = nn.Linear(d, d)
        # spatial stage
        self.s_blocks = nn.ModuleList([MixerBlock(d, CrossScan2D(d, height, width), drop) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d)
        self.readout = nn.Linear(2 * d, d)

    def forward(self, x):
        b, steps, channels, height, width = x.shape
        length = height * width
        pixel = x.permute(0, 3, 4, 1, 2).reshape(b * length, steps, channels)      # (N, T, V)
        # variable stage: one token per factor and day, attention across factors within each day
        tokens = pixel.unsqueeze(-1) * self.feat_weight + self.feat_bias            # (N, T, V, d)
        z = tokens.reshape(-1, channels, tokens.shape[-1])
        for block in self.v_blocks:  # activation checkpointing only saves memory, outputs are unchanged
            z = (torch.utils.checkpoint.checkpoint(block, z, use_reentrant=False)
                 if self.training and self.act_ckpt else block(z))
        day_context = self.v_out(self.v_norm(z).mean(1)).reshape(pixel.shape[0], steps, -1)
        # temporal stage
        p = self.pixel_embed(pixel.reshape(b * length, steps * channels))
        z = self.day_embed(pixel) + self.day_pos + day_context
        for block in self.t_blocks:
            z = block(z)
        z = self.t_norm(z)
        p = p + self.t_out(torch.cat([z[:, -1], z.mean(1)], -1))
        # spatial stage
        p = p.reshape(b, length, -1)
        for block in self.s_blocks:
            p = block(p)
        p = self.out_norm(p)
        return self.readout(torch.cat([p[:, length // 2], p.mean(1)], -1))


# --------------------------------------------------------------------------- retrieval graphs

class FireRegime(nn.Module):
    """Causal fire-history descriptor r (Q = 30) from complete years strictly before the query year.

    Per spatial scale (13x13 patch, central 3x3): 12 monthly rates, a +-15 day seasonal
    rate around the query day of year, the annual rate and the fraction of years with
    any detection. Rates are pixel-detections per observed year, log1p transformed.
    """

    def __init__(self):
        super().__init__()
        days = [HISTORY_START + dt.timedelta(days=i) for i in range(HISTORY_DAYS)]
        year = torch.tensor([d.year for d in days])
        month = torch.tensor([d.month - 1 for d in days])
        self.register_buffer("day_year", year, persistent=False)
        self.register_buffer("day_doy", torch.tensor([d.timetuple().tm_yday for d in days]), persistent=False)
        self.register_buffer("month_onehot", F.one_hot(month, 12).float(), persistent=False)
        self.register_buffer("year_onehot", F.one_hot(year - 2000, HISTORY_YEARS).float(), persistent=False)

    def forward(self, burn, year, doy):
        if burn.ndim == 5:
            burn = burn[:, 0]
        if burn.shape[-1] != HISTORY_DAYS:
            raise ValueError(f"Unexpected burn history {tuple(burn.shape)}")
        events = (burn > 0).float()
        valid = (self.day_year[None] < year[:, None]).float()
        years = (year - 2000).clamp(0, HISTORY_YEARS).float().clamp(min=1.0)
        gap = (self.day_doy[None] - doy[:, None]).abs()
        season = (torch.minimum(gap, 365 - gap) <= 15).float()
        features = []
        for series in (events.sum((1, 2)), events[:, CENTER - 1:CENTER + 2, CENTER - 1:CENTER + 2].sum((1, 2))):
            series = series * valid
            monthly = series @ self.month_onehot / years[:, None]
            seasonal = (series * season).sum(1) / years
            annual = series.sum(1) / years
            any_year = (series @ self.year_onehot > 0).float().sum(1) / years
            features += [torch.log1p(monthly), torch.log1p(seasonal)[:, None],
                         torch.log1p(annual)[:, None], any_year[:, None]]
        return torch.cat(features, 1)


def clean_input(x_enc):
    return torch.nan_to_num(x_enc.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-10, 10)


def query_descriptors(regime, x_enc, x_mark_enc, burn_history):
    """Central-cell trajectory e, fire-history descriptor r, year and day of year of each sample."""
    year = (x_mark_enc[:, -1, 0] * 24 + 2000).round().long()
    doy = (x_mark_enc[:, -1, -1] * 365 + 1).round().long()
    env = x_enc[:, :, :, CENTER, CENTER].reshape(x_enc.shape[0], -1)
    return env, regime(burn_history, year, doy), year, doy


@torch.no_grad()
def library_descriptors(x_enc, x_mark_enc, burn_history):
    """Library entries for training samples, preprocessed exactly as the model inputs.

    Concatenate over all 2000-2020 training rows together with the labels y and save
    np.savez(path, env=..., regime=..., label=..., year=..., doy=...).
    """
    regime = FireRegime().to(x_enc.device)
    env, reg, year, doy = query_descriptors(regime, clean_input(x_enc), x_mark_enc, burn_history.float())
    return dict(env=env, regime=reg, year=year, doy=doy)


def generalized_tanimoto(q, m):
    """Tanimoto coefficient for non-negative vectors; two all-zero vectors are identical (1)."""
    dot = q @ m.T
    denom = q.square().sum(1, keepdim=True) + m.square().sum(1)[None] - dot
    both_zero = denom <= 1e-12
    return torch.where(both_zero, torch.ones_like(dot), dot / denom.clamp(min=1e-12))


class RetrievalGraph(nn.Module):
    """Bipartite target -> library graph with values E(y_j) + phi(k_i - k_j) (paper Secs. 4.1.2-4.1.4).

    kind="fire": fire-inertia graph, neighbours by the Tanimoto coefficient of fire-history
                 descriptors, ties broken by day-of-year proximity; adds a projection of the own key.
    kind="env":  ecoclimatic graph, neighbours by the squared distance of learned keys of the
                 central-cell trajectory.
    Library samples from the target year are always masked, so training never retrieves the
    target itself or same-season events, and test targets (2021-2024) see the full library.
    """

    def __init__(self, kind, in_dim, d, k, dropout=0.1):
        super().__init__()
        self.kind, self.k = kind, k
        self.norm = nn.LayerNorm(in_dim)
        self.key = nn.Linear(in_dim, d)
        self.label = nn.Embedding(2, d)                                        # E(0), E(1)
        self.T = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(2 * d, d, bias=False))               # phi
        self.weight_dropout = nn.Dropout(dropout)
        if kind == "fire":
            self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
            self.self_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, q_raw, q_year, q_doy, m_raw, m_y, m_year, m_doy):
        kq = self.key(self.norm(q_raw))
        km = self.key(self.norm(m_raw))
        if self.kind == "fire":
            similarity = generalized_tanimoto(q_raw, m_raw)
            season = torch.cos(2 * math.pi * (q_doy[:, None] - m_doy[None]).float() / 365.0)
            select = similarity + 1e-3 * season
            score = similarity / self.log_temperature.exp()
        else:
            score = -(kq.square().sum(1, keepdim=True) - 2 * kq @ km.T + km.square().sum(1)[None])
            select = score
        select = select.masked_fill(m_year[None] == q_year[:, None], float("-inf"))
        _, index = select.topk(self.k, dim=1)                                  # neighbour set N_i
        weights = self.weight_dropout(torch.softmax(score.gather(1, index), dim=1))   # a_ij
        values = self.label(m_y[index]) + self.T(kq[:, None] - km[index])
        context = (weights[..., None] * values).sum(1)
        if self.kind == "fire":
            context = context + self.self_proj(kq)
        return context


# --------------------------------------------------------------------------- one branch

class Branch(nn.Module):
    """One complete GTSV network; with_graphs=False gives the local branch."""

    def __init__(self, cfg, library=None, with_graphs=True):
        super().__init__()
        d = int(cfg["d"])
        self.backbone = Backbone(d, int(cfg["depth"]), float(cfg["drop_path"]), cfg["steps"], cfg["channels"],
                                 cfg["height"], cfg["width"], act_ckpt=cfg.get("act_ckpt", True))
        self.time = nn.Linear(4, d)
        self.regime = FireRegime()
        self.graphs = "bm" if with_graphs else ""
        if self.graphs:
            for name in ("env", "regime", "y", "year", "doy"):
                self.register_buffer("mem_" + name, library[name])
            self.fire_graph = RetrievalGraph("fire", self.mem_regime.shape[1], d, int(cfg["k"]))
            self.env_graph = RetrievalGraph("env", self.mem_env.shape[1], d, int(cfg["k"]))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.3), nn.Linear(d, 1))
        # Weight EMA (timm ModelEmaV2); shadow weights are buffers so they are saved in checkpoints.
        self.ema_decay = float(cfg["ema"])
        self._ema_call = False
        if self.ema_decay > 0:
            for name, param in self.named_parameters():
                self.register_buffer("ema__" + name.replace(".", "__"), param.detach().clone())
            self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self):
        steps = float(self.ema_steps)
        decay = min(self.ema_decay, (1.0 + steps) / (10.0 + steps))
        for name, param in self.named_parameters():
            getattr(self, "ema__" + name.replace(".", "__")).mul_(decay).add_(param.detach(), alpha=1 - decay)
        self.ema_steps += 1

    def forward(self, x_enc, x_mark_enc, burn_history=None):
        if self.ema_decay > 0 and not self._ema_call:
            if self.training:
                self._ema_update()  # shadow follows the weights left by the previous optimiser step
            else:
                weights = {n: getattr(self, "ema__" + n.replace(".", "__")) for n, _ in self.named_parameters()}
                self._ema_call = True
                try:
                    return torch.func.functional_call(self, weights, (x_enc, x_mark_enc), dict(burn_history=burn_history))
                finally:
                    self._ema_call = False
        x_enc = clean_input(x_enc)
        doy = (x_mark_enc[:, -1, -1] * 365 + 1).round()
        angle = 2 * math.pi * (doy - 1) / 365.0
        h = self.backbone(x_enc) + self.time(torch.stack(
            [angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos()], 1))   # seasonal date encoding d
        if self.graphs:
            if burn_history is None:
                raise ValueError("Retrieval graphs need burn_history")
            env, regime, year, qdoy = query_descriptors(self.regime, x_enc, x_mark_enc, burn_history.float())
            h = h + self.fire_graph(regime, year, qdoy, self.mem_regime, self.mem_y, self.mem_year, self.mem_doy)
            h = h + self.env_graph(env, year, qdoy, self.mem_env, self.mem_y, self.mem_year, self.mem_doy)
        return self.head(h)


# --------------------------------------------------------------------------- full model

def load_library(path):
    memory = np.load(path)
    return dict(env=torch.from_numpy(memory["env"]).float(), regime=torch.from_numpy(memory["regime"]).float(),
                y=torch.from_numpy(memory["label"]).long(), year=torch.from_numpy(memory["year"]).long(),
                doy=torch.from_numpy(memory["doy"]).long())


class Model(nn.Module):
    """GTSV-Mamba, one local and two graph-context branches (1,889,773 parameters)."""

    def __init__(self, configs=None, memory_path=None, **overrides):
        super().__init__()
        cfg = dict(FINAL_CONFIG, **overrides)
        memory_path = memory_path or getattr(configs, "gtsv_memory", None)
        library = load_library(memory_path) if "g" in cfg["branches"] else None
        if "g" in cfg["branches"] and memory_path is None:
            raise ValueError("Graph-context branches need the library .npz (memory_path or configs.gtsv_memory)")
        self.branches = nn.ModuleList([Branch(cfg, library, with_graphs=(kind == "g")) for kind in cfg["branches"]])

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, burn_history=None):
        logits = torch.stack([b(x_enc, x_mark_enc, burn_history=burn_history) for b in self.branches])  # (K, B, 1)
        if self.training:
            pick = torch.randint(len(self.branches), (logits.shape[1],), device=logits.device)
            return logits[pick, torch.arange(logits.shape[1], device=logits.device)]
        return logits.mean(0)
