#!/usr/bin/env python3
"""Importance-weight diagnostics.

Two figures per dataset:

  iw_vs_size:    p_approx(c|q)/p_exact(c|q) vs cluster size n_c, for 2LS and S-2LS.
                 Motivates the size correction.
  iw_vs_dirvar:  same y, x = q^T Σ_c q (q-dependent dispersion), for S-2LS and SD-2LS.
                 Motivates the dispersion correction.

Both aggregate IW per cluster by averaging over NQ queries.

Usage:
    python plot_iw.py [dataset...]
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logsumexp
import os
import sys
from data import load_dataset
from plot_style import (
    METHOD_COLORS, METHOD_LABELS, DATASET_LABELS,
    FS_SUPTITLE, FS_TITLE, FS_AXLABEL, FS_TICK,
)

sns.set_theme(style="whitegrid", font_scale=1.1)

C = 1024
TAU = 0.1
NQ = 50

# Y-axis label uses paper notation R_i^{(N)}(q). Items in a cluster share the
# cluster's IW; we show one dot per cluster.
IW_LABEL = r"$R_i^{(N)}(\mathbf{q})$"


def precompute(ds_name):
    """Returns dict of arrays needed by both IW plots."""
    ds = load_dataset(ds_name, n_queries=NQ)
    V = ds["vectors"].astype(np.float64)
    Q = ds["queries"].astype(np.float64)
    qidx = ds["query_indices"]
    cl = np.load(f"clusters/{ds_name}/C{C}.npz")
    means = cl["centroids"].astype(np.float64)
    asgn = cl["assignments"].astype(np.int32)[:len(V)]
    counts = cl["counts"].astype(np.int64)

    log_counts = np.log(np.maximum(counts, 1).astype(np.float64))

    # Per-query, per-cluster true log Z_c (with self masked)
    sims = V @ Q.T  # (N, NQ)
    sims_q_masked = sims.copy()
    for i, qi in enumerate(qidx):
        sims_q_masked[qi, i] = -np.inf

    true_logZc = np.full((NQ, C), -np.inf, dtype=np.float64)
    for i in range(NQ):
        for c in range(C):
            idx = np.where(asgn == c)[0]
            if len(idx) == 0:
                continue
            true_logZc[i, c] = logsumexp(sims_q_masked[idx, i] / TAU)

    # q^T Σ_c q for all (q, c). Pre-compute Σ_c flat once.
    d = V.shape[1]
    covs_flat = np.zeros((C, d * d), dtype=np.float64)
    for c in range(C):
        idx = np.where(asgn == c)[0]
        if len(idx) < 2:
            continue
        block = V[idx]
        m = block.mean(axis=0)
        r = block - m
        covs_flat[c] = ((r.T @ r) / len(idx)).ravel()
    qq_flat = np.einsum("qi,qj->qij", Q, Q).reshape(NQ, d * d)
    qcq = qq_flat @ covs_flat.T  # (NQ, C)

    centroid_scores = (Q @ means.T) / TAU  # (NQ, C)

    return dict(
        means=means, asgn=asgn, counts=counts, log_counts=log_counts,
        true_logZc=true_logZc, qcq=qcq, centroid_scores=centroid_scores,
    )


def compute_iw(method, pre):
    """Returns (C,) array: IW averaged over queries, for given method."""
    cs = pre["centroid_scores"]
    lc = pre["log_counts"][None, :]
    if method == "centroid":
        log_wc = cs
    elif method == "ivf0":
        log_wc = lc + cs
    elif method == "ivf2":
        log_wc = lc + cs + pre["qcq"] / (2 * TAU ** 2)
    else:
        raise ValueError(method)
    log_p_approx = log_wc - logsumexp(log_wc, axis=1, keepdims=True)
    log_p_exact = pre["true_logZc"] - logsumexp(pre["true_logZc"], axis=1, keepdims=True)
    iw = np.exp(log_p_approx - log_p_exact)
    return iw.mean(axis=0)


def plot_panel(ax, x, y, color, label, xlabel, valid, ymin, ymax):
    """One scatter panel with IW=1 reference and shaded over/undersample regions."""
    xv, yv = x[valid], y[valid]
    xlo, xhi = xv.min(), xv.max()
    # Shading: over the IW=1 line is oversampled (red), under is undersampled (green).
    ax.fill_between([xlo, xhi], 1.0, ymax, color="#E74C3C", alpha=0.06, zorder=0)
    ax.fill_between([xlo, xhi], ymin, 1.0, color="#27AE60", alpha=0.06, zorder=0)
    ax.scatter(xv, yv, s=15, alpha=0.55, color=color, edgecolors="none", zorder=2)
    ax.axhline(1.0, color="grey", lw=1.0, alpha=0.7, zorder=1)
    ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FS_AXLABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.text(0.02, 0.98, "oversampled", transform=ax.transAxes,
            ha="left", va="top", fontsize=18, color="#C0392B", alpha=0.45)
    ax.text(0.02, 0.02, "undersampled", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=18, color="#1E8449", alpha=0.45)


METHODS = ["centroid", "ivf0", "ivf2"]
METHOD_TAGS = {"centroid": "2ls", "ivf0": "s2ls", "ivf2": "sd2ls"}


def _save_single(ds_name, method_key, x, y, valid, xlabel, kind, ymin, ymax):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    plot_panel(ax, x, y, METHOD_COLORS[method_key], METHOD_LABELS[method_key],
               xlabel, valid, ymin, ymax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(IW_LABEL, fontsize=FS_AXLABEL)
    plt.tight_layout()
    out = f"results/figures/{ds_name}/iw_{kind}_{METHOD_TAGS[method_key]}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    print(f"Saved {out}")
    plt.close()


def _shared_ylim(iws, valid):
    lo = min(iw[valid].min() for iw in iws)
    hi = max(iw[valid].max() for iw in iws)
    return lo * 0.8, hi * 1.25


def plot_iw_vs_size(ds_name):
    pre = precompute(ds_name)
    counts = pre["counts"]
    valid = counts > 0
    iws = {m: compute_iw(m, pre) for m in METHODS}
    ymin, ymax = _shared_ylim(list(iws.values()), valid)
    for m in METHODS:
        _save_single(ds_name, m, counts, iws[m], valid, "Cluster size", "size", ymin, ymax)


def plot_iw_vs_dirvar(ds_name):
    pre = precompute(ds_name)
    counts = pre["counts"]
    qcq_mean = pre["qcq"].mean(axis=0)
    valid = counts > 0
    iws = {m: compute_iw(m, pre) for m in METHODS}
    ymin, ymax = _shared_ylim(list(iws.values()), valid)
    for m in METHODS:
        _save_single(ds_name, m, qcq_mean, iws[m], valid,
                     r"$\sigma_k^2(\mathbf{q})$", "dirvar", ymin, ymax)


if __name__ == "__main__":
    datasets = sys.argv[1:] or list(DATASET_LABELS.keys())
    which = os.environ.get("IW_PLOT", "both")
    for ds in datasets:
        print(f"\n=== {ds} ===")
        if which in ("size", "both"):
            plot_iw_vs_size(ds)
        if which in ("dirvar", "both"):
            plot_iw_vs_dirvar(ds)
