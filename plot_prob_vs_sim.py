#!/usr/bin/env python3
"""Plot p(i|q) vs q^T x_i to show that naive two-level softmax
assigns wildly different probabilities to items with the same similarity,
and that S-2LS / SD-2LS correct this.

Two panels per figure: high-similarity zoom (top 1%) and full range.
One figure per method.
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
    FS_SUPTITLE, FS_TITLE, FS_AXLABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
)

sns.set_theme(style="whitegrid", font_scale=1.1)

C = 1024
TAU = 0.1
NQ = 20
QUERY_IDX = {"glove-100": 15}  # default 0
N_PLOT = 200_000
RANK = 10
HSM_LEAF = 1024


# ---------------------------------------------------------------------------
# On-the-fly precompute (replaces old cache files: sims_nq20, true_logZc, means)
# ---------------------------------------------------------------------------

def precompute(ds_name, qi):
    """Returns (q, sim, asgn, counts, means, true_logZc_q) for a single query."""
    ds = load_dataset(ds_name, n_queries=NQ)
    V = ds["vectors"]
    q = ds["queries"][qi].astype(np.float64)
    qidx_global = int(ds["query_indices"][qi])

    cl = np.load(f"clusters/{ds_name}/C{C}.npz")
    means = cl["centroids"].astype(np.float64)
    asgn = cl["assignments"].astype(np.int32)[:len(V)]
    counts = cl["counts"].astype(np.int64)

    sim = V.astype(np.float64) @ q
    sim_masked = sim.copy()
    sim_masked[qidx_global] = -np.inf

    true_logZc_q = np.full(C, -np.inf, dtype=np.float64)
    for c in range(C):
        idx = np.where(asgn == c)[0]
        if len(idx) == 0:
            continue
        lo = sim_masked[idx] / TAU
        if np.all(np.isneginf(lo)):
            continue
        true_logZc_q[c] = logsumexp(lo)

    return q, qidx_global, sim, sim_masked, asgn, counts, means, true_logZc_q


def compute_exact_probs(sim_masked):
    logits = sim_masked / TAU
    return np.exp(logits - logsumexp(logits))


def compute_approx_probs(sim_masked, asgn, log_wc, true_logZc_q, qidx_global):
    """Two-level softmax probs given arbitrary cluster log-weights."""
    log_p_cluster = log_wc - logsumexp(log_wc)
    log_p = log_p_cluster[asgn] + (sim_masked / TAU - true_logZc_q[asgn])
    p = np.exp(log_p)
    p[qidx_global] = 0.0
    return p


def compute_cluster_log_weights(method, q, means, counts, dir_var_q=None):
    centroid_scores = q @ means.T / TAU
    valid = np.isfinite(centroid_scores)
    centroid_scores[~valid] = -np.inf
    log_counts = np.log(np.maximum(counts, 1).astype(np.float64))

    if method == "centroid":
        return centroid_scores
    if method == "ivf0":
        return log_counts + centroid_scores
    if method == "ivf2":
        return log_counts + centroid_scores + dir_var_q / (2 * TAU ** 2)
    raise ValueError(method)


def compute_dir_var(q, V, asgn, counts):
    """q^T Σ_c q with rank-RANK projection (cheap)."""
    d = V.shape[1]
    Vd = V.astype(np.float64)
    cluster_sums = np.zeros((C, d), dtype=np.float64)
    np.add.at(cluster_sums, asgn, Vd)
    safe_counts = np.maximum(counts, 1).astype(np.float64)
    cmeans = cluster_sums / safe_counts[:, None]
    residuals = Vd - cmeans[asgn]
    pooled_cov = (residuals.T @ residuals) / len(Vd)
    eigvals, eigvecs = np.linalg.eigh(pooled_cov)
    basis = eigvecs[:, -RANK:]
    q_proj = basis.T @ q
    q_proj_sq = q_proj ** 2
    projected = residuals @ basis
    projected_sq = projected ** 2
    cluster_sum_sq = np.zeros((C, RANK), dtype=np.float64)
    np.add.at(cluster_sum_sq, asgn, projected_sq)
    variances = cluster_sum_sq / safe_counts[:, None]
    return variances @ q_proj_sq  # (C,)


# ---------------------------------------------------------------------------
# Plot helpers (preserved from the original)
# ---------------------------------------------------------------------------

def _plot_panel(ax, sim, p_exact, p_approx, idx, method_name, color, xlim,
                point_size=3, kl_div=None):
    order_idx = np.argsort(sim[idx])
    ax.plot(sim[idx][order_idx], p_exact[idx][order_idx],
            color="black", lw=2.2, zorder=2, label="Exact softmax")

    ax.scatter(sim[idx], p_approx[idx],
               s=point_size, alpha=0.45, color=color, edgecolors="none",
               rasterized=True, zorder=1, label=method_name)

    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$\mathbf{q}^\top \mathbf{x}_i$", fontsize=FS_AXLABEL)
    ax.set_ylabel(r"$p(i \mid \mathbf{q})$", fontsize=FS_AXLABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)

    ax.legend(loc="upper right", fontsize=FS_LEGEND, markerscale=4,
              framealpha=0.9)


def plot_prob_vs_sim_overlay(ds_name, method_key, p_approx, p_exact, sim,
                             idx_full, xlim_full, ylim_full):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    label = METHOD_LABELS[method_key]
    color = METHOD_COLORS[method_key]

    _plot_panel(ax, sim, p_exact, p_approx, idx_full, label, color,
                xlim=xlim_full)
    if ylim_full:
        ax.set_ylim(*ylim_full)
    ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")

    plt.tight_layout()
    method_tag = {"centroid": "2ls", "ivf0": "s2ls", "ivf2": "sd2ls"}[method_key]
    out = f"results/figures/{ds_name}/prob_vs_sim_{method_tag}.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    print(f"Saved {out}")
    plt.close()


METHODS = ["centroid", "ivf0", "ivf2"]  # 2LS (intro), S-2LS, SD-2LS — no Hierarchical here


def plot_prob_vs_sim(ds_name, method_filter=None):
    qi = QUERY_IDX.get(ds_name, 0)
    q, qidx_global, sim, sim_masked, asgn, counts, means, true_logZc_q = precompute(ds_name, qi)
    p_exact = compute_exact_probs(sim_masked)

    rng = np.random.RandomState(0)
    pos_indices = np.where(sim > sim.min())[0]
    pos_indices = pos_indices[pos_indices != qidx_global]
    idx_full = rng.choice(pos_indices, size=min(N_PLOT, len(pos_indices)), replace=False)

    methods = method_filter or METHODS

    # Need vectors only if hierarchical or ivf2
    V = None
    dir_var_q = None
    hsm = None
    if "ivf2" in methods or "hierarchical" in methods:
        ds = load_dataset(ds_name, n_queries=NQ)
        V = ds["vectors"]
    if "ivf2" in methods:
        print("  Computing directional variances...")
        dir_var_q = compute_dir_var(q, V, asgn, counts)
    if "hierarchical" in methods:
        from ivf_softmax import HierarchicalSoftmax
        cache = f"cache/{ds_name}/hsm_leaf{HSM_LEAF}.npz"
        if os.path.exists(cache):
            hsm = HierarchicalSoftmax.load(cache)
        else:
            print(f"  Building HSM tree (leaf={HSM_LEAF})...")
            hsm = HierarchicalSoftmax(V.astype(np.float32), tau=TAU, leaf_size=HSM_LEAF)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            hsm.save(cache)
        hsm.tau = TAU

    # Compute approx probs for each method (need shared y-limits)
    all_p_approx = {}
    for m in methods:
        if m == "hierarchical":
            log_p = hsm.log_probs(q.astype(np.float32))
            log_p[qidx_global] = -np.inf
            all_p_approx[m] = np.exp(log_p)
        else:
            log_wc = compute_cluster_log_weights(m, q, means, counts, dir_var_q=dir_var_q)
            all_p_approx[m] = compute_approx_probs(sim_masked, asgn, log_wc,
                                                    true_logZc_q, qidx_global)

    # Shared y-limits across methods (over full population, not the subsample,
    # so highest-similarity / extreme-probability items aren't clipped)
    all_f = np.concatenate([p_exact] + list(all_p_approx.values()))
    ylim_full = (all_f[all_f > 0].min() * 0.5, all_f.max() * 2)
    sim_no_self = np.delete(sim, qidx_global)
    xlim_full = (sim_no_self.max() * 1.005, sim_no_self.min() * 0.99)

    for m in methods:
        print(f"  Plotting {METHOD_LABELS[m]}...")
        plot_prob_vs_sim_overlay(ds_name, m, all_p_approx[m], p_exact, sim,
                                 idx_full, xlim_full=xlim_full, ylim_full=ylim_full)


if __name__ == "__main__":
    datasets = sys.argv[1:2] or list(DATASET_LABELS.keys())
    methods = sys.argv[2:] or None
    for ds_name in datasets:
        print(f"\n=== {ds_name} ===")
        plot_prob_vs_sim(ds_name, method_filter=methods)
