#!/usr/bin/env python3
"""Evaluate KL(approx || exact) and per-query latency for every method, every dataset,
at the per-dataset τ triplet (low/mid/high). Fixed C=1024 across all datasets.

Methods: exact, topk (FAISS IVFFlat k=1000), 2LS (centroid), hierarchical, S-2LS, SD-2LS.

Conventions:
  - KL(approx || exact) = sum p_approx (log p_approx - log p_exact). Robust to top-k zeros.
  - Query indices are MASKED from the candidate pool (set logits to -inf).
  - SD-2LS uses FULL-RANK per-cluster Σ_c (d×d) for honest latency reporting.

Output: results/eval_table.json (merged with any existing).
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
import json
import os
import time
import faiss
from scipy.special import logsumexp
from data import load_dataset

C = 1024
HSM_LEAF = 1024
TOPK_K = 1000
NQ = 1000

# Shared τ triplet across all 5 datasets — works because GloVe is now normalized.
TAU_LIST = [0.05, 0.1, 0.2]
TAUS = {ds: TAU_LIST for ds in
        ["glove-100", "vk-lsvd", "yambda", "synth-balanced", "synth-unbalanced"]}


def kl_approx_exact(log_p_approx, log_p_exact):
    """KL(approx || exact). Finite when approx has zeros (top-k case)."""
    p_approx = np.exp(log_p_approx)
    mask = p_approx > 1e-30
    return float(np.sum(p_approx[mask] * (log_p_approx[mask] - log_p_exact[mask])))


def precompute_clustering(ds_name, vectors):
    cl = np.load(f"clusters/{ds_name}/C{C}.npz")
    means = cl["centroids"].astype(np.float64)
    asgn = cl["assignments"].astype(np.int32)[:len(vectors)]
    counts = cl["counts"].astype(np.int64)
    log_counts = np.log(np.maximum(counts, 1).astype(np.float64))
    cluster_indices = [np.where(asgn == c)[0] for c in range(C)]

    d = vectors.shape[1]
    vectors_f64 = vectors.astype(np.float64)
    covs = np.zeros((C, d, d), dtype=np.float64)
    for c in range(C):
        idx = cluster_indices[c]
        if len(idx) < 2:
            continue
        block = vectors_f64[idx]
        mean = block.mean(axis=0)
        r = block - mean
        covs[c] = (r.T @ r) / len(idx)

    # Pre-flatten covariances for fast q^T Σ_c q via single GEMV:
    #   q^T Σ_c q = <q q^T, Σ_c>_F = vec(q q^T) · vec(Σ_c)
    # Stacking vec(Σ_c) into a (C, d²) matrix lets us compute all C scores in one BLAS call.
    covs_flat = covs.reshape(C, d * d).astype(np.float64)

    return dict(
        means=means, asgn=asgn, counts=counts, log_counts=log_counts,
        cluster_indices=cluster_indices, covs=covs, covs_flat=covs_flat,
    )


def build_topk_index(vectors):
    N, d = vectors.shape
    nlist = int(2 ** round(np.log2(np.sqrt(N))))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors.astype(np.float32))
    index.add(vectors.astype(np.float32))
    index.nprobe = max(1, nlist // 32)
    return index, nlist


def eval_dataset(ds_name):
    print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")
    ds = load_dataset(ds_name, n_queries=NQ)
    vectors = ds["vectors"]
    queries = ds["queries"].astype(np.float64)
    qidx = ds["query_indices"]
    N, d = vectors.shape
    print(f"  N={N:,}  d={d}  nq={NQ}")

    pc = precompute_clustering(ds_name, vectors)
    means, asgn, counts = pc["means"], pc["asgn"], pc["counts"]
    log_counts, cluster_indices, covs = pc["log_counts"], pc["cluster_indices"], pc["covs"]
    covs_flat = pc["covs_flat"]
    means_f32 = means.astype(np.float32)
    vectors_f32 = vectors.astype(np.float32)
    vectors_f64 = vectors.astype(np.float64)

    from ivf_softmax import HierarchicalSoftmax
    cache_path = f"cache/{ds_name}/hsm_leaf{HSM_LEAF}.npz"
    if os.path.exists(cache_path):
        hsm = HierarchicalSoftmax.load(cache_path)
    else:
        print(f"  Building hierarchical tree (leaf={HSM_LEAF})...")
        t0 = time.perf_counter()
        hsm = HierarchicalSoftmax(vectors_f32, tau=1.0, leaf_size=HSM_LEAF)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        hsm.save(cache_path)
        print(f"    {time.perf_counter()-t0:.1f}s")

    print(f"  Building FAISS IVFFlat (k={TOPK_K})...")
    t0 = time.perf_counter()
    index, nlist = build_topk_index(vectors)
    print(f"    nlist={nlist}, nprobe={index.nprobe}, built in {time.perf_counter()-t0:.1f}s")

    out = {"dataset": ds_name, "N": int(N), "d": int(d), "C": C, "nq": NQ, "by_tau": {}}

    for tau in TAUS[ds_name]:
        print(f"\n  τ={tau}")
        method_kls = {m: [] for m in ["exact", "topk", "centroid", "hierarchical", "ivf0", "ivf2"]}
        method_times = {m: [] for m in method_kls}

        for i, q in enumerate(queries):
            q_self_idx = int(qidx[i])
            q_f32 = q.astype(np.float32)
            sims_q = vectors_f64 @ q

            # Mask: exclude the query itself from being a candidate
            logits_exact = sims_q / tau
            logits_exact[q_self_idx] = -np.inf
            log_p_exact = logits_exact - logsumexp(logits_exact)

            # ---- exact (KL=0 by definition) ----
            method_kls["exact"].append(0.0)
            t0 = time.perf_counter()
            logits = vectors_f32 @ q_f32 / tau
            logits[q_self_idx] = -np.inf
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            np.random.choice(len(p), p=p)
            method_times["exact"].append((time.perf_counter() - t0) * 1000)

            # ---- topk ----
            sims_tk, ids_tk = index.search(q_f32.reshape(1, -1), TOPK_K + 1)  # +1 in case self is hit
            sims_tk = sims_tk[0].astype(np.float64)
            ids_tk = ids_tk[0]
            keep = (ids_tk >= 0) & (ids_tk != q_self_idx)
            sims_tk = sims_tk[keep][:TOPK_K] / tau
            ids_tk = ids_tk[keep][:TOPK_K]
            log_p_tk = np.full(N, -np.inf, dtype=np.float64)
            log_p_tk[ids_tk] = sims_tk - logsumexp(sims_tk)
            method_kls["topk"].append(kl_approx_exact(log_p_tk, log_p_exact))
            t0 = time.perf_counter()
            s, ids = index.search(q_f32.reshape(1, -1), TOPK_K + 1)
            s = s[0].astype(np.float64) / tau
            ids = ids[0]
            keep = (ids >= 0) & (ids != q_self_idx)
            s = s[keep][:TOPK_K]
            s -= s.max()
            pp = np.exp(s)
            pp /= pp.sum()
            np.random.choice(len(pp), p=pp)
            method_times["topk"].append((time.perf_counter() - t0) * 1000)

            # ---- 2LS family: shared per-query precomputation ----
            true_logZc = np.full(C, -np.inf, dtype=np.float64)
            for c in range(C):
                idx = cluster_indices[c]
                if len(idx) > 0:
                    logits_c = sims_q[idx] / tau
                    if q_self_idx in idx:
                        logits_c = logits_c[idx != q_self_idx]
                        if len(logits_c) == 0:
                            continue
                    true_logZc[c] = logsumexp(logits_c)

            centroid_scores = (means @ q) / tau
            valid_c = np.isfinite(centroid_scores)
            centroid_scores[~valid_c] = -np.inf

            qcq = covs_flat @ np.outer(q, q).ravel()  # full-rank q^T Σ_c q via flat GEMV

            for method, log_wc_fn in [
                ("centroid", lambda: centroid_scores),
                ("ivf0",     lambda: log_counts + centroid_scores),
                ("ivf2",     lambda: log_counts + centroid_scores + qcq / (2 * tau ** 2)),
            ]:
                log_wc = log_wc_fn()
                log_p_cluster = log_wc - logsumexp(log_wc)
                # within-cluster log probs (exact softmax over members, query masked)
                log_p_approx = np.full(N, -np.inf, dtype=np.float64)
                for c in range(C):
                    idx = cluster_indices[c]
                    if len(idx) == 0 or not np.isfinite(true_logZc[c]):
                        continue
                    logits_c = sims_q[idx] / tau
                    keep_mask = idx != q_self_idx
                    log_p_approx[idx[keep_mask]] = (
                        log_p_cluster[c] + logits_c[keep_mask] - true_logZc[c]
                    )
                method_kls[method].append(kl_approx_exact(log_p_approx, log_p_exact))

                # latency
                t0 = time.perf_counter()
                cs = q_f32 @ means_f32.T / tau
                if method == "ivf0":
                    cs = log_counts + cs
                elif method == "ivf2":
                    qcq32 = covs_flat @ np.outer(q, q).ravel()
                    cs = log_counts + cs + qcq32 / (2 * tau ** 2)
                cs -= cs.max()
                pc_ = np.exp(cs)
                pc_ /= pc_.sum()
                chosen = np.random.choice(len(pc_), p=pc_)
                members = cluster_indices[chosen]
                cluster_sims = vectors_f32[members] @ q_f32
                lc = cluster_sims / tau
                lc -= lc.max()
                pw = np.exp(lc)
                pw /= pw.sum()
                np.random.choice(len(pw), p=pw)
                method_times[method].append((time.perf_counter() - t0) * 1000)

            # ---- hierarchical ----
            hsm.tau = tau
            log_p_h = hsm.log_probs(q.astype(np.float32))
            log_p_h[q_self_idx] = -np.inf
            method_kls["hierarchical"].append(kl_approx_exact(log_p_h, log_p_exact))
            t0 = time.perf_counter()
            hsm.sample(q.astype(np.float32), 1)
            method_times["hierarchical"].append((time.perf_counter() - t0) * 1000)

        row = {}
        for m in method_kls:
            kls = np.array(method_kls[m])
            ts = np.array(method_times[m])
            row[m] = dict(
                kl_mean=float(kls.mean()), kl_std=float(kls.std()),
                kl_median=float(np.median(kls)),
                kl_ci95=float(1.96 * kls.std() / np.sqrt(len(kls))),
                time_mean_ms=float(ts.mean()), time_std_ms=float(ts.std()),
                time_ci95_ms=float(1.96 * ts.std() / np.sqrt(len(ts))),
            )
            print(f"    {m:>14s}  KL={kls.mean():.5f} ± {kls.std():.5f}   "
                  f"time={ts.mean():.3f} ± {ts.std():.3f} ms")
        out["by_tau"][str(tau)] = row

    return out


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] or list(TAUS.keys())
    out_path = "results/eval_table.json"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    for name in datasets:
        all_results[name] = eval_dataset(name)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {out_path}")
