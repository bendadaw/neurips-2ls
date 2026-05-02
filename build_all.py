#!/usr/bin/env python3
"""Build clusters and hierarchical trees for every dataset used in the paper.

Produces:
  clusters/<ds>/C1024.npz      (k-means clustering, used by 2LS / S-2LS / SD-2LS / table)
  cache/<ds>/hsm_leaf1024.npz  (hierarchical-softmax tree, used by table)

Usage:
    python build_all.py [dataset...]    # default: all 5 datasets
"""
import os
import sys
import time
import numpy as np
import faiss
from data import load_dataset

DATASETS = sys.argv[1:] or [
    "glove-100", "vk-lsvd", "yambda", "synth-balanced", "synth-unbalanced",
]
C = 1024
HSM_LEAF = 1024


def build_clustering(ds_name, vectors):
    out = f"clusters/{ds_name}/C{C}.npz"
    if os.path.exists(out):
        print(f"  clusters C={C}  (cached)")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.perf_counter()
    d = vectors.shape[1]
    km = faiss.Kmeans(d, C, niter=25, verbose=False, seed=42, spherical=False)
    km.train(vectors.astype(np.float32))
    centroids = km.centroids
    _, asgn = km.index.search(vectors.astype(np.float32), 1)
    asgn = asgn.ravel().astype(np.int32)
    counts = np.bincount(asgn, minlength=C).astype(np.int64)
    np.savez(out, centroids=centroids, assignments=asgn, counts=counts)
    print(f"  clusters C={C}  {time.perf_counter()-t0:.1f}s")


def build_hsm(ds_name, vectors):
    out = f"cache/{ds_name}/hsm_leaf{HSM_LEAF}.npz"
    if os.path.exists(out):
        print(f"  HSM leaf={HSM_LEAF}  (cached)")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from ivf_softmax import HierarchicalSoftmax
    t0 = time.perf_counter()
    hsm = HierarchicalSoftmax(vectors.astype(np.float32), tau=0.1, leaf_size=HSM_LEAF)
    hsm.save(out)
    print(f"  HSM leaf={HSM_LEAF}  {time.perf_counter()-t0:.1f}s")


for ds_name in DATASETS:
    print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")
    ds = load_dataset(ds_name, n_queries=1000)
    vectors = ds["vectors"]
    print(f"  N={len(vectors):,}, d={vectors.shape[1]}")
    build_clustering(ds_name, vectors)
    build_hsm(ds_name, vectors)
