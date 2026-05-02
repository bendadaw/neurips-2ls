"""Dataset loading.

The loader returns the FULL vector array and a set of query indices into it.
Queries are NOT removed from the vector array. Callers (clustering, eval) decide
whether to mask out query indices at compute time.

Normalization policy:
  - glove-100:        normalized to unit sphere (paper-standard).
  - vk-lsvd:          raw (already nearly on sphere, norms 0.94 ± 0.03).
  - yambda:           the provider's `normalized_embed` column (unit-norm).
  - synth-balanced:   GMM, 1M pts, K=1024, exactly equal component sizes, d=100, σ=0.1.
  - synth-unbalanced: GMM, 1M pts, K=1024, LogNormal(σ_w=0.8) component weights.
"""

import numpy as np
import h5py
import os

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def _query_indices(n_total, n_queries, seed=42):
    """Deterministic seeded sample of n_queries indices from [0, n_total)."""
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=n_queries, replace=False).astype(np.int64)


def load_dataset(name, n_queries=1000, query_indices=None):
    """Return dict with keys: vectors (full array, no holdout), query_indices (int64),
    queries (= vectors[query_indices], for convenience), name, n, d.

    The vector array is NOT modified to remove queries. If you don't want a query
    sampled in your eval, mask it explicitly in the caller.
    """
    if name == "glove-100":
        V = _load_glove100_vectors()
    elif name == "vk-lsvd":
        V = _load_vk_lsvd_vectors()
    elif name == "yambda":
        V = _load_yambda_vectors()
    elif name in ("synth-balanced", "synth-unbalanced"):
        V = _load_synth_vectors(name)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if query_indices is None:
        query_indices = _query_indices(len(V), n_queries)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)

    return {
        "vectors": V,
        "queries": V[query_indices].copy(),
        "query_indices": query_indices,
        "name": name,
        "n": V.shape[0],
        "d": V.shape[1],
    }


def _load_glove100_vectors():
    path = os.path.join(DATASETS_DIR, "glove-100", "glove-100-angular.hdf5")
    with h5py.File(path, "r") as f:
        V = np.array(f["train"], dtype=np.float32)
    norms = np.linalg.norm(V, axis=1)
    V = V[norms > 1e-8]
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def _load_vk_lsvd_vectors():
    path = os.path.join(DATASETS_DIR, "vk-lsvd", "metadata", "item_embeddings.npz")
    V = np.load(path)["embedding"].astype(np.float32)
    norms = np.linalg.norm(V, axis=1)
    return V[norms > 1e-8]


def _load_yambda_vectors():
    import pyarrow.parquet as pq
    path = os.path.join(DATASETS_DIR, "yambda", "embeddings.parquet")
    table = pq.read_table(path, columns=["normalized_embed"])
    flat = table.column("normalized_embed").combine_chunks()
    values = flat.values.to_numpy(zero_copy_only=False).astype(np.float32)
    n = len(flat)
    d = len(flat[0].as_py())
    V = values.reshape(n, d)
    norms = np.linalg.norm(V, axis=1)
    return V[norms > 1e-8]


def _load_synth_vectors(name):
    path = os.path.join(DATASETS_DIR, name, "data.npz")
    return np.load(path)["vectors"].astype(np.float32)
