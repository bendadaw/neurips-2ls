"""IVF Softmax sampling: exact, top-k, centroid, and IVF (zero + second-order).

All cluster-based methods use un-normalized centroids (cluster means x_bar).
This absorbs the first-order Taylor correction into the dot product.

Three IVF-family variants:
  - Centroid:  softmax over <q, x_bar_c>/tau (no cluster sizes)
  - IVF-0:     log(n_c) + <q, x_bar_c>/tau  (adds count weighting)
  - IVF-2:     IVF-0 + (1/2tau^2) q^T Cov_c q  (adds low-rank covariance)
"""

import numpy as np
from scipy.special import logsumexp
import faiss


# ---------------------------------------------------------------------------
# Exact softmax (baseline)
# ---------------------------------------------------------------------------

class ExactSoftmax:
    """Exact softmax over all database vectors. O(Nd) per query."""

    def __init__(self, vectors, tau=1.0):
        self.vectors = vectors.astype(np.float64)
        self.tau = tau
        self.N, self.d = vectors.shape

    def log_probs(self, query):
        """Return log p(i|q) for all i. Shape (N,)."""
        logits = self.vectors @ query / self.tau
        return logits - logsumexp(logits)

    def sample(self, query, n_samples):
        """Sample n_samples indices. One matmul, one choice."""
        logits = self.vectors @ query / self.tau
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return np.random.choice(self.N, size=n_samples, replace=True, p=p)


# ---------------------------------------------------------------------------
# Top-k truncated softmax (approximate MIPS via FAISS IVF)
# ---------------------------------------------------------------------------

class TopKSoftmax:
    """Top-k truncated softmax. Approximate MIPS via FAISS IVF, softmax over top-k."""

    def __init__(self, vectors, tau=1.0, k=100, nlist=256, nprobe=16):
        self.vectors = vectors
        self.tau = tau
        self.N, self.d = vectors.shape
        self.k = k

        quantizer = faiss.IndexFlatIP(self.d)
        self.index = faiss.IndexIVFFlat(quantizer, self.d, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = nprobe

    def log_probs(self, query):
        """Approximate log p(i|q): softmax over top-k, -inf elsewhere."""
        sims, ids = self.index.search(query.reshape(1, -1).astype(np.float32), self.k)
        sims = sims[0].astype(np.float64) / self.tau
        ids = ids[0]
        log_p = np.full(self.N, -np.inf, dtype=np.float64)
        log_p[ids] = sims - logsumexp(sims)
        return log_p

    def sample(self, query, n_samples):
        """Approximate MIPS top-k, then softmax sample."""
        sims, ids = self.index.search(query.reshape(1, -1).astype(np.float32), self.k)
        sims = sims[0].astype(np.float64) / self.tau
        ids = ids[0]
        sims -= sims.max()
        p = np.exp(sims)
        p /= p.sum()
        local_indices = np.random.choice(self.k, size=n_samples, replace=True, p=p)
        return ids[local_indices]


# ---------------------------------------------------------------------------
# Shared IVF clustering base
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class Cluster:
    """Single cluster: indices, mean, count, covariance."""
    __slots__ = ('indices', 'mean', 'count', 'covariance')

    def __init__(self, indices, mean, count, covariance):
        self.indices = indices       # (n_c,) int64 — original vector indices
        self.mean = mean             # (d,) float64 — un-normalized mean
        self.count = count           # int
        self.covariance = covariance # (d, d) float64 — within-cluster covariance


class Clustering:
    """A set of clusters over a vector database. Precomputes and caches all
    per-cluster statistics needed by IVF softmax methods.

    Build from vectors + assignments, or load from disk.
    """

    def __init__(self, vectors, assignments):
        self.N, self.d = vectors.shape
        assignments = np.asarray(assignments, dtype=np.int32)
        self.C = int(assignments.max()) + 1

        # Sort once — O(N log N) — everything else is slicing
        order = np.argsort(assignments, kind='mergesort')
        sorted_asgn = assignments[order]
        counts = np.bincount(sorted_asgn, minlength=self.C).astype(np.int64)
        offsets = np.zeros(self.C + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        # Contiguous reordered vectors
        self.reordered = vectors[order].copy()
        self.idx_map = order
        self.offsets = offsets

        # Build per-cluster objects
        self.clusters = []
        for c in range(self.C):
            s, e = offsets[c], offsets[c + 1]
            idx = order[s:e]
            n_c = int(counts[c])
            if n_c > 0:
                block = self.reordered[s:e].astype(np.float64)
                mean = block.mean(axis=0)
                residuals = block - mean
                cov = (residuals.T @ residuals) / n_c
            else:
                mean = np.zeros(self.d, dtype=np.float64)
                cov = np.zeros((self.d, self.d), dtype=np.float64)
            self.clusters.append(Cluster(idx, mean, n_c, cov))

        # Stacked arrays for vectorized operations
        self.means = np.array([cl.mean for cl in self.clusters], dtype=np.float64)   # (C, d)
        self.counts = counts                                                          # (C,)
        self.log_counts = np.log(np.maximum(counts, 1).astype(np.float64))           # (C,)

    def save(self, path):
        """Save clustering to npz."""
        covs = np.array([cl.covariance for cl in self.clusters])  # (C, d, d)
        np.savez(path,
                 means=self.means,
                 counts=self.counts,
                 covariances=covs,
                 idx_map=self.idx_map,
                 offsets=self.offsets)

    @classmethod
    def load(cls, path, vectors):
        """Load pre-saved clustering and attach to vectors."""
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.N, obj.d = vectors.shape
        obj.reordered = vectors[data['idx_map']].copy()
        obj.idx_map = data['idx_map']
        obj.offsets = data['offsets']
        obj.means = data['means'].astype(np.float64)
        obj.counts = data['counts'].astype(np.int64)
        obj.C = len(obj.counts)
        obj.log_counts = np.log(np.maximum(obj.counts, 1).astype(np.float64))
        covs = data['covariances']
        obj.clusters = []
        for c in range(obj.C):
            s, e = obj.offsets[c], obj.offsets[c + 1]
            obj.clusters.append(Cluster(
                indices=obj.idx_map[s:e],
                mean=obj.means[c],
                count=int(obj.counts[c]),
                covariance=covs[c].astype(np.float64),
            ))
        return obj

    @classmethod
    def build(cls, vectors, n_clusters, spherical=False):
        """Run FAISS k-means and build clustering."""
        N, d = vectors.shape
        km = faiss.Kmeans(d, n_clusters, niter=20, verbose=False, spherical=spherical)
        km.train(vectors)
        _, asgn = km.index.search(vectors, 1)
        return cls(vectors, asgn.ravel())


def load_clustering(dataset_name, n_clusters):
    """Load a precomputed clustering from clusters/<dataset>/<C>.npz."""
    import os
    path = os.path.join(os.path.dirname(__file__), "clusters", dataset_name, f"C{n_clusters}.npz")
    data = np.load(path)
    return data["centroids"], data["assignments"], data["counts"]


def _build_ivf(vectors, n_clusters, spherical=False, precomputed=None):
    """K-means clustering, return (means, inv_lists, counts, assignments, reordered, offsets).

    Centroids are un-normalized cluster means x_bar_c.
    Vectors are reordered so each cluster is contiguous in memory.
    If spherical=True, uses spherical k-means (centroids normalized each iteration).
    If precomputed=(centroids, assignments, counts), skip k-means.
    """
    N, d = vectors.shape

    if precomputed is not None:
        _, assignments, counts = precomputed
        counts = counts.astype(np.int64)
    else:
        kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False, spherical=spherical)
        kmeans.train(vectors)
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.ravel()
        counts = np.zeros(n_clusters, dtype=np.int64)

    inv_lists = []
    means = np.zeros((n_clusters, d), dtype=np.float64)

    for c in range(n_clusters):
        members = np.where(assignments == c)[0]
        inv_lists.append(members)
        if precomputed is None:
            counts[c] = len(members)
        if len(members) > 0:
            means[c] = vectors[members].astype(np.float64).mean(axis=0)

    # Reorder vectors so each cluster is contiguous
    order = np.concatenate(inv_lists)
    reordered = vectors[order].copy()  # contiguous C-order array
    # Map from reordered index back to original index
    idx_map = order
    # Offsets: cluster c spans reordered[offsets[c]:offsets[c+1]]
    offsets = np.zeros(n_clusters + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return means.astype(np.float32), inv_lists, counts, assignments, reordered, idx_map, offsets


# ---------------------------------------------------------------------------
# Centroid Softmax baseline (no cluster-size weighting)
# ---------------------------------------------------------------------------

class CentroidSoftmax:
    """Softmax over un-normalized cluster means, then exact within-cluster sampling.

    P(c|q) = exp(<q, x_bar_c>/tau) / sum_j exp(<q, x_bar_j>/tau)
    """

    def __init__(self, vectors, n_clusters, tau=1.0, spherical=False, precomputed=None):
        self.vectors = vectors
        self.tau = tau
        self.N, self.d = vectors.shape
        self.n_clusters = n_clusters
        self.means, self.inv_lists, _, _, self.reordered, self.idx_map, self.offsets = _build_ivf(vectors, n_clusters, spherical=spherical, precomputed=precomputed)

    def cluster_log_weights(self, query):
        return (self.means @ query / self.tau).astype(np.float64)

    def log_probs(self, query):
        log_Zc = self.cluster_log_weights(query)
        log_Z_total = logsumexp(log_Zc)
        all_logits = self.vectors.astype(np.float64) @ query / self.tau
        log_p = np.empty(self.N, dtype=np.float64)
        for c in range(self.n_clusters):
            members = self.inv_lists[c]
            if len(members) == 0:
                continue
            logits_c = all_logits[members]
            log_p[members] = (log_Zc[c] - log_Z_total) + (logits_c - logsumexp(logits_c))
        return log_p

    def sample(self, query, n_samples):
        log_Zc = self.cluster_log_weights(query)
        log_Zc -= log_Zc.max()
        p_cluster = np.exp(log_Zc)
        p_cluster /= p_cluster.sum()
        c = np.random.choice(self.n_clusters, p=p_cluster)
        s, e = self.offsets[c], self.offsets[c + 1]
        logits = self.reordered[s:e].astype(np.float64) @ query / self.tau
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        local = np.random.choice(e - s, size=n_samples, replace=True, p=p)
        return self.idx_map[s + local]


# ---------------------------------------------------------------------------
# IVF Softmax (zero-order and second-order)
# ---------------------------------------------------------------------------

class IVFSoftmax:
    """IVF-based approximate softmax sampling with un-normalized cluster means.

    order=0: log(n_c) + <q, x_bar_c>/tau
    order=2: order=0 + (1/2tau^2) * q^T Cov_c q  (low-rank covariance correction)

    Args:
        vectors: (N, d) unit-normed float32 array.
        n_clusters: number of IVF clusters.
        tau: temperature.
        rank: rank for low-rank covariance approximation (0 = no second-order).
    """

    def __init__(self, vectors, n_clusters, tau=1.0, rank=0, spherical=False, precomputed=None):
        self.vectors = vectors
        self.tau = tau
        self.N, self.d = vectors.shape
        self.n_clusters = n_clusters
        self.rank = rank

        self.means, self.inv_lists, self.counts, _, self.reordered, self.idx_map, self.offsets = _build_ivf(vectors, n_clusters, spherical=spherical, precomputed=precomputed)
        self.log_counts = np.log(np.maximum(self.counts, 1).astype(np.float64))

        if rank > 0:
            self._build_low_rank_covariance(rank)
        else:
            self.global_basis = None
            self.cluster_variances = None

    def _build_low_rank_covariance(self, rank):
        """Compute shared global basis U and per-cluster projected variances.

        Residuals are r_i = x_i - x_bar_c.
        Since E[r] = 0 by construction, Cov = E[rr^T].
        """
        d = self.d

        # Pooled covariance across all clusters
        pooled_cov = np.zeros((d, d), dtype=np.float64)
        total = 0
        for c in range(self.n_clusters):
            members = self.inv_lists[c]
            if len(members) < 2:
                continue
            X = self.vectors[members].astype(np.float64)
            residuals = X - self.means[c].astype(np.float64)
            pooled_cov += residuals.T @ residuals
            total += len(members)

        if total > 0:
            pooled_cov /= total

        # Top-r eigenvectors as shared basis
        eigvals, eigvecs = np.linalg.eigh(pooled_cov)
        rank = min(rank, d)
        self.global_basis = eigvecs[:, -rank:].astype(np.float32)  # (d, rank)

        # Per-cluster projected variances
        self.cluster_variances = np.zeros((self.n_clusters, rank), dtype=np.float64)
        for c in range(self.n_clusters):
            members = self.inv_lists[c]
            if len(members) < 2:
                continue
            X = self.vectors[members].astype(np.float64)
            residuals = X - self.means[c].astype(np.float64)
            projected = residuals @ self.global_basis.astype(np.float64)  # (n_c, rank)
            self.cluster_variances[c] = np.mean(projected ** 2, axis=0)

    def cluster_log_weights(self, query, order=0):
        """Compute log Z_c estimates for each cluster."""
        dots = self.means @ query / self.tau  # (C,)
        log_w = self.log_counts + dots

        if order >= 2 and self.global_basis is not None:
            q_proj = self.global_basis.T @ query  # (rank,)
            q_proj_sq = q_proj ** 2
            correction = self.cluster_variances @ q_proj_sq / (2.0 * self.tau ** 2)
            log_w += correction

        return log_w.astype(np.float64)

    def log_probs(self, query, order=0):
        """Approximate log p(i|q) for all i. O(Nd) — for quality measurement only."""
        log_Zc = self.cluster_log_weights(query, order=order)
        log_Z_total = logsumexp(log_Zc)
        all_logits = self.vectors.astype(np.float64) @ query / self.tau
        log_p = np.empty(self.N, dtype=np.float64)
        for c in range(self.n_clusters):
            members = self.inv_lists[c]
            if len(members) == 0:
                continue
            logits_c = all_logits[members]
            log_p[members] = (log_Zc[c] - log_Z_total) + (logits_c - logsumexp(logits_c))
        return log_p

    def sample(self, query, n_samples, order=0):
        """Two-stage: pick cluster, then exact softmax within cluster."""
        log_Zc = self.cluster_log_weights(query, order=order)
        log_Zc -= log_Zc.max()
        p_cluster = np.exp(log_Zc)
        p_cluster /= p_cluster.sum()
        c = np.random.choice(self.n_clusters, p=p_cluster)
        s, e = self.offsets[c], self.offsets[c + 1]
        logits = self.reordered[s:e].astype(np.float64) @ query / self.tau
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        local = np.random.choice(e - s, size=n_samples, replace=True, p=p)
        return self.idx_map[s + local]

    def sample_batch(self, query, n_samples, order=0):
        """Alias for sample with order parameter (used by experiment runner)."""
        return self.sample(query, n_samples, order=order)

    def sample_topp(self, query, n_samples, p_threshold=0.9, order=0):
        """Top-p sampling: probe clusters until cumulative mass >= p_threshold.

        1. Compute cluster weights — O(Cd)
        2. Sort, take smallest set with cumsum >= p_threshold — adaptive nprobe
        3. Exact softmax within those clusters — O(n_probed * n_c * d)
        4. Renormalize and sample

        Returns (samples, n_probed) for diagnostics.
        """
        log_Zc = self.cluster_log_weights(query, order=order)
        log_Zc -= log_Zc.max()
        p_cluster = np.exp(log_Zc)
        p_cluster /= p_cluster.sum()

        # Sort clusters by probability descending
        order_idx = np.argsort(-p_cluster)
        cumsum = np.cumsum(p_cluster[order_idx])
        # Smallest set with cumsum >= p_threshold (at least 1 cluster)
        n_probed = int(np.searchsorted(cumsum, p_threshold)) + 1
        n_probed = min(n_probed, self.n_clusters)
        selected = order_idx[:n_probed]

        # Gather all members from selected clusters
        all_members = np.concatenate([self.inv_lists[c] for c in selected])
        # Exact softmax over gathered vectors
        logits = self.vectors[all_members].astype(np.float64) @ query / self.tau
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        local_indices = np.random.choice(len(all_members), size=n_samples, replace=True, p=p)
        return all_members[local_indices], n_probed


# ---------------------------------------------------------------------------
# Hierarchical Softmax (binary k-means tree)
# ---------------------------------------------------------------------------

class HierarchicalSoftmax:
    """Hierarchical softmax via recursive binary k-means tree.

    Parameter-free variant: each internal node carries the mean of its descendant
    subtree as routing representative; routing score is <q, mean_child>/tau.

    Sampling cost: O(d * depth + leaf_size * d) per query, where
    depth ≈ log2(N / leaf_size).
    """

    def __init__(self, vectors, tau=1.0, leaf_size=256):
        self.vectors = vectors
        self.tau = tau
        self.N, self.d = vectors.shape
        self.leaf_size = leaf_size

        # Contiguous storage for leaf vectors
        self._reordered = np.empty((self.N, self.d), dtype=np.float32)
        self._idx_map = np.empty(self.N, dtype=np.int64)
        self._next_pos = 0

        # Tree stored as flat list of dicts during build, then flattened
        self._nodes_list = []
        self._leaves_placed = 0
        self._build(np.arange(self.N))
        del self._leaves_placed
        self._flatten()

    def _build(self, indices):
        """Recursively build binary tree. Returns node index."""
        node_id = len(self._nodes_list)
        vecs = self.vectors[indices]
        mean = vecs.astype(np.float64).mean(axis=0).astype(np.float32)
        count = len(indices)

        node = {
            'mean': mean,
            'log_count': np.log(float(max(count, 1))),
            'left': -1, 'right': -1,
            'leaf_start': -1, 'leaf_end': -1,
        }
        self._nodes_list.append(node)

        if count <= self.leaf_size:
            s = self._next_pos
            self._reordered[s:s + count] = vecs
            self._idx_map[s:s + count] = indices
            self._next_pos += count
            node['leaf_start'] = s
            node['leaf_end'] = s + count
            self._leaves_placed += count
            if self._leaves_placed % (self.N // 20) < count:
                print(f"    Tree build: {self._leaves_placed / self.N:.0%} ({self._leaves_placed}/{self.N})")
            return node_id

        # Binary k-means split
        kmeans = faiss.Kmeans(self.d, 2, niter=10, verbose=False)
        kmeans.train(vecs)
        _, asgn = kmeans.index.search(vecs, 1)
        left_mask = asgn.ravel() == 0

        if left_mask.all() or (~left_mask).all():
            s = self._next_pos
            self._reordered[s:s + count] = vecs
            self._idx_map[s:s + count] = indices
            self._next_pos += count
            node['leaf_start'] = s
            node['leaf_end'] = s + count
            return node_id

        node['left'] = self._build(indices[left_mask])
        node['right'] = self._build(indices[~left_mask])
        return node_id

    def _flatten(self):
        """Convert list-of-dicts to struct-of-arrays for fast traversal."""
        n = len(self._nodes_list)
        self._means = np.array([nd['mean'] for nd in self._nodes_list], dtype=np.float32)   # (n, d)
        self._log_counts = np.array([nd['log_count'] for nd in self._nodes_list], dtype=np.float64)
        self._left = np.array([nd['left'] for nd in self._nodes_list], dtype=np.int32)
        self._right = np.array([nd['right'] for nd in self._nodes_list], dtype=np.int32)
        self._leaf_start = np.array([nd['leaf_start'] for nd in self._nodes_list], dtype=np.int64)
        self._leaf_end = np.array([nd['leaf_end'] for nd in self._nodes_list], dtype=np.int64)
        # Precompute which nodes are leaves
        self._is_leaf = self._left == -1
        # Keep nodes list for save compat, then discard
        self.nodes = self._nodes_list
        del self._nodes_list

    def log_probs(self, query):
        """Approximate log p(i|q) for all i via iterative BFS."""
        log_p = np.empty(self.N, dtype=np.float64)
        query64 = query.astype(np.float64)
        # All node scores at once: means @ query / tau (parameter-free, no count weighting)
        all_scores = (self._means.astype(np.float64) @ query64) / self.tau

        # BFS with stack
        stack = [(0, 0.0)]  # (node_id, log_prefix)
        while stack:
            nid, log_prefix = stack.pop()
            if self._is_leaf[nid]:
                s, e = self._leaf_start[nid], self._leaf_end[nid]
                logits = self._reordered[s:e].astype(np.float64) @ query64 / self.tau
                log_p[self._idx_map[s:e]] = log_prefix + (logits - logsumexp(logits))
            else:
                lid, rid = self._left[nid], self._right[nid]
                log_L, log_R = all_scores[lid], all_scores[rid]
                log_total = np.logaddexp(log_L, log_R)
                stack.append((lid, log_prefix + log_L - log_total))
                stack.append((rid, log_prefix + log_R - log_total))
        return log_p

    def sample(self, query, n_samples):
        """Sample n indices by traversing the tree with binomial splitting."""
        query64 = query.astype(np.float64)
        all_scores = (self._means.astype(np.float64) @ query64) / self.tau

        out = np.empty(n_samples, dtype=np.int64)
        stack = [(0, n_samples, 0)]  # (node_id, n, offset)
        while stack:
            nid, n, offset = stack.pop()
            if n == 0:
                continue
            if self._is_leaf[nid]:
                s, e = self._leaf_start[nid], self._leaf_end[nid]
                logits = self._reordered[s:e].astype(np.float64) @ query64 / self.tau
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum()
                local = np.random.choice(e - s, size=n, replace=True, p=p)
                out[offset:offset + n] = self._idx_map[s + local]
            else:
                lid, rid = self._left[nid], self._right[nid]
                log_L, log_R = all_scores[lid], all_scores[rid]
                max_lr = max(log_L, log_R)
                p_left = np.exp(log_L - max_lr) / (np.exp(log_L - max_lr) + np.exp(log_R - max_lr))
                n_left = np.random.binomial(n, p_left)
                stack.append((lid, n_left, offset))
                stack.append((rid, n - n_left, offset + n_left))
        return out

    def save(self, path):
        """Save tree to npz (flat arrays, no pickle)."""
        np.savez(path, means=self._means, log_counts=self._log_counts,
                 left=self._left, right=self._right,
                 leaf_start=self._leaf_start, leaf_end=self._leaf_end,
                 reordered=self._reordered, idx_map=self._idx_map,
                 tau=self.tau, leaf_size=self.leaf_size)

    @classmethod
    def load(cls, path, vectors=None):
        """Load tree from npz."""
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.tau = float(data['tau'])
        obj.leaf_size = int(data['leaf_size'])
        obj._reordered = data['reordered']
        obj._idx_map = data['idx_map']
        obj.N = len(obj._idx_map)
        obj.d = data['means'].shape[1]
        obj.vectors = vectors
        obj._means = data['means']
        obj._log_counts = data['log_counts']
        obj._left = data['left'].astype(np.int32)
        obj._right = data['right'].astype(np.int32)
        obj._leaf_start = data['leaf_start']
        obj._leaf_end = data['leaf_end']
        obj._is_leaf = obj._left == -1
        # Reconstruct nodes list for backward compat
        obj.nodes = []
        for i in range(len(obj._means)):
            obj.nodes.append({
                'mean': obj._means[i], 'log_count': obj._log_counts[i],
                'left': int(obj._left[i]), 'right': int(obj._right[i]),
                'leaf_start': int(obj._leaf_start[i]), 'leaf_end': int(obj._leaf_end[i]),
            })
        return obj


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def kl_divergence(log_p_exact, log_p_approx):
    """KL(approx || exact) = sum p_approx * (log p_approx - log p_exact).

    This direction is finite even when p_approx has zeros (0*log(0) = 0),
    which makes it usable for top-k truncated distributions.
    """
    p_approx = np.exp(log_p_approx)
    mask = p_approx > 1e-30
    return np.sum(p_approx[mask] * (log_p_approx[mask] - log_p_exact[mask]))


def l1_distance(log_p_exact, log_p_approx):
    """L1 distance between the two distributions."""
    return np.sum(np.abs(np.exp(log_p_exact) - np.exp(log_p_approx)))
