#!/usr/bin/env python3
"""Build the two synthetic GMM datasets and save to datasets/.

- synth-balanced:  K=1024, exactly N/K points per component (no random sampling).
- synth-unbalanced: K=1024, mixture weights ~ exp(0.8 * Z), Z~N(0,1), normalized.

Both: N=1,000,000, d=100, isotropic per-component covariance σ=0.1, component
means uniform on the unit sphere in R^d. Queries (1000) drawn from the same mixture.
Vectors are NOT normalized after generation — these are unnormalized GMMs.
"""
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
N = 1_000_000
NQ = 1000
D = 100
K = 1024
SIGMA = 0.1
LOGNORMAL_SIGMA = 0.8
SEED = 42


def sample_sphere(k, d, rng):
    g = rng.standard_normal((k, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g


def gen(name, weights, n=N, nq=NQ, d=D, k=K, sigma=SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    means = sample_sphere(k, d, rng)  # (K, d) on unit sphere

    # Cumulative-count assignment for exact balance, OR multinomial for weighted
    if weights is None:
        # Exactly balanced: floor(n/k) each, distribute remainder
        base = n // k
        rem = n - base * k
        comp_sizes = np.full(k, base, dtype=np.int64)
        comp_sizes[:rem] += 1
        assert comp_sizes.sum() == n
    else:
        # Weighted multinomial draw of cluster-sizes from weights — gives the
        # cleanest realization of the desired imbalance shape.
        comp_sizes = rng.multinomial(n, weights)

    # Generate points: stack per-component
    vectors = np.empty((n, d), dtype=np.float32)
    components = np.empty(n, dtype=np.int32)
    pos = 0
    for c in range(k):
        nc = int(comp_sizes[c])
        if nc == 0:
            continue
        noise = rng.standard_normal((nc, d)).astype(np.float32) * sigma
        vectors[pos:pos + nc] = means[c].astype(np.float32) + noise
        components[pos:pos + nc] = c
        pos += nc
    assert pos == n

    # Shuffle
    perm = rng.permutation(n)
    vectors = vectors[perm]
    components = components[perm]

    # Queries from same mixture (separate RNG stream from the data, but same sigma + means)
    if weights is None:
        q_comp = rng.integers(0, k, size=nq)
    else:
        q_comp = rng.choice(k, size=nq, p=weights)
    q_noise = rng.standard_normal((nq, d)).astype(np.float32) * sigma
    queries = means[q_comp].astype(np.float32) + q_noise

    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    np.savez(
        os.path.join(out, 'data.npz'),
        vectors=vectors,
        queries=queries,
        components=components,
        query_components=q_comp.astype(np.int32),
        component_means=means.astype(np.float32),
        component_sizes=comp_sizes.astype(np.int64),
        sigma=sigma,
    )
    print(f"  {name}: N={n} d={d} K={k} comp_size CV={comp_sizes.std()/comp_sizes.mean():.3f} "
          f"min={comp_sizes.min()} max={comp_sizes.max()}")


if __name__ == '__main__':
    print("Generating synth-balanced...")
    gen('synth-balanced', weights=None)

    print("Generating synth-unbalanced...")
    rng_w = np.random.default_rng(SEED + 1)  # weights RNG independent of data RNG
    logw = LOGNORMAL_SIGMA * rng_w.standard_normal(K)
    w = np.exp(logw)
    w /= w.sum()
    gen('synth-unbalanced', weights=w)

    print("Done.")
