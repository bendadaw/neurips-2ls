# Two-Level Softmax Sampling Done Right

Code for our NeurIPS 2026 paper
**"Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion."**

**Authors:** Walid Bendada, Guillaume Salha-Galvan

This repository reproduces every figure and table in the paper. All experiments
run on CPU; no GPU is required.

## Repository layout

```
build_synth.py             Generates the two synthetic GMM datasets
build_all.py               Builds k-means clusters and HSM trees for every dataset
data.py                    Dataset loaders
ivf_softmax.py             2LS / S-2LS / SD-2LS / hierarchical-softmax implementations
eval_table.py              Runs the main KL / latency experiment (Table 1)
render_table.py            Formats results/eval_table.json into LaTeX tables
plot_iw.py                 Importance-weight diagnostics (R vs cluster size and R vs sigma_k^2(q))
plot_prob_vs_sim.py        p(i|q) vs q^T x_i scatter plots
plot_cluster_size_hist.py  Cluster-size histograms
plot_style.py              Shared colors / labels / fontsizes
scripts/download_data.sh   Fetches the three real datasets
```

## Reproduction

### 1. Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`huggingface-cli` is also required for VK-LSVD and YAMBDA:
```
pip install -U "huggingface_hub[cli]"
```

### 2. Get the data

Real datasets (downloaded into `datasets/`):
```
bash scripts/download_data.sh
```
Sources: GloVe-100-angular (ann-benchmarks),
[deepvk/VK-LSVD](https://huggingface.co/datasets/deepvk/VK-LSVD),
[yandex/yambda](https://huggingface.co/datasets/yandex/yambda).

Synthetic datasets (generated locally, seed=42):
```
python build_synth.py
```

### 3. Build clusters and hierarchical-softmax trees

```
python build_all.py
```
Produces `clusters/<ds>/C1024.npz` and `cache/<ds>/hsm_leaf1024.npz` for every
dataset. This step takes a while (especially for YAMBDA).

### 4. Reproduce the table

```
python eval_table.py            # writes results/eval_table.json
python render_table.py          # writes results/tables.tex
```
With `NQ = 1000` queries per dataset (set in `eval_table.py`), this is the slow
step — expect a few hours on the full set; you can pass dataset names as
arguments to run a subset, e.g. `python eval_table.py synth-balanced`.

### 5. Reproduce the figures

```
python plot_cluster_size_hist.py
python plot_iw.py
python plot_prob_vs_sim.py
```
Each script accepts dataset names as arguments and writes PDFs into
`results/figures/<ds>/`. Per dataset, the paper uses nine figures:

```
iw_size_2ls.pdf       iw_dirvar_2ls.pdf       prob_vs_sim_2ls.pdf
iw_size_s2ls.pdf      iw_dirvar_s2ls.pdf      prob_vs_sim_s2ls.pdf
iw_size_sd2ls.pdf     iw_dirvar_sd2ls.pdf     prob_vs_sim_sd2ls.pdf
```
plus `cluster_size_hist.pdf`.

## Notes

- All randomness is seeded; per-dataset query indices are deterministic
  (`data.py:_query_indices`, seed=42).
- KL is computed in the "approx || exact" direction, consistent with the paper.
- The clustering uses FAISS k-means at `C=1024` with `niter=25`, seed=42; the
  hierarchical baseline uses leaf size 1024.
- The synthetic datasets use seed=42 and an independent stream for the
  log-normal mixture weights (seed=43).

## License

Released for the sole purpose of reviewing the associated paper. See `LICENSE`.
