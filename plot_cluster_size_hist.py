#!/usr/bin/env python3
"""Histogram of cluster sizes at C=1024, one figure per dataset.

Usage:
    python plot_cluster_size_hist.py [dataset...]
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from plot_style import DATASET_LABELS, FS_SUPTITLE, FS_AXLABEL, FS_TICK

sns.set_theme(style="whitegrid", font_scale=1.1)

C = 1024


def plot_one(ds_name):
    counts = np.load(f"clusters/{ds_name}/C{C}.npz")["counts"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.histplot(counts, bins=60, color="#5B8DB8", ax=ax,
                 edgecolor="white", linewidth=0.3)
    ax.set_title(f"{DATASET_LABELS.get(ds_name, ds_name)}", fontsize=18, fontweight="bold")
    ax.set_xlabel("Cluster size", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    out = f"results/figures/{ds_name}/cluster_size_hist.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    datasets = sys.argv[1:] or list(DATASET_LABELS.keys())
    for ds in datasets:
        plot_one(ds)
