"""Shared visual style for all plots: method colors, dataset labels, fontsizes."""

# Algorithm colors (consistent across every figure in the paper).
METHOD_COLORS = {
    "exact":         "#7F8C8D",  # grey
    "topk":          "#F39C12",  # orange
    "centroid":      "#A0522D",  # brown      — 2LS
    "hierarchical":  "#9B59B6",  # purple
    "ivf0":          "#3498DB",  # blue      — S-2LS
    "ivf2":          "#27AE60",  # green     — SD-2LS
}

METHOD_LABELS = {
    "exact":         "Exact",
    "topk":          "Top-k",
    "centroid":      "2LS",
    "hierarchical":  "Hierarchical",
    "ivf0":          "S-2LS (ours)",
    "ivf2":          "SD-2LS (ours)",
}

DATASET_LABELS = {
    "glove-100":         "GloVe-100",
    "vk-lsvd":           "VK-LSVD",
    "yambda":            "YAMBDA",
    "synth-balanced":    "Synth balanced",
    "synth-unbalanced":  "Synth unbalanced",
}

# Font sizes (consistent across plots — readable at print size).
FS_SUPTITLE = 22
FS_TITLE = 20
FS_AXLABEL = 22
FS_LEGEND = 17
FS_TICK = 17
FS_ANNOT = 16
