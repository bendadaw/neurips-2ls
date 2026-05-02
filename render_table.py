#!/usr/bin/env python3
"""Render results/eval_table.json to LaTeX. Six tables: KL and latency at each tau.

Datasets are rows, methods are columns. Bold = best, underline = 2nd best.
Latency tables also include Exact and a speedup-vs-exact column. KL excludes
exact (KL=0 by definition) and excludes topk's "speedup" column (it's an
approximate method, not the reference).

Usage:
    python render_table.py            # writes results/tables.tex
    python render_table.py --stdout   # also prints to stdout
"""
import json
import argparse
import os

from plot_style import METHOD_LABELS, DATASET_LABELS

DATASETS = ["glove-100", "vk-lsvd", "yambda", "synth-balanced", "synth-unbalanced"]
TAUS = ["0.05", "0.1", "0.2"]
KL_METHODS = ["topk", "centroid", "hierarchical", "ivf0", "ivf2"]
LAT_METHODS = ["exact", "topk", "centroid", "hierarchical", "ivf0", "ivf2"]


def fmt_kl(mean, ci):
    """KL ± 95% CI. Use 4 sig figs; cap tiny CIs."""
    if mean < 1e-4:
        return f"{mean:.1e} $\\pm$ {ci:.0e}"
    return f"{mean:.4f} $\\pm$ {ci:.4f}"


def fmt_lat(mean, ci):
    if mean >= 100:
        return f"{mean:.1f} $\\pm$ {ci:.1f}"
    if mean >= 10:
        return f"{mean:.2f} $\\pm$ {ci:.2f}"
    return f"{mean:.3f} $\\pm$ {ci:.3f}"


def fmt_speedup(s):
    if s >= 100:
        return f"{s:.0f}$\\times$"
    if s >= 10:
        return f"{s:.1f}$\\times$"
    return f"{s:.2f}$\\times$"


def rank_decorate(values, fmts):
    """Bold best (lowest), underline 2nd best. Inputs are pre-formatted strings."""
    finite = [(i, v) for i, v in enumerate(values) if v is not None]
    finite.sort(key=lambda x: x[1])
    out = list(fmts)
    if len(finite) >= 1:
        i = finite[0][0]
        out[i] = f"\\textbf{{{out[i]}}}"
    if len(finite) >= 2:
        i = finite[1][0]
        out[i] = f"\\underline{{{out[i]}}}"
    return out


def kl_table(data, tau):
    header_methods = " & ".join(METHOD_LABELS[m] for m in KL_METHODS)
    lines = [
        f"% KL(approx || exact) at tau={tau}. Lower is better. Bold = best, underlined = 2nd best.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{KL$(p_{{\\text{{approx}}}} \\,\\|\\, p_{{\\text{{exact}}}})$ at $\\tau={tau}$, $C=1024$, $n_q=100$. Lower is better; mean $\\pm$ 95\\% CI. \\textbf{{Bold}} = best, \\underline{{underlined}} = second best.}}",
        f"\\label{{tab:kl-tau{tau.replace('.','')}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * len(KL_METHODS)}}}",
        "\\toprule",
        f"Dataset & {header_methods} \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        if ds not in data:
            continue
        row = data[ds]["by_tau"].get(tau)
        if row is None:
            continue
        means = [row[m]["kl_mean"] for m in KL_METHODS]
        cis = [1.96 * row[m]["kl_std"] / (row[m].get("nq", 100) ** 0.5)
               if "kl_ci95" not in row[m] else row[m]["kl_ci95"] for m in KL_METHODS]
        # data already has time_ci95_ms but not kl_ci95; compute from std and nq.
        nq = data[ds].get("nq", 100)
        cis = [1.96 * row[m]["kl_std"] / (nq ** 0.5) for m in KL_METHODS]
        fmts = [fmt_kl(m, c) for m, c in zip(means, cis)]
        decorated = rank_decorate(means, fmts)
        lines.append(f"{DATASET_LABELS[ds]} & {' & '.join(decorated)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def lat_table(data):
    """Single latency table: mean over tau (latency is tau-invariant in practice)."""
    header_methods = " & ".join(METHOD_LABELS[m] for m in LAT_METHODS)
    lines = [
        "% Per-query latency (ms), averaged across tau. Speedup = exact / method.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-query latency in ms, averaged over $\\tau \\in \\{0.05, 0.1, 0.2\\}$, $C=1024$, $n_q=100$. Mean $\\pm$ 95\\% CI. Speedup is relative to Exact softmax. \\textbf{Bold} = fastest approximate method, \\underline{underlined} = second fastest (excludes Exact). Latency is essentially $\\tau$-invariant: cluster-sampling and member-softmax cost is dominated by linear-algebra structure, not the temperature.}",
        "\\label{tab:latency}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * len(LAT_METHODS)}}}",
        "\\toprule",
        f"Dataset & {header_methods} \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        if ds not in data:
            continue
        by_tau = data[ds]["by_tau"]
        # Average mean and CI across taus (CIs averaged is a fine approximation for invariant quantities).
        means, cis = [], []
        for m in LAT_METHODS:
            ms = [by_tau[t][m]["time_mean_ms"] for t in TAUS if t in by_tau]
            cs = [by_tau[t][m]["time_ci95_ms"] for t in TAUS if t in by_tau]
            means.append(sum(ms) / len(ms))
            cis.append(sum(cs) / len(cs))
        exact_mean = means[0]
        # Format each cell. For approximate methods, append speedup parenthetical.
        cells = []
        for m, mean, ci in zip(LAT_METHODS, means, cis):
            base = fmt_lat(mean, ci)
            if m == "exact":
                cells.append(base)
            else:
                spd = exact_mean / mean if mean > 0 else 0
                cells.append(f"{base} ({fmt_speedup(spd)})")
        # Rank only the approximate methods on latency.
        approx_means = means[1:]
        ranked = rank_decorate(approx_means, cells[1:])
        cells = [cells[0]] + ranked
        lines.append(f"{DATASET_LABELS[ds]} & {' & '.join(cells)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/eval_table.json")
    ap.add_argument("--output", default="results/tables.tex")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    parts = ["% Auto-generated by render_table.py from results/eval_table.json", ""]
    for tau in TAUS:
        parts.append(kl_table(data, tau))
    parts.append(lat_table(data))
    out = "\n".join(parts)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(out)
    print(f"Wrote {args.output}")
    if args.stdout:
        print()
        print(out)


if __name__ == "__main__":
    main()
