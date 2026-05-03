#!/usr/bin/env python3
"""
Step 2: Analyze framework classification results and generate figures.

Input:  framework_classifications.jsonl (from step1)
Output: Multiple publication-quality figures

Figures:
  - Figure A: Heatmap — Deonto/Conseq scores × coalition condition × phase
  - Figure B: Framework shift (Δ consequentialist) by intergroup pressure level
  - Figure C: Sacred value intensity by coalition condition
  - Figure D: Framework transition Sankey/alluvial
  - Figure E: Intergroup pressure gradient — continuous scores
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})


def load_classifications(path="framework_classifications.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("base_deonto") is not None and r.get("final_deonto") is not None:
                    records.append(r)
    return pd.DataFrame(records)


def add_derived_columns(df):
    """Add intergroup pressure level and derived metrics."""

    # Parse ratio into n_sup, n_opp
    def parse_ratio(ratio_str):
        parts = ratio_str.split(":")
        return int(parts[0]), int(parts[1])

    df[["n_sup", "n_opp"]] = df["ratio"].apply(
        lambda x: pd.Series(parse_ratio(x))
    )
    total_peers = df["n_sup"] + df["n_opp"]

    # Intergroup pressure level (0 = pure ingroup, 1 = pure outgroup)
    df["intergroup_pressure"] = df["n_opp"] / total_peers

    # Coalition labels
    def coalition_label(row):
        r = row["ratio"]
        if r in ("3:0", "2:0"):
            return "Full Support"
        elif r in ("2:1", "1:0"):  # 1:0 shouldn't exist but just in case
            return "Majority Support"
        elif r in ("1:2", "0:1"):
            return "Majority Opposition"
        elif r in ("0:3", "0:2"):
            return "Full Opposition"
        elif r == "1:1":
            return "Equal Split"
        else:
            return "Other"
    df["coalition"] = df.apply(coalition_label, axis=1)

    # Intergroup category (for the narrative)
    def intergroup_category(row):
        r = row["ratio"]
        if r in ("3:0", "2:0"):
            return "No intergroup\n(pure ingroup)"
        elif r in ("2:1", "1:1"):
            return "Partial intergroup\n(mixed coalition)"
        elif r in ("1:2",):
            return "Majority outgroup\n(coalition minority)"
        elif r in ("0:3", "0:2"):
            return "Full intergroup\n(isolated)"
        else:
            return "Other"
    df["intergroup_cat"] = df.apply(intergroup_category, axis=1)

    # Shifts
    df["delta_deonto"] = df["final_deonto"] - df["base_deonto"]
    df["delta_conseq"] = df["final_conseq"] - df["base_conseq"]
    df["delta_sacred"] = df["final_sacred"] - df["base_sacred"]

    # Consequentialist dominance ratio
    df["base_conseq_dominance"] = df["base_conseq"] - df["base_deonto"]
    df["final_conseq_dominance"] = df["final_conseq"] - df["final_deonto"]
    df["delta_conseq_dominance"] = df["final_conseq_dominance"] - df["base_conseq_dominance"]

    return df


# ═══════════════════════════════════════════════════════════════════
# FIGURE A: HEATMAP — Framework Scores × Condition × Phase
# ═══════════════════════════════════════════════════════════════════
def figure_A_heatmap(df):
    """
    Heatmap comparing baseline vs final deontological & consequentialist scores
    across coalition conditions. This is the key visualization for the Greene narrative.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Order conditions by intergroup pressure
    ratio_order = ["3:0", "2:0", "2:1", "1:1", "1:2", "0:2", "0:3"]
    ratio_order = [r for r in ratio_order if r in df["ratio"].values]

    # Build heatmap data
    metrics = ["base_deonto", "base_conseq", "final_deonto", "final_conseq",
               "delta_deonto", "delta_conseq", "base_sacred", "final_sacred", "delta_sacred"]

    # ── Left panel: Baseline vs Final scores ──
    ax = axes[0]
    col_labels = ["Deonto\n(base)", "Conseq\n(base)", "Deonto\n(final)", "Conseq\n(final)"]
    data_cols = ["base_deonto", "base_conseq", "final_deonto", "final_conseq"]

    heatmap_data = np.zeros((len(ratio_order), len(data_cols)))
    for i, ratio in enumerate(ratio_order):
        sub = df[df["ratio"] == ratio]
        for j, col in enumerate(data_cols):
            heatmap_data[i, j] = sub[col].mean()

    im = ax.imshow(heatmap_data, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(len(ratio_order)))
    ax.set_yticklabels([f"{r}\n(n={len(df[df['ratio']==r])})" for r in ratio_order], fontsize=9)
    ax.set_ylabel("Support : Opposition Ratio", fontsize=11)
    ax.set_title("A. Framework Scores by Condition\n(Baseline vs Final)", fontweight="bold")

    # Annotate cells
    for i in range(len(ratio_order)):
        for j in range(len(data_cols)):
            val = heatmap_data[i, j]
            color = "white" if val > 0.6 or val < 0.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    # Divider between base and final
    ax.axvline(1.5, color="white", linewidth=3)

    plt.colorbar(im, ax=ax, label="Score (0-1)", shrink=0.8)

    # ── Right panel: Deltas (shifts) ──
    ax = axes[1]
    delta_labels = ["Δ Deonto", "Δ Conseq", "Δ Sacred"]
    delta_cols = ["delta_deonto", "delta_conseq", "delta_sacred"]

    delta_data = np.zeros((len(ratio_order), len(delta_cols)))
    for i, ratio in enumerate(ratio_order):
        sub = df[df["ratio"] == ratio]
        for j, col in enumerate(delta_cols):
            delta_data[i, j] = sub[col].mean()

    max_abs = max(abs(delta_data.min()), abs(delta_data.max()), 0.15)
    im2 = ax.imshow(delta_data, cmap="RdBu_r", aspect="auto",
                     vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(delta_labels)))
    ax.set_xticklabels(delta_labels, fontsize=10, ha="center")
    ax.set_yticks(range(len(ratio_order)))
    ax.set_yticklabels([r for r in ratio_order], fontsize=9)
    ax.set_title("B. Framework Shifts (Δ = Final − Baseline)\n"
                 "Red = increase, Blue = decrease", fontweight="bold")

    for i in range(len(ratio_order)):
        for j in range(len(delta_cols)):
            val = delta_data[i, j]
            sign = "+" if val > 0 else ""
            color = "white" if abs(val) > max_abs * 0.5 else "black"
            ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im2, ax=ax, label="Δ Score", shrink=0.8)

    # Arrow annotation: intergroup pressure gradient
    fig.text(0.02, 0.5, "← More ingroup support    |    More outgroup pressure →",
             fontsize=9, rotation=90, va="center", ha="center", color="#666666",
             fontstyle="italic")

    fig.suptitle("Coalition Structure and Moral Framework:\n"
                 "Intergroup Pressure Drives Consequentialist Shift",
                 fontsize=14, fontweight="bold", y=1.04)

    plt.tight_layout()
    plt.savefig("fig_framework_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# FIGURE B: Framework Shift Bars by Intergroup Category
# ═══════════════════════════════════════════════════════════════════
def figure_B_framework_shift(df):
    """Bar chart: Δ deonto, Δ conseq, Δ sacred by intergroup category."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cat_order = [
        "No intergroup\n(pure ingroup)",
        "Partial intergroup\n(mixed coalition)",
        "Majority outgroup\n(coalition minority)",
        "Full intergroup\n(isolated)",
    ]
    cat_order = [c for c in cat_order if c in df["intergroup_cat"].values]
    colors = ["#27ae60", "#8e44ad", "#e67e22", "#c0392b"][:len(cat_order)]

    for ax_idx, (metric, label, title) in enumerate([
        ("delta_conseq", "Δ Consequentialist Score", "A. Consequentialist Shift"),
        ("delta_deonto", "Δ Deontological Score", "B. Deontological Shift"),
        ("delta_sacred", "Δ Sacred Value Intensity", "C. Sacred Language Shift"),
    ]):
        ax = axes[ax_idx]
        means = [df[df["intergroup_cat"] == cat][metric].mean() for cat in cat_order]
        sems = [df[df["intergroup_cat"] == cat][metric].sem() for cat in cat_order]
        ns = [len(df[df["intergroup_cat"] == cat]) for cat in cat_order]

        bars = ax.bar(range(len(cat_order)), means,
                      yerr=[s * 1.96 for s in sems], capsize=5,
                      color=colors, edgecolor="black", linewidth=1, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(cat_order)))
        ax.set_xticklabels(cat_order, fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=12)

        for i, (m, s, n) in enumerate(zip(means, sems, ns)):
            sign = "+" if m > 0 else ""
            y_pos = m + s * 1.96 + 0.01 if m > 0 else m - s * 1.96 - 0.01
            va = "bottom" if m > 0 else "top"
            ax.text(i, y_pos, f"{sign}{m:.3f}\n(n={n})", ha="center", va=va,
                    fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Moral Framework Migration Under Intergroup Pressure\n"
                 "Greene's prediction: opposition → consequentialist increase, "
                 "deontological decrease, sacred language reduction",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig("fig_framework_shift_bars.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# FIGURE C: Paired Dot Plot — Base vs Final Scores
# ═══════════════════════════════════════════════════════════════════
def figure_C_paired_dots(df):
    """Paired dot plot: baseline vs final consequentialist score by condition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ratio_order = ["3:0", "2:0", "2:1", "1:1", "1:2", "0:2", "0:3"]
    ratio_order = [r for r in ratio_order if r in df["ratio"].values]
    y_pos = np.arange(len(ratio_order))

    # Panel A: Consequentialist scores
    for i, ratio in enumerate(ratio_order):
        sub = df[df["ratio"] == ratio]
        bm = sub["base_conseq"].mean()
        bs = sub["base_conseq"].sem() * 1.96
        fm = sub["final_conseq"].mean()
        fs = sub["final_conseq"].sem() * 1.96

        # Arrow from baseline to final
        ax1.errorbar(bm, i - 0.1, xerr=bs, fmt="o", color="#3498db",
                    markersize=8, capsize=3, linewidth=1.2, zorder=3)
        ax1.errorbar(fm, i + 0.1, xerr=fs, fmt="s", color="#e74c3c",
                    markersize=8, capsize=3, linewidth=1.2, zorder=3)
        ax1.annotate("", xy=(fm, i), xytext=(bm, i),
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=1, alpha=0.6))

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{r} (n={len(df[df['ratio']==r])})" for r in ratio_order])
    ax1.set_xlabel("Consequentialist Score", fontsize=11)
    ax1.set_title("A. Consequentialist Score\nBaseline (●) → Final (■)", fontweight="bold")
    ax1.legend(["Baseline", "Final"], fontsize=9, loc="lower right")

    # Panel B: Deontological scores
    for i, ratio in enumerate(ratio_order):
        sub = df[df["ratio"] == ratio]
        bm = sub["base_deonto"].mean()
        bs = sub["base_deonto"].sem() * 1.96
        fm = sub["final_deonto"].mean()
        fs = sub["final_deonto"].sem() * 1.96

        ax2.errorbar(bm, i - 0.1, xerr=bs, fmt="o", color="#3498db",
                    markersize=8, capsize=3, linewidth=1.2, zorder=3)
        ax2.errorbar(fm, i + 0.1, xerr=fs, fmt="s", color="#e74c3c",
                    markersize=8, capsize=3, linewidth=1.2, zorder=3)
        ax2.annotate("", xy=(fm, i), xytext=(bm, i),
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=1, alpha=0.6))

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ratio_order)
    ax2.set_xlabel("Deontological Score", fontsize=11)
    ax2.set_title("B. Deontological Score\nBaseline (●) → Final (■)", fontweight="bold")

    # Gradient annotation
    fig.text(0.02, 0.5, "← Ingroup    Outgroup →", fontsize=9, rotation=90,
             va="center", ha="center", color="#888888", fontstyle="italic")

    fig.suptitle("Framework Migration: Baseline → Post-Discussion\n"
                 "Arrows show direction and magnitude of shift",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("fig_framework_paired.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# FIGURE D: Intergroup Pressure Gradient (continuous)
# ═══════════════════════════════════════════════════════════════════
def figure_D_gradient(df):
    """Scatter + trend: framework shift vs intergroup pressure (continuous)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    pressure_vals = sorted(df["intergroup_pressure"].unique())

    for ax, metric, label, title, color in [
        (ax1, "delta_conseq", "Δ Consequentialist", "A. Consequentialist Shift", "#e74c3c"),
        (ax2, "delta_deonto", "Δ Deontological", "B. Deontological Shift", "#3498db"),
        (ax3, "delta_sacred", "Δ Sacred Intensity", "C. Sacred Value Shift", "#8e44ad"),
    ]:
        means = []
        cis = []
        for p in pressure_vals:
            sub = df[df["intergroup_pressure"] == p]
            means.append(sub[metric].mean())
            cis.append(1.96 * sub[metric].sem())

        ax.errorbar(pressure_vals, means, yerr=cis,
                    fmt="o-", color=color, markersize=8, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)

        # Trend line
        x_arr = np.array(pressure_vals)
        y_arr = np.array(means)
        if len(x_arr) >= 3:
            z = np.polyfit(x_arr, y_arr, 1)
            p_line = np.poly1d(z)
            ax.plot(x_arr, p_line(x_arr), "--", color=color, alpha=0.4, linewidth=1.5)
            # Annotate slope
            ax.text(0.95, 0.05, f"slope = {z[0]:+.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color=color, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))

        ax.set_xlabel("Intergroup Pressure\n(fraction of peers opposing)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Moral Framework as a Function of Intergroup Pressure\n"
                 "Continuous pressure gradient from pure ingroup → pure outgroup",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig("fig_framework_gradient.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# FIGURE E: Framework Transition Matrix
# ═══════════════════════════════════════════════════════════════════
def figure_E_transition(df):
    """Framework transition matrix: base_framework → final_framework by condition."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ratio_groups = {
        "Full Support\n(3:0 / 2:0)": df[df["ratio"].isin(["3:0", "2:0"])],
        "Mixed\n(2:1 / 1:1)": df[df["ratio"].isin(["2:1", "1:1"])],
        "Majority Opp\n(1:2)": df[df["ratio"].isin(["1:2"])],
        "Full Opposition\n(0:3 / 0:2)": df[df["ratio"].isin(["0:3", "0:2"])],
    }

    fw_labels = ["Deontological", "Hybrid", "Consequentialist", "Neutral"]

    for ax, (group_name, sub) in zip(axes.flat, ratio_groups.items()):
        n = len(sub)
        if n == 0:
            ax.set_title(f"{group_name} (n=0)")
            continue

        # Build transition matrix
        matrix = np.zeros((len(fw_labels), len(fw_labels)))
        for _, row in sub.iterrows():
            bf = row.get("base_framework", "Neutral")
            ff = row.get("final_framework", "Neutral")
            if bf in fw_labels and ff in fw_labels:
                bi = fw_labels.index(bf)
                fi = fw_labels.index(ff)
                matrix[bi, fi] += 1

        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_pct = matrix / row_sums * 100

        im = ax.imshow(matrix_pct, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(fw_labels)))
        ax.set_xticklabels(["Deonto", "Hybrid", "Conseq", "Neutral"], fontsize=8, rotation=30)
        ax.set_yticks(range(len(fw_labels)))
        ax.set_yticklabels(["Deonto", "Hybrid", "Conseq", "Neutral"], fontsize=8)
        ax.set_xlabel("Final Framework", fontsize=9)
        ax.set_ylabel("Baseline Framework", fontsize=9)
        ax.set_title(f"{group_name}\n(n={n})", fontweight="bold", fontsize=11)

        for i in range(len(fw_labels)):
            for j in range(len(fw_labels)):
                val = matrix_pct[i, j]
                count = int(matrix[i, j])
                if count > 0:
                    color = "white" if val > 50 else "black"
                    ax.text(j, i, f"{val:.0f}%\n({count})", ha="center", va="center",
                            fontsize=7.5, color=color, fontweight="bold")

    fig.suptitle("Framework Transition Matrices by Coalition Condition\n"
                 "Row = baseline framework, Column = final framework (% of row)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("fig_framework_transitions.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════
def print_summary(df):
    print("=" * 70)
    print("FRAMEWORK ANALYSIS SUMMARY (LLM-CLASSIFIED)")
    print("=" * 70)

    for ratio in sorted(df["ratio"].unique()):
        sub = df[df["ratio"] == ratio]
        n = len(sub)
        print(f"\nRatio {ratio} (n={n}):")
        print(f"  Baseline: deonto={sub['base_deonto'].mean():.3f}, "
              f"conseq={sub['base_conseq'].mean():.3f}, "
              f"sacred={sub['base_sacred'].mean():.3f}")
        print(f"  Final:    deonto={sub['final_deonto'].mean():.3f}, "
              f"conseq={sub['final_conseq'].mean():.3f}, "
              f"sacred={sub['final_sacred'].mean():.3f}")
        print(f"  Δ deonto={sub['delta_deonto'].mean():+.4f} ± {sub['delta_deonto'].sem():.4f}")
        print(f"  Δ conseq={sub['delta_conseq'].mean():+.4f} ± {sub['delta_conseq'].sem():.4f}")
        print(f"  Δ sacred={sub['delta_sacred'].mean():+.4f} ± {sub['delta_sacred'].sem():.4f}")

    print("\n" + "=" * 70)
    print("INTERGROUP CATEGORY SUMMARY")
    print("=" * 70)
    for cat in df["intergroup_cat"].unique():
        sub = df[df["intergroup_cat"] == cat]
        n = len(sub)
        print(f"\n{cat.replace(chr(10), ' ')} (n={n}):")
        print(f"  Δ conseq = {sub['delta_conseq'].mean():+.4f} ± {sub['delta_conseq'].sem():.4f}")
        print(f"  Δ deonto = {sub['delta_deonto'].mean():+.4f} ± {sub['delta_deonto'].sem():.4f}")
        print(f"  Δ sacred = {sub['delta_sacred'].mean():+.4f} ± {sub['delta_sacred'].sem():.4f}")


def main():
    print("Loading classifications...")
    df = load_classifications()
    print(f"Loaded {len(df)} classified pairs")

    print("Computing derived columns...")
    df = add_derived_columns(df)

    print(f"Sources: {dict(df['source'].value_counts())}")
    print(f"Ratios: {dict(df['ratio'].value_counts())}")

    # Generate all figures
    print("\nFigure A: Framework Heatmap...")
    figure_A_heatmap(df)

    print("Figure B: Framework Shift Bars...")
    figure_B_framework_shift(df)

    print("Figure C: Paired Dot Plot...")
    figure_C_paired_dots(df)

    print("Figure D: Pressure Gradient...")
    figure_D_gradient(df)

    print("Figure E: Transition Matrices...")
    figure_E_transition(df)

    print_summary(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
