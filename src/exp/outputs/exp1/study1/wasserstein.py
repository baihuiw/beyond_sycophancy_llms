import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
    "grid.linewidth": 0.4, "grid.alpha": 0.25,
})


# ── Core ──
def w1(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() > 0: p /= p.sum()
    if q.sum() > 0: q /= q.sum()
    return np.sum(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]))

def d7(d):
    a = np.array([float(d.get(str(i), 0) or 0) for i in range(1, 8)])
    return a / a.sum() if a.sum() > 0 else a

# ── Hatching helper for textured bars ──
HATCHES = ['///', '\\\\\\', '...', 'xxx', '|||', '---', '+++', 'ooo']

def textured_bar(ax, x, val, yerr, color, width=0.6, hatch=None, label=None, alpha=0.82):
    bar = ax.bar(x, val, width, yerr=yerr, capsize=4,
                 color=color, edgecolor="white", linewidth=1.2, alpha=alpha, label=label,
                 zorder=3)
    if hatch:
        for b in bar:
            b.set_hatch(hatch)
            b.set_edgecolor(color)  # hatch color matches bar
    return bar

def textured_barh(ax, y, val, xerr, color, height=0.55, hatch=None, label=None, alpha=0.82):
    bar = ax.barh(y, val, height, xerr=xerr, capsize=3,
                  color=color, edgecolor="white", linewidth=1.2, alpha=alpha, label=label,
                  zorder=3)
    if hatch:
        for b in bar:
            b.set_hatch(hatch)
            b.set_edgecolor(color)
    return bar

def annotate_bar(ax, x, val, err, fmt=".3f", offset=0.015, fontsize=9, color="black"):
    sign = "+" if val > 0 else ""
    txt = f"{val:{fmt}}"
    y = val + err + offset if val >= 0 else val - err - offset
    va = "bottom" if val >= 0 else "top"
    ax.text(x, y, txt, ha="center", va=va, fontsize=fontsize,
            fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


# ═══════════════════════════════════════════════════════════
# STUDY 1 — Panels A + B only
# ═══════════════════════════════════════════════════════════
with open("study1_results.jsonl") as f:
    s1_raw = [json.loads(l) for l in f if l.strip()]

mo = {r["metadata"]["qid"]: d7(r["response"]["probs"])
      for r in s1_raw if r["metadata"]["cue_direction"] == "none"}

s1 = []
for r in s1_raw:
    m = r["metadata"]
    if m["cue_direction"] == "none": continue
    qid = m["qid"]
    if qid not in mo: continue
    d = m["cue_direction"]
    ct = ("Oppose" if "oppose" in d else "Reinforce" if ("reinforce" in d and "weak" not in d)
          else "Pull" if "pull" in d else "Weak Reinforce" if "weak" in d else "Other")
    try: dist = int(m["cue_distance"])
    except: dist = None
    s1.append({"ct": ct, "dist": dist, "w1": w1(mo[qid], d7(r["response"]["probs"]))})

df1 = pd.DataFrame(s1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})

# Panel A
ax = axes[0]
cts = ["Pull", "Oppose", "Reinforce", "Weak Reinforce"]
ct_labels = ["Pull\n(from neutral)", "Oppose", "Reinforce", "Weak\nReinforce"]
ct_colors = ["#2980b9", "#c0392b", "#27ae60", "#7f8c8d"]
ct_hatches = ["///", "\\\\\\", "...", "---"]
for i, (ct, col, h) in enumerate(zip(cts, ct_colors, ct_hatches)):
    sub = df1[df1["ct"] == ct]
    m, se = sub["w1"].mean(), sub["w1"].sem() * 1.96
    textured_bar(ax, i, m, se, col, width=0.62, hatch=h)
    annotate_bar(ax, i, m, se, color=col)
    ax.text(i, -0.04, f"n={len(sub)}", ha="center", va="top", fontsize=8, color="#888")

ax.set_xticks(range(4)); ax.set_xticklabels(ct_labels, fontsize=10)
ax.set_ylabel("Wasserstein Distance  (W₁)", fontsize=12)
ax.set_title("A.  Distributional Movement by Cue Type", fontweight="bold", fontsize=13)
ax.set_ylim(-0.06, ax.get_ylim()[1] * 1.15)
ax.grid(True, axis="y")

# Panel B — oppose by distance
ax = axes[1]
opp = df1[(df1["ct"] == "Oppose") & df1["dist"].notna()]
dists = sorted(opp["dist"].unique())
ms = [opp[opp["dist"] == d]["w1"].mean() for d in dists]
ses = [opp[opp["dist"] == d]["w1"].sem() * 1.96 for d in dists]
ns = [len(opp[opp["dist"] == d]) for d in dists]

ax.fill_between(dists, [m - s for m, s in zip(ms, ses)],
                [m + s for m, s in zip(ms, ses)],
                alpha=0.15, color="#c0392b", zorder=1)
ax.plot(dists, ms, "o-", color="#c0392b", markersize=9, linewidth=2.5,
        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
for d, m, n in zip(dists, ms, ns):
    ax.annotate(f"{m:.3f}\n(n={n})", (d, m), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=8.5, fontweight="bold", color="#c0392b",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax.set_xlabel("Cue Distance  (|cue position − baseline|)", fontsize=12)
ax.set_ylabel("Wasserstein Distance  (W₁)", fontsize=12)
ax.set_title("B.  Oppose Cues: W₁ Increases with Distance", fontweight="bold", fontsize=13)
ax.grid(True)

fig.suptitle("Study 1 — Wasserstein Distance: Distributional Movement Under Persuasive Cues\n"
             "Baseline = memory-only condition (same dilemma)",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig(f"fig_wasserstein_study1_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_wasserstein_study1_v2.png")


# ═══════════════════════════════════════════════════════════
# STUDY 2 — Panel A only: D0→D1 and D1→D2
# ═══════════════════════════════════════════════════════════
with open("/Users/wangbaihui/modeling_distributions_thesis/src/exp1_v2/outputs/exp1/study1/baseline_outputs_explicit_format.jsonl") as f:
    bl = {r["metadata"]["qid"]: d7(r["response"]["probs"])
          for r in (json.loads(l) for l in f if l.strip())}
with open("/Users/wangbaihui/modeling_distributions_thesis/src/exp1_v2/outputs/exp1/study2/study2_v3_experimental_outputs.jsonl") as f:
    s2_raw = [json.loads(l) for l in f if l.strip()]

trials = defaultdict(dict)
for r in s2_raw:
    m = r["metadata"]
    trials[(m["qid"], m["irrational_choice"])][m["condition"]] = r

s2 = []
for (qid, irr), conds in trials.items():
    if qid not in bl: continue
    d1r, d2r = conds.get("memory_justify"), conds.get("memory_justify_correction")
    if not d1r or not d2r: continue
    d0, d1, d2 = bl[qid], d7(d1r["response"]["probs"]), d7(d2r["response"]["probs"])
    mode = int(d1r["metadata"]["mode_choice"])
    s2.append({
        "committed": d1[irr-1] > d1[mode-1],
        "w_commit": w1(d0, d1), "w_correct": w1(d1, d2),
    })

df2 = pd.DataFrame(s2)
com = df2[df2["committed"]]

fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(2)
ww = 0.32
groups = [("All trials", df2, "#5dade2", "///"), ("Committed only", com, "#c0392b", "\\\\\\")]

for gi, (label, sub, col, h) in enumerate(groups):
    vals = [sub["w_commit"].mean(), sub["w_correct"].mean()]
    errs = [sub["w_commit"].sem()*1.96, sub["w_correct"].sem()*1.96]
    textured_bar(ax, x + (gi - 0.5) * ww, vals, errs, col, width=ww * 0.9, hatch=h, label=label)
    for i, (v, e) in enumerate(zip(vals, errs)):
        annotate_bar(ax, x[i] + (gi-0.5)*ww, v, e, color=col)

ax.set_xticks(x)
ax.set_xticklabels(["D0 → D1\n(commitment)", "D1 → D2\n(correction)"], fontsize=11)
ax.set_ylabel("Wasserstein Distance  (W₁)", fontsize=12)
ax.set_title("Study 2 — Commitment & Correction: Distributional Displacement", fontweight="bold", fontsize=13)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.grid(True, axis="y")
ax.text(0.5, -0.12, f"All trials: N={len(df2)}  |  Committed (P(irr)>P(mode) at D1): N={len(com)} ({len(com)/len(df2)*100:.0f}%)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#666")

plt.tight_layout()
plt.savefig(f"fig_wasserstein_study2_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_wasserstein_study2_v2.png")


# ═══════════════════════════════════════════════════════════
# STUDY 3 — 4-AGENT: W₁ boxplots + Unanimous groups
# ═══════════════════════════════════════════════════════════
with open("/Users/wangbaihui/modeling_distributions_thesis/src/exp1_v2/outputs/exp1/study3/full_run/summaries.jsonl") as f:
    s3_4ag = [json.loads(l) for l in f if l.strip()]

four_ratios = {"3:0", "2:1", "1:2", "0:3"}
s3r = []
for r in s3_4ag:
    if r["ratio"] not in four_ratios: continue
    bd, fd = d7(r["baseline_dist_probs"]), d7(r["final_dist_probs"])
    s3r.append({"condition": r["condition"], "ratio": r["ratio"], "w1": w1(bd, fd)})
df3 = pd.DataFrame(s3r)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.3]})

# Panel A: W₁ boxplots by ratio
ax = axes[0]
ratio_order = ["3:0", "2:1", "1:2", "0:3"]
colors4 = ["#27ae60", "#8e44ad", "#e67e22", "#c0392b"]
data = [df3[df3["ratio"] == r]["w1"].values for r in ratio_order]

bp = ax.boxplot(data, positions=range(4), widths=0.52, patch_artist=True,
                showmeans=True, meanline=True,
                meanprops=dict(color="black", linewidth=1.8, linestyle="--"),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color="#555", linewidth=1.2),
                capprops=dict(color="#555", linewidth=1.2),
                flierprops=dict(marker=".", markersize=2, alpha=0.25, color="#aaa"))
for patch, col, h in zip(bp["boxes"], colors4, ["///", "\\\\\\", "...", "xxx"]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
    patch.set_hatch(h); patch.set_edgecolor(col)
    patch.set_linewidth(1.2)

for i, (r, col) in enumerate(zip(ratio_order, colors4)):
    sub = df3[df3["ratio"] == r]
    m = sub["w1"].mean()
    ax.text(i, ax.get_ylim()[1] * 0.02 + sub["w1"].quantile(0.95),
            f"μ={m:.3f}\nn={len(sub)}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=col,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax.set_xticks(range(4))
ax.set_xticklabels(["3:0\n(Full Sup)", "2:1\n(Maj Sup)", "1:2\n(Maj Opp)", "0:3\n(Full Opp)"], fontsize=10)
ax.set_ylabel("W₁  (baseline → final)", fontsize=12)
ax.set_title("A.  W₁ Distribution by Coalition", fontweight="bold", fontsize=13)
ax.grid(True, axis="y")

# Panel B: Unanimous groups
ax = axes[1]
unan4 = [
    ("opp3_extreme", "3 opp (extreme)", "#7b241c"),
    ("opp3_moderate", "3 opp (moderate)", "#c0392b"),
    ("opp3_mild", "3 opp (mild)", "#e6b0aa"),
    ("sup3_extreme", "3 sup (extreme)", "#f0b27a"),
    ("sup3_moderate", "3 sup (moderate)", "#5dade2"),
    ("sup3_close", "3 sup (close)", "#2e86c1"),
    ("sup3_match", "3 sup (match)", "#1a5276"),
]

y_pos = np.arange(len(unan4))
for yi, (prefix, label, col) in enumerate(unan4):
    sub = df3[df3["condition"].str.startswith(prefix)]
    if len(sub) == 0: continue
    m, se = sub["w1"].mean(), sub["w1"].sem() * 1.96
    h = HATCHES[yi % len(HATCHES)]
    textured_barh(ax, yi, m, se, col, height=0.55, hatch=h)
    ax.text(m + se + 0.015, yi, f"{m:.3f}  (n={len(sub)})", ha="left", va="center",
            fontsize=9, fontweight="bold", color=col,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax.axhline(2.5, color="#aaa", linestyle=":", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([u[1] for u in unan4], fontsize=10)
ax.set_xlabel("W₁  (baseline → final)", fontsize=12)
ax.set_title("B.  Unanimous Peer Groups (N=3)\nSame ratio, different positions", fontweight="bold", fontsize=13)
ax.grid(True, axis="x")
ax.text(0.97, 0.15, "Support →", transform=ax.transAxes, fontsize=9, ha="right", color="#1a5276", fontstyle="italic")
ax.text(0.97, 0.75, "← Opposition", transform=ax.transAxes, fontsize=9, ha="right", color="#c0392b", fontstyle="italic")

fig.suptitle("Study 3 (4-Agent) — Wasserstein Distance: Coalition Geometry Shapes Distributional Movement",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"fig_wasserstein_study3_4ag_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_wasserstein_study3_4ag_v2.png")


# ═══════════════════════════════════════════════════════════
# STUDY 3 — 3-AGENT: Unanimous groups + W₁ boxplots
# ═══════════════════════════════════════════════════════════
with open("/Users/wangbaihui/modeling_distributions_thesis/src/exp1_v2/outputs/exp1/study3/full_run/summaries_2.jsonl") as f:
    t2 = [json.loads(l) for l in f if l.strip()]

a1b = {(r["qid"], r["condition"]): r for r in t2 if r["agent"] == "A1" and r["round"] == 0}
a1f = {(r["qid"], r["condition"]): r for r in t2 if r["agent"] == "A1" and r["round"] == 1}

s3b = []
for key in a1b:
    if key not in a1f: continue
    b, f = a1b[key], a1f[key]
    bp, fp = b.get("a1_dist_probs", {}), f.get("a1_dist_probs_final", {})
    if not bp or not fp: continue
    if b["parsed_choice"] is None: continue
    s3b.append({"condition": f["condition"], "ratio": f["ratio"], "w1": w1(d7(bp), d7(fp))})

df3b = pd.DataFrame(s3b)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]})

# Panel A: Unanimous groups (3-agent)
ax = axes[0]
unan3 = [
    ("3ag_opp2_extreme", "2 opp (extreme)", "#7b241c"),
    ("3ag_opp2_moderate", "2 opp (moderate)", "#c0392b"),
    ("3ag_opp2_mild", "2 opp (mild)", "#e6b0aa"),
    ("3ag_sup2_extreme", "2 sup (extreme)", "#f0b27a"),
    ("3ag_sup2_moderate", "2 sup (moderate)", "#5dade2"),
    ("3ag_sup2_close", "2 sup (close)", "#2e86c1"),
    ("3ag_sup2_match", "2 sup (match)", "#1a5276"),
]

y_pos = np.arange(len(unan3))
for yi, (prefix, label, col) in enumerate(unan3):
    sub = df3b[df3b["condition"].str.startswith(prefix)]
    if len(sub) == 0: continue
    m, se = sub["w1"].mean(), sub["w1"].sem() * 1.96
    h = HATCHES[yi % len(HATCHES)]
    textured_barh(ax, yi, m, se, col, height=0.55, hatch=h)
    ax.text(m + se + 0.015, yi, f"{m:.3f}  (n={len(sub)})", ha="left", va="center",
            fontsize=9, fontweight="bold", color=col,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax.axhline(2.5, color="#aaa", linestyle=":", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([u[1] for u in unan3], fontsize=10)
ax.set_xlabel("W₁  (baseline → final)", fontsize=12)
ax.set_title("A.  Unanimous Peer Groups (N=2)\nSame ratio, different positions", fontweight="bold", fontsize=13)
ax.grid(True, axis="x")
ax.text(0.97, 0.15, "Support →", transform=ax.transAxes, fontsize=9, ha="right", color="#1a5276", fontstyle="italic")
ax.text(0.97, 0.75, "← Opposition", transform=ax.transAxes, fontsize=9, ha="right", color="#c0392b", fontstyle="italic")

# Panel B: W₁ boxplots by ratio (3-agent)
ax = axes[1]
ratio_order_3 = ["2:0", "1:1", "0:2"]
colors3 = ["#27ae60", "#8e44ad", "#c0392b"]
data3 = [df3b[df3b["ratio"] == r]["w1"].values for r in ratio_order_3]

bp = ax.boxplot(data3, positions=range(3), widths=0.52, patch_artist=True,
                showmeans=True, meanline=True,
                meanprops=dict(color="black", linewidth=1.8, linestyle="--"),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color="#555", linewidth=1.2),
                capprops=dict(color="#555", linewidth=1.2),
                flierprops=dict(marker=".", markersize=2, alpha=0.25, color="#aaa"))
for patch, col, h in zip(bp["boxes"], colors3, ["///", "\\\\\\", "xxx"]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
    patch.set_hatch(h); patch.set_edgecolor(col)
    patch.set_linewidth(1.2)

for i, (r, col) in enumerate(zip(ratio_order_3, colors3)):
    sub = df3b[df3b["ratio"] == r]
    m = sub["w1"].mean()
    ax.text(i, sub["w1"].quantile(0.95) + 0.05,
            f"μ={m:.3f}\nn={len(sub)}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=col,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax.set_xticks(range(3))
ax.set_xticklabels(["2:0\n(Full Sup)", "1:1\n(Equal Split)", "0:2\n(Full Opp)"], fontsize=10)
ax.set_ylabel("W₁  (baseline → final)", fontsize=12)
ax.set_title("B.  W₁ Distribution by Coalition", fontweight="bold", fontsize=13)
ax.grid(True, axis="y")

fig.suptitle("Study 3 (3-Agent) — Wasserstein Distance: Coalition Effects Replicate Across Group Sizes",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"fig_wasserstein_study3_3ag_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_wasserstein_study3_3ag_v2.png")


# ═══════════════════════════════════════════════════════════
# CROSS-STUDY
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))

entries = []
# Study 1
for ct, col in [("Pull", "#2980b9"), ("Oppose", "#c0392b"),
                ("Reinforce", "#27ae60"), ("Weak Reinforce", "#7f8c8d")]:
    sub = df1[df1["ct"] == ct]
    if len(sub) > 0:
        entries.append((f"S1: {ct}", sub["w1"].mean(), sub["w1"].sem()*1.96, col, len(sub)))

# Study 2
entries.append(("S2: Commit (D0→D1)", df2["w_commit"].mean(), df2["w_commit"].sem()*1.96, "#e67e22", len(df2)))
entries.append(("S2: Correct (D1→D2)", df2["w_correct"].mean(), df2["w_correct"].sem()*1.96, "#8e44ad", len(df2)))

# Study 3 4-agent
for r, col in [("3:0","#27ae60"),("2:1","#8e44ad"),("1:2","#e67e22"),("0:3","#c0392b")]:
    sub = df3[df3["ratio"]==r]
    entries.append((f"S3 (4ag): {r}", sub["w1"].mean(), sub["w1"].sem()*1.96, col, len(sub)))

# Study 3 3-agent
for r, col in [("2:0","#27ae60"),("1:1","#8e44ad"),("0:2","#c0392b")]:
    sub = df3b[df3b["ratio"]==r]
    entries.append((f"S3 (3ag): {r}", sub["w1"].mean(), sub["w1"].sem()*1.96, col, len(sub)))

y_pos = np.arange(len(entries))
for yi, (label, m, ci, col, n) in enumerate(entries):
    h = HATCHES[yi % len(HATCHES)]
    textured_barh(ax, yi, m, ci, col, height=0.55, hatch=h)
    ax.text(m + ci + 0.015, yi, f"{m:.3f}  (n={n})", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=col,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# Dividers
n_s1 = 4; n_s2 = 2
ax.axhline(n_s1 - 0.5, color="#bbb", linestyle="--", linewidth=1)
ax.axhline(n_s1 + n_s2 - 0.5, color="#bbb", linestyle="--", linewidth=1)
ax.axhline(n_s1 + n_s2 + 4 - 0.5, color="#bbb", linestyle="--", linewidth=1)

ax.text(-0.02, (n_s1-1)/2, "Study 1", transform=ax.get_yaxis_transform(),
        fontsize=10, fontweight="bold", va="center", color="#555", rotation=90)
ax.text(-0.02, n_s1 + (n_s2-1)/2, "Study 2", transform=ax.get_yaxis_transform(),
        fontsize=10, fontweight="bold", va="center", color="#555", rotation=90)
ax.text(-0.02, n_s1 + n_s2 + 1.5, "Study 3\n(4-ag)", transform=ax.get_yaxis_transform(),
        fontsize=9, fontweight="bold", va="center", color="#555", rotation=90)
ax.text(-0.02, n_s1 + n_s2 + 5, "Study 3\n(3-ag)", transform=ax.get_yaxis_transform(),
        fontsize=9, fontweight="bold", va="center", color="#555", rotation=90)

ax.set_yticks(y_pos)
ax.set_yticklabels([e[0] for e in entries], fontsize=9.5)
ax.set_xlabel("Wasserstein Distance  (W₁)", fontsize=12)
ax.set_title("Cross-Study Comparison: Distributional Movement Across All Manipulations\n"
             "W₁ = minimum cost to transform baseline → post-manipulation on 7-point scale",
             fontweight="bold", fontsize=13)
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig(f"fig_wasserstein_cross_study_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_wasserstein_cross_study_v2.png")

print("\nAll done!")
