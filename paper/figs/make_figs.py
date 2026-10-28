"""Generate the paper figures (vector PDF) from our measured numbers. Run: python make_figs.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})
OUT = __import__("os").path.dirname(__file__)


# ---------- Fig 1: framework schematic ----------
def framework():
    fig, ax = plt.subplots(figsize=(7.8, 3.0)); ax.axis("off"); ax.set_xlim(0, 10.8); ax.set_ylim(0, 6.0)
    def box(x, y, w, h, t, c, ec="#333", lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    linewidth=lw, edgecolor=ec, facecolor=c))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.8)
    def arrow(x1, y1, x2, y2, color="#555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9))

    # Input
    box(0.1, 2.4, 1.4, 1.2, "Query\nBundle $S$", "#eef3fb")

    # Three input streams / estimators
    box(1.9, 4.3, 2.5, 1.3, "Pairwise Base\n$s_{\\mathrm{pair}}, s_{\\mathrm{cat}}, s_{\\mathrm{style}}$\n$s_{\\mathrm{bridge}}, s_{\\mathrm{contrast}}$", "#fde9e0")
    box(1.9, 2.4, 2.5, 1.3, "(1) Simplicial Incidence\nSurplus $\\hat{\\Delta}_{\\mathrm{inc}}$\n$\\mathbf{1}_{|b \\cap S| \\geq 2} \\cdot \\beta_{\\mathrm{clique}}$", "#e8f6ea")
    box(1.9, 0.5, 2.5, 1.3, "(2) Quadratic Concordance\n$\\hat{R}_{\\gamma}$ (Rao Diversity)\n$e_2(S, c) \\cdot (1 - \\mathbf{p}_S^{\\!\\top} G_S \\mathbf{p}_S)^{\\gamma}$", "#fff4d6")

    arrow(1.5, 3.0, 1.9, 4.95)
    arrow(1.5, 3.0, 1.9, 3.05)
    arrow(1.5, 3.0, 1.9, 1.15)

    # Fusion into Collaborative Y
    box(4.8, 2.0, 1.8, 2.4, "Collaborative\nEvidence\n$Y(c \\mid S) =$\n$\\sum_k w_k\\, z(s_k)$", "#e9eef2")
    arrow(4.4, 4.95, 4.8, 3.7)
    arrow(4.4, 3.05, 4.8, 3.2)
    arrow(4.4, 1.15, 4.8, 2.6)

    # Metadata Prior
    box(4.8, 4.8, 1.8, 1.0, "Metadata Prior\n$\\mu_0(c \\mid S)$", "#efe7f7")

    # Estimator 3: EB Shrinkage
    box(7.0, 2.2, 2.2, 2.6, "(3) Conjugate EB\nShrinkage $\\hat{\\theta}_{S,c}$\n$(1 - u_c)Y + u_c \\mu_0$\n\n$u_c = \\frac{\\kappa_f}{\\kappa_f + f_c}$\nExposure Router", "#f5e6d3", ec="#b35400", lw=1.2)
    arrow(6.6, 3.2, 7.0, 3.2)
    arrow(6.6, 5.3, 7.5, 4.8)

    # Output ranking
    box(9.5, 2.9, 1.2, 1.2, "Ranked\nItems", "#eef3fb")
    arrow(9.2, 3.5, 9.5, 3.5)

    ax.text(5.7, 0.4, "Validation-selected weights $w$ + finite-sample variance routing", ha="center", fontsize=7.2, style="italic", color="#555")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_framework.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- Fig 2: regime heatmap (validation-selected weights per dataset) ----------
def regime():
    sigs = ["cooc", "ppmi+csvd", "ontology", "content", "contrast"]
    data = ["POG", "POG_dense", "Spotify", "Sp_sparse"]
    W = np.array([
        [1.0, 0.0, 1.0, 0.5, 0.0],   # POG: cooc, -, cat, style, -
        [1.0, 0.0, 1.0, 0.25, 1.0],  # dense: cooc, -, cat+softcat, imgstyle, contrast
        [1.0, 1.0, 0.0, 1.0, 1.0],   # spotify: cooc, ppmi+csvd, -, audio+style, contrast
        [1.0, 0.75, 0.0, 1.0, 0.5],  # sparse
    ])
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    im = ax.imshow(W, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sigs))); ax.set_xticklabels(sigs, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(data, fontsize=8)
    for i in range(len(data)):
        for j in range(len(sigs)):
            ax.text(j, i, f"{W[i,j]:.2g}" if W[i, j] else "·", ha="center", va="center",
                    fontsize=7.5, color="white" if W[i, j] > 0.6 else "#333")
    ax.text(-0.5, -0.9, "fashion = complementarity (ontology) · music = similarity (collab+audio)",
            fontsize=6.8, style="italic", color="#444")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="selected weight")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_regime.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- Fig 3: accuracy vs compute (POG, full-ranking R@20) ----------
EPS_PARAMS = 0.02  # log-axis sentinel for the zero-trainable-parameter closed-form core (labelled "0")


def pareto():
    # (params in millions [measured], R@20 on POG full-ranking, label, marker). Ours is 0 trainable params
    # (encoder off by default); it is drawn at a left-edge sentinel on the log axis and annotated "0 params".
    pts = [(EPS_PARAMS, 0.0653, "Ours (0 params)", "o", "#c0392b"),
           (9.48, 0.0284, "CLHE", "s", "#2c7fb8"),
           (16.0, 0.0375, "RaMen", "^", "#31a354")]
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    for p, r, lab, mk, c in pts:
        ax.scatter(p, r, s=70, marker=mk, color=c, edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(lab, (p, r), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("trainable parameters (millions, log)")
    ax.set_ylabel("POG  Recall@20  (full-ranking)")
    ax.set_xlim(0.012, 40); ax.set_ylim(0.015, 0.095); ax.grid(alpha=0.3, lw=0.5)
    ax.text(0.016, 0.083, "0 params  &  better (0.0653 vs 0.0375)", fontsize=7.5, style="italic", color="#c0392b")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pareto.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- Fig 5: DDBC protocol critique (F1 on its own 100-candidate pool) ----------
def ddbc():
    groups = ["POG\nk=4", "Spotify\nk=30", "k=60", "k=90"]
    ddbc_v = [0.139, 0.282, 0.285, 0.287]
    cooc_v = [0.507, 0.471, 0.493, 0.487]
    ours_v = [0.168, 0.518, 0.537, 0.531]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.bar(x - w, ddbc_v, w, label="DDBC (ICLR'26)", color="#8e8e8e", edgecolor="k", lw=0.5)
    ax.bar(x,     cooc_v, w, label="co-occurrence (0 params)", color="#4a90d9", edgecolor="k", lw=0.5)
    ax.bar(x + w, ours_v, w, label="Ours", color="#c0392b", edgecolor="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("F1 (DDBC's 100-cand. protocol)")
    ax.legend(fontsize=7.0, frameon=False, loc="upper left")
    ax.text(-0.4, 0.60, "a parameter-free baseline beats the diffusion SOTA on its own metric",
            fontsize=6.6, style="italic", color="#444")
    ax.set_ylim(0, 0.62); ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ddbc.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- Fig 6: trainable-parameter comparison (measured) ----------
def params():
    names = ["Ours", "DDBC", "CLHE", "RaMen", "Bundle-MLLM"]
    pm = [EPS_PARAMS, 0.79, 9.5, 16.0, 7000.0]   # Ours = 0 trainable params; drawn at a log-axis sentinel, labelled "0"
    labels = ["0", "0.79M", "9.5M", "16M", "7000M"]
    colors = ["#c0392b", "#8e8e8e", "#2c7fb8", "#31a354", "#b07cc6"]
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    y = np.arange(len(names))
    ax.barh(y, pm, color=colors, edgecolor="k", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("trainable parameters (millions, log)")
    for yi, v, lab in zip(y, pm, labels):
        ax.text(v * 1.3, yi, lab, va="center", fontsize=7.5)
    ax.set_xlim(0.012, 40000); ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_params.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- Fig 4: cold/warm stratified recall (POG) ----------
def stratified():
    methods = ["popularity", "co-occurrence", "content-kNN", "Ours"]
    cold = [0.000, 0.000, 0.026, 0.074]
    warm = [0.037, 0.045, 0.012, 0.078]
    x = np.arange(len(methods)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.bar(x - w / 2, cold, w, label="cold items", color="#4a90d9", edgecolor="k", lw=0.5)
    ax.bar(x + w / 2, warm, w, label="warm items", color="#f0a500", edgecolor="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("POG  Recall@20"); ax.legend(fontsize=7.5, frameon=False)
    ax.text(0.0, 0.082, "collaborative = 0 on cold; EB shrinkage lifts cold to 0.074", fontsize=6.8, style="italic", color="#444")
    ax.set_ylim(0, 0.095); ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_stratified.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- Fig 7: weight-sensitivity sweep, two regimes (POG vs Spotify) ----------
# 100% measured: reads the JSON written by `WSWEEP=1 BUNDLE_DS=<ds> fullrank_eval.py`
# (per-signal test Recall@20 with all other weights / higher-order / EB held at the selected values).
def wsweep():
    import json, os
    RES = os.path.join(OUT, "..", "..", "results")
    # readable labels for the real internal signal keys; curves with a flat zero contribution are dropped
    LAB = {"cooc": "co-occurrence", "ppmi": "PPMI", "csvd": "PPMI-SVD", "cat": "taxonomy",
           "softcat": "soft-taxonomy", "style": "content", "audiostyle": "content (audio)",
           "imgstyle": "content (image)", "bridge": "content-bridge", "simp": "set-reasoner",
           "diff2": "diff-of-2", "pop": "popularity", "homo": "artist/album"}
    DASH = {"ppmi", "csvd", "softcat", "diff2"}
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#17becf",
               "#8c8c8c", "#bcbd22", "#e377c2", "#7f7f7f", "#1b9e77"]

    def panel(ax, path, title):
        d = json.load(open(path))
        V = d["grid"]; sel = d["selected_r20"]
        # order curves by how much they move the score, keep the legend legible
        items = [(k, y) for k, y in d["curves"].items() if any(v == v for v in y)]
        items.sort(key=lambda kv: -(max(v for v in kv[1] if v == v) - min(v for v in kv[1] if v == v)))
        for i, (k, y) in enumerate(items):
            yy = [v if v == v else None for v in y]
            ax.plot(V, yy, "--" if k in DASH else "-", color=palette[i % len(palette)],
                    marker="o", ms=3, lw=1.3, label=LAB.get(k, k))
        ax.axhline(sel, ls=":", lw=0.8, color="#444")
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("weight on one signal"); ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="lower center")
        return sel

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    s1 = panel(a1, f"{RES}/wsweep_pog.json", "POG (fashion): complementarity")
    s2 = panel(a2, f"{RES}/wsweep_spotify.json", "Spotify (music): similarity")
    a1.set_ylabel("Recall@20")
    fig.text(0.5, -0.03, "which signal matters flips by regime: taxonomy/content on fashion, "
             "co-occurrence/PPMI on music (dotted line = full model at selected weights)",
             ha="center", fontsize=6.4, style="italic", color="#444")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_wsweep.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  wsweep: POG selected R@20={s1:.4f}, Spotify selected R@20={s2:.4f} (measured)")


framework(); regime(); pareto(); stratified(); ddbc(); params(); wsweep()
print("wrote fig_framework/regime/pareto/stratified/ddbc/params/wsweep .pdf")
