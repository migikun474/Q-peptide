"""Generate publication-quality figures from executed experiment records.

Reads only what is on disk in results/experiments/. If a report is missing, the figure
that depends on it is skipped with a message -- nothing is plotted from invented data.

Run:  python -m backend.experiments.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/experiments")
FIGS = Path("results/figures")

# A small consistent style: dark-on-light, colourblind-safe, no chartjunk.
PALETTE = {
    "exact": "#444444",
    "p1": "#0072B2",
    "p2": "#009E73",
    "p3": "#CC79A7",
    "uniform": "#999999",
    "sa": "#009E73",
    "greedy": "#E69F00",
    "local": "#D55E00",
    "random": "#999999",
    "accent": "#0072B2",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.autolayout": True,
    })


def _load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        print(f"  skipped: {p} not found")
        return None
    return json.loads(p.read_text())


def fig_depth_study(run: dict, tag: str) -> None:
    """QAOA depth vs energy gap, success probability and circuit cost (spec section 36)."""
    rows = run.get("qaoa", {}).get("depth_study", [])
    if not rows:
        print("  skipped: no depth study")
        return
    uniform = run.get("exact", {}).get("uniform_random_success_probability")

    optimizers = sorted({r["optimizer"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    for opt in optimizers:
        sub = sorted([r for r in rows if r["optimizer"] == opt], key=lambda r: r["p"])
        ps = [r["p"] for r in sub]
        axes[0].plot(ps, [abs(r["absolute_gap"]) + 1e-16 for r in sub],
                     "o-", label=opt, lw=1.6, ms=5)
        axes[1].plot(ps, [r["success_probability"] for r in sub],
                     "o-", label=opt, lw=1.6, ms=5)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("QAOA depth $p$")
    axes[0].set_ylabel(r"$|E_{\rm found} - E^*|$")
    axes[0].set_title("Depth vs optimality gap")
    axes[0].legend()

    if uniform:
        axes[1].axhline(uniform, color=PALETTE["uniform"], ls="--", lw=1.2,
                        label="uniform random")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("QAOA depth $p$")
    axes[1].set_ylabel("success probability")
    axes[1].set_title("Depth vs success probability")
    axes[1].legend()

    sub = sorted([r for r in rows if r["optimizer"] == optimizers[0]],
                 key=lambda r: r["p"])
    ps = [r["p"] for r in sub]
    axes[2].plot(ps, [r["logical_depth"] for r in sub], "o-",
                 color=PALETTE["exact"], label="logical depth", lw=1.6, ms=5)
    axes[2].plot(ps, [r.get("transpiled_depth") or np.nan for r in sub], "s-",
                 color=PALETTE["p1"], label="transpiled depth", lw=1.6, ms=5)
    axes[2].plot(ps, [r.get("transpiled_two_qubit_gates") or np.nan for r in sub], "^-",
                 color=PALETTE["p3"], label="transpiled 2q gates", lw=1.6, ms=5)
    axes[2].set_xlabel("QAOA depth $p$")
    axes[2].set_ylabel("count")
    axes[2].set_title("Depth vs circuit cost")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(ps)
    fig.suptitle(f"QAOA depth study — {tag}", y=1.04, fontsize=10)
    out = FIGS / f"qaoa_depth_study_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_scaling(scaling: dict) -> None:
    """Success probability and classical gaps across problem sizes."""
    pts = [p for p in scaling.get("points", []) if p.get("exact", {}).get("optimal_energy") is not None]
    if not pts:
        print("  skipped: no evaluated scaling points")
        return
    ns = [p["n_mutation_vars"] for p in pts]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    for p_depth, colour in (("p1", PALETTE["p1"]), ("p2", PALETTE["p2"]), ("p3", PALETTE["p3"])):
        vals = [p.get("qaoa", {}).get(p_depth, {}).get("success_probability") for p in pts]
        axes[0].plot(ns, vals, "o-", color=colour, label=f"QAOA {p_depth}", lw=1.6, ms=5)
    axes[0].plot(ns, [p["exact"]["uniform_random_success_probability"] for p in pts],
                 "--", color=PALETTE["uniform"], label="uniform random", lw=1.4)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("candidate mutations $N$")
    axes[0].set_ylabel("success probability")
    axes[0].set_title("Problem size vs success probability")
    axes[0].legend()

    mult = [p.get("qaoa", {}).get("p3", {}).get("success_probability_vs_uniform") for p in pts]
    axes[1].plot(ns, mult, "o-", color=PALETTE["p3"], lw=1.8, ms=6)
    axes[1].axhline(1.0, color=PALETTE["uniform"], ls="--", lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("candidate mutations $N$")
    axes[1].set_ylabel(r"QAOA $p{=}3$ success / uniform")
    axes[1].set_title("Concentration over random sampling")

    for key, colour, label in (
        ("simulated_annealing", PALETTE["sa"], "annealing"),
        ("greedy", PALETTE["greedy"], "greedy"),
        ("local_search", PALETTE["local"], "local search"),
        ("random_sampling", PALETTE["random"], "random"),
    ):
        gaps = [max(abs(p.get("classical", {}).get(key, {}).get("absolute_gap", 0.0)), 1e-16)
                for p in pts]
        axes[2].plot(ns, gaps, "o-", color=colour, label=label, lw=1.5, ms=4)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("candidate mutations $N$")
    axes[2].set_ylabel(r"$|E_{\rm found} - E^*|$")
    axes[2].set_title("Classical optimality gap")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(ns)
    fig.suptitle("Scaling benchmark (exact optimum available at every size)",
                 y=1.04, fontsize=10)
    out = FIGS / "scaling_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_pareto(run: dict, tag: str) -> None:
    """Pareto front over (normalised activity, negated normalised hemolysis)."""
    pareto = run.get("pareto", {})
    pts = pareto.get("all_points", [])
    if not pts:
        print("  skipped: no Pareto points")
        return
    parent = run.get("parent_analysis", {})

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    dom = [p for p in pts if not p["is_pareto_optimal"]]
    front = sorted([p for p in pts if p["is_pareto_optimal"]],
                   key=lambda p: p["activity_norm"])

    ax.scatter([p["activity_norm"] for p in dom], [-p["hemolysis_norm"] for p in dom],
               s=18, c="#bbbbbb", edgecolors="none", label="candidates")
    ax.plot([p["activity_norm"] for p in front], [-p["hemolysis_norm"] for p in front],
            "o-", color=PALETTE["p2"], ms=7, lw=1.5, label="Pareto front")
    ax.scatter([parent.get("activity_normalized")], [-parent.get("hemolysis_normalized", 0)],
               s=120, marker="*", c="#E69F00", edgecolors="#7a5300",
               zorder=5, label="parent")

    ax.set_xlabel("normalised predicted activity  →  better")
    ax.set_ylabel("− normalised predicted hemolysis  →  better")
    ax.set_title(f"Pareto front — {tag}\n(model predictions only)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    out = FIGS / f"pareto_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_landscape(run: dict, tag: str) -> None:
    """Individual effects and the pairwise interaction matrix."""
    ls = run.get("landscape", {})
    singles = ls.get("individual_effects", [])
    pairs = ls.get("pairwise_effects", [])
    if not singles:
        print("  skipped: no landscape")
        return
    n = ls.get("n_variables", len(singles))
    labels = [s["label"] for s in singles]
    deltas = [s["delta_i"] for s in singles]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    colours = ["#009E73" if d >= 0 else "#D55E00" for d in deltas]
    axes[0].bar(range(n), deltas, color=colours)
    axes[0].axhline(0, color="#444", lw=0.8)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axes[0].set_ylabel(r"$\Delta_i = S(P \oplus m_i) - S(P)$")
    axes[0].set_title("Individual mutation effects")

    M = np.full((n, n), np.nan)
    for p in pairs:
        M[p["i"], p["j"]] = p["delta_ij"]
        M[p["j"], p["i"]] = p["delta_ij"]
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    im = axes[1].imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[1].set_xticks(range(n)); axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_title("Pairwise interactions $\\Delta_{ij}$\n(grey = same-position conflict)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"ML mutation landscape — {tag}", y=1.02, fontsize=10)
    out = FIGS / f"mutation_landscape_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_surrogate(run: dict, tag: str) -> None:
    """Surrogate vs true ML score, the measurement that sets the validity regime."""
    sv = run.get("surrogate_validation", {})
    cands = run.get("candidates", [])
    if not cands:
        print("  skipped: no candidates")
        return

    sur = np.array([c["surrogate_delta_score"] for c in cands])
    true = np.array([c["direct_ml_delta_score"] for c in cands])
    nmut = np.array([c["n_mutations"] for c in cands])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    sc = axes[0].scatter(true, sur, c=nmut, cmap="viridis", s=22, edgecolors="none")
    lim = [min(true.min(), sur.min()) - 0.05, max(true.max(), sur.max()) + 0.05]
    axes[0].plot(lim, lim, "k--", lw=1, label="exact agreement")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel(r"true $\Delta S$ from the ML models")
    axes[0].set_ylabel(r"second-order surrogate $\widehat{\Delta S}$")
    axes[0].set_title("Surrogate vs truth")
    axes[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes[0], label="mutations selected", fraction=0.046)

    per_k = sv.get("exhaustive_by_mutation_count", {})
    ks, rhos, ranks = [], [], []
    for k, rec in sorted(per_k.items(), key=lambda t: int(t[0])):
        if rec.get("status") != "evaluated":
            continue
        ks.append(int(k))
        rhos.append(rec.get("spearman_rho") or 0.0)
        tp = rec.get("surrogate_top_pick", {})
        ranks.append(tp.get("percentile") or 0.0)
    if ks:
        x = np.arange(len(ks))
        axes[1].bar(x - 0.2, rhos, 0.4, color=PALETTE["p1"], label="Spearman ρ")
        axes[1].bar(x + 0.2, [r / 100 for r in ranks], 0.4, color=PALETTE["p3"],
                    label="top-pick percentile / 100")
        axes[1].axhline(0, color="#444", lw=0.8)
        axes[1].set_xticks(x); axes[1].set_xticklabels([f"k={k}" for k in ks])
        axes[1].set_ylim(-1, 1)
        axes[1].set_ylabel("value")
        axes[1].set_title("Fidelity within a fixed mutation count\n(exhaustive)")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "exact regime:\nno k ≥ 3 to evaluate",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.suptitle(f"Second-order surrogate fidelity — {tag}", y=1.02, fontsize=10)
    out = FIGS / f"surrogate_fidelity_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_noise(run: dict, tag: str) -> None:
    """Noise degradation of expectation and success probability."""
    noise = run.get("noise", {})
    if noise.get("status") != "evaluated":
        print("  skipped: noise study not evaluated")
        return
    runs = noise["runs"]
    names = list(runs)
    exp = [runs[n]["expectation"] for n in names]
    succ = [runs[n]["success_probability"] for n in names]
    feas = [runs[n]["feasible_probability"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(names))
    axes[0].bar(x, exp, color=PALETTE["p1"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel(r"$\langle H_C \rangle$")
    axes[0].set_title("Noise vs expected energy (lower is better)")

    axes[1].bar(x - 0.2, succ, 0.4, color=PALETTE["p2"], label="success probability")
    axes[1].bar(x + 0.2, feas, 0.4, color=PALETTE["greedy"], label="feasible probability")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Noise vs solution quality")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Simulated noise study — {tag} (Aer, documented rates)",
                 y=1.04, fontsize=10)
    out = FIGS / f"noise_study_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    _style()
    FIGS.mkdir(parents=True, exist_ok=True)
    print("generating figures from executed experiment records\n")

    for tag in ("K2", "K3"):
        run = _load(f"final_{tag}")
        if not run:
            continue
        print(f"final_{tag}:")
        fig_landscape(run, tag)
        fig_depth_study(run, tag)
        fig_surrogate(run, tag)
        fig_pareto(run, tag)
        fig_noise(run, tag)
        print()

    scaling = _load("scaling_benchmark")
    if scaling:
        print("scaling_benchmark:")
        fig_scaling(scaling)

    print(f"\nfigures in {FIGS}/")


if __name__ == "__main__":
    main()
