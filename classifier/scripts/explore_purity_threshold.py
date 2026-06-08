#!/usr/bin/env python3
"""Per-class purity/efficiency vs softmax threshold, and the contamination
composition (which true classes leak in) vs threshold, evaluated under the
data-driven class prior (exp argmax proportions). Exploratory: shows how
tightening one class's probability cut sharpens it and how the other classes'
contributions to its selected sample fall off. Output -> figures/dev/."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO   = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "checkpoints/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"
OUT    = REPO.parent / "figures/dev/purity_threshold_scan.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# data-driven prior: argmax class counts in the experimental data (J/psi:psi':DY:Comb)
DATA_N = np.array([751, 199, 1511, 663], float)
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
THR    = np.linspace(0.25, 0.97, 60)

b      = np.load(BUNDLE, allow_pickle=True)
yt     = b["y_test"].astype(int)
proba  = b["y_proba"].astype(np.float64)
names  = [str(x) for x in b["class_names"]]
am     = proba.argmax(1)
K      = len(names)

# per-event weight reweighting the (balanced) MC test set to the data prior
native_N = np.array([(yt == i).sum() for i in range(K)], float)
w        = (DATA_N / native_N)[yt]

def wsum(mask):
    return w[mask].sum()

fig, axes = plt.subplots(K, 2, figsize=(13, 4.0 * K))
fig.suptitle("Per-class purity vs softmax threshold (data-driven prior "
             f"{int(DATA_N[0])}:{int(DATA_N[1])}:{int(DATA_N[2])}:{int(DATA_N[3])})",
             fontsize=14, fontweight="bold", y=0.995)

for i, name in enumerate(names):
    purity, eff = [], []
    comp = np.zeros((len(THR), K))            # true-class fractions among selected
    for j, t in enumerate(THR):
        sel = (am == i) & (proba[:, i] >= t)
        den = wsum(sel)
        purity.append(wsum(sel & (yt == i)) / den if den > 0 else np.nan)
        eff.append(wsum(sel & (yt == i)) / wsum(yt == i))
        if den > 0:
            for k in range(K):
                comp[j, k] = wsum(sel & (yt == k)) / den
    purity = np.array(purity)

    # ── left: purity + efficiency ────────────────────────────────────────────
    axL = axes[i, 0]
    axL.plot(THR, purity, color=COLORS[i], lw=2.4, label="purity (precision)")
    axL.plot(THR, eff, color=COLORS[i], lw=1.8, ls="--", label="efficiency (recall)")
    for lvl, ls in [(0.90, ":"), (0.95, "-.")]:
        axL.axhline(lvl, color="grey", lw=1.0, ls=ls, alpha=0.7)
        ok = np.where(purity >= lvl)[0]
        if len(ok):
            t90 = THR[ok[0]]
            axL.axvline(t90, color="black", lw=0.9, ls=ls, alpha=0.6)
            axL.annotate(f"{lvl:.0%}@t={t90:.2f}\n(eff {eff[ok[0]]:.2f})",
                         xy=(t90, lvl), xytext=(4, -10 if lvl == 0.90 else 6),
                         textcoords="offset points", fontsize=8, color="black")
    axL.set_title(f"{name}  (argmax purity = {purity[0]:.3f})", fontweight="bold")
    axL.set_xlabel("softmax threshold $t$"); axL.set_ylabel("metric")
    axL.set_ylim(0, 1.02); axL.set_xlim(THR[0], THR[-1])
    axL.grid(alpha=0.3, ls="--"); axL.legend(loc="lower left", fontsize=9)

    # ── right: contamination composition (stacked) ───────────────────────────
    axR = axes[i, 1]
    axR.stackplot(THR, *[comp[:, k] for k in range(K)],
                  colors=[COLORS[k] for k in range(K)],
                  labels=[f"true {names[k]}" for k in range(K)], alpha=0.85)
    axR.set_title(f"composition of the '{name}'-selected sample", fontweight="bold")
    axR.set_xlabel("softmax threshold $t$"); axR.set_ylabel("fraction of selected")
    axR.set_ylim(0, 1); axR.set_xlim(THR[0], THR[-1])
    axR.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.tight_layout(rect=(0, 0, 1, 0.99))
fig.savefig(OUT, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT}")

# console summary: threshold to reach 90% / 95% purity per class
print("\nclass         argmax   t(90%)  eff90   t(95%)  eff95")
for i, name in enumerate(names):
    row = f"{name:12s}"
    pur0 = None
    cells = []
    for lvl in (0.90, 0.95):
        purity = np.array([
            (wsum((am == i) & (proba[:, i] >= t) & (yt == i)) /
             max(wsum((am == i) & (proba[:, i] >= t)), 1e-12))
            for t in THR])
        eff = np.array([
            wsum((am == i) & (proba[:, i] >= t) & (yt == i)) / wsum(yt == i)
            for t in THR])
        if pur0 is None:
            pur0 = purity[0]
        ok = np.where(purity >= lvl)[0]
        cells.append(f"t={THR[ok[0]]:.2f}  {eff[ok[0]]:.2f}" if len(ok) else "  --      -- ")
    print(f"{row}  {pur0:.3f}   {cells[0]}   {cells[1]}")
