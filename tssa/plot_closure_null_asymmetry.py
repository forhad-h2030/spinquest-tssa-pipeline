#!/usr/bin/env python3
"""Null-asymmetry closure on the MC test set: random spin (A_N^true=0),
data-driven proportions, 3-seed ensemble J/psi selection -> closure bias vs
threshold. MC twin of plot_false_asymmetry_data.py."""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "post_processing"))
import torch
from classify import _load_model, _infer_one   # noqa: E402

BUNDLE = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"
CKPTS  = [REPO / f"classifier/checkpoints/boot_00{i}/ml_input_multiclass_M_26_march_19.best.pth"
          for i in range(3)]
OUT    = REPO / "figures/note/closure_null_asymmetry.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

ETA, F, P = 0.6, 0.18, 0.70
PREF       = 1.0 / (ETA * F * P)
LO, HI     = 2.2, 4.2
WORKING_PT = 0.635
EXP_N      = [751, 199, 1511, 663]
THRESHOLDS = np.unique(np.append(np.linspace(0.30, 0.90, 37), WORKING_PT))
N_EXP      = 5000
MIN_N      = 5
rng        = np.random.default_rng(2024)

device = torch.device("cpu")

# ── load MC test bundle, recover raw features, ensemble J/psi score ─────────────
b = np.load(BUNDLE, allow_pickle=True)
X_test = b["X_test"].astype(np.float32)
y_test = b["y_test"].astype(int)

ck0 = torch.load(CKPTS[0], map_location=device, weights_only=False)
sc0 = ck0["scaler"]
mean0 = sc0["mean"].astype(np.float32).flatten()
std0  = sc0["std"].astype(np.float32).flatten()
X_raw = X_test * std0 + mean0                       # un-standardize (boot_000)

mass = X_raw[:, 4]                                  # rec_dimu_M
px   = X_raw[:, 14] + X_raw[:, 15]                  # pos+neg px_st1 -> dimuon px proxy
left = px > 0

# 3-seed ensemble J/psi probability
probs = []
for c in CKPTS:
    m, sc, _ = _load_model(c, device)
    probs.append(_infer_one(m, sc, X_raw, device)[:, 0])
p_jpsi = np.mean(probs, axis=0)
print(f"[INFO] MC test events: {len(mass)}   ensemble scored")

class_idx = [np.where(y_test == c)[0] for c in range(4)]


def an(nuL, nuR, ndL, ndR):
    if min(nuL, nuR, ndL, ndR) <= 0:
        return np.nan
    A = math.sqrt(nuL*ndR); B = math.sqrt(ndL*nuR)
    return PREF * (A - B) / (A + B)


# ── pseudo-experiments ──────────────────────────────────────────────────────────
bias = np.full(len(THRESHOLDS), np.nan)
sem  = np.full(len(THRESHOLDS), np.nan)
sig1 = np.full(len(THRESHOLDS), np.nan)
an_store = [[] for _ in THRESHOLDS]

for e in range(N_EXP):
    idxs = np.concatenate([rng.choice(class_idx[c], size=EXP_N[c], replace=True)
                           for c in range(4)])
    p_s   = p_jpsi[idxs]; m_s = mass[idxs]; l_s = left[idxs]
    spin_up = rng.random(len(idxs)) < 0.5            # random spin tag
    inwin   = (m_s >= LO) & (m_s <= HI)
    for t, thr in enumerate(THRESHOLDS):
        sel = (p_s >= thr) & inwin
        su = sel & spin_up; sd = sel & ~spin_up
        nuL = float((su & l_s).sum()); nuR = float((su & ~l_s).sum())
        ndL = float((sd & l_s).sum()); ndR = float((sd & ~l_s).sum())
        a = an(nuL, nuR, ndL, ndR)
        if np.isfinite(a):
            an_store[t].append(a)

for t in range(len(THRESHOLDS)):
    arr = np.array(an_store[t])
    if len(arr) < MIN_N:
        continue
    bias[t] = 0.0 - arr.mean()
    sem[t]  = arr.std(ddof=1) / np.sqrt(len(arr))
    sig1[t] = arr.std(ddof=1)

iw = int(np.argmin(np.abs(THRESHOLDS - WORKING_PT)))
print(f"[RESULT] t={THRESHOLDS[iw]:.3f}: closure bias = {bias[iw]:+.4f} ± {sem[iw]:.4f}  "
      f"(single-meas σ = ±{sig1[iw]:.3f})")

# ── figure (2-panel, mirrors false_asymmetry_data.png) ──────────────────────────
fig = plt.figure(figsize=(11, 9))
gs  = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)

ax0 = fig.add_subplot(gs[0])
ax0.axhline(0, color="black", lw=1.0, ls="--", zorder=1)
ax0.fill_between(THRESHOLDS, -sig1, sig1, color="grey", alpha=0.18, lw=0, zorder=1,
                 label=r"single-measurement stat. error ($\pm\sigma_{A_N}$)")
ax0.errorbar(THRESHOLDS, bias, yerr=sem, fmt="o", ms=5, capsize=3, elinewidth=1.3,
             color="crimson", zorder=4,
             label=r"closure bias  $\delta_{\rm closure}=\langle A_N\rangle - A_N^{\rm true}$")
ax0.axvline(WORKING_PT, color="darkgreen", lw=1.6, ls=":",
            label=fr"working point $t={WORKING_PT}$")
ax0.set_ylabel(r"closure bias  $\delta_{\rm closure}$")
ax0.set_title("Classifier closure (null-asymmetry) test on MC test set\n"
              fr"random spin ($A_N^{{\rm true}}=0$), data-driven proportions, "
              fr"{N_EXP:,} pseudo-experiments/point", fontsize=11, fontweight="bold")
ax0.legend(loc="upper left", fontsize=9, framealpha=0.92)
ax0.grid(alpha=0.3, ls="--")

ax1 = fig.add_subplot(gs[1])
ax1.axhline(0, color="black", lw=1.0, ls="--", zorder=1)
ax1.errorbar(THRESHOLDS, bias, yerr=sem, fmt="o", ms=5, capsize=3, elinewidth=1.3,
             color="crimson", zorder=4)
ax1.axvline(WORKING_PT, color="darkgreen", lw=1.6, ls=":")
ax1.annotate(fr"$\delta_{{\rm closure}} = {bias[iw]:+.3f} \pm {sem[iw]:.3f}$",
             xy=(WORKING_PT, bias[iw]), xytext=(0.60, 0.85),
             textcoords="axes fraction", fontsize=11, color="darkgreen",
             bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", ec="darkgreen", alpha=0.95))
m = np.isfinite(bias)
yspan = max(1e-3, 1.3 * np.nanmax(np.abs(bias[m]) + sem[m]))
ax1.set_ylim(-yspan, yspan)
ax1.set_xlabel(r"DNN $p_{J/\psi}$ threshold")
ax1.set_ylabel(r"closure bias (zoom)")
ax1.set_title("Zoom: bias with error on the mean", fontsize=10)
ax1.grid(alpha=0.3, ls="--")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT}")
