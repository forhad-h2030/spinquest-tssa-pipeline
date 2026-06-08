#!/usr/bin/env python3
"""Final A_N comparison: DNN vs fit-based, on the same processed events, at the
working point (fit value is threshold-independent). Statistical errors only."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO    = Path(__file__).resolve().parents[1]
PRED_UP = REPO / "post_processing/output/pred_up.npz"
PRED_DN = REPO / "post_processing/output/pred_down.npz"
OUT     = REPO / "figures/note/an_compare_dnn_fit.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

ETA, F, P = 0.6, 0.18, 0.70
PREF = 1.0 / (ETA * F * P)
LO, HI = 2.2, 4.2
THR    = 0.635

# fit-based result (RooFit simultaneous fit on shared events; threshold-independent)
AN_FIT, dAN_FIT = 0.418, 0.852

up = np.load(PRED_UP, allow_pickle=True)
dn = np.load(PRED_DN, allow_pickle=True)
pu, Mu, pxu, ptu = (up["y_proba"][:,0].astype(float), up["M"].astype(float),
                    up["px_dimu"].astype(float), up["pt_dimu"].astype(float))
pd, Md, pxd, ptd = (dn["y_proba"][:,0].astype(float), dn["M"].astype(float),
                    dn["px_dimu"].astype(float), dn["pt_dimu"].astype(float))

su = (pu >= THR) & (Mu >= LO) & (Mu <= HI)
sd = (pd >= THR) & (Md >= LO) & (Md <= HI)
NuL = float((su & (pxu > 0)).sum()); NuR = float((su & (pxu <= 0)).sum())
NdL = float((sd & (pxd > 0)).sum()); NdR = float((sd & (pxd <= 0)).sum())

A = math.sqrt(NuL*NdR); B = math.sqrt(NdL*NuR); den = A + B
AN_DNN = PREF * (A - B)/den
# Poisson propagation through the geometric-mean estimator (matches asymmetry.py)
dAN_DNN = PREF * (A * B / den**2) * math.sqrt(1/NuL + 1/NuR + 1/NdL + 1/NdR)

# x position = mean dimuon pT of the selected sample; x-error bar = +-3 sigma
# of that pT distribution (clipped at 0 since pT >= 0).
pt_sel = np.concatenate([ptu[su], ptd[sd]])
pt_mean, pt_std = pt_sel.mean(), pt_sel.std()
xlo = max(0.0, pt_mean - 3 * pt_std)
xhi = pt_mean + 3 * pt_std
xerr = [[pt_mean - xlo], [xhi - pt_mean]]
print(f"DNN A_N={AN_DNN:+.3f}±{dAN_DNN:.3f}  N={int(NuL+NuR+NdL+NdR)}  "
      f"pT mean={pt_mean:.2f}  sigma={pt_std:.2f}  +-3sigma=[{xlo:.2f},{xhi:.2f}]")

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.axhline(0, color="black", lw=1.0, ls="--", zorder=1)
# fit point (placed at same mean pT, slightly offset for visibility)
ax.errorbar([pt_mean], [AN_FIT], yerr=[dAN_FIT], xerr=xerr,
            fmt="s", ms=11, color="#E67E22", capsize=6, elinewidth=2.0, zorder=4,
            label=fr"Fit-based: $A_N = {AN_FIT:+.3f} \pm {dAN_FIT:.3f}$")
# DNN point
ax.errorbar([pt_mean], [AN_DNN], yerr=[dAN_DNN], xerr=xerr,
            fmt="o", ms=12, color="#2471A3", capsize=6, elinewidth=2.0, zorder=5,
            label=fr"AI-based (DNN): $A_N = {AN_DNN:+.3f} \pm {dAN_DNN:.3f}$")

ax.set_xlim(0, max(3.0, xhi + 0.3))   # axis follows the +-3sigma bar, not min/max
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$p_T^{\mu^+\mu^-}$  [GeV/$c$]", fontsize=12)
ax.set_ylabel(r"$A_N$", fontsize=13)
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3, ls="--")
ax.text(0.02, 0.03,
        fr"$\eta=0.6,\ f=0.18,\ \langle P\rangle=0.7$;  statistical errors only;  "
        fr"marker at mean $p_T$, bar = $\pm3\sigma$ of the $p_T$ distribution ($t={THR}$)",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod", alpha=0.9))

fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT}")
