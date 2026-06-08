#!/usr/bin/env python3
"""False-asymmetry (null) test on the real polarized data: pool DNN-selected
J/psi, randomly relabel spin (exact hypergeometric, real up/down totals fixed),
recompute A_N over a threshold scan. Bias should be ~0 (real-data analogue of the
MC closure). See note Systematic Studies."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parents[1]
PRED_UP = REPO / "post_processing/output/pred_up.npz"
PRED_DN = REPO / "post_processing/output/pred_down.npz"
OUT     = REPO / "figures/note/false_asymmetry_data.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── physics + selection (match tssa/asymmetry.py) ───────────────────────────────
ETA, F, P_AVG = 0.6, 0.18, 0.70
PREF          = 1.0 / (ETA * F * P_AVG)          # ≈ 13.23
MASS_MIN, MASS_MAX = 2.2, 4.2                     # counting window
WORKING_PT    = 0.635                             # data-driven working point (90% purity)

THRESHOLDS = np.unique(np.append(np.linspace(0.30, 0.90, 37), WORKING_PT))
N_RESAMPLE = 20000
MIN_N      = 10        # require at least this many up/down and left/right events
rng        = np.random.default_rng(12345)


def _an_vec(N_uL, N_uR, N_dL, N_dR):
    """Vectorised geometric-mean A_N; NaN where any state is empty."""
    N_uL = N_uL.astype(np.float64); N_uR = N_uR.astype(np.float64)
    N_dL = N_dL.astype(np.float64); N_dR = N_dR.astype(np.float64)
    good = (N_uL > 0) & (N_uR > 0) & (N_dL > 0) & (N_dR > 0)
    A = np.sqrt(np.where(good, N_uL * N_dR, 1.0))
    B = np.sqrt(np.where(good, N_dL * N_uR, 1.0))
    a_raw = (A - B) / (A + B)
    out = PREF * a_raw
    out[~good] = np.nan
    return out


# ── load real selected events (pool up + down) ──────────────────────────────────
pup = np.load(PRED_UP, allow_pickle=True)
pdn = np.load(PRED_DN, allow_pickle=True)

M    = np.concatenate([pup["M"],        pdn["M"]]).astype(np.float64)
px   = np.concatenate([pup["px_dimu"],  pdn["px_dimu"]]).astype(np.float64)
pjp  = np.concatenate([pup["y_proba"][:, 0], pdn["y_proba"][:, 0]]).astype(np.float64)
# true labels just for bookkeeping (NOT used in the scramble): 0=up, 1=down
true_spin = np.concatenate([np.zeros(len(pup["M"]), int),
                            np.ones(len(pdn["M"]), int)])

print(f"[INFO] pooled events: up={len(pup['M'])}  down={len(pdn['M'])}  "
      f"total={len(M)}")

# ── threshold scan ──────────────────────────────────────────────────────────────
bias, sem, single_sig, n_up, n_dn = [], [], [], [], []
for thr in THRESHOLDS:
    sel = (pjp >= thr) & (M >= MASS_MIN) & (M <= MASS_MAX)
    left = px[sel] > 0
    L_tot = int(left.sum())
    R_tot = int((~left).sum())
    nu = int((true_spin[sel] == 0).sum())   # keep real up total fixed
    nd = int((true_spin[sel] == 1).sum())
    n_up.append(nu); n_dn.append(nd)

    if min(L_tot, R_tot, nu, nd) < MIN_N:
        bias.append(np.nan); sem.append(np.nan); single_sig.append(np.nan)
        continue

    # exact scramble via hypergeometric draw of N_uL
    N_uL = rng.hypergeometric(ngood=L_tot, nbad=R_tot, nsample=nu, size=N_RESAMPLE)
    N_uR = nu - N_uL
    N_dL = L_tot - N_uL
    N_dR = R_tot - N_uR
    an = _an_vec(N_uL, N_uR, N_dL, N_dR)
    valid = np.isfinite(an)
    nv = int(valid.sum())
    if nv == 0:
        bias.append(np.nan); sem.append(np.nan); single_sig.append(np.nan)
        continue
    mean_an = float(np.nanmean(an))
    std_an  = float(np.nanstd(an))
    bias.append(0.0 - mean_an)                 # false asymmetry = 0 - <A_N>
    sem.append(std_an / np.sqrt(nv))           # error on the mean
    single_sig.append(std_an)                  # single-measurement stat error

bias       = np.array(bias)
sem        = np.array(sem)
single_sig = np.array(single_sig)

# value at the working point
iw = int(np.argmin(np.abs(THRESHOLDS - WORKING_PT)))
print(f"[RESULT] working point t={THRESHOLDS[iw]:.3f}: "
      f"false A_N = {bias[iw]:+.4f} ± {sem[iw]:.4f} (err-on-mean), "
      f"single-meas σ = ±{single_sig[iw]:.3f}, "
      f"N_up={n_up[iw]}, N_dn={n_dn[iw]}")

# ── figure: top = full scan + faint stat band, bottom = zoom on bias ────────────
fig = plt.figure(figsize=(11, 9))
gs  = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)

# top panel
ax0 = fig.add_subplot(gs[0])
ax0.axhline(0, color="black", lw=1.0, ls="--", zorder=1)
ax0.fill_between(THRESHOLDS, -single_sig, single_sig, color="royalblue",
                 alpha=0.12, lw=0, zorder=1,
                 label=r"single-measurement stat. error ($\pm\sigma_{A_N}$)")
ax0.errorbar(THRESHOLDS, bias, yerr=sem, fmt="o", ms=5, capsize=3,
             elinewidth=1.3, color="crimson", zorder=4,
             label=r"false asymmetry  $\delta = -\langle A_N\rangle_{\rm scramble}$")
ax0.axvline(WORKING_PT, color="darkgreen", lw=1.6, ls=":",
            label=fr"working point $t={WORKING_PT}$")
ax0.set_ylabel(r"false $A_N$")
ax0.set_title("False-asymmetry (spin-label scrambling) test on real polarized data\n"
              fr"pooled $J/\psi$ candidates, mass $\in[{MASS_MIN},{MASS_MAX}]$ GeV/$c^2$,"
              fr" {N_RESAMPLE:,} scrambles/point", fontsize=11, fontweight="bold")
ax0.legend(loc="upper left", fontsize=9, framealpha=0.92)
ax0.grid(alpha=0.3, ls="--")

# bottom panel: zoom on the bias (err-on-mean visible)
ax1 = fig.add_subplot(gs[1])
ax1.axhline(0, color="black", lw=1.0, ls="--", zorder=1)
ax1.errorbar(THRESHOLDS, bias, yerr=sem, fmt="o", ms=5, capsize=3,
             elinewidth=1.3, color="crimson", zorder=4)
ax1.axvline(WORKING_PT, color="darkgreen", lw=1.6, ls=":")
ax1.annotate(fr"$\delta = {bias[iw]:+.3f} \pm {sem[iw]:.3f}$",
             xy=(WORKING_PT, bias[iw]), xytext=(0.62, 0.85),
             textcoords="axes fraction", fontsize=11, color="darkgreen",
             bbox=dict(boxstyle="round,pad=0.3", fc="honeydew",
                       ec="darkgreen", alpha=0.95))
m = np.isfinite(bias)
yspan = max(1e-3, 1.3 * np.nanmax(np.abs(bias[m]) + sem[m]))
ax1.set_ylim(-yspan, yspan)
ax1.set_xlabel(r"DNN $p_{J/\psi}$ threshold")
ax1.set_ylabel(r"false $A_N$  (zoom)")
ax1.set_title("Zoom: residual bias with error on the mean", fontsize=10)
ax1.grid(alpha=0.3, ls="--")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT}")
