#!/usr/bin/env python3
"""A_N vs DNN J/psi threshold, with MC purity (+-3sigma window) on a second axis;
the working point is marked."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO    = Path(__file__).resolve().parents[1]
BUNDLE  = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"
CKPT    = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.best.pth"
PRED_UP = REPO / "post_processing/output/pred_up.npz"
PRED_DN = REPO / "post_processing/output/pred_down.npz"
OUT     = REPO / "figures/note/an_vs_purity.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# physics
ETA   = 0.6
F_DIL = 0.18
P_POL = 0.70
SCALE = 1.0 / (ETA * F_DIL * P_POL)

TARGET_PURITY = 0.90
THRESHOLDS    = np.linspace(0.25, 0.90, 200)
N_BOOT        = 200
EXP_N         = [751, 199, 1511, 663]   # data-driven (argmax of exp data)

# mass-fit AN result (from RooFit simultaneous fit, ml-mode / shared events)
AN_FIT    =  0.418
dAN_FIT   =  0.852


# ── Gaussian fit helper ────────────────────────────────────────────────────────
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian(centers, counts, fit_range=(2.8, 3.8)):
    mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    xf, yf = centers[mask], counts[mask].astype(float)
    try:
        # reconstructed J/psi peak (~3.32), shifted above PDG 3.097
        p0 = [yf.max(), 3.30, 0.20]
        popt, pcov = curve_fit(gaussian, xf, yf, p0=p0,
                               bounds=([0, 2.9, 0.05], [np.inf, 3.6, 0.6]))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, True
    except Exception:
        return None, None, False


# ── load MC bundle & derive ±3σ window ────────────────────────────────────────
print("Loading MC bundle ...")
bundle   = np.load(BUNDLE, allow_pickle=True)
y_test   = bundle["y_test"].astype(np.int64)
y_proba  = bundle["y_proba"].astype(np.float64)
X_test   = bundle["X_test"].astype(np.float64)
cnames   = list(bundle["class_names"])
K        = len(cnames)

ckpt   = torch.load(CKPT, map_location="cpu", weights_only=False)
scaler = ckpt["scaler"]
m_mean = float(scaler["mean"].flatten()[4])
m_std  = float(scaler["std"].flatten()[4])
mass_mc = X_test[:, 4] * m_std + m_mean

MASS_BINS = np.linspace(2.0, 5.9, 60)
MASS_CTRS = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])
h_jpsi, _ = np.histogram(mass_mc[y_test == 0], bins=MASS_BINS)
popt_mc, _, ok_mc = fit_gaussian(MASS_CTRS, h_jpsi)
if ok_mc:
    MU_JPSI, SIG_JPSI = float(popt_mc[1]), float(popt_mc[2])
else:
    MU_JPSI, SIG_JPSI = 3.32, 0.20
MASS_WIN = (MU_JPSI - 3 * SIG_JPSI, MU_JPSI + 3 * SIG_JPSI)
print(f"  J/ψ MC: μ={MU_JPSI:.3f}  σ={SIG_JPSI:.3f}  window={MASS_WIN[0]:.3f}–{MASS_WIN[1]:.3f}")


# ── MC bootstrap: purity vs threshold (within ±3σ window) ─────────────────────
print("Running MC bootstrap purity scan ...")
class_idx = [np.where(y_test == c)[0] for c in range(K)]
amax_mc   = y_proba.argmax(axis=1)        # predicted class (argmax)
rng       = np.random.default_rng(42)
pur_mat   = np.zeros((N_BOOT, len(THRESHOLDS)))

for b in range(N_BOOT):
    idxs, labs = [], []
    for c, (cidx, n) in enumerate(zip(class_idx, EXP_N)):
        ch = rng.choice(cidx, size=n, replace=True)
        idxs.append(ch); labs.append(np.full(n, c, dtype=np.int64))
    sidx    = np.concatenate(idxs)
    slabels = np.concatenate(labs)
    ps      = y_proba[sidx, 0]
    is_jp   = amax_mc[sidx] == 0
    m_boot  = mass_mc[sidx]
    is_sig  = slabels == 0
    in_win  = (m_boot >= MASS_WIN[0]) & (m_boot <= MASS_WIN[1])
    for t, thr in enumerate(THRESHOLDS):
        acc = is_jp & (ps >= thr)
        tp  = float(( acc &  is_sig & in_win).sum())
        fp  = float(( acc & ~is_sig & in_win).sum())
        pur_mat[b, t] = tp / (tp + fp + 1e-12)

pur_mean = pur_mat.mean(axis=0)
pur_std  = pur_mat.std(axis=0)

# find 95% purity threshold
idx_tgt = np.where(pur_mean >= TARGET_PURITY)[0]
THR_95  = float(THRESHOLDS[idx_tgt[0]]) if len(idx_tgt) else float(THRESHOLDS[-1])
PUR_95  = float(pur_mean[idx_tgt[0]])   if len(idx_tgt) else TARGET_PURITY
# pin to the canonical working point from the purity stress test
THR_95  = 0.635
PUR_95  = float(pur_mean[np.argmin(np.abs(THRESHOLDS - THR_95))])
print(f"  90% purity threshold: t={THR_95:.3f}  purity={PUR_95:.4f}")


# ── load experimental data ─────────────────────────────────────────────────────
print("Loading experimental data ...")
up = np.load(PRED_UP, allow_pickle=True)
dn = np.load(PRED_DN, allow_pickle=True)

p_up  = up["y_proba"].astype(np.float64)[:, 0]
px_up = up["px_dimu"].astype(np.float64)
M_up  = up["M"].astype(np.float64)
p_dn  = dn["y_proba"].astype(np.float64)[:, 0]
px_dn = dn["px_dimu"].astype(np.float64)
M_dn  = dn["M"].astype(np.float64)
# predicted class == J/psi (argmax); required so a probability cut below 0.5 is
# not applied to events whose argmax is another class (matches asymmetry.py)
jp_up = up["y_proba"].argmax(axis=1) == 0
jp_dn = dn["y_proba"].argmax(axis=1) == 0
CNT_LO, CNT_HI = 2.2, 4.2   # A_N counting window (matches asymmetry.py)


# ── A_N at a hard threshold (predicted J/psi + counting window + px split) ─────
def an_at(thr):
    su = jp_up & (p_up >= thr) & (M_up >= CNT_LO) & (M_up <= CNT_HI)
    sd = jp_dn & (p_dn >= thr) & (M_dn >= CNT_LO) & (M_dn <= CNT_HI)
    nuL = float((su & (px_up >  0)).sum()); nuR = float((su & (px_up <= 0)).sum())
    ndL = float((sd & (px_dn >  0)).sum()); ndR = float((sd & (px_dn <= 0)).sum())
    ntot = nuL + nuR + ndL + ndR
    if min(nuL, nuR, ndL, ndR) < 1:
        return np.nan, np.nan, ntot
    a = np.sqrt(nuL * ndR); b = np.sqrt(ndL * nuR); den = a + b
    an  = SCALE * (a - b) / den
    err = SCALE * (a * b / den**2) * np.sqrt(1/nuL + 1/nuR + 1/ndL + 1/ndR)
    return an, err, ntot

# curve: scan over the grid
AN_vals = np.full(len(THRESHOLDS), np.nan)
AN_err  = np.full(len(THRESHOLDS), np.nan)
N_sel   = np.zeros(len(THRESHOLDS))
for t, thr in enumerate(THRESHOLDS):
    AN_vals[t], AN_err[t], N_sel[t] = an_at(thr)

# working point: evaluate exactly at THR_95 (no grid rounding)
AN_WP, dAN_WP, _ = an_at(THR_95)
print(f"  AN at t={THR_95:.3f} (exact): {AN_WP:+.4f} ± {dAN_WP:.4f}")


# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()

# purity band on right axis (drawn first so AN sits on top)
col_pur = "#8E44AD"
ax2.plot(THRESHOLDS, pur_mean, color=col_pur, lw=1.8, ls="--", alpha=0.8,
         label="MC purity (±3σ window)")
ax2.fill_between(THRESHOLDS, pur_mean - pur_std, pur_mean + pur_std,
                 color=col_pur, alpha=0.12)
ax2.axhline(TARGET_PURITY, color=col_pur, lw=0.8, ls=":", alpha=0.6)
ax2.set_ylabel("MC J/ψ purity (within ±3σ)", color=col_pur, fontsize=11)
ax2.tick_params(axis="y", labelcolor=col_pur)
ax2.set_ylim(0.4, 1.10)

# mass-fit AN — shown as a full error bar at x=0.88 and a horizontal band
col_fit = "magenta"
ax1.errorbar([0.885], [AN_FIT], yerr=[dAN_FIT],
             fmt="D", ms=7, lw=2.0, capsize=6, capthick=2,
             color=col_fit, zorder=7,
             label=rf"Mass-fit $A_N = {AN_FIT:.3f} \pm {dAN_FIT:.3f}$ (RooFit stat.)")
ax1.axhspan(AN_FIT - dAN_FIT, AN_FIT + dAN_FIT,
            color=col_fit, alpha=0.07, zorder=3)

# AN with error bars on left axis
col_an = "#1A5276"
valid  = np.isfinite(AN_vals)
ax1.errorbar(THRESHOLDS[valid], AN_vals[valid], yerr=AN_err[valid],
             fmt="o-", ms=3.5, lw=1.5, capsize=2,
             color=col_an,
             label=r"DNN $A_N$ (stat. from DNN-selected counts)",
             zorder=5)
ax1.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)

# footnote clarifying error source
ax1.text(0.02, 0.03,
         "DNN error bars: Poisson stat. propagated through\n"
         r"geometric-mean estimator on DNN-selected counts",
         transform=ax1.transAxes, fontsize=7.5, va="bottom", color="dimgray",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.6))

# mark the working-point threshold; annotation points at A_N(THR_95) computed exactly
ax1.axvline(THR_95, color="darkred", lw=1.5, ls="--", alpha=0.85, zorder=6)
ax1.annotate(
    f"$t = {THR_95:.3f}$\n(90% purity)",
    xy=(THR_95, AN_WP if np.isfinite(AN_WP) else 0),
    xytext=(THR_95 - 0.12, ax1.get_ylim()[0] if not np.isnan(ax1.get_ylim()[0]) else -0.5),
    fontsize=8.5, color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
)

ax1.set_xlabel(r"$p_{J/\psi}$ threshold", fontsize=12)
ax1.set_ylabel(r"$A_N$", color=col_an, fontsize=12)
ax1.tick_params(axis="y", labelcolor=col_an)
ax1.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
ax1.set_ylim(-1.5, 1.5)
ax1.grid(axis="y", ls="--", alpha=0.3)

# second x-axis showing N selected (top)
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
# pick a few tick positions
tick_ts = THRESHOLDS[::25]
ax3.set_xticks(tick_ts)
ax3.set_xticklabels([f"{int(N_sel[np.argmin(np.abs(THRESHOLDS-t))]):,}" for t in tick_ts],
                    fontsize=7.5, rotation=30)
ax3.set_xlabel("N selected events (up+down)", fontsize=9)

# combined legend
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper right")

fig.suptitle(
    r"$A_N$ vs DNN threshold with MC purity"
    "\nSpinQuest 2024 commissioning  |  "
    r"$\eta=0.6,\; f=0.18,\; \langle P\rangle=0.70$",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT}")
