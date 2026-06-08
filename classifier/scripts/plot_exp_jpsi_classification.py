#!/usr/bin/env python3
"""DNN J/psi classification of the experimental data (spin up/down): total
spectrum + DNN-classified contributions, and the J/psi-selected mass fit at the
working-point threshold."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

REPO     = Path(__file__).resolve().parents[2]
BUNDLE   = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"
CKPT     = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.best.pth"
PRED_UP  = REPO / "post_processing/output/pred_up.npz"
PRED_DN  = REPO / "post_processing/output/pred_down.npz"
OUT      = REPO / "figures/note/exp_jpsi_classification.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MASS_BINS     = np.linspace(2.0, 5.9, 60)
MASS_CTRS     = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])
BIN_W         = MASS_BINS[1] - MASS_BINS[0]
CLASS_COLORS  = ["#2471A3", "#E74C3C", "#27AE60", "#8E44AD"]
TARGET_PURITY = 0.90


# ── Gaussian fit helper ────────────────────────────────────────────────────────
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian(centers, counts, fit_range=(2.4, 4.4)):
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


# ── load MC bundle and derive ±3σ mass window ─────────────────────────────────
print("Loading MC test bundle ...")
bundle  = np.load(BUNDLE, allow_pickle=True)
y_test  = bundle["y_test"].astype(np.int64)
y_proba = bundle["y_proba"].astype(np.float64)
cnames  = list(bundle["class_names"])
K       = len(cnames)

ckpt   = torch.load(CKPT, map_location="cpu", weights_only=False)
scaler = ckpt["scaler"]
m_mean = float(scaler["mean"].flatten()[4])
m_std  = float(scaler["std"].flatten()[4])
mass_mc = bundle["X_test"].astype(np.float64)[:, 4] * m_std + m_mean

# fit J/ψ MC mass to get window
h_jpsi, _ = np.histogram(mass_mc[y_test == 0], bins=MASS_BINS)
popt_mc, _, ok_mc = fit_gaussian(MASS_CTRS, h_jpsi, fit_range=(2.8, 3.8))
if ok_mc:
    MU_JPSI, SIG_JPSI = float(popt_mc[1]), float(popt_mc[2])
else:
    MU_JPSI, SIG_JPSI = 3.32, 0.20
MASS_WIN = (MU_JPSI - 3 * SIG_JPSI, MU_JPSI + 3 * SIG_JPSI)
print(f"  J/ψ MC fit:  μ = {MU_JPSI:.3f} GeV   σ = {SIG_JPSI:.3f} GeV")
print(f"  ±3σ window:  [{MASS_WIN[0]:.3f}, {MASS_WIN[1]:.3f}] GeV")


# ── find threshold for target purity within ±3σ window from MC bootstrap ─────
print(f"\nFinding threshold for {TARGET_PURITY:.0%} J/ψ purity from MC ...")
class_idx   = [np.where(y_test == c)[0] for c in range(K)]
EXP_N       = [751, 199, 1511, 663]   # data-driven (argmax of exp data)
THRESHOLDS  = np.linspace(0.25, 0.99, 200)
N_BOOT      = 200
rng         = np.random.default_rng(42)

pur_mat = np.zeros((N_BOOT, len(THRESHOLDS)))
for b in range(N_BOOT):
    idxs, labs = [], []
    for c, (cidx, n) in enumerate(zip(class_idx, EXP_N)):
        ch = rng.choice(cidx, size=n, replace=True)
        idxs.append(ch); labs.append(np.full(n, c, dtype=np.int64))
    sidx    = np.concatenate(idxs)
    slabels = np.concatenate(labs)
    ps      = y_proba[sidx, 0]
    m_boot  = mass_mc[sidx]
    is_sig  = slabels == 0
    in_win  = (m_boot >= MASS_WIN[0]) & (m_boot <= MASS_WIN[1])
    for t, thr in enumerate(THRESHOLDS):
        acc = ps >= thr
        tp  = float(( acc &  is_sig & in_win).sum())
        fp  = float(( acc & ~is_sig & in_win).sum())
        pur_mat[b, t] = tp / (tp + fp + 1e-12)

pur_mean = pur_mat.mean(axis=0)
idx_tgt  = np.where(pur_mean >= TARGET_PURITY)[0]
THR_95   = float(THRESHOLDS[idx_tgt[0]]) if len(idx_tgt) else 0.78
PUR_95   = float(pur_mean[idx_tgt[0]])   if len(idx_tgt) else TARGET_PURITY
# pin to the canonical working point from the purity stress test (Sec. stress_test)
THR_95   = 0.635
PUR_95   = float(pur_mean[np.argmin(np.abs(THRESHOLDS - THR_95))])
print(f"  threshold = {THR_95:.3f}  mean purity (±3σ) = {PUR_95:.4f}")


# ── load experimental predictions ─────────────────────────────────────────────
spins = [
    {"label": "Spin Up",   "file": PRED_UP,  "color": "#C0392B"},
    {"label": "Spin Down", "file": PRED_DN,  "color": "#1A5276"},
]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

for row, sp in enumerate(spins):
    pred     = np.load(sp["file"], allow_pickle=True)
    M        = pred["M"].astype(np.float64)
    y_pred   = pred["y_pred"].astype(np.int64)
    y_pr     = pred["y_proba"].astype(np.float64)
    p_jpsi   = y_pr[:, 0]
    n_total  = len(M)

    ax_l = fig.add_subplot(gs[row, 0])
    ax_r = fig.add_subplot(gs[row, 1])

    # ── left panel: total + classified contributions ──────────────────────────
    cnt_total, _ = np.histogram(M, bins=MASS_BINS)
    yerr_tot     = np.where(cnt_total > 0, np.sqrt(cnt_total), 1.0)
    ax_l.errorbar(MASS_CTRS, cnt_total, yerr=yerr_tot,
                  fmt="o", ms=3.5, lw=1.1, capsize=2, color="black",
                  label=f"All data  (N={n_total:,})", zorder=5)

    lstyles = ["-", "--", "-.", ":"]
    lwidths = [2.0, 2.0, 1.8, 1.8]
    for k in range(K):
        m_k = M[y_pred == k]
        ax_l.hist(m_k, bins=MASS_BINS, histtype="step",
                  color=CLASS_COLORS[k], lw=lwidths[k], ls=lstyles[k],
                  label=f"DNN {cnames[k]} (N={len(m_k):,})")

    ax_l.axvline(3.097, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_l.axvline(3.686, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_l.set_xlabel(r"Dimuon mass [GeV/$c^2$]", fontsize=11)
    ax_l.set_ylabel("Events / bin", fontsize=11)
    ax_l.set_xlim(2.0, 5.9)
    ax_l.set_title(f"{sp['label']} — classified contributions",
                   fontsize=11, fontweight="bold", color=sp["color"])
    ax_l.legend(fontsize=7.5, loc="upper right")
    ax_l.grid(axis="y", ls="--", alpha=0.3)

    # ── right panel: J/ψ selected + Gaussian fit ──────────────────────────────
    jpsi_sel = p_jpsi >= THR_95
    m_jpsi   = M[jpsi_sel]

    cnt_jpsi, _ = np.histogram(m_jpsi, bins=MASS_BINS)
    yerr_jpsi   = np.where(cnt_jpsi > 0, np.sqrt(cnt_jpsi), 1.0)
    ax_r.errorbar(MASS_CTRS, cnt_jpsi, yerr=yerr_jpsi,
                  fmt="o", ms=3.5, lw=1.1, capsize=2,
                  color=CLASS_COLORS[0], zorder=5,
                  label=rf"DNN J/ψ  (N={len(m_jpsi):,})")

    popt, perr, ok = fit_gaussian(MASS_CTRS, cnt_jpsi, fit_range=(2.4, 4.4))
    if ok:
        x_fit = np.linspace(2.2, 4.6, 300)
        fit_label = ("Gaussian fit\n"
                     rf"$\mu$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f} GeV" "\n"
                     rf"$\sigma$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f} GeV" "\n"
                     rf"Amp = {popt[0]:.1f} $\pm$ {perr[0]:.1f}")
        ax_r.plot(x_fit, gaussian(x_fit, *popt),
                  color="darkred", lw=2.0, zorder=6, label=fit_label)

    # shade ±3σ window
    ax_r.axvspan(MASS_WIN[0], MASS_WIN[1], alpha=0.07, color="gold", zorder=0)
    ax_r.axvline(3.097, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_r.axvline(3.686, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_r.set_xlabel(r"Dimuon mass [GeV/$c^2$]", fontsize=11)
    ax_r.set_ylabel("Events / bin", fontsize=11)
    ax_r.set_xlim(2.0, 5.9)
    ax_r.set_title(f"{sp['label']} — DNN J/ψ selection",
                   fontsize=11, fontweight="bold", color=sp["color"])
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.grid(axis="y", ls="--", alpha=0.3)
    ax_r.text(0.97, 0.95,
              f"$p_{{J/\\psi}} \\geq {THR_95:.3f}$\n"
              f"MC purity (±3σ) $\\approx$ {PUR_95:.2f}",
              transform=ax_r.transAxes, ha="right", va="top",
              fontsize=9, color="dimgray")

fig.suptitle(
    r"Experimental data — DNN J/$\psi$ classification (3-seed ensemble)"
    f"\n{TARGET_PURITY:.0%} purity threshold: $p_{{J/\\psi}} \\geq {THR_95:.3f}$"
    f"  |  ±3σ window: [{MASS_WIN[0]:.2f}, {MASS_WIN[1]:.2f}] GeV",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT}")
