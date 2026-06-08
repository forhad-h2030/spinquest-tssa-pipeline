#!/usr/bin/env python3
"""Purity stress test: J/psi purity & F1 vs DNN threshold (+-3sigma window,
bootstrap band) plus the DNN-selected J/psi mass fit. Derives the working-point
threshold and the +-3sigma window centered on the reconstructed J/psi peak."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

REPO   = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"
CKPT   = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.best.pth"
OUT    = REPO / "figures/note/jpsi_purity_mass.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

N_BOOT        = 200
TARGET_PURITY = 0.90
THRESHOLDS    = np.linspace(0.25, 0.90, 200)
MASS_BINS     = np.linspace(2.0, 5.9, 60)
MASS_CTRS  = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])
BIN_W      = MASS_BINS[1] - MASS_BINS[0]

# Data-driven scenario: proportions from argmax classification of the exp data
# (J/psi:psi':DY:Comb = 751:199:1511:663)
EXP_N      = [751, 199, 1511, 663]
CLASS_COLORS = ["#2471A3", "#E74C3C", "#27AE60", "#8E44AD"]

# ── load bundle ───────────────────────────────────────────────────────────────
bundle  = np.load(BUNDLE, allow_pickle=True)
X_test  = bundle["X_test"].astype(np.float64)
y_test  = bundle["y_test"].astype(np.int64)
y_proba = bundle["y_proba"].astype(np.float64)
cnames  = list(bundle["class_names"])
K       = len(cnames)

ckpt    = torch.load(CKPT, map_location="cpu", weights_only=False)
scaler  = ckpt["scaler"]
m_mean  = float(scaler["mean"].flatten()[4])
m_std   = float(scaler["std"].flatten()[4])
mass    = X_test[:, 4] * m_std + m_mean

rng       = np.random.default_rng(42)
class_idx = [np.where(y_test == c)[0] for c in range(K)]
p_jpsi    = y_proba[:, 0]


# ── Gaussian fit ──────────────────────────────────────────────────────────────
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian(centers, counts, fit_range=(2.7, 3.5)):
    mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    xf, yf = centers[mask], counts[mask].astype(float)
    try:
        # start at the reconstructed J/psi peak (~3.32, shifted above PDG 3.097
        # by the target-vertex mass hypothesis), not at the PDG value
        p0 = [yf.max(), 3.30, 0.20]
        popt, pcov = curve_fit(gaussian, xf, yf, p0=p0,
                               bounds=([0, 2.9, 0.05], [np.inf, 3.6, 0.6]))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, True
    except Exception:
        return None, None, False


# ── derive ±3σ mass window from true J/ψ MC distribution ─────────────────────
jpsi_mass_mc = mass[y_test == 0]
h_jpsi, _    = np.histogram(jpsi_mass_mc, bins=MASS_BINS)
popt_mc, _, ok_mc = fit_gaussian(MASS_CTRS, h_jpsi, fit_range=(2.8, 3.8))
if ok_mc:
    MU_JPSI  = float(popt_mc[1])
    SIG_JPSI = float(popt_mc[2])
else:
    MU_JPSI, SIG_JPSI = 3.32, 0.20   # reconstructed-peak fallback
MASS_WIN = (MU_JPSI - 3 * SIG_JPSI, MU_JPSI + 3 * SIG_JPSI)
print(f"J/ψ MC fit:  μ = {MU_JPSI:.3f} GeV   σ = {SIG_JPSI:.3f} GeV")
print(f"±3σ window:  [{MASS_WIN[0]:.3f}, {MASS_WIN[1]:.3f}] GeV")


# ── bootstrap purity / F1 within ±3σ mass window (Nominal) ───────────────────
def bootstrap_curves(exp_counts):
    pur = np.zeros((N_BOOT, len(THRESHOLDS)))
    f1  = np.zeros((N_BOOT, len(THRESHOLDS)))
    for b in range(N_BOOT):
        idxs, labs = [], []
        for c, (cidx, n) in enumerate(zip(class_idx, exp_counts)):
            ch = rng.choice(cidx, size=n, replace=True)
            idxs.append(ch); labs.append(np.full(n, c, dtype=np.int64))
        sidx    = np.concatenate(idxs)
        slabels = np.concatenate(labs)
        ps      = y_proba[sidx, 0]
        m_boot  = mass[sidx]
        is_sig  = slabels == 0
        in_win  = (m_boot >= MASS_WIN[0]) & (m_boot <= MASS_WIN[1])
        for t, thr in enumerate(THRESHOLDS):
            acc = ps >= thr
            tp  = float(( acc &  is_sig & in_win).sum())
            fp  = float(( acc & ~is_sig & in_win).sum())
            fn  = float((~acc &  is_sig & in_win).sum())
            pur[b, t] = tp / (tp + fp + 1e-12)
            f1 [b, t] = (2*tp) / (2*tp + fp + fn + 1e-12)
    return pur, f1


def sample_mass(exp_counts):
    idxs, labs = [], []
    for c, (cidx, n) in enumerate(zip(class_idx, exp_counts)):
        ch = rng.choice(cidx, size=n, replace=True)
        idxs.append(ch); labs.append(np.full(n, c, dtype=np.int64))
    sidx = np.concatenate(idxs)
    return mass[sidx], np.concatenate(labs), y_proba[sidx], sidx


# ── run bootstrap ─────────────────────────────────────────────────────────────
print("Running bootstrap (Nominal)...")
pur_b, f1_b = bootstrap_curves(EXP_N)
pur_mean = pur_b.mean(axis=0); pur_std = pur_b.std(axis=0)
f1_mean  = f1_b.mean(axis=0);  f1_std  = f1_b.std(axis=0)

# threshold for target purity
idx_pur  = np.where(pur_mean >= TARGET_PURITY)[0]
opt_t    = float(THRESHOLDS[idx_pur[0]])  if len(idx_pur) else float(THRESHOLDS[-1])
opt_pur  = float(pur_mean[idx_pur[0]])    if len(idx_pur) else float(pur_mean[-1])
opt_f1   = float(f1_mean[idx_pur[0]])     if len(idx_pur) else float(f1_mean[-1])
print(f"  threshold={opt_t:.3f}  Purity={opt_pur:.4f}  F1={opt_f1:.3f}  (target {TARGET_PURITY:.0%})")

rng_mass = np.random.default_rng(99)
m_s, labs_s, proba_s, sidx_s = sample_mass(EXP_N)
p_s = proba_s[:, 0]

# ── unscale pT and pz for sampled events ──────────────────────────────────────
sc_mean = np.array(scaler["mean"]).flatten()
sc_std  = np.array(scaler["std"]).flatten()
mT_s  = X_test[sidx_s, 9] * sc_std[9] + sc_mean[9]
pz_s  = X_test[sidx_s, 3] * sc_std[3] + sc_mean[3]
pt_s  = np.sqrt(np.clip(mT_s**2 - m_s**2, 0, None))

# ── figure ────────────────────────────────────────────────────────────────────
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
ax_scan = fig.add_subplot(gs[0, :])
ax_comp = fig.add_subplot(gs[1, 0])
ax_sel  = fig.add_subplot(gs[1, 1])

# ── panel 0: purity & F1 scan ─────────────────────────────────────────────────
col = "#1A5276"
ax_scan.plot(THRESHOLDS, pur_mean, color=col, lw=2.0, ls="-",  label="Purity")
ax_scan.fill_between(THRESHOLDS, pur_mean-pur_std, pur_mean+pur_std, color=col, alpha=0.15)
ax_scan.plot(THRESHOLDS, f1_mean,  color=col, lw=2.0, ls="--", label="F1")
ax_scan.fill_between(THRESHOLDS, f1_mean-f1_std,  f1_mean+f1_std,  color=col, alpha=0.10)
ax_scan.axvline(opt_t, color="darkred", lw=1.2, ls="--", alpha=0.85,
               label=rf"{TARGET_PURITY:.0%} purity: $p \geq {opt_t:.3f}$")
ax_scan.axhline(TARGET_PURITY, color="darkred", lw=0.8, ls=":", alpha=0.5)
ax_scan.annotate(f"p={opt_t:.3f}\n(purity={opt_pur:.2f})",
                 xy=(opt_t, opt_pur),
                 xytext=(opt_t + 0.03, opt_pur - 0.10),
                 fontsize=8.5, color="darkred",
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
ax_scan.set_xlabel("Min J/ψ softmax threshold", fontsize=11)
ax_scan.set_ylabel("Metric", fontsize=11)
ax_scan.set_title("J/ψ Purity & F1 vs threshold (within ±3σ mass window)\n"
                  r"Data-driven: J/$\psi$=751, $\psi'$=199, DY=1511, Comb=663"
                  f"   |   mass window: [{MASS_WIN[0]:.2f}, {MASS_WIN[1]:.2f}] GeV",
                  fontsize=10, fontweight="bold")
ax_scan.set_xlim(0.25, 0.90); ax_scan.set_ylim(0, 1.05)
ax_scan.legend(fontsize=9, loc="lower right")
ax_scan.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_scan.grid(axis="y", ls="--", alpha=0.35)
unc_note = ("±1σ: std over 200 bootstrap\nresamplings (sampling uncertainty)")
ax_scan.text(0.03, 0.03, unc_note, transform=ax_scan.transAxes,
             fontsize=7.5, va="bottom", color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.7))

# ── panel 1: true components ──────────────────────────────────────────────────
lstyles = ["-", "--", "-.", ":"]
lwidths = [2.0, 2.0, 1.8, 1.8]
for k in range(K):
    m_k = m_s[labs_s == k]
    ax_comp.hist(m_k, bins=MASS_BINS, histtype="step",
                 color=CLASS_COLORS[k], lw=lwidths[k], ls=lstyles[k],
                 label=f"True {cnames[k]} (N={len(m_k):,})")

cnt_total, _ = np.histogram(m_s, bins=MASS_BINS)
yerr = np.where(cnt_total > 0, np.sqrt(cnt_total), 1.0)
ax_comp.errorbar(MASS_CTRS, cnt_total, yerr=yerr,
                 fmt="o", ms=3.5, lw=1.1, capsize=2, color="black",
                 label=f"Total (N={len(m_s):,})", zorder=5)
ax_comp.axvline(3.097, color="gray", lw=0.8, ls=":", alpha=0.7)
ax_comp.axvline(3.686, color="gray", lw=0.8, ls=":", alpha=0.7)
ax_comp.set_xlabel(r"Dimuon mass [GeV/$c^2$]", fontsize=11)
ax_comp.set_ylabel("Events / bin", fontsize=11)
ax_comp.set_xlim(2.0, 5.9)
ax_comp.set_title("True components", fontsize=11, fontweight="bold")
ax_comp.legend(fontsize=8, loc="upper right")
ax_comp.grid(axis="y", ls="--", alpha=0.3)

# ── panel 2: DNN J/ψ selection + Gaussian fit ─────────────────────────────────
jpsi_sel     = p_s >= opt_t
m_jpsi       = m_s[jpsi_sel]
is_true_jpsi = labs_s == 0
in_win_s     = (m_s >= MASS_WIN[0]) & (m_s <= MASS_WIN[1])
tp  = float((jpsi_sel &  is_true_jpsi & in_win_s).sum())
fp  = float((jpsi_sel & ~is_true_jpsi & in_win_s).sum())
purity_opt   = tp / (tp + fp + 1e-12)

cnt_jpsi, _ = np.histogram(m_jpsi, bins=MASS_BINS)
yerr_jpsi   = np.where(cnt_jpsi > 0, np.sqrt(cnt_jpsi), 1.0)
ax_sel.errorbar(MASS_CTRS, cnt_jpsi, yerr=yerr_jpsi,
                fmt="o", ms=3.5, lw=1.1, capsize=2,
                color=CLASS_COLORS[0], zorder=5,
                label=rf"DNN J/ψ  (N={len(m_jpsi):,})")

popt, perr, ok = fit_gaussian(MASS_CTRS, cnt_jpsi, fit_range=(2.4, 4.4))
if ok:
    print(f"  selected-J/ψ fit: μ = {popt[1]:.3f} ± {perr[1]:.3f}  "
          f"σ = {popt[2]:.3f} ± {perr[2]:.3f}  (N={len(m_jpsi)})")
    x_fit  = np.linspace(2.2, 4.6, 300)
    fit_label = (rf"Gaussian fit" "\n"
                 rf"$\mu$ = {popt[1]:.3f} ± {perr[1]:.3f} GeV" "\n"
                 rf"$\sigma$ = {popt[2]:.3f} ± {perr[2]:.3f} GeV" "\n"
                 rf"Amp = {popt[0]:.1f} ± {perr[0]:.1f}")
    ax_sel.plot(x_fit, gaussian(x_fit, *popt),
                color="darkred", lw=2.0, zorder=6, label=fit_label)

# shade the ±3σ window
ax_sel.axvspan(MASS_WIN[0], MASS_WIN[1], alpha=0.07, color="gold", zorder=0)
ax_sel.axvline(3.097, color="gray", lw=0.8, ls=":", alpha=0.7)
ax_sel.axvline(3.686, color="gray", lw=0.8, ls=":", alpha=0.7)
ax_sel.set_xlabel(r"Dimuon mass [GeV/$c^2$]", fontsize=11)
ax_sel.set_ylabel("Events / bin", fontsize=11)
ax_sel.set_xlim(2.0, 5.9)
ax_sel.set_title("DNN J/ψ selection", fontsize=11, fontweight="bold")
ax_sel.legend(fontsize=8, loc="upper right")
ax_sel.grid(axis="y", ls="--", alpha=0.3)
ax_sel.text(0.97, 0.95,
            f"$p_{{J/\\psi}} \\geq {opt_t:.3f}$\n"
            f"Purity (±3σ): {purity_opt:.3f}",
            transform=ax_sel.transAxes, ha="right", va="top", fontsize=9,
            color="dimgray")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT}")
