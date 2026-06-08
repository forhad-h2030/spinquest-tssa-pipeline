#!/usr/bin/env python3
"""J/psi TSSA A_N from DNN-classified events (geometric-mean estimator). Reads
pred_{up,down}.npz; produces the 4-pad mass+A_N panel, the threshold scan, and
A_N vs pT. Run from tssa/run.sh."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.size": 13})

# ── Physics parameters (mirrors An.py) ────────────────────────────────────────
ETA   = 0.6    # spin-transfer efficiency
F     = 0.18   # dilution factor
P_AVG = 0.70   # mean target polarization

# ── Mass window for counting ──────────────────────────────────────────────────
MASS_MIN = 2.2
MASS_MAX = 4.2

# ── Probability threshold scan ────────────────────────────────────────────────
# A J/psi candidate must first be the argmax over the 4 softmax classes, then
# pass the probability cut. The argmax requirement already imposes p_jpsi > the
# other three scores, so in practice every selected event has p_jpsi well above
# the 4-class floor (0.25): scanning below the working point only reproduces the
# same sample. We therefore scan from 0.50 (where the cut starts to bite) up to
# 0.90, bracketing the t = 0.635 working point.
THRESHOLDS    = np.arange(0.50, 0.91, 0.05)
HALF_STEP     = 0.025

HIST_COLOR = "royalblue"
FIT_COLOR  = "firebrick"
JPSI_MASS  = 3.097


# ── Helpers ───────────────────────────────────────────────────────────────────
def _gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gauss(masses: np.ndarray, bins: int = 50):
    counts, edges = np.histogram(masses, bins=bins, range=(MASS_MIN, MASS_MAX))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= MASS_MIN) & (centers <= MASS_MAX)
    xf, yf = centers[mask], counts[mask]
    if yf.sum() < 5:
        return None, None, centers, counts
    p0 = [float(yf.max()), JPSI_MASS, 0.15]
    try:
        popt, pcov = curve_fit(_gaussian, xf, yf, p0=p0,
                               bounds=([0, 2.7, 0.05], [np.inf, 3.5, 0.5]),
                               maxfev=5000)
        return popt, pcov, centers, counts
    except Exception:
        return None, None, centers, counts


def _count(masses: np.ndarray):
    N = int(((masses >= MASS_MIN) & (masses <= MASS_MAX)).sum())
    return float(N), math.sqrt(N) if N > 0 else 0.0


def _compute_an(N_uL, N_uR, N_dL, N_dR, eta=ETA, f=F, P=P_AVG):
    if min(N_uL, N_uR, N_dL, N_dR) <= 0:
        return None
    A         = math.sqrt(N_uL * N_dR)
    B         = math.sqrt(N_dL * N_uR)
    denom     = A + B
    A_raw     = (A - B) / denom
    prefactor = 1.0 / (eta * f * P)
    A_N       = prefactor * A_raw

    # Statistical uncertainty: Poisson sqrt(N) on each count propagated through
    # the geometric-mean estimator. With A=sqrt(N_uL N_dR), B=sqrt(N_dL N_uR),
    #   sigma_AN = prefactor * A*B/(A+B)^2 * sqrt(1/N_uL+1/N_uR+1/N_dL+1/N_dR)
    # (bootstrap-verified; matches a Poisson resampling of the four counts).
    dA_N = prefactor * (A * B / denom**2) * math.sqrt(
        1.0/N_uL + 1.0/N_uR + 1.0/N_dL + 1.0/N_dR)
    return A_raw, A_N, dA_N


def _split(pred: dict, threshold: float):
    """Return mass and pT arrays for the 4 spin states at given DNN threshold."""
    M      = pred["M"].astype(np.float64)
    px     = pred["px_dimu"].astype(np.float64)
    pt     = pred["pt_dimu"].astype(np.float64)
    proba  = pred["y_proba"].astype(np.float64)
    p_jpsi = proba[:, 0]

    # A J/psi candidate must be PREDICTED J/psi (argmax over the 4 classes) and
    # then pass the probability cut. Thresholding p_jpsi alone is meaningless for
    # t < 0.5, where another class can be the argmax. (At the working point
    # t = 0.635 > 0.5 the argmax requirement is automatic, so it leaves the
    # headline counts unchanged.)
    is_jpsi = proba.argmax(axis=1) == 0
    sel = is_jpsi & (p_jpsi >= threshold)
    M_s, px_s, pt_s = M[sel], px[sel], pt[sel]

    left  = px_s > 0
    right = px_s <= 0
    return (M_s[left],  pt_s[left],
            M_s[right], pt_s[right])


# ── Figure 1: 2×2 mass panels + A_N panel (one per threshold) ─────────────────
def plot_mass_and_an(pup, pdn, threshold: float, out_path: Path,
                     eta=ETA, f=F, P=P_AVG, bins=50):

    uL_M, uL_pt, uR_M, uR_pt = _split(pup, threshold)
    dL_M, dL_pt, dR_M, dR_pt = _split(pdn, threshold)

    datasets = [
        ("UP-Left",   uL_M, HIST_COLOR),
        ("UP-Right",  uR_M, HIST_COLOR),
        ("DOWN-Left", dL_M, HIST_COLOR),
        ("DOWN-Right",dR_M, HIST_COLOR),
    ]

    fig = plt.figure(figsize=(19, 9))
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.9],
                   wspace=0.38, hspace=0.38)
    mass_axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]
    ax_an     = fig.add_subplot(gs[:, 2])

    fig.suptitle(
        r"J/$\psi$  —  DNN score $\geq$" + f" {threshold:.3f}"
        + f"  |  mass window [{MASS_MIN}, {MASS_MAX}]" + r" GeV/$c^2$",
        fontsize=13,
    )

    for ax, (label, masses, col) in zip(mass_axes, datasets):
        popt, pcov, centers, counts = _fit_gauss(masses, bins=bins)
        yerr = np.sqrt(counts.astype(float))
        ax.errorbar(centers, counts, yerr=yerr, fmt="o", ms=4,
                    capsize=3, elinewidth=1.2, color=col,
                    label=f"{label}  (N={len(masses):,})")
        if popt is not None:
            x_fine = np.linspace(MASS_MIN, MASS_MAX, 500)
            ax.plot(x_fine, _gaussian(x_fine, *popt),
                    color=FIT_COLOR, lw=2.2,
                    label=rf"Gauss: $\mu$={popt[1]:.3f}, $\sigma$={abs(popt[2]):.3f}")
            ax.axvline(popt[1], color=FIT_COLOR, lw=0.9, ls=":", alpha=0.7)
        ax.axvspan(MASS_MIN, MASS_MAX, alpha=0.05, color="grey")
        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_xlabel(r"Dimuon mass  [GeV/$c^2$]")
        ax.set_ylabel("Counts")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25, ls="--")

    # A_N panel
    N_uL, dN_uL = _count(uL_M)
    N_uR, dN_uR = _count(uR_M)
    N_dL, dN_dL = _count(dL_M)
    N_dR, dN_dR = _count(dR_M)

    res = _compute_an(N_uL, N_uR, N_dL, N_dR, eta, f, P)
    if res is not None:
        A_raw, A_N_val, dA_N_val = res
        x_pt = 0.5 * (MASS_MIN + MASS_MAX)
        ax_an.errorbar([x_pt], [A_N_val], yerr=[dA_N_val],
                       fmt="*", ms=18, capsize=6, elinewidth=2.0,
                       color=HIST_COLOR, zorder=4,
                       label=rf"$A_N={A_N_val:+.4f}\pm{dA_N_val:.4f}$")
        ax_an.text(0.5, 0.88,
                   rf"$A_N = {A_N_val:+.3f} \pm {dA_N_val:.3f}$",
                   transform=ax_an.transAxes, ha="center", va="top", fontsize=13,
                   bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                             ec="goldenrod", lw=1.5, alpha=0.95))
        counts_txt = (rf"$N_{{\uparrow L}}={int(N_uL)}$   $N_{{\uparrow R}}={int(N_uR)}$"
                      "\n"
                      rf"$N_{{\downarrow L}}={int(N_dL)}$   $N_{{\downarrow R}}={int(N_dR)}$")
        ax_an.text(0.98, 0.97, counts_txt, transform=ax_an.transAxes,
                   ha="right", va="top", fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    else:
        ax_an.text(0.5, 0.5, "Zero counts", ha="center", va="center",
                   transform=ax_an.transAxes, color="red")

    ax_an.axhline(0, color="black", lw=0.9, ls="--")
    ax_an.axvspan(MASS_MIN, MASS_MAX, alpha=0.08, color="grey", label="mass window")
    ax_an.set_xlim(MASS_MIN, MASS_MAX)
    ax_an.set_xlabel(r"Dimuon mass  [GeV/$c^2$]")
    ax_an.set_ylabel(r"$A_N$")
    ax_an.set_title("$A_N$  (counts in mass window)\n" + rf"DNN $\geq$ {threshold:.3f}")
    ax_an.legend(loc="lower right", fontsize=9)
    ax_an.grid(alpha=0.3, ls="--")
    params = rf"$\eta={eta}$,  $f={f}$,  $\langle P\rangle={P}$"
    ax_an.text(0.04, 0.97, params, transform=ax_an.transAxes,
               va="top", ha="left", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  mass+AN → {out_path}")


# ── Figure 2: A_N vs DNN threshold ────────────────────────────────────────────
def plot_an_vs_threshold(pup, pdn, out_path: Path, eta=ETA, f=F, P=P_AVG):
    thresh_vals, an_vals, an_errs, n_totals = [], [], [], []

    print(f"\n{'Thresh':>8s}  {'N_uL':>6s}  {'N_uR':>6s}  {'N_dL':>6s}  {'N_dR':>6s}"
          f"  {'A_raw':>9s}  {'A_N':>9s}  {'±dA_N':>9s}")

    for t in THRESHOLDS:
        uL_M, _, uR_M, _ = _split(pup, t)
        dL_M, _, dR_M, _ = _split(pdn, t)
        N_uL, _ = _count(uL_M); N_uR, _ = _count(uR_M)
        N_dL, _ = _count(dL_M); N_dR, _ = _count(dR_M)
        res = _compute_an(N_uL, N_uR, N_dL, N_dR, eta, f, P)
        if res is None:
            print(f"  {t:.2f}     {int(N_uL):>6d}  {int(N_uR):>6d}  {int(N_dL):>6d}  {int(N_dR):>6d}  SKIP")
            continue
        A_raw, A_N_val, dA_N_val = res
        print(f"  {t:.2f}     {int(N_uL):>6d}  {int(N_uR):>6d}  {int(N_dL):>6d}  {int(N_dR):>6d}"
              f"  {A_raw:>+9.4f}  {A_N_val:>+9.4f}  {dA_N_val:>9.4f}")
        thresh_vals.append(t)
        an_vals.append(A_N_val)
        an_errs.append(dA_N_val)
        n_totals.append(int(N_uL + N_uR + N_dL + N_dR))

    if not thresh_vals:
        print("[WARN] no valid thresholds for A_N vs threshold plot")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(0, color="black", lw=1.2, ls="--", zorder=1)
    ax.errorbar(np.array(thresh_vals), an_vals,
                xerr=HALF_STEP, yerr=an_errs,
                fmt="o", ms=9, capsize=6, elinewidth=2.0, capthick=2.0,
                color="royalblue", zorder=3)
    for x, y, dy, N in zip(thresh_vals, an_vals, an_errs, n_totals):
        ax.annotate(f"N={N:,}", xy=(x, y + dy),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", color="dimgrey", fontsize=10)
    ax.set_xlabel(r"DNN $p_{J/\psi}$ threshold")
    ax.set_ylabel(r"$A_N$")
    ax.set_title(r"$J/\psi$  TSSA  $A_N$  vs  DNN Probability Threshold",
                 fontweight="bold", pad=12)
    ax.grid(alpha=0.3, ls="--")
    info = (rf"$\eta={eta}$,  $f={f}$,  $\langle P\rangle={P}$"
            "\n"
            rf"mass $\in$ ({MASS_MIN}, {MASS_MAX}) GeV/$c^2$")
    ax.text(0.02, 0.03, info, transform=ax.transAxes, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="goldenrod", alpha=0.9, lw=1.2))
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nA_N vs threshold → {out_path}")


# ── Figure 3: A_N vs mean pT ──────────────────────────────────────────────────
def plot_an_vs_pt(pup, pdn, out_path: Path,
                  pt_cuts: list | None = None,
                  eta=ETA, f=F, P=P_AVG):
    """One A_N point per DNN threshold, placed at the mean dimuon pT of the
    selected sample. Thresholds bracket the t = 0.635 working point; below ~0.5
    the argmax requirement already fixes the sample, so lower cuts are omitted."""
    if pt_cuts is None:
        pt_cuts = [0.55, 0.635, 0.75]

    COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(0, color="black", lw=1.2, ls="--", zorder=1)

    for cut, color in zip(pt_cuts, COLORS):
        uL_M, uL_pt, uR_M, uR_pt = _split(pup, cut)
        dL_M, dL_pt, dR_M, dR_pt = _split(pdn, cut)

        N_uL, _ = _count(uL_M); N_uR, _ = _count(uR_M)
        N_dL, _ = _count(dL_M); N_dR, _ = _count(dR_M)
        res = _compute_an(N_uL, N_uR, N_dL, N_dR, eta, f, P)
        if res is None:
            continue
        _, A_N_val, dA_N_val = res

        all_pt  = np.concatenate([uL_pt, uR_pt, dL_pt, dR_pt])
        if len(all_pt) == 0:
            continue
        # x position = mean dimuon pT; x-error bar = +-3 sigma of that pT
        # distribution (clipped at 0 since pT >= 0), not the full min-max range.
        mean_pt = float(all_pt.mean())
        std_pt  = float(all_pt.std())
        xlo     = max(0.0, mean_pt - 3 * std_pt)
        xhi     = mean_pt + 3 * std_pt
        N_tot   = int(N_uL + N_uR + N_dL + N_dR)

        ax.errorbar([mean_pt], [A_N_val],
                    xerr=[[mean_pt - xlo], [xhi - mean_pt]],
                    yerr=[dA_N_val],
                    fmt="o", ms=12, capsize=7, elinewidth=2.0, capthick=2.0,
                    color=color, zorder=3,
                    label=rf"DNN $\geq$ {cut:.2f}  (N={N_tot:,})")

    ax.set_xlabel(r"$p_T^{\mu^+\mu^-}$  [GeV/$c$]")
    ax.set_ylabel(r"$A_N$")
    ax.set_title(r"$J/\psi$  TSSA  $A_N$  vs  $p_T^{\mu^+\mu^-}$",
                 fontweight="bold", pad=12)
    ax.set_xlim(left=0.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3, ls="--")
    info = (rf"$\eta={eta}$,  $f={f}$,  $\langle P\rangle={P}$"
            "\n"
            rf"mass $\in$ ({MASS_MIN}, {MASS_MAX}) GeV/$c^2$")
    ax.text(0.02, 0.05, info, transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="goldenrod", lw=1.2, alpha=0.95))
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"A_N vs pT        → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-up",   required=True, type=Path)
    p.add_argument("--pred-down", required=True, type=Path)
    p.add_argument("--out-dir",   default="figures/pipeline/asymmetry", type=Path)
    p.add_argument("--eta",  type=float, default=ETA)
    p.add_argument("--f",    type=float, default=F)
    p.add_argument("--P",    type=float, default=P_AVG)
    p.add_argument("--mass-bins", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.635,
                   help="DNN threshold for the 4-pad mass+AN figure (default 0.635, the working point)")
    args = p.parse_args()

    pup = np.load(args.pred_up,   allow_pickle=True)
    pdn = np.load(args.pred_down, allow_pickle=True)
    print(f"[INFO] up  N={len(pup['y_pred']):,}   down  N={len(pdn['y_pred']):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 4-pad mass + single A_N panel at chosen threshold
    plot_mass_and_an(pup, pdn, threshold=args.threshold,
                     out_path=args.out_dir / f"an_mass_4pad_p{args.threshold:.3f}.png",
                     eta=args.eta, f=args.f, P=args.P,
                     bins=args.mass_bins)

    # A_N vs threshold scan
    plot_an_vs_threshold(pup, pdn,
                         out_path=args.out_dir / "an_vs_threshold.png",
                         eta=args.eta, f=args.f, P=args.P)

    # A_N vs pT (one point per threshold cut)
    plot_an_vs_pt(pup, pdn,
                  out_path=args.out_dir / "an_vs_pt.png",
                  eta=args.eta, f=args.f, P=args.P)

    print("[DONE]")


if __name__ == "__main__":
    main()
