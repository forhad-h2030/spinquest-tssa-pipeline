#!/usr/bin/env python3
"""Combined DNN + RooFit overlay: 2x2 dimuon-mass panels (data + DNN-selected
J/psi + RooFit PDF) plus an A_N comparison panel. Needs fit_params.json from the
fit step."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm as sp_norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.size": 11})

MASS_MIN   = 2.0
MASS_MAX   = 5.9
N_BINS     = 60
BIN_W      = (MASS_MAX - MASS_MIN) / N_BINS
BIN_EDGES  = np.linspace(MASS_MIN, MASS_MAX, N_BINS + 1)
BIN_CTRS   = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
RATIO      = 3.686 / 3.097

ETA, F, P_AVG = 0.6, 0.18, 0.70

DNN_THRESHOLD = 0.30

_REPO = Path(__file__).resolve().parent.parent
FEAT_UP   = _REPO / "post_processing/output/features_up.npz"
FEAT_DOWN = _REPO / "post_processing/output/features_down.npz"
PRED_UP   = _REPO / "post_processing/output/pred_up.npz"
PRED_DOWN = _REPO / "post_processing/output/pred_down.npz"
FIT_JSON  = _REPO / "figures/tssa_fit/fit_params.json"
OUT_DIR   = _REPO / "figures/tssa"

# Panel order: (spin, side, row, col, label)
PANELS = [
    ("up",   "left",  0, 0, r"Spin $\uparrow$,  Left  ($p_x > 0$)"),
    ("up",   "right", 0, 1, r"Spin $\uparrow$,  Right  ($p_x < 0$)"),
    ("down", "left",  1, 0, r"Spin $\downarrow$,  Left  ($p_x > 0$)"),
    ("down", "right", 1, 1, r"Spin $\downarrow$,  Right  ($p_x < 0$)"),
]


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _exp_pdf(M: np.ndarray, tau: float) -> np.ndarray:
    """Exponential PDF normalised over [MASS_MIN, MASS_MAX]."""
    if abs(tau) < 1e-9:
        norm = MASS_MAX - MASS_MIN
    else:
        norm = (np.exp(tau * MASS_MAX) - np.exp(tau * MASS_MIN)) / tau
    return np.exp(tau * M) / norm


def roofit_curve(M: np.ndarray, params: dict, mean1: float, sigma1: float) -> tuple:
    """
    Return (total, jpsi, psip, bkg) bin-count curves for a single panel.
    All curves are in units of events/bin.
    """
    mean2  = mean1 * RATIO
    sigma2 = sigma1 * RATIO

    g1  = sp_norm.pdf(M, mean1, sigma1)
    g2  = sp_norm.pdf(M, mean2, sigma2)
    bkg = _exp_pdf(M, params["tau"])

    jpsi = params["N_sig1"] * g1  * BIN_W
    psip = params["N_sig2"] * g2  * BIN_W
    bkgc = params["N_bkg"]  * bkg * BIN_W
    return jpsi + psip + bkgc, jpsi, psip, bkgc


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_panels(feat_up: Path, feat_down: Path,
                 pred_up: Path, pred_down: Path,
                 threshold: float) -> dict:
    """Return per-panel dicts with 'all_M', 'dnn_M', 'N_all', 'N_dnn'."""
    panels = {}
    for spin, feat_path, pred_path in [("up",   feat_up,   pred_up),
                                        ("down", feat_down, pred_down)]:
        feat = np.load(feat_path, allow_pickle=True)
        pred = np.load(pred_path, allow_pickle=True)

        M_all  = feat["M"].astype(np.float64)
        px_all = feat["px_dimu"].astype(np.float64)
        sel_w  = (M_all >= MASS_MIN) & (M_all <= MASS_MAX)
        M_all, px_all = M_all[sel_w], px_all[sel_w]

        M_dnn  = pred["M"].astype(np.float64)
        px_dnn = pred["px_dimu"].astype(np.float64)
        p_jpsi = pred["y_proba"].astype(np.float64)[:, 0]
        sel_d  = (p_jpsi >= threshold) & (M_dnn >= MASS_MIN) & (M_dnn <= MASS_MAX)
        M_dnn, px_dnn = M_dnn[sel_d], px_dnn[sel_d]

        for side, mask_fn in [("left",  lambda px: px > 0),
                               ("right", lambda px: px <= 0)]:
            key = f"{spin}_{side}"
            ma  = M_all[mask_fn(px_all)]
            md  = M_dnn[mask_fn(px_dnn)]
            panels[key] = {"all_M": ma, "dnn_M": md}

    return panels


def _dnn_an(panels: dict) -> tuple[float, float] | None:
    def _cnt(arr): return float(len(arr))
    N_uL = _cnt(panels["up_left"]["dnn_M"])
    N_uR = _cnt(panels["up_right"]["dnn_M"])
    N_dL = _cnt(panels["down_left"]["dnn_M"])
    N_dR = _cnt(panels["down_right"]["dnn_M"])
    if min(N_uL, N_uR, N_dL, N_dR) <= 0:
        return None
    A = math.sqrt(N_uL * N_dR)
    B = math.sqrt(N_dL * N_uR)
    A_raw = (A - B) / (A + B)
    A_N   = A_raw / (ETA * F * P_AVG)
    dNuL, dNuR = math.sqrt(N_uL), math.sqrt(N_uR)
    dNdL, dNdR = math.sqrt(N_dL), math.sqrt(N_dR)
    t1  = (B/A)**2 * ((N_uR*dNuL)**2 + (N_dL*dNdR)**2)
    t2  = (A/B)**2 * ((N_dR*dNdL)**2 + (N_uL*dNuR)**2)
    dA_N = (1.0/(ETA*F*P_AVG)) * math.sqrt(t1+t2) / (A+B)**2
    return A_N, dA_N


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_overlay(feat_up: Path, feat_down: Path,
                 pred_up: Path, pred_down: Path,
                 fit_json: Path, out_dir: Path,
                 threshold: float = DNN_THRESHOLD):

    with open(fit_json) as fh:
        fp = json.load(fh)

    mean1, sigma1 = fp["mean1"], fp["sigma1"]
    fit_an  = fp["A_N"]
    fit_dan = fp["dA_N"]

    panels = _load_panels(feat_up, feat_down, pred_up, pred_down, threshold)

    dnn_res = _dnn_an(panels)
    dnn_an  = dnn_res[0] if dnn_res else None
    dnn_dan = dnn_res[1] if dnn_res else None

    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.85],
                   wspace=0.38, hspace=0.40)

    mass_axes = {}
    for spin, side, row, col, _ in PANELS:
        mass_axes[f"{spin}_{side}"] = fig.add_subplot(gs[row, col])
    ax_an = fig.add_subplot(gs[:, 2])

    for spin, side, _, _, label in PANELS:
        key = f"{spin}_{side}"
        ax  = mass_axes[key]
        d   = panels[key]
        pars = fp["panels"][key]

        # all-events histogram (data points)
        cnt_all, _ = np.histogram(d["all_M"], bins=BIN_EDGES)
        yerr = np.where(cnt_all > 0, np.sqrt(cnt_all), 1.0)
        ax.errorbar(BIN_CTRS, cnt_all, yerr=yerr,
                    fmt="o", ms=3.5, lw=1.1, capsize=2,
                    color="black", label=f"Data  (N={len(d['all_M']):,})", zorder=3)

        # DNN-selected filled histogram
        cnt_dnn, _ = np.histogram(d["dnn_M"], bins=BIN_EDGES)
        ax.bar(BIN_CTRS, cnt_dnn, width=BIN_W * 0.9,
               color="#2471A3", alpha=0.45,
               label=rf"DNN $p_{{J/\psi}}\geq{threshold:.2f}$  (N={len(d['dnn_M']):,})",
               zorder=2)

        # RooFit PDF curves
        M_fine = np.linspace(MASS_MIN, MASS_MAX, 600)
        total, jpsi_c, psip_c, bkg_c = roofit_curve(M_fine, pars, mean1, sigma1)
        ax.plot(M_fine, total,  color="#1A5276", lw=2.0, label="RooFit total",  zorder=4)
        ax.plot(M_fine, jpsi_c, color="#27AE60", lw=1.6, ls="--", label=r"$J/\psi$",   zorder=4)
        ax.plot(M_fine, psip_c, color="#8E44AD", lw=1.2, ls="--", label=r"$\psi(2S)$", zorder=4)
        ax.plot(M_fine, bkg_c,  color="#C0392B", lw=1.4, ls=":",  label="Background", zorder=4)

        # J/ψ peak lines
        ax.axvline(3.097, color="gray", lw=0.7, ls=":", alpha=0.7)
        ax.axvline(3.686, color="gray", lw=0.7, ls=":", alpha=0.7)

        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"Dimuon mass  [GeV/$c^2$]", fontsize=10)
        ax.set_ylabel(f"Events / {BIN_W*1000:.0f} MeV", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
        ax.grid(alpha=0.25, ls="--")

        # N_sig annotation
        ax.text(0.03, 0.97,
                rf"$N_{{J/\psi}}^{{\rm fit}} = {pars['N_sig1']:.0f} \pm {pars['dN_sig1']:.0f}$",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFACD", ec="goldenrod",
                          lw=0.8, alpha=0.9))

    # ── A_N comparison panel ─────────────────────────────────────────────────
    methods  = []
    an_vals  = []
    an_errs  = []
    colors   = []

    if fit_an is not None:
        methods.append("RooFit")
        an_vals.append(fit_an)
        an_errs.append(fit_dan)
        colors.append("#1A5276")

    if dnn_an is not None:
        methods.append(rf"DNN ($t={threshold:.2f}$)")
        an_vals.append(dnn_an)
        an_errs.append(dnn_dan)
        colors.append("#2471A3")

    y_pos = np.arange(len(methods))
    ax_an.axvline(0, color="black", lw=1.0, ls="--", zorder=1)
    for y, val, err, col in zip(y_pos, an_vals, an_errs, colors):
        ax_an.errorbar([val], [y], xerr=[err],
                       fmt="D", ms=10, capsize=7, elinewidth=2.0, capthick=2.0,
                       color=col, zorder=3)
        ax_an.text(val + err * 1.1, float(y),
                   rf"${val:+.3f} \pm {err:.3f}$",
                   va="center", ha="left", fontsize=10, color=col)

    ax_an.set_yticks(y_pos)
    ax_an.set_yticklabels(methods, fontsize=11)
    ax_an.set_xlabel(r"$A_N$", fontsize=12)
    ax_an.set_title(r"$A_N$ comparison", fontsize=12, fontweight="bold")
    ax_an.set_ylim(-0.6, len(methods) - 0.4)
    ax_an.grid(axis="x", alpha=0.3, ls="--")
    params_txt = (rf"$\eta={ETA}$,  $f={F}$,  $\langle P\rangle={P_AVG}$")
    ax_an.text(0.05, 0.05, params_txt, transform=ax_an.transAxes,
               fontsize=9, va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    fig.suptitle(
        r"$J/\psi$ TSSA — DNN selection vs RooFit simultaneous mass fit"
        "\n"
        r"SpinQuest 2024 commissioning data",
        fontsize=12, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"overlay_dnn_roofit_t{threshold:.3f}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved → {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-up",    default=FEAT_UP,   type=Path)
    p.add_argument("--feat-down",  default=FEAT_DOWN, type=Path)
    p.add_argument("--pred-up",    default=PRED_UP,   type=Path)
    p.add_argument("--pred-down",  default=PRED_DOWN, type=Path)
    p.add_argument("--fit-params", default=FIT_JSON,  type=Path)
    p.add_argument("--out-dir",    default=OUT_DIR,   type=Path)
    p.add_argument("--threshold",  default=DNN_THRESHOLD, type=float)
    args = p.parse_args()

    plot_overlay(
        feat_up=args.feat_up, feat_down=args.feat_down,
        pred_up=args.pred_up, pred_down=args.pred_down,
        fit_json=args.fit_params,
        out_dir=args.out_dir,
        threshold=args.threshold,
    )
    print("[DONE]")


if __name__ == "__main__":
    main()
