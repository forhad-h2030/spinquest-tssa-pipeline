#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
MC_BUNDLE = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"
MC_CKPT   = REPO / "classifier/checkpoints/boot_000/ml_input_multiclass_M_26_march_19.best.pth"

# ── physics normalization (target/beam property — shared across processes) ────
ETA, F, P_AVG = 0.6, 0.18, 0.70
PREF = 1.0 / (ETA * F * P_AVG)

# data-driven class prior (exp argmax composition) used to weight the MC purity
DATA_PRIOR = np.array([751, 199, 1511, 663], float)

MASS_WINDOW = (2.2, 4.2)
CLASSES = {
    "jpsi": dict(idx=0, label=r"$J/\psi$",     window=MASS_WINDOW, color="#1f77b4"),
    "psip": dict(idx=1, label=r"$\psi(2S)$",   window=MASS_WINDOW, color="#d62728"),
    "dy":   dict(idx=2, label="DY",            window=MASS_WINDOW, color="#2ca02c"),
    "comb": dict(idx=3, label="Combinatoric",  window=MASS_WINDOW, color="#9467bd"),
}
THRESHOLDS = np.round(np.arange(0.30, 0.91, 0.05), 3)
WORKING_POINT = 0.80          # uniform probability cut applied to every class
PLAIN = {"jpsi": "J/psi", "psip": "psi(2S)", "dy": "DY", "comb": "Comb"}


def load_mc_purity_inputs():
    """MC test bundle + reconstructed mass (via the checkpoint scaler at the
    rec_dimu_M index 4) + per-event weights to the data-driven prior."""
    if not (MC_BUNDLE.exists() and MC_CKPT.exists()):
        return None
    b  = np.load(MC_BUNDLE, allow_pickle=True)
    ck = torch.load(MC_CKPT, map_location="cpu", weights_only=False)
    yt = b["y_test"].astype(int)
    proba = b["y_proba"].astype(np.float64)
    sc = ck["scaler"]
    m_mean = float(np.array(sc["mean"]).flatten()[4])
    m_std  = float(np.array(sc["std"]).flatten()[4])
    mass = b["X_test"][:, 4].astype(np.float64) * m_std + m_mean
    native = np.array([(yt == i).sum() for i in range(4)], float)
    w = (DATA_PRIOR / native)[yt]
    return dict(yt=yt, proba=proba, mass=mass, w=w)


def purity_curve(mc, cfg):
    """MC purity of class `cfg` vs threshold, data-prior weighted, in-window."""
    am = mc["proba"].argmax(1)
    idx, (lo, hi) = cfg["idx"], cfg["window"]
    inwin = (mc["mass"] >= lo) & (mc["mass"] <= hi)
    pur = []
    for t in THRESHOLDS:
        sel = (am == idx) & (mc["proba"][:, idx] >= t) & inwin
        den = mc["w"][sel].sum()
        pur.append(mc["w"][sel & (mc["yt"] == idx)].sum() / den if den > 0 else np.nan)
    return np.array(pur)


def _compute_an(N_uL, N_uR, N_dL, N_dR):
    """Geometric-mean A_N and its Poisson statistical error (cf. asymmetry.py)."""
    if min(N_uL, N_uR, N_dL, N_dR) <= 0:
        return None
    A = math.sqrt(N_uL * N_dR)
    B = math.sqrt(N_dL * N_uR)
    den = A + B
    A_raw = (A - B) / den
    A_N = PREF * A_raw
    dA_N = PREF * (A * B / den**2) * math.sqrt(
        1.0 / N_uL + 1.0 / N_uR + 1.0 / N_dL + 1.0 / N_dR)
    return A_raw, A_N, dA_N


def _counts(pred, idx, threshold, window):
    """Four spin/side counts for class `idx` at the given threshold and window.
    Selection = argmax over the 4 classes == idx AND p_idx >= threshold."""
    M = pred["M"].astype(np.float64)
    px = pred["px_dimu"].astype(np.float64)
    proba = pred["y_proba"].astype(np.float64)
    sel = (proba.argmax(1) == idx) & (proba[:, idx] >= threshold) \
        & (M >= window[0]) & (M <= window[1])
    NL = float((sel & (px > 0)).sum())
    NR = float((sel & (px <= 0)).sum())
    return NL, NR


def scan_class(pup, pdn, cfg):
    """Threshold scan of A_N for one class; returns the per-threshold arrays."""
    out = {"t": [], "an": [], "dan": [], "n": []}
    print(f"\n=== {cfg['label']}  (window {cfg['window']}) ===")
    print(f"{'t':>6s} {'N_uL':>6s} {'N_uR':>6s} {'N_dL':>6s} {'N_dR':>6s} "
          f"{'A_raw':>9s} {'A_N':>9s} {'±dA_N':>9s}")
    for t in THRESHOLDS:
        NuL, NuR = _counts(pup, cfg["idx"], t, cfg["window"])
        NdL, NdR = _counts(pdn, cfg["idx"], t, cfg["window"])
        res = _compute_an(NuL, NuR, NdL, NdR)
        if res is None:
            print(f"{t:>6.2f} {int(NuL):>6d} {int(NuR):>6d} {int(NdL):>6d} {int(NdR):>6d}  SKIP")
            continue
        A_raw, A_N, dA_N = res
        print(f"{t:>6.2f} {int(NuL):>6d} {int(NuR):>6d} {int(NdL):>6d} {int(NdR):>6d} "
              f"{A_raw:>+9.4f} {A_N:>+9.4f} {dA_N:>9.4f}")
        out["t"].append(t); out["an"].append(A_N); out["dan"].append(dA_N)
        out["n"].append(int(NuL + NuR + NdL + NdR))
    return out


def working_point_summary(pup, pdn, mc, wp=WORKING_POINT):
    """At a single uniform probability cut: per-component A_N (exp data) and the
    J/psi-tag contamination composition from the MC confusion matrix."""
    jwin = CLASSES["jpsi"]["window"]
    print(f"\n{'='*64}\nWorking point: uniform cut p >= {wp:.2f}")
    print(f"Counting mass window (all channels): [{jwin[0]}, {jwin[1]}] GeV/c^2")
    print(f"{'='*64}")

    print("\nComponent TSSA A_N (experimental data):")
    print(f"  {'channel':12s} {'A_N':>9s} {'+-stat':>9s} {'N':>7s}")
    comp = {}                                    # channel key -> (A_N, dA_N)
    for c, cfg in CLASSES.items():
        NuL, NuR = _counts(pup, cfg["idx"], wp, cfg["window"])
        NdL, NdR = _counts(pdn, cfg["idx"], wp, cfg["window"])
        res = _compute_an(NuL, NuR, NdL, NdR)
        if res is None:
            continue
        _, a, da = res
        comp[c] = (a, da)
        print(f"  {PLAIN[c]:12s} {a:>+9.3f} {da:>9.3f} {int(NuL+NuR+NdL+NdR):>7d}")

    if mc is None:
        print("\n[WARN] MC bundle unavailable — contamination proportions skipped.")
        return
    idx, (lo, hi) = CLASSES["jpsi"]["idx"], jwin
    am = mc["proba"].argmax(1)
    sel = (am == idx) & (mc["proba"][:, idx] >= wp) & (mc["mass"] >= lo) & (mc["mass"] <= hi)
    tot = mc["w"][sel].sum()
    print(f"\nJ/psi-tag contamination from the simulated confusion matrix")
    print(f"  (argmax=J/psi, p>=%.2f, mass in [%.1f, %.1f], data-driven prior):" % (wp, lo, hi))
    keys = ["jpsi", "psip", "dy", "comb"]
    frac = {}                                    # channel key -> contamination fraction
    for c in keys:
        frac[c] = mc["w"][sel & (mc["yt"] == CLASSES[c]["idx"])].sum() / tot
        tag = "  (signal purity)" if c == "jpsi" else ""
        print(f"    true {PLAIN[c]:8s}: {frac[c]:.4f}{tag}")
    fbkg = 1.0 - frac["jpsi"]
    print(f"    background fraction f = {fbkg:.4f}")

    # ── background-corrected J/psi asymmetry: Eqs (4)-(5) ─────────────────────
    # A_bgr = (1/f) sum_bkg frac_j * A_j   (sim weights x measured channel A_N)
    bkg = [c for c in keys if c != "jpsi" and c in comp]
    Abgr = sum(frac[c] * comp[c][0] for c in bkg) / fbkg
    dAbgr = math.sqrt(sum((frac[c] * comp[c][1])**2 for c in bkg)) / fbkg
    Aincl, dAincl = comp["jpsi"]
    Ajpsi = (Aincl - fbkg * Abgr) / (1.0 - fbkg)
    dAjpsi = math.sqrt(dAincl**2 + fbkg**2 * dAbgr**2) / (1.0 - fbkg)
    print(f"\nBackground-corrected J/psi A_N  (Eqs. 4-5):")
    print(f"    A_incl (tag)   = {Aincl:+.3f} +- {dAincl:.3f}")
    print(f"    A_bgr (combined) = {Abgr:+.3f} +- {dAbgr:.3f}")
    print(f"    A_jpsi = (A_incl - f*A_bgr)/(1-f) = {Ajpsi:+.3f} +- {dAjpsi:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-up",   default=REPO / "post_processing/output/pred_up.npz", type=Path)
    p.add_argument("--pred-down", default=REPO / "post_processing/output/pred_down.npz", type=Path)
    p.add_argument("--out-dir",   default=REPO / "figures/tssa", type=Path)
    p.add_argument("--classes", nargs="+", default=list(CLASSES),
                   choices=list(CLASSES), help="which classes to scan")
    args = p.parse_args()

    pup = np.load(args.pred_up, allow_pickle=True)
    pdn = np.load(args.pred_down, allow_pickle=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {c: scan_class(pup, pdn, CLASSES[c]) for c in args.classes}
    mc = load_mc_purity_inputs()
    if mc is None:
        print("[WARN] MC bundle/checkpoint not found — purity overlay skipped.")

    working_point_summary(pup, pdn, mc)

    n = len(args.classes)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 4.6 * nrow),
                             squeeze=False)
    for ax, c in zip(axes.flat, args.classes):
        cfg, r = CLASSES[c], results[c]
        ax.axhline(0, color="black", lw=1.0, ls="--", zorder=1)
        ax.errorbar(r["t"], r["an"], yerr=r["dan"], fmt="o", ms=7, capsize=5,
                    elinewidth=1.6, color=cfg["color"], zorder=3, label=r"$A_N$ (data)")
        for x, y, dy, nn in zip(r["t"], r["an"], r["dan"], r["n"]):
            ax.annotate(f"{nn}", xy=(x, y + dy), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="dimgrey")
        ax.set_title(f"{cfg['label']}  TSSA $A_N$ vs DNN threshold",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(r"DNN $p_{\rm class}$ threshold")
        ax.set_ylabel(r"$A_N$")
        ax.grid(alpha=0.3, ls="--")
        # second axis: MC class purity (data-driven prior) at each threshold
        if mc is not None:
            axp = ax.twinx()
            pur = purity_curve(mc, cfg)
            axp.plot(THRESHOLDS, pur, color="black", lw=1.8, ls="--",
                     marker="s", ms=4, label="MC purity")
            axp.set_ylabel("purity", color="black")
            axp.set_ylim(0, 1.02)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axp.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        else:
            ax.legend(loc="upper left", fontsize=8)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(r"Per-class TSSA $A_N$ (data) and MC purity vs DNN threshold  "
                 r"($\eta=0.6,\ f=0.18,\ \langle P\rangle=0.70$; stat. errors)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out_dir / "an_vs_threshold_by_class.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
