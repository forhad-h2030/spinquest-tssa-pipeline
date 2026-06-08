#!/usr/bin/env python3
"""Inclusive dimuon mass (0-10 GeV) before/after removing DNN J/psi and psi(2S)
candidates. Regenerates the inclusive sample on first run (extract 0-10 +
classify, cached in post_processing/output/); pass --force to rebuild."""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO      = Path(__file__).resolve().parents[2]
CONFIG    = REPO / "config/pipeline.sh"
CKPT_DIR  = REPO / "classifier/checkpoints"
OUTDIR    = REPO / "post_processing/output"
FIGDIR    = REPO / "figures/note"
OUT_TOTAL = FIGDIR / "inclusive_dimuon_mass.png"        # total only (Fig. 15)
OUT       = FIGDIR / "inclusive_jpsi_psip_removed.png"  # total + background (Fig. 16)
FIGDIR.mkdir(parents=True, exist_ok=True)

BLUE = "#1f77b4"   # total exp-data colour, shared by both figures

THR_JPSI = 0.635   # 90% J/psi purity (window centered on reco peak ~3.32)
THR_PSIP = 0.724   # 90% psi(2S) purity (window centered on reco peak ~3.80)
BINS = np.arange(0.0, 10.0001, 0.1)


def _cfg(var: str) -> str:
    """Read a shell variable assignment from config/pipeline.sh."""
    m = re.search(rf'^{var}="?([^"\n]+)"?', CONFIG.read_text(), re.M)
    if not m:
        raise RuntimeError(f"{var} not found in {CONFIG}")
    return m.group(1)


def _ensure_inclusive(force: bool = False):
    """Extract (mass 0-10) + classify both spins; cache in post_processing/output."""
    roots = {"up": _cfg("ROOT_UP"), "down": _cfg("ROOT_DOWN")}
    preds = {}
    for spin, root in roots.items():
        feat = OUTDIR / f"incl_features_{spin}.npz"
        pred = OUTDIR / f"incl_pred_{spin}.npz"
        if force or not pred.exists():
            print(f"[gen] {spin}: extract (mass 0-10) + classify ...", flush=True)
            subprocess.run([sys.executable, str(REPO / "post_processing/extract.py"),
                            "--input", root, "--output", str(feat),
                            "--spin", spin, "--mass-min", "0", "--mass-max", "10"],
                           check=True)
            subprocess.run([sys.executable, str(REPO / "post_processing/classify.py"),
                            "--features", str(feat), "--ckpt-dir", str(CKPT_DIR),
                            "--output", str(pred)], check=True)
        preds[spin] = np.load(pred, allow_pickle=True)
    return preds["up"], preds["down"]


ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="rebuild the inclusive predictions")
args = ap.parse_args()

up, dn = _ensure_inclusive(force=args.force)
M     = np.concatenate([up["M"],             dn["M"]]).astype(float)
p_jp  = np.concatenate([up["y_proba"][:, 0], dn["y_proba"][:, 0]]).astype(float)
p_pp  = np.concatenate([up["y_proba"][:, 1], dn["y_proba"][:, 1]]).astype(float)

rem_jp = p_jp >= THR_JPSI
rem_pp = p_pp >= THR_PSIP
keep   = ~rem_jp & ~rem_pp           # both resonances removed
N_tot, N_jp, N_pp, N_keep = len(M), int(rem_jp.sum()), int(rem_pp.sum()), int(keep.sum())
print(f"[INFO] total={N_tot}  J/psi removed={N_jp}  psi(2S) removed={N_pp}  remaining={N_keep}")

def style(ax):
    ax.axvline(3.097, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(3.686, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$M_{\mu\mu}$  [GeV/$c^2$]", fontsize=12)
    ax.set_ylabel("Events / 0.1 GeV", fontsize=12)
    ax.grid(axis="y", ls="--", alpha=0.3)

# ── Figure 15: total exp data only (blue) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(M, bins=BINS, histtype="step", lw=1.8, color=BLUE,
        label=f"Total events  (N={N_tot:,})")
style(ax)
ax.set_title("Inclusive dimuon invariant-mass spectrum\n"
             "SpinQuest 2024 commissioning (spin-up + spin-down)",
             fontsize=11, fontweight="bold")
ax.text(0.97, 0.97,
        f"N = {N_tot:,}\nmean = {M.mean():.3f} GeV\nstd = {M.std():.3f} GeV",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
fig.savefig(OUT_TOTAL, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT_TOTAL}")

# ── Figure 16: total (blue) + background after J/psi + psi(2S) removal ───────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(M, bins=BINS, histtype="step", lw=1.8, color=BLUE,
        label=f"Total events  (N={N_tot:,})")
ax.hist(M[keep], bins=BINS, histtype="stepfilled", lw=1.6,
        edgecolor="#c0392b", facecolor="#c0392b", alpha=0.30,
        label=fr"Background, $J/\psi$+$\psi(2S)$ removed  (N={N_keep:,})")
style(ax)
ax.set_title("Inclusive dimuon mass — before and after DNN $J/\\psi$ + $\\psi(2S)$ removal\n"
             "SpinQuest 2024 commissioning (spin-up + spin-down)",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT}")
