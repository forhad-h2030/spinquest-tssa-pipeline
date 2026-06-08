#!/usr/bin/env python3
"""Legacy in-repo RooFit J/psi TSSA A_N (independent of the DNN): simultaneous
mass fit on features_{up,down}.npz -> A_N via the geometric-mean estimator. The
analysis note's fit uses ana-spinquest-fit/fit/fit_mode_final.py --data ml."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ── Physics parameters ────────────────────────────────────────────────────────
ETA   = 0.6
F     = 0.18
P_AVG = 0.70

MASS_MIN = 2.0
MASS_MAX = 5.9


def _load_panels(feat_up: Path, feat_down: Path) -> dict[str, np.ndarray]:
    """
    Read features NPZs and return mass arrays for all four spin states.
    No DNN scores used — pure kinematics only.
    """
    up   = np.load(feat_up,   allow_pickle=True)
    down = np.load(feat_down, allow_pickle=True)

    results = {}
    for spin, data in [("up", up), ("down", down)]:
        M  = data["M"].astype(np.float64)
        px = data["px_dimu"].astype(np.float64)

        sel = (M >= MASS_MIN) & (M <= MASS_MAX)
        M, px = M[sel], px[sel]

        results[f"{spin}_left"]  = M[px > 0]   # px > 0 = left
        results[f"{spin}_right"] = M[px <= 0]  # px ≤ 0 = right

    for key, arr in results.items():
        print(f"  [{key}]  {len(arr):,} events in mass window")

    return results


def _compute_an(N_uL, dN_uL, N_uR, dN_uR,
                N_dL, dN_dL, N_dR, dN_dR,
                eta=ETA, f=F, P=P_AVG):
    """Geometric-mean estimator with full error propagation from fit errors."""
    if min(N_uL, N_uR, N_dL, N_dR) <= 0:
        return None

    A         = math.sqrt(N_uL * N_dR)
    B         = math.sqrt(N_dL * N_uR)
    denom     = A + B
    A_raw     = (A - B) / denom
    prefactor = 1.0 / (eta * f * P)
    A_N       = prefactor * A_raw

    t1   = (B / A) ** 2 * ((N_uR * dN_uL) ** 2 + (N_dL * dN_dR) ** 2)
    t2   = (A / B) ** 2 * ((N_dR * dN_dL) ** 2 + (N_uL * dN_uR) ** 2)
    dA_N = prefactor * math.sqrt(t1 + t2) / denom ** 2

    return A_raw, A_N, dA_N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-up",   required=True, type=Path)
    p.add_argument("--feat-down", required=True, type=Path)
    p.add_argument("--out-dir",   default="figures/tssa_fit", type=Path)
    p.add_argument("--eta", type=float, default=ETA)
    p.add_argument("--f",   type=float, default=F)
    p.add_argument("--P",   type=float, default=P_AVG)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Import RooFit engine ─────────────────────────────────────────────────
    # Imported here so the script fails early with a clear message if ROOT
    # is not available, rather than after loading data.
    try:
        from fit_common import fit_and_save, PANELS, panel_suffix
        from ROOT import RooRealVar, RooArgList
        import ROOT
        ROOT.gROOT.SetBatch(True)
    except ImportError as e:
        raise SystemExit(
            f"[ERROR] ROOT / PyROOT not available: {e}\n"
            "  Load ROOT before running: module load root  (on Rivanna)\n"
            "  or activate an environment with PyROOT installed."
        )

    # ── Load mass arrays ─────────────────────────────────────────────────────
    print("\n── Loading data ──────────────────────────────────────────────")
    mass_arrays = _load_panels(args.feat_up, args.feat_down)

    # ── Run simultaneous RooFit ──────────────────────────────────────────────
    fit_plot = args.out_dir / "fit_4panel.png"
    print("\n── Running simultaneous fit ──────────────────────────────────")
    fit_and_save(mass_arrays, str(fit_plot))

    # ── Re-run fit to recover parameter objects for A_N ──────────────────────
    # fit_and_save() draws the canvas but does not return parameters.
    # We re-run the final fit here in a lightweight way to extract N_sig.
    print("\n── Extracting signal yields for A_N ──────────────────────────")
    from fit_common import (make_roo_dataset, build_panel_model,
                             MASS_MIN as FMIN, MASS_MAX as FMAX, N_BINS, RATIO)
    from ROOT import (RooRealVar, RooFormulaVar, RooDataSet, RooSimultaneous,
                      RooCategory, RooFit, RooArgSet, RooArgList)

    mass   = RooRealVar("rec_dimu_M", "mass", FMIN, FMAX)
    mass.setBins(N_BINS)
    mean1  = RooRealVar("mean1",  "mean",  3.10, 2.90, 3.40)
    sigma1 = RooRealVar("sigma1", "sigma", 0.10, 0.04, 0.30)
    mean2  = RooFormulaVar("mean2",  "psi' mean",  f"@0*{RATIO:.8f}", RooArgList(mean1))
    sigma2 = RooFormulaVar("sigma2", "psi' sigma", f"@0*{RATIO:.8f}", RooArgList(sigma1))

    datasets, models, panel_pars, keep = {}, {}, {}, []
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        datasets[suf] = make_roo_dataset(f"ds_{suf}", mass, mass_arrays[f"{spin}_{side}"])
        mdl, pars, shapes = build_panel_model(suf, mass, mean1, sigma1, mean2, sigma2)
        models[suf], panel_pars[suf] = mdl, pars
        keep.extend(shapes)

    sample   = RooCategory("sample2", "sample")
    for spin, side, _, _ in PANELS:
        sample.defineType(panel_suffix(spin, side))

    combData = RooDataSet(
        "combData2", "data", RooArgSet(mass, sample),
        RooFit.Index(sample),
        RooFit.Import("up_right",   datasets["up_right"]),
        RooFit.Import("up_left",    datasets["up_left"]),
        RooFit.Import("down_right", datasets["down_right"]),
        RooFit.Import("down_left",  datasets["down_left"]),
    )

    simPdf = RooSimultaneous("simPdf2", "simPdf", sample)
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        simPdf.addPdf(models[suf], suf)

    simPdf.fitTo(combData, RooFit.PrintLevel(-1))

    # ── Extract N_sig ± err per panel ────────────────────────────────────────
    yields = {}
    print(f"\n  {'panel':>12s}  {'N_sig':>8s}  {'err':>8s}")
    for spin, side, _, _ in PANELS:
        suf   = panel_suffix(spin, side)
        nsig1 = panel_pars[suf][1]
        N     = nsig1.getVal()
        dN    = nsig1.getError()
        yields[f"{spin}_{side}"] = (N, dN)
        print(f"  {suf:>12s}  {N:>8.1f}  {dN:>8.1f}")

    # ── Compute A_N ─────────────────────────────────────────────────────────
    N_uL,  dN_uL  = yields["up_left"]
    N_uR,  dN_uR  = yields["up_right"]
    N_dL,  dN_dL  = yields["down_left"]
    N_dR,  dN_dR  = yields["down_right"]

    res = _compute_an(N_uL, dN_uL, N_uR, dN_uR,
                      N_dL, dN_dL, N_dR, dN_dR,
                      args.eta, args.f, args.P)

    print("\n── A_N result (RooFit-based) ─────────────────────────────────")
    if res is None:
        print("  [WARN] zero or negative yield in at least one panel — cannot compute A_N")
    else:
        A_raw, A_N, dA_N = res
        print(f"  A_raw = {A_raw:+.4f}")
        print(f"  A_N   = {A_N:+.4f} ± {dA_N:.4f}")
        print(f"  (eta={args.eta}, f={args.f}, P={args.P})")

        # save result to text file for easy comparison
        result_file = args.out_dir / "an_result.txt"
        with open(result_file, "w") as fp:
            fp.write("# RooFit-based A_N extraction\n")
            fp.write(f"# eta={args.eta}  f={args.f}  P={args.P}\n")
            fp.write(f"N_uL  = {N_uL:.1f} +/- {dN_uL:.1f}\n")
            fp.write(f"N_uR  = {N_uR:.1f} +/- {dN_uR:.1f}\n")
            fp.write(f"N_dL  = {N_dL:.1f} +/- {dN_dL:.1f}\n")
            fp.write(f"N_dR  = {N_dR:.1f} +/- {dN_dR:.1f}\n")
            fp.write(f"A_raw = {A_raw:+.6f}\n")
            fp.write(f"A_N   = {A_N:+.6f}\n")
            fp.write(f"dA_N  = {dA_N:.6f}\n")
        print(f"\n  result saved → {result_file}")

        # ── Save fit parameters to JSON for overlay plot ─────────────────
        fit_params = {
            "mean1":  mean1.getVal(),
            "sigma1": sigma1.getVal(),
            "ratio":  float(3.686 / 3.097),
            "mass_min": float(MASS_MIN),
            "mass_max": float(MASS_MAX),
            "panels": {},
            "A_N":   A_N,
            "dA_N":  dA_N,
            "A_raw": A_raw,
            "eta": args.eta, "f": args.f, "P": args.P,
        }
        for spin, side, _, _ in PANELS:
            suf = panel_suffix(spin, side)
            tau_p, nsig1_p, nsig2_p, nbkg_p = panel_pars[suf]
            fit_params["panels"][suf] = {
                "N_sig1": nsig1_p.getVal(), "dN_sig1": nsig1_p.getError(),
                "N_sig2": nsig2_p.getVal(), "dN_sig2": nsig2_p.getError(),
                "N_bkg":  nbkg_p.getVal(),  "dN_bkg":  nbkg_p.getError(),
                "tau":    tau_p.getVal(),
            }
        params_file = args.out_dir / "fit_params.json"
        with open(params_file, "w") as fp:
            json.dump(fit_params, fp, indent=2)
        print(f"  fit params saved → {params_file}")

    print(f"  fit plot   → {fit_plot}")
    print("[DONE]")


if __name__ == "__main__":
    main()
