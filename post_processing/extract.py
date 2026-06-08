#!/usr/bin/env python3
"""Reconstructed ROOT (ktracker flat tree) -> 18 DNN input features (+ M, px_dimu,
event IDs), applying the dimuon quality cuts via uproot. Run from
post_processing/run.sh; cut list and feature order are in utils/features.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot
import awkward as ak

MUON_MASS = 0.1056          # GeV
FEATURE_NAMES = [
    "rec_dimu_y",
    "rec_dimu_eta",
    "rec_dimu_E",
    "rec_dimu_pz",
    "rec_dimu_M",
    "rec_mu_theta_pos",
    "rec_mu_theta_neg",
    "rec_mu_open_angle",
    "rec_mu_dpt",
    "rec_dimu_mT",
    "rec_mu_Epos",
    "rec_mu_Eneg",
    "rec_track_pos_x_st1",
    "rec_track_neg_x_st1",
    "rec_track_pos_px_st1",
    "rec_track_neg_px_st1",
    "rec_dz_vtx",
    "rec_mu_deltaR",
]

# Jagged dimuon branches (one var-length list per event)
DIMU_BRANCHES = [
    "rec_dimuon_px_pos_tgt", "rec_dimuon_py_pos_tgt", "rec_dimuon_pz_pos_tgt",
    "rec_dimuon_px_neg_tgt", "rec_dimuon_py_neg_tgt", "rec_dimuon_pz_neg_tgt",
    "rec_dimuon_x_pos_st1",  "rec_dimuon_x_neg_st1",
    "rec_dimuon_px_pos_st1", "rec_dimuon_px_neg_st1",
    "rec_dimuon_z_pos_vtx",  "rec_dimuon_z_neg_vtx",
    # quality-cut branches
    "rec_dimuon_y_pos_st1",  "rec_dimuon_y_neg_st1",
    "rec_dimuon_py_pos_st1", "rec_dimuon_py_neg_st1",
    "rec_dimuon_chisq_target_pos",   "rec_dimuon_chisq_target_neg",
    "rec_dimuon_chisq_dump_pos",     "rec_dimuon_chisq_dump_neg",
    "rec_dimuon_chisq_upstream_pos", "rec_dimuon_chisq_upstream_neg",
]

# Scalar (one per event) metadata branches
META_BRANCHES = ["eventID", "runID", "spillID"]

# Optional trigger branch — applied as event-level mask if present in the tree
FPGA_BRANCH = "fpga_bits"


def _wrap_phi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def extract(root_path: Path, tree_name: str, mass_min: float, mass_max: float,
            spin: str) -> dict:

    tree: uproot.behaviors.TTree.HasBranches = uproot.open(  # type: ignore[assignment]
        f"{root_path}:{tree_name}"
    )
    avail = set(tree.keys())
    missing = [b for b in DIMU_BRANCHES + META_BRANCHES if b not in avail]
    if missing:
        raise RuntimeError(f"Missing branches: {missing}")

    has_fpga = FPGA_BRANCH in avail
    if not has_fpga:
        print(f"[WARN] branch '{FPGA_BRANCH}' not found — FPGA trigger cut skipped")

    # Load jagged dimuon branches + scalar metadata (+ trigger if available)
    extra = [FPGA_BRANCH] if has_fpga else []
    data = tree.arrays(DIMU_BRANCHES + META_BRANCHES + extra, library="ak")

    # ── optional FPGA trigger cut (event-level, before flattening) ──────────
    if has_fpga:
        fpga = ak.to_numpy(data[FPGA_BRANCH]).astype(np.int64)
        evt_sel = (fpga & 0x1) != 0
        data = data[evt_sel]
        n_trig = int(evt_sel.sum())
        print(f"[INFO] FPGA bit-0 trigger: {n_trig:,} / {len(evt_sel):,} events pass")

    # ── flatten jagged arrays ────────────────────────────────────────────────
    # counts[i] = number of dimuons in event i; used to broadcast metadata
    counts = ak.to_numpy(ak.num(data["rec_dimuon_px_pos_tgt"], axis=1))
    n_events = len(counts)
    n_dimu   = int(counts.sum())
    print(f"[INFO] {root_path.name}: {n_events:,} events → {n_dimu:,} dimuon candidates")

    def flat(branch: str) -> np.ndarray:
        return ak.to_numpy(ak.flatten(data[branch])).astype(np.float64)

    pxp = flat("rec_dimuon_px_pos_tgt")
    pyp = flat("rec_dimuon_py_pos_tgt")
    pzp = flat("rec_dimuon_pz_pos_tgt")
    pxn = flat("rec_dimuon_px_neg_tgt")
    pyn = flat("rec_dimuon_py_neg_tgt")
    pzn = flat("rec_dimuon_pz_neg_tgt")

    # Broadcast scalar metadata to match each dimuon
    eventID = np.repeat(ak.to_numpy(data["eventID"]), counts)
    runID   = np.repeat(ak.to_numpy(data["runID"]),   counts)
    spillID = np.repeat(ak.to_numpy(data["spillID"]), counts)

    # ── compute kinematics ───────────────────────────────────────────────────
    Ep = np.sqrt(pxp**2 + pyp**2 + pzp**2 + MUON_MASS**2)
    En = np.sqrt(pxn**2 + pyn**2 + pzn**2 + MUON_MASS**2)

    Ed  = Ep + En
    pxd = pxp + pxn
    pyd = pyp + pyn
    pzd = pzp + pzn
    M2  = Ed**2 - pxd**2 - pyd**2 - pzd**2
    M   = np.sqrt(np.maximum(M2, 0.0))

    # ── quality cuts ─────────────────────────────────────────────────────────
    z_pos_vtx  = flat("rec_dimuon_z_pos_vtx")
    z_neg_vtx  = flat("rec_dimuon_z_neg_vtx")
    y_st1_pos  = flat("rec_dimuon_y_pos_st1")
    y_st1_neg  = flat("rec_dimuon_y_neg_st1")
    py_st1_pos = flat("rec_dimuon_py_pos_st1")
    py_st1_neg = flat("rec_dimuon_py_neg_st1")
    x_st1_pos  = flat("rec_dimuon_x_pos_st1")
    x_st1_neg  = flat("rec_dimuon_x_neg_st1")
    chi2_tgt_pos = flat("rec_dimuon_chisq_target_pos")
    chi2_tgt_neg = flat("rec_dimuon_chisq_target_neg")
    chi2_dmp_pos = flat("rec_dimuon_chisq_dump_pos")
    chi2_dmp_neg = flat("rec_dimuon_chisq_dump_neg")
    chi2_ups_pos = flat("rec_dimuon_chisq_upstream_pos")
    chi2_ups_neg = flat("rec_dimuon_chisq_upstream_neg")

    sel_quality = (
        (z_pos_vtx > -600) & (z_neg_vtx > -600) &
        (np.abs(y_st1_pos) > 3) & (np.abs(y_st1_neg) > 3) &
        (chi2_tgt_pos > 0) & (chi2_tgt_neg > 0) &
        (chi2_dmp_pos - chi2_tgt_pos > 0) & (chi2_dmp_neg - chi2_tgt_neg > 0) &
        (chi2_ups_pos - chi2_tgt_pos > 0) & (chi2_ups_neg - chi2_tgt_neg > 0) &
        (py_st1_pos * py_st1_neg < 0) &
        (np.abs(x_st1_pos) < 25) & (np.abs(x_st1_neg) < 25)
    )
    n_quality = int(sel_quality.sum())
    print(f"[INFO] quality cuts → {n_quality:,} candidates  (removed {n_dimu - n_quality:,})")

    # ── mass window ──────────────────────────────────────────────────────────
    sel = sel_quality & (M >= mass_min) & (M <= mass_max)
    n_pass = int(sel.sum())
    print(f"[INFO] mass window [{mass_min}, {mass_max}] GeV → {n_pass:,} candidates")

    def s(arr: np.ndarray) -> np.ndarray: return arr[sel]

    pxp, pyp, pzp = s(pxp), s(pyp), s(pzp)
    pxn, pyn, pzn = s(pxn), s(pyn), s(pzn)
    Ep, En = s(Ep), s(En)
    Ed, pxd, pyd, pzd, M = s(Ed), s(pxd), s(pyd), s(pzd), s(M)

    # ── derived features ─────────────────────────────────────────────────────
    pt_d     = np.sqrt(pxd**2 + pyd**2)
    dimu_y   = 0.5 * np.log((Ed + pzd) / np.maximum(Ed - pzd, 1e-12))
    theta_d  = np.arctan2(pt_d, pzd)
    dimu_eta = -np.log(np.tan(theta_d / 2.0 + 1e-12))
    dimu_mT  = np.sqrt(M**2 + pt_d**2)

    theta_pos = np.arctan(pxp / (pzp + 1e-12))
    theta_neg = np.arctan(pxn / (pzn + 1e-12))
    pt_pos    = np.sqrt(pxp**2 + pyp**2)
    pt_neg    = np.sqrt(pxn**2 + pyn**2)
    dpt       = pt_pos - pt_neg

    pp        = np.sqrt(pxp**2 + pyp**2 + pzp**2)
    pn        = np.sqrt(pxn**2 + pyn**2 + pzn**2)
    cos_open  = (pxp*pxn + pyp*pyn + pzp*pzn) / np.maximum(pp * pn, 1e-12)
    open_angle = np.arccos(np.clip(cos_open, -1.0, 1.0))

    eta_pos   = -np.log(np.tan(np.arctan2(pt_pos, pzp) / 2.0 + 1e-12))
    eta_neg   = -np.log(np.tan(np.arctan2(pt_neg, pzn) / 2.0 + 1e-12))
    phi_pos   = np.arctan2(pyp, pxp)
    phi_neg   = np.arctan2(pyn, pxn)
    deltaR    = np.sqrt((eta_pos - eta_neg)**2 + _wrap_phi(phi_pos - phi_neg)**2)

    st1_x_pos  = s(x_st1_pos)
    st1_x_neg  = s(x_st1_neg)
    st1_px_pos = s(flat("rec_dimuon_px_pos_st1"))
    st1_px_neg = s(flat("rec_dimuon_px_neg_st1"))
    dz_vtx     = s(z_pos_vtx) - s(z_neg_vtx)

    X = np.column_stack([
        dimu_y, dimu_eta, Ed, pzd, M,
        theta_pos, theta_neg, open_angle,
        dpt, dimu_mT, Ep, En,
        st1_x_pos, st1_x_neg, st1_px_pos, st1_px_neg,
        dz_vtx, deltaR,
    ])

    meta = {
        "input_file":    str(root_path),
        "tree_name":     tree_name,
        "spin":          spin,
        "mass_min":      mass_min,
        "mass_max":      mass_max,
        "n_events_raw":  n_events,
        "n_dimu_raw":    n_dimu,
        "n_dimu_pass":   n_pass,
        "feature_names": FEATURE_NAMES,
    }

    return {
        "X":       X,
        "M":       M,
        "px_dimu": pxd,           # dimuon px: > 0 = left, < 0 = right
        "pt_dimu": pt_d,          # dimuon pT = sqrt(px²+py²)
        "eventID": s(eventID),
        "runID":   s(runID),
        "spillID": s(spillID),
        "meta":    meta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, type=Path)
    p.add_argument("--output",   required=True, type=Path)
    p.add_argument("--spin",     default="unknown", choices=["up", "down", "unknown"])
    p.add_argument("--tree",     default="tree")
    p.add_argument("--mass-min", type=float, default=2.0)
    p.add_argument("--mass-max", type=float, default=6.0)
    args = p.parse_args()

    result = extract(args.input, args.tree, args.mass_min, args.mass_max, args.spin)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X             = result["X"].astype(np.float32),
        M             = result["M"].astype(np.float32),
        px_dimu       = result["px_dimu"].astype(np.float32),
        pt_dimu       = result["pt_dimu"].astype(np.float32),
        eventID       = result["eventID"],
        runID         = result["runID"],
        spillID       = result["spillID"],
        spin          = np.array(args.spin),
        feature_names = np.array(FEATURE_NAMES, dtype=object),
        meta_json     = np.array(json.dumps(result["meta"])),
    )
    print(f"[INFO] saved → {args.output}  shape={result['X'].shape}")


if __name__ == "__main__":
    main()
