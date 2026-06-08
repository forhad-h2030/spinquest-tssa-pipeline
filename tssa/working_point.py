#!/usr/bin/env python3
"""Canonical working-point calculator: spin-state counts -> A_N with Poisson
stat error and the 3-seed ensemble model error, at the given threshold(s).
Usage: python3 tssa/working_point.py [t1 t2 ...]   (default 0.635)."""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "post_processing"))
import torch
from classify import _load_model, _infer_one   # noqa: E402

ETA, F, P = 0.6, 0.18, 0.70
PREF      = 1.0 / (ETA * F * P)
LO, HI    = 2.2, 4.2
CKPTS = [REPO / f"classifier/checkpoints/boot_00{i}/ml_input_multiclass_M_26_march_19.best.pth"
         for i in range(3)]
device = torch.device("cpu")


def _load_features():
    fu = np.load(REPO / "post_processing/output/features_up.npz",   allow_pickle=True)
    fd = np.load(REPO / "post_processing/output/features_down.npz", allow_pickle=True)
    return (fu["X"].astype(np.float32), fu["M"].astype(float), fu["px_dimu"].astype(float),
            fd["X"].astype(np.float32), fd["M"].astype(float), fd["px_dimu"].astype(float))


def _scores(Xu, Xd):
    """Return per-seed (3,N) J/psi probabilities for up and down samples."""
    pu, pd = [], []
    for c in CKPTS:
        m, sc, _ = _load_model(c, device)
        pu.append(_infer_one(m, sc, Xu, device)[:, 0])
        pd.append(_infer_one(m, sc, Xd, device)[:, 0])
    return np.array(pu), np.array(pd)


def _counts(p_up, p_dn, Mu, pxu, Md, pxd, t):
    su = (p_up >= t) & (Mu >= LO) & (Mu <= HI)
    sd = (p_dn >= t) & (Md >= LO) & (Md <= HI)
    return (float((su & (pxu > 0)).sum()), float((su & (pxu <= 0)).sum()),
            float((sd & (pxd > 0)).sum()), float((sd & (pxd <= 0)).sum()))


def _an(nuL, nuR, ndL, ndR):
    if min(nuL, nuR, ndL, ndR) <= 0:
        return None, None
    A = math.sqrt(nuL * ndR); B = math.sqrt(ndL * nuR); den = A + B
    AN = PREF * (A - B) / den
    # Poisson propagation: sigma = PREF * A*B/(A+B)^2 * sqrt(sum 1/N_i)
    dAN = PREF * (A * B / den**2) * math.sqrt(1/nuL + 1/nuR + 1/ndL + 1/ndR)
    return AN, dAN


def report(thresholds):
    Xu, Mu, pxu, Xd, Md, pxd = _load_features()
    pj_up, pj_dn = _scores(Xu, Xd)
    mean_up, mean_dn = pj_up.mean(0), pj_dn.mean(0)
    for t in thresholds:
        c = _counts(mean_up, mean_dn, Mu, pxu, Md, pxd, t)
        AN_c, dAN_c = _an(*c)
        seed_AN = []
        for s in range(len(CKPTS)):
            cs = _counts(pj_up[s], pj_dn[s], Mu, pxu, Md, pxd, t)
            a, _ = _an(*cs)
            seed_AN.append(a)
        model = float(np.std(seed_AN, ddof=1))
        print(f"\nt = {t:.3f}")
        print(f"  counts  NuL={c[0]:.0f} NuR={c[1]:.0f} NdL={c[2]:.0f} NdR={c[3]:.0f}  "
              f"(up={c[0]+c[1]:.0f}, dn={c[2]+c[3]:.0f}, tot={sum(c):.0f})")
        print(f"  A_N = {AN_c:+.3f} ± {dAN_c:.3f} (stat) ± {model:.3f} (model)")


if __name__ == "__main__":
    ts = [float(x) for x in sys.argv[1:]] or [0.635]
    report(ts)
