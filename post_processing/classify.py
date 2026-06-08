#!/usr/bin/env python3
"""Apply the 3-seed DNN ensemble to a features NPZ (per-seed scaler -> softmax ->
averaged probabilities) and write pred_*.npz with y_proba/y_pred. Classes:
J/psi(0), psi(2S)(1), DY(2), Combinatoric(3). Run from post_processing/run.sh."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── import model class from project utils ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.core_train_multiclass import ParticleClassifierMulticlass, ParticleResNetMulticlass

CKPT_GLOB_DEFAULT = "boot_*/ml_input_multiclass_M_26_march_19.best.pth"


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg         = ckpt["cfg"]
    input_dim   = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    class_names = list(ckpt["class_names"])
    scaler      = ckpt["scaler"]          # {"mean": (1,18), "std": (1,18)}
    model_type  = ckpt.get("model_type", "dnn")

    if model_type == "resnet":
        model = ParticleResNetMulticlass(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg.get("num_layers", 4),
            dropout_rate=cfg.get("dropout_rate", 0.1),
        )
    else:
        model = ParticleClassifierMulticlass(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg.get("num_layers", 4),
            dropout_rate=cfg.get("dropout_rate", 0.1),
            flat=cfg.get("flat", True),
        )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    return model, scaler, class_names


@torch.no_grad()
def _infer_one(model, scaler, X_raw: np.ndarray, device: torch.device,
               batch_size: int = 8192) -> np.ndarray:
    mean = scaler["mean"].astype(np.float32)
    std  = scaler["std"].astype(np.float32)
    X    = (X_raw.astype(np.float32) - mean) / std

    probas = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        logits = model(batch)
        probas.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probas, axis=0)   # (N, K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path, help="NPZ from 01_extract_exp.py")
    p.add_argument("--ckpt-dir", required=True, type=Path, help="Dir containing boot_* subdirs")
    p.add_argument("--output",   required=True, type=Path)
    p.add_argument("--ckpt-glob", default=CKPT_GLOB_DEFAULT)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device",   default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # ── load features ────────────────────────────────────────────────────────
    feat = np.load(args.features, allow_pickle=True)
    X    = feat["X"].astype(np.float32)
    M    = feat["M"]
    spin = str(feat["spin"])
    print(f"[INFO] features : {args.features}  shape={X.shape}  spin={spin}")

    # ── guard: feature order must match the training order exactly ────────────
    # The checkpoints store only a positional scaler (mean/std vectors) and an
    # input_dim — NOT the feature names — so the network and scaler are indexed
    # purely by column position. If extract.py ever emits a different order, the
    # scaling and inference would be silently wrong. Fail loudly instead.
    from utils.features import FEATURE_NAMES
    if "feature_names" in feat:
        got = [str(x) for x in feat["feature_names"]]
        if got != list(FEATURE_NAMES):
            raise ValueError(
                "Feature order in the input NPZ does not match the training "
                "order (utils.features.FEATURE_NAMES).\n"
                f"  got: {got}\n  expected: {list(FEATURE_NAMES)}")
    else:
        print("[WARN] features NPZ has no 'feature_names'; cannot verify column order")
    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature count {X.shape[1]} != expected {len(FEATURE_NAMES)} "
            "(utils.features.FEATURE_NAMES)")

    # ── find checkpoints ─────────────────────────────────────────────────────
    ckpts = sorted(args.ckpt_dir.glob(args.ckpt_glob))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found with pattern '{args.ckpt_glob}' in {args.ckpt_dir}")
    print(f"[INFO] found {len(ckpts)} checkpoint(s)")

    # ── ensemble inference ───────────────────────────────────────────────────
    all_probas: list[np.ndarray] = []
    class_names: list[str] = []

    for ckpt_path in ckpts:
        print(f"  loading {ckpt_path.parent.name}/{ckpt_path.name} ...", flush=True)
        model, scaler, cnames = _load_model(ckpt_path, device)
        if not class_names:
            class_names = cnames
        proba = _infer_one(model, scaler, X, device, args.batch_size)
        all_probas.append(proba)
        print(f"    → proba shape {proba.shape}  mean_max={proba.max(axis=1).mean():.3f}")

    # Average softmax across seeds (deep ensemble)
    y_proba = np.mean(all_probas, axis=0)   # (N, K)
    y_pred  = y_proba.argmax(axis=1)        # (N,)

    print(f"\n[INFO] Ensemble predicted class fractions:")
    for k, name in enumerate(class_names):
        frac = (y_pred == k).mean()
        print(f"  {name:12s}: {frac*100:.1f}%  ({(y_pred==k).sum():,} events)")

    # ── save ─────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "features_path": str(args.features),
        "ckpt_dir":      str(args.ckpt_dir),
        "n_seeds":       len(ckpts),
        "ckpt_files":    [str(c) for c in ckpts],
        "spin":          spin,
        "class_names":   class_names,
        "n_events":      int(len(y_pred)),
    }
    np.savez_compressed(
        args.output,
        y_proba      = y_proba.astype(np.float32),
        y_pred       = y_pred.astype(np.int32),
        M            = M.astype(np.float32),
        px_dimu      = feat["px_dimu"],
        pt_dimu      = feat["pt_dimu"],
        spin         = np.array(spin),
        class_names  = np.array(class_names, dtype=object),
        eventID      = feat["eventID"],
        runID        = feat["runID"],
        spillID      = feat["spillID"],
        meta_json    = np.array(json.dumps(meta)),
    )
    print(f"[INFO] saved → {args.output}")


if __name__ == "__main__":
    main()
