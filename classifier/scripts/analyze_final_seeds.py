#!/usr/bin/env python3
"""Analyze the 3-seed final DNN: per-class efficiency (averaged, +-1sigma) and
the mean confusion matrix. Env: RESULT_DIR (boot_* parent), OUT_DIR."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

REPO_ROOT   = Path(__file__).resolve().parents[1]
RESULT_DIR  = Path(os.environ.get(
    "RESULT_DIR",
    str(Path.home() / "spinsquest-multiclass" / "spinquest-combinatoric-bkg"
        / "adamw_onecycle_dnn")
))
OUT_DIR     = Path(os.environ.get("OUT_DIR", str(RESULT_DIR)))
BUNDLE_NAME = os.environ.get(
    "BUNDLE_NAME",
    "ml_input_multiclass_M_26_march_19.test_bundle.npz"
)

CLASS_NAMES = ["J/psi", "psi(2S)", "DY", "Comb"]
K = 4

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load each seed independently ─────────────────────────────────────────────
boot_dirs = sorted(RESULT_DIR.glob("boot_*"))
if not boot_dirs:
    print(f"[ERROR] no boot_* dirs found in {RESULT_DIR}")
    sys.exit(1)

seed_results = []
for bd in boot_dirs:
    bpath = bd / BUNDLE_NAME
    if not bpath.exists():
        print(f"[WARN] skipping {bd.name}: {bpath} not found")
        continue
    bundle = np.load(bpath, allow_pickle=True)
    y_test  = bundle["y_test"].astype(np.int64)
    y_pred  = bundle["y_pred"].astype(np.int64)
    y_proba = bundle["y_proba"].astype(np.float64)

    cm      = confusion_matrix(y_test, y_pred, labels=list(range(K)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc     = float((y_pred == y_test).mean())
    diag    = np.array([cm_norm[i, i] for i in range(K)])

    seed_results.append({
        "tag": bd.name,
        "cm": cm, "cm_norm": cm_norm, "acc": acc, "diag": diag,
        "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
    })
    print(f"[INFO] {bd.name}  acc={acc:.4f}  diag={diag}")

n_seeds = len(seed_results)
print(f"\n[INFO] loaded {n_seeds} seeds")

# ── summary statistics ────────────────────────────────────────────────────────
diags = np.stack([r["diag"] for r in seed_results])   # (n_seeds, K)
accs  = np.array([r["acc"]  for r in seed_results])

mean_diag = diags.mean(axis=0)
std_diag  = diags.std(axis=0, ddof=1) if n_seeds > 1 else diags.std(axis=0)
mean_acc  = accs.mean()
std_acc   = accs.std(ddof=1) if n_seeds > 1 else accs.std()

print(f"\n── Per-class efficiency (mean ± std over {n_seeds} seeds) ──")
header = f"{'Seed':10s}" + "".join(f"  {n:>10s}" for n in CLASS_NAMES) + "  {'Overall':>8s}"
print(header); print("─" * len(header))
for r in seed_results:
    row = f"{r['tag']:10s}" + "".join(f"  {v:>10.3f}" for v in r["diag"]) + f"  {r['acc']:>8.4f}"
    print(row)
print("─" * len(header))
mean_row = f"{'Mean':10s}" + "".join(f"  {v:>10.3f}" for v in mean_diag) + f"  {mean_acc:>8.4f}"
std_row  = f"{'Std':10s}"  + "".join(f"  {v:>10.3f}" for v in std_diag)  + f"  {std_acc:>8.4f}"
print(mean_row); print(std_row)

# print classification report for the best seed
best = max(seed_results, key=lambda r: r["acc"])
print(f"\n── Classification report ({best['tag']}, best acc={best['acc']:.4f}) ──")
print(classification_report(best["y_test"], best["y_pred"],
                             target_names=CLASS_NAMES, digits=3))

# ── figure 1: per-seed confusion matrices ─────────────────────────────────────
ncols = n_seeds
fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
if ncols == 1:
    axes = [axes]

for ax, r in zip(axes, seed_results):
    cm_n = r["cm_norm"]
    im = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if cm_n[i,j] > 0.6 else "black")
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(f"{r['tag']}\nacc = {r['acc']:.4f}", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    "AdamW+OneCycle DNN — 3 seeds, final tuned data\n"
    f"Mean acc = {mean_acc:.4f} ± {std_acc:.4f}",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out1 = OUT_DIR / "confusion_matrix_final_3seeds.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[INFO] per-seed CMs → {out1}")

# ── figure 2: mean confusion matrix ───────────────────────────────────────────
mean_cm_norm = np.stack([r["cm_norm"] for r in seed_results]).mean(axis=0)

fig2, ax2 = plt.subplots(figsize=(5.5, 5.0))
im2 = ax2.imshow(mean_cm_norm, vmin=0, vmax=1, cmap="Blues")
for i in range(K):
    for j in range(K):
        val  = mean_cm_norm[i, j]
        sd   = np.stack([r["cm_norm"] for r in seed_results])[:, i, j].std(ddof=1) if n_seeds > 1 else 0.0
        txt  = f"{val:.2f}\n±{sd:.2f}"
        ax2.text(j, i, txt, ha="center", va="center",
                 fontsize=9, color="white" if val > 0.6 else "black")
ax2.set_xticks(range(K)); ax2.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
ax2.set_yticks(range(K)); ax2.set_yticklabels(CLASS_NAMES)
ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
ax2.set_title(
    f"Mean confusion matrix ({n_seeds} seeds)\n"
    f"AdamW+OneCycle DNN, final tuned MC",
    fontsize=11, fontweight="bold"
)
fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
out2 = OUT_DIR / "confusion_matrix_final_mean.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[INFO] mean CM        → {out2}")

# ── figure 3: per-class efficiency bar chart with error bars ──────────────────
fig3, ax3 = plt.subplots(figsize=(7, 4))
x  = np.arange(K)
w  = 0.55
colors = ["#2471A3", "#E74C3C", "#27AE60", "#8E44AD"]

bars = ax3.bar(x, mean_diag, width=w,
               color=colors[:K], alpha=0.85,
               yerr=std_diag, capsize=5, ecolor="black", linewidth=1.2)
for bar, val, sd in zip(bars, mean_diag, std_diag):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + sd + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax3.set_xticks(x); ax3.set_xticklabels(CLASS_NAMES, fontsize=11)
ax3.set_ylabel("Efficiency (recall)", fontsize=11)
ax3.set_ylim(0.50, 1.02)
ax3.set_title(
    f"Per-class efficiency — AdamW+OneCycle DNN\n"
    f"Mean over {n_seeds} seeds (±1σ), final tuned MC\n"
    f"Overall acc = {mean_acc:.4f} ± {std_acc:.4f}",
    fontsize=10, fontweight="bold"
)
ax3.grid(axis="y", ls="--", alpha=0.3)
plt.tight_layout()
out3 = OUT_DIR / "efficiency_final_seeds.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[INFO] efficiency bar → {out3}")

print("\n[DONE] All figures saved.")
