#!/usr/bin/env python3
"""3-seed mean confusion matrix with the MC test sample resampled to the
data-driven class proportions (J/psi and psi(2S) from argmax; DY taken at its
95%-purity working point, the lower-confidence DY counted as combinatoric ->
751:199:1176:998). Plain argmax decision, no per-event threshold. Cells show
counts scaled to the experimental total (3124) and the row-normalized recall;
column purity (precision) is annotated under each predicted class. Output ->
figures/dev/."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO   = Path(__file__).resolve().parents[1]
CKPT   = REPO / "checkpoints"
OUT    = REPO.parent / "figures/dev/confusion_matrix_dataprior.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

PRIOR  = np.array([751, 199, 1176, 998], float)   # data-driven composition (DY @ 95% wp)
names  = ["J/psi", "psi(2S)", "DY", "Comb"]
K      = len(names)

# accumulate a weighted confusion matrix per seed, then average
cms = []
for bd in sorted(CKPT.glob("boot_*")):
    f = bd / "ml_input_multiclass_M_26_march_19.test_bundle.npz"
    if not f.exists():
        continue
    d   = np.load(f, allow_pickle=True)
    yt  = d["y_test"].astype(int)
    pr  = d["y_proba"].astype(np.float64)
    pred = pr.argmax(1)                            # plain argmax, no cut
    nat = np.array([(yt == i).sum() for i in range(K)], float)
    w   = (PRIOR / nat)[yt]                         # reweight test set to data prior
    cm  = np.array([[w[(yt == t) & (pred == p)].sum() for p in range(K)]
                    for t in range(K)])
    cms.append(cm * PRIOR.sum() / cm.sum())         # scale to 3124
n_seeds = len(cms)
cm   = np.mean(cms, 0)
cm_sd = np.std(cms, 0, ddof=1) if n_seeds > 1 else np.zeros_like(cm)

recall = np.diag(cm) / cm.sum(1)
purity = np.diag(cm) / cm.sum(0)
rownorm = cm / cm.sum(1, keepdims=True)

fig, ax = plt.subplots(figsize=(8.2, 7.2))
im = ax.imshow(rownorm, cmap="Blues", vmin=0, vmax=1)
for t in range(K):
    for p in range(K):
        ax.text(p, t, f"{cm[t, p]:.0f}\n{rownorm[t, p]:.2f}", ha="center", va="center",
                fontsize=10, color="white" if rownorm[t, p] > 0.55 else "black")
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels([f"pred {n}\npurity {purity[p]:.3f}" for p, n in enumerate(names)],
                   fontsize=9.5)
ax.set_yticklabels([f"true {n}\nrecall {recall[t]:.3f}" for t, n in enumerate(names)],
                   fontsize=9.5)
ax.set_xlabel("predicted class  (column = purity / precision)", fontsize=11)
ax.set_ylabel("true class  (row = recall / efficiency)", fontsize=11)
ax.tick_params(axis="x", pad=6)
ax.set_title("Confusion matrix evaluated on the MC test sample "
             "(data-driven class proportions)\n"
             f"mean of {n_seeds} seeds, counts scaled to 3124",
             fontsize=10.5, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12, label="row-normalized (recall)")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {OUT}")
print("purity:", {names[p]: round(purity[p], 3) for p in range(K)})
print("recall:", {names[t]: round(recall[t], 3) for t in range(K)})
