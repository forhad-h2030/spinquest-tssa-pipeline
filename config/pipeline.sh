#!/usr/bin/env bash
# config/pipeline.sh — All tunable parameters for the TSSA extraction pipeline.
# Sourced by pipeline/run.sh.  Edit here; do not hard-code paths in the scripts.

# ── Input data ────────────────────────────────────────────────────────────────
ROOT_UP="/Users/spin/ana-spinquest-fit/data/exp_data_up_may25_2026.root"
ROOT_DOWN="/Users/spin/ana-spinquest-fit/data/exp_data_down_may25_2026.root"

# ── Repo root (derived from this file's location — do not edit) ───────────────
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── DNN checkpoints (boot_*/  subdirectories live here) ───────────────────────
# After training on Rivanna, rsync boot_*/ into classifier/checkpoints/
# See classifier/README.md for sync instructions.
CKPT_DIR="$_REPO_ROOT/classifier/checkpoints"

# One test bundle NPZ for MC validation (any seed — same model)
MC_BUNDLE="$CKPT_DIR/boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz"

# ── Kinematics ────────────────────────────────────────────────────────────────
MASS_MIN=2.0
MASS_MAX=5.9

# ── DNN threshold for A_N extraction (J/ψ softmax score) ─────────────────────
# Nominal working point: t=0.635 (90% J/ψ purity within the +-3σ window centered
# on the simulated J/ψ peak μ≈3.32, σ≈0.20; data-driven proportions).
AN_THRESHOLD=0.635

# ── DY working point (DY softmax score) ──────────────────────────────────────
# t=0.685 gives 95% DY purity under the data-driven prior (751:199:1511:663),
# at ~52% true-DY efficiency. Above this cut the DY↔combinatoric overlap drops
# out of the DY sample (keeps 1177/1511 argmax-DY events on the exp data).
DY_THRESHOLD=0.685

# ── Physics parameters for A_N ────────────────────────────────────────────────
AN_ETA=0.6    # spin-transfer efficiency
AN_F=0.18     # dilution factor
AN_P=0.70     # mean target polarization

# ── Output directories (all absolute) ────────────────────────────────────────
WORK_DIR="$_REPO_ROOT/post_processing/output"
FIG_DIR="$_REPO_ROOT/figures"
FIG_NOTE="$FIG_DIR/note"          # figures that appear in the analysis note
FIG_ASYMMETRY="$FIG_DIR/tssa"     # asymmetry.py raw output (4-pad / scans)
FIG_DEV="$FIG_DIR/dev"            # legacy / exploratory plots (not in the note)

# ── Analysis note figures directory ──────────────────────────────────────────
NOTE_FIGS_DIR="/Users/spin/analysis_note_spinquest/figures"

# ── Python interpreter ────────────────────────────────────────────────────────
PYTHON=python3    # override: PYTHON="conda run -n myenv python3"
