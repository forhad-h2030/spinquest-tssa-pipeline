#!/usr/bin/env bash
# tssa_fit/run.sh — RooFit-based J/ψ TSSA extraction
# Requires ROOT/PyROOT. On Rivanna: module load root
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/config/pipeline.sh"

log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$REPO_ROOT/figures/tssa_fit"

log "RooFit A_N extraction"
$PYTHON "$REPO_ROOT/tssa_fit/extract_an.py" \
    --feat-up   "$WORK_DIR/features_up.npz" \
    --feat-down "$WORK_DIR/features_down.npz" \
    --out-dir   "$REPO_ROOT/figures/tssa_fit" \
    --eta       "$AN_ETA" \
    --f         "$AN_F" \
    --P         "$AN_P"

log "copying figures to analysis note"
cp "$REPO_ROOT/figures/tssa_fit/fit_4panel.png"      "$NOTE_FIGS_DIR/fit_mode_final.png"      2>/dev/null && log "  copied fit_mode_final.png"      || true
cp "$REPO_ROOT/figures/tssa_fit/fit_4panel_hist.png" "$NOTE_FIGS_DIR/fit_mode_final_hist.png" 2>/dev/null && log "  copied fit_mode_final_hist.png" || true

log "done — figures/tssa_fit/"
