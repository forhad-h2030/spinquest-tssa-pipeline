#!/usr/bin/env bash
# post_processing/run.sh — extract features and classify both spin states
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/config/pipeline.sh"

SCRIPT_DIR="$REPO_ROOT/post_processing"
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$WORK_DIR"

log "extract — spin up"
$PYTHON "$SCRIPT_DIR/extract.py" \
    --input "$ROOT_UP" --output "$WORK_DIR/features_up.npz" \
    --spin up --mass-min "$MASS_MIN" --mass-max "$MASS_MAX"

log "extract — spin down"
$PYTHON "$SCRIPT_DIR/extract.py" \
    --input "$ROOT_DOWN" --output "$WORK_DIR/features_down.npz" \
    --spin down --mass-min "$MASS_MIN" --mass-max "$MASS_MAX"

log "classify — spin up"
$PYTHON "$SCRIPT_DIR/classify.py" \
    --features "$WORK_DIR/features_up.npz" \
    --ckpt-dir "$CKPT_DIR" \
    --output   "$WORK_DIR/pred_up.npz"

log "classify — spin down"
$PYTHON "$SCRIPT_DIR/classify.py" \
    --features "$WORK_DIR/features_down.npz" \
    --ckpt-dir "$CKPT_DIR" \
    --output   "$WORK_DIR/pred_down.npz"

log "done — predictions in $WORK_DIR/"
