#!/usr/bin/env bash
# tssa/run.sh — compute J/psi TSSA A_N from classified experimental data
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/config/pipeline.sh"

SCRIPT_DIR="$REPO_ROOT/tssa"
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$FIG_ASYMMETRY"

log "asymmetry A_N"
$PYTHON "$SCRIPT_DIR/asymmetry.py" \
    --pred-up   "$WORK_DIR/pred_up.npz" \
    --pred-down "$WORK_DIR/pred_down.npz" \
    --out-dir   "$FIG_ASYMMETRY" \
    --threshold "$AN_THRESHOLD" \
    --eta       "$AN_ETA" \
    --f         "$AN_F" \
    --P         "$AN_P"

log "A_N vs threshold + MC purity"
$PYTHON "$SCRIPT_DIR/plot_an_vs_purity.py"

log "A_N comparison: DNN vs fit"
$PYTHON "$SCRIPT_DIR/plot_an_compare.py"

log "false-asymmetry null test (real data)"
$PYTHON "$SCRIPT_DIR/plot_false_asymmetry_data.py"

log "closure null test (MC, 3-seed ensemble)"
$PYTHON "$SCRIPT_DIR/plot_closure_null_asymmetry.py"

log "overlay: DNN + RooFit comparison"
FIT_PARAMS="$FIG_DIR/tssa_fit/fit_params.json"
if [ -f "$FIT_PARAMS" ]; then
    $PYTHON "$SCRIPT_DIR/overlay.py" \
        --feat-up    "$WORK_DIR/features_up.npz" \
        --feat-down  "$WORK_DIR/features_down.npz" \
        --pred-up    "$WORK_DIR/pred_up.npz" \
        --pred-down  "$WORK_DIR/pred_down.npz" \
        --fit-params "$FIT_PARAMS" \
        --out-dir    "$FIG_ASYMMETRY" \
        --threshold  "$AN_THRESHOLD"
else
    log "WARNING: $FIT_PARAMS not found — run tssa_fit/run.sh first to generate it"
fi

log "copying figures to analysis note"
# 4-spin mass+A_N panel: copied under the name used in the note
AN_MASS="$FIG_ASYMMETRY/an_mass_4pad_p${AN_THRESHOLD}.png"
[ -f "$AN_MASS" ] && cp "$AN_MASS" "$NOTE_FIGS_DIR/an_mass_4spin.png" && log "  copied an_mass_4spin.png"

for f in \
    "$FIG_ASYMMETRY/an_vs_threshold.png" \
    "$FIG_ASYMMETRY/an_vs_pt.png" \
    "$FIG_ASYMMETRY/overlay_dnn_roofit_t${AN_THRESHOLD}.png" \
    "$FIG_NOTE/an_vs_purity.png" \
    "$FIG_NOTE/an_compare_dnn_fit.png" \
    "$FIG_NOTE/false_asymmetry_data.png" \
    "$FIG_NOTE/closure_null_asymmetry.png"
do
    [ -f "$f" ] && cp "$f" "$NOTE_FIGS_DIR/" && log "  copied $(basename $f)"
done

log "done — figures in $FIG_ASYMMETRY/"
