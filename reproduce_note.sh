#!/usr/bin/env bash
# Regenerate the analysis-note figures (DNN + RooFit) into reproduction/,
# named as in the note, for cross-checking. Needs the Rivanna checkpoints.
# See README ("Reproduce the analysis note").
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/config/pipeline.sh"

OUTDIR="$REPO_ROOT/reproduction"
FIT_REPO="/Users/spin/ana-spinquest-fit/fit"
ROOT_PY="${ROOT_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python}"
mkdir -p "$OUTDIR"

log()    { echo "[$(date +%H:%M:%S)] $*"; }
gather() { [ -f "$1" ] && cp "$1" "$OUTDIR/$2" && log "  collected $2" || log "  [skip] $1"; }

# 0. preflight: checkpoints + input data must exist
miss=0
for b in boot_000 boot_001 boot_002; do
  [ -f "$CKPT_DIR/$b/ml_input_multiclass_M_26_march_19.best.pth" ] || { echo "MISSING: $b checkpoint"; miss=1; }
done
[ -f "$MC_BUNDLE" ] && [ -f "$ROOT_UP" ] && [ -f "$ROOT_DOWN" ] || { echo "MISSING: MC bundle or exp ROOT files"; miss=1; }
if [ "$miss" = 1 ]; then
  echo "Copy the trained checkpoints from Rivanna into $CKPT_DIR/ (bash sync_from_rivanna.sh), then re-run."
  exit 1
fi

# 1. processing
log "stage 1: extract + classify"
bash "$REPO_ROOT/post_processing/run.sh"

# 2. classifier validation (simulation)
log "stage 2: classifier validation figures"
RESULT_DIR="$CKPT_DIR" OUT_DIR="$FIG_NOTE" $PYTHON "$REPO_ROOT/classifier/scripts/analyze_final_seeds.py"
$PYTHON "$REPO_ROOT/classifier/scripts/plot_jpsi_purity_mass.py"
$PYTHON "$REPO_ROOT/classifier/scripts/plot_exp_jpsi_classification.py"
$PYTHON "$REPO_ROOT/classifier/scripts/plot_inclusive_jpsi_removed.py"

# 3. TSSA A_N (experimental data)
log "stage 3: TSSA A_N figures"
$PYTHON "$REPO_ROOT/tssa/asymmetry.py" \
    --pred-up "$WORK_DIR/pred_up.npz" --pred-down "$WORK_DIR/pred_down.npz" \
    --out-dir "$FIG_ASYMMETRY" --threshold "$AN_THRESHOLD" --eta "$AN_ETA" --f "$AN_F" --P "$AN_P"
$PYTHON "$REPO_ROOT/tssa/plot_an_vs_purity.py"
$PYTHON "$REPO_ROOT/tssa/plot_an_compare.py"
$PYTHON "$REPO_ROOT/tssa/plot_false_asymmetry_data.py"
$PYTHON "$REPO_ROOT/tssa/plot_closure_null_asymmetry.py"

# 4. RooFit cross-check (needs ROOT)
log "stage 4: RooFit cross-check"
if [ -x "$ROOT_PY" ]; then
  ( cd "$FIT_REPO" && "$ROOT_PY" fit_mode_final.py --data ml )
else
  log "  SKIP fit: no ROOT python at $ROOT_PY (set ROOT_PY=...)"
fi

# 5. collect into reproduction/ with note filenames
log "stage 5: collecting into $OUTDIR"
for fig in confusion_matrix_final_mean jpsi_purity_mass exp_jpsi_classification \
           inclusive_dimuon_mass inclusive_jpsi_psip_removed \
           an_vs_purity an_compare_dnn_fit false_asymmetry_data closure_null_asymmetry; do
  gather "$FIG_NOTE/$fig.png" "$fig.png"
done
gather "$FIG_ASYMMETRY/an_mass_4pad_p${AN_THRESHOLD}.png" "an_mass_4spin.png"
gather "$FIT_REPO/fit_mode_final_ml.png"      "fit_mode_final.png"
gather "$FIT_REPO/fit_mode_final_ml_hist.png" "fit_mode_final_hist.png"

log "done. Cross-check $OUTDIR/ against analysis_note_spinquest/figures/ (working point t=$AN_THRESHOLD)."
