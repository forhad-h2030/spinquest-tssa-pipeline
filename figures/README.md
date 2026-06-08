# figures/ — output layout

All of `figures/` is **gitignored** (regenerable). Outputs are split by purpose
so it is clear which plots belong to the analysis note and which are
exploratory. The clean, note-named set is collected separately into
`../reproduction/` by `reproduce_note.sh`.

| Directory | Config var | Contents |
|-----------|------------|----------|
| `figures/note/` | `FIG_NOTE` | the figures that appear in the analysis note |
| `figures/tssa/` | `FIG_ASYMMETRY` | `asymmetry.py` raw output (4-pad mass+A_N, threshold/pT scans, overlay) |
| `figures/dev/`  | `FIG_DEV` | legacy / exploratory plots **not** in the note |

## `figures/note/` — analysis-note figures and their scripts

| Figure | Script |
|--------|--------|
| `confusion_matrix_dataprior.png` | `classifier/scripts/confusion_matrix_dataprior.py` |
| `jpsi_purity_mass.png` | `classifier/scripts/plot_jpsi_purity_mass.py` |
| `exp_jpsi_classification.png` | `classifier/scripts/plot_exp_jpsi_classification.py` |
| `inclusive_dimuon_mass.png`, `inclusive_jpsi_psip_removed.png` | `classifier/scripts/plot_inclusive_jpsi_removed.py` |
| `an_mass_4spin.png` (from `figures/tssa/an_mass_4pad_p<t>.png`) | `tssa/asymmetry.py` |
| `an_vs_purity.png` | `tssa/plot_an_vs_purity.py` |
| `an_compare_dnn_fit.png` | `tssa/plot_an_compare.py` |
| `false_asymmetry_data.png` | `tssa/plot_false_asymmetry_data.py` |
| `closure_null_asymmetry.png` | `tssa/plot_closure_null_asymmetry.py` |
| `fit_mode_final.png`, `fit_mode_final_hist.png` | `ana-spinquest-fit/fit/fit_mode_final.py --data ml` (separate repo) |

## `figures/dev/` — exploratory (not in the note)

Diagnostic plots produced by `classifier/scripts/confusion_matrix_dataprior.py`
(per-class purity/recall at the data-driven proportions) and
`classifier/scripts/explore_purity_threshold.py` (per-class purity/efficiency
and contamination vs softmax threshold). Kept for reference; the note's
validation figures come from the `figures/note/` scripts above.

## Reproduce everything

`bash reproduce_note.sh` runs the full chain and collects the note figures
(with note filenames) into `../reproduction/` for cross-checking against
`analysis_note_spinquest/figures/`.
