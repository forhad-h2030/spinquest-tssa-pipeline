# SpinQuest TSSA Pipeline

End-to-end pipeline for extracting the J/ψ Transverse Single-Spin Asymmetry
(A_N) from SpinQuest (E1039) experimental data using a 4-class DNN classifier.

---

## Overview

```
ROOT data (exp)
     │
     ▼
post_processing/      extract features + apply DNN → pred_{up,down}.npz
     │
     └──▶ tssa/       compute J/ψ A_N → asymmetry plots
```

The classifier is validated on the held-out MC test bundle separately
(`classifier/scripts/`, collected by `reproduce_note.sh`).

The DNN is trained separately on Rivanna and synced locally via
`sync_from_rivanna.sh` (see `classifier/README.md`).

---

## Repository structure

```
spinquest-tssa-pipeline/
├── classifier/               DNN training (runs on Rivanna)
│   ├── README.md             ← start here for training + sync instructions
│   ├── checkpoints/          synced .pth files land here (gitignored)
│   ├── scripts/              Python training scripts
│   └── multi_class_*.sh      SLURM submit scripts
│
├── post_processing/          apply model to experimental ROOT files
│   ├── README.md
│   ├── extract.py            ROOT → feature NPZ (with quality cuts)
│   ├── classify.py           feature NPZ → probability NPZ (DNN ensemble)
│   └── run.sh                runs both steps for spin up + down
│
├── tssa/                     ML-based A_N extraction
│   ├── README.md
│   ├── asymmetry.py
│   └── run.sh
│
├── tssa_fit/                 RooFit-based A_N extraction (sanity check)
│   ├── README.md
│   ├── fit_common.py         RooFit simultaneous fit engine
│   ├── extract_an.py         reads features NPZ → fit → A_N
│   └── run.sh
│
├── utils/
│   └── core_train_multiclass.py   shared model class (DNN + ResNet)
│
├── config/
│   └── pipeline.sh           all paths and physics parameters
│
└── sync_from_rivanna.sh      pull checkpoints from Rivanna (gitignored)
```

---

## Quick start

### Prerequisites

```bash
pip install torch uproot awkward numpy scipy matplotlib
```

### Step 1 — Get the trained checkpoints

The pipeline needs the 3-seed DNN checkpoints (`classifier/checkpoints/boot_00{0,1,2}/*.best.pth`
and the matching `test_bundle.npz`).

**If the model is already trained, skip the training — just sync the
checkpoints** (or skip Step 1 entirely if they are already present in
`classifier/checkpoints/`):

```bash
# On local machine — pull an existing trained run from Rivanna
bash sync_from_rivanna.sh                 # lists available runs, then exits
bash sync_from_rivanna.sh classifier/outputs_adamw_<jobid>/adamw_onecycle_dnn
```

**Only if you need to (re)train**, submit the job on Rivanna first, then sync
(see **`classifier/README.md`** for full instructions):

```bash
# On Rivanna — submit from repo root
sbatch classifier/multi_class_final_dnn.sh
# then sync as above once it completes
```

### Step 2 — Edit config

Set input ROOT file paths in `config/pipeline.sh`:

```bash
ROOT_UP="/path/to/exp_data_up.root"
ROOT_DOWN="/path/to/exp_data_down.root"
```

These are the experimental data files; the trained model uses them to estimate
the per-event class probabilities. To process different files, edit only these
two lines.

### Step 3 — Extract features and classify

```bash
bash post_processing/run.sh
```

Outputs saved to `post_processing/output/pred_{up,down}.npz`.
Run once — reused by tssa without rerunning the model.

### Step 4 — Validate the classifier on MC

The classifier-validation figures (confusion matrix, J/ψ purity stress test,
null-asymmetry closure) are produced by `reproduce_note.sh` (stage 2), or
individually from `classifier/scripts/` (`analyze_final_seeds.py`,
`plot_jpsi_purity_mass.py`, ...).

### Step 5 — Extract A_N (ML-based)

```bash
bash tssa/run.sh
```

Figures saved to `figures/tssa/`.

### Step 6 — Sanity check with RooFit (requires ROOT)

```bash
bash tssa_fit/run.sh
```

Runs a simultaneous mass fit on the raw mass spectrum (no DNN scores used).
Compare `figures/tssa_fit/an_result.txt` against the ML result.

---

## Reproduce the analysis note

`reproduce_note.sh` regenerates the note's data figures from the trained
checkpoints and writes them — named as in the note — into `reproduction/`
(it does **not** touch the note's own `figures/`; cross-check by hand):

```bash
# 1. copy the trained checkpoints from Rivanna into classifier/checkpoints/
#    (each boot_00{0,1,2}/ with *.best.pth and the test_bundle.npz)
# 2. then:
bash reproduce_note.sh
```

The 12 figures it produces (scripts are under `classifier/scripts/` or `tssa/`;
the two `fit_*` come from the separate repo `ana-spinquest-fit/fit/`):

| Figure | Script | What it shows |
|--------|--------|---------------|
| `confusion_matrix_final_mean.png` | `analyze_final_seeds.py` | per-class confusion matrix (MC, mean over 3 seeds) |
| `jpsi_purity_mass.png` | `plot_jpsi_purity_mass.py` | J/ψ purity stress test, sets the working-point threshold |
| `exp_jpsi_classification.png` | `plot_exp_jpsi_classification.py` | DNN J/ψ classification of the experimental data |
| `inclusive_dimuon_mass.png` | `plot_inclusive_jpsi_removed.py` | inclusive dimuon mass spectrum (exp, 0–10 GeV) |
| `inclusive_jpsi_psip_removed.png` | `plot_inclusive_jpsi_removed.py` | same spectrum after removing DNN J/ψ + ψ(2S) |
| `an_mass_4spin.png` | `asymmetry.py` | four spin-state mass panels and the resulting A_N |
| `an_vs_purity.png` | `plot_an_vs_purity.py` | A_N vs DNN threshold, overlaid with MC purity |
| `an_compare_dnn_fit.png` | `plot_an_compare.py` | DNN A_N vs fit-based A_N, same events |
| `false_asymmetry_data.png` | `plot_false_asymmetry_data.py` | false-asymmetry (null) test on the real data |
| `closure_null_asymmetry.png` | `plot_closure_null_asymmetry.py` | null-asymmetry closure test on the MC |
| `fit_mode_final.png` | `fit_mode_final.py --data ml` | RooFit simultaneous J/ψ mass fit, 4 panels |
| `fit_mode_final_hist.png` | `fit_mode_final.py --data ml` | raw mass histograms feeding the fit |

Notes:
- Working point **t = 0.635** (90% J/ψ purity); run `python3 tssa/working_point.py` for the headline A_N.
- The RooFit stage needs a ROOT-enabled Python; set `ROOT_PY=/path/to/root_env/bin/python` if not at the default.
- The fit uses the Module-2 fit (`ana-spinquest-fit/fit/fit_mode_final.py --data ml`, the one synced to the shared `features_*.npz`) — **not** the in-repo `tssa_fit/` path.
- Static figures (schematic diagrams, hotspot-scan ellipses, tuning overlays) are **not** regenerated here.

---

## Git workflow

Only **code** is committed — no data, weights, or figures.

```
edit code locally  →  git push  →  git pull on Rivanna  →  train
                                                              │
                                            bash sync_from_rivanna.sh
                                                              │
                                                    classifier/checkpoints/
```
