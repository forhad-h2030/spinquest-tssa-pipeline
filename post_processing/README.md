# Post-processing — Classification of experimental data

Applies the trained DNN ensemble to reconstructed ROOT files and saves the
per-event class probabilities to NPZ files.  Run this once per data file;
the output NPZs are reused by `tssa/` without re-running the model.

---

## Prerequisites

- Checkpoints synced to `classifier/checkpoints/` (see `classifier/README.md`)
- Python packages: `uproot`, `awkward`, `numpy`, `torch`

---

## Usage

### Run everything (both spin states)

```bash
bash post_processing/run.sh
```

### Run individual steps

**Step 1 — Extract features from ROOT**

```bash
python3 post_processing/extract.py \
    --input  /path/to/exp_data_up.root \
    --output post_processing/output/features_up.npz \
    --spin   up

python3 post_processing/extract.py \
    --input  /path/to/exp_data_down.root \
    --output post_processing/output/features_down.npz \
    --spin   down
```

**Step 2 — Classify (DNN ensemble inference)**

```bash
python3 post_processing/classify.py \
    --features post_processing/output/features_up.npz \
    --ckpt-dir classifier/checkpoints \
    --output   post_processing/output/pred_up.npz

python3 post_processing/classify.py \
    --features post_processing/output/features_down.npz \
    --ckpt-dir classifier/checkpoints \
    --output   post_processing/output/pred_down.npz
```

---

## Quality cuts applied in extract.py

| Cut | Branch |
|-----|--------|
| FPGA bit-0 trigger (if branch present) | `fpga_bits & 0x1 != 0` |
| z_track > −600 cm | `rec_dimuon_z_pos/neg_vtx` |
| \|y_st1\| > 3 cm | `rec_dimuon_y_pos/neg_st1` |
| chi2_tgt > 0 | `rec_dimuon_chisq_target_pos/neg` |
| chi2_dump − chi2_tgt > 0 | `rec_dimuon_chisq_dump_pos/neg` |
| chi2_ups − chi2_tgt > 0 | `rec_dimuon_chisq_upstream_pos/neg` |
| py_st1_pos × py_st1_neg < 0 | `rec_dimuon_py_pos/neg_st1` |
| \|x_st1\| < 25 cm | `rec_dimuon_x_pos/neg_st1` |
| mass window | 2.0 – 5.9 GeV (configurable via `config/pipeline.sh`) |

---

## Output format

Each `pred_{up,down}.npz` contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `y_proba` | (N, 4) | Softmax probabilities per class |
| `y_pred`  | (N,)   | Predicted class index (argmax) |
| `M`       | (N,)   | Dimuon invariant mass [GeV] |
| `px_dimu` | (N,)   | Dimuon px [GeV/c]  (>0 = left, <0 = right) |
| `pt_dimu` | (N,)   | Dimuon pT [GeV/c] |
| `eventID` | (N,)   | Event ID |
| `runID`   | (N,)   | Run ID |
| `spillID` | (N,)   | Spill ID |
| `spin`    | scalar | "up" or "down" |
| `class_names` | (4,) | ["J/psi", "psi(2S)", "DY", "Combinatoric"] |

The `output/` directory is `.gitignored`.
