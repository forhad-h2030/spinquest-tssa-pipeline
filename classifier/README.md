# Classifier

4-class DNN that separates J/ψ, ψ(2S), Drell-Yan, and combinatoric background
from SpinQuest dimuon data.  Training runs on Rivanna (GPU cluster); the
resulting checkpoint files are synced locally and used by `post_processing/`.

---

## Step 1 — Train on Rivanna

### 1a. Clone the repo on Rivanna

```bash
git clone https://github.com/<your-org>/spinquest-tssa-pipeline.git
cd spinquest-tssa-pipeline
```

### 1b. Submit the training job

Always submit from the **repo root** — SLURM uses that directory as the working
directory and all output paths are relative to it:

```bash
cd spinquest-tssa-pipeline          # must be repo root
sbatch classifier/multi_class_final_dnn.sh
```

Training writes checkpoints to `classifier/outputs_*/adamw_onecycle_dnn/`.
Each seed produces:

```
adamw_onecycle_dnn/
    boot_000/ml_input_multiclass_M_26_march_19.best.pth
    boot_001/ml_input_multiclass_M_26_march_19.best.pth
    boot_002/ml_input_multiclass_M_26_march_19.best.pth
```

Each `.pth` file contains: model weights, architecture config, fitted feature
scaler (mean/std), class names, and validation metrics.

---

## Step 2 — Sync checkpoints locally

Run from your **local machine** (repo root):

```bash
# List available training runs on Rivanna
bash sync_from_rivanna.sh

# Sync the run you want (replace <jobid> with the SLURM job ID)
bash sync_from_rivanna.sh classifier/outputs_final_dnn_<jobid>/plain_dnn
```

The script connects to `dgy5cd@login.hpc.virginia.edu` and pulls only
`best.pth` and `test_bundle.npz` — skipping logs and plots.

After syncing, the local tree should look like:

```
classifier/checkpoints/
    boot_000/ml_input_multiclass_M_26_march_19.best.pth
    boot_000/ml_input_multiclass_M_26_march_19.test_bundle.npz
    boot_001/...
    boot_002/...
```

The `checkpoints/` directory is `.gitignored` — weights are never committed.

---

## Step 3 — Update config

Set `CKPT_DIR` in `config/pipeline.sh` to point to the synced checkpoints:

```bash
CKPT_DIR="classifier/checkpoints"
```

---

## Model architecture

| Parameter      | Value |
|----------------|-------|
| Architecture   | Flat MLP (DNN) |
| Hidden dim     | 512 |
| Layers         | 4 |
| Dropout        | 0.1 |
| Loss           | Cross-entropy + label smoothing (0.05) |
| Optimizer      | AdamW |
| Scheduler      | OneCycleLR |
| Epochs         | 300 |
| Seeds (ensemble) | 3 (boot_000, boot_001, boot_002) |
| Input features | 18 dimuon kinematic variables |
| Output classes | J/ψ (0), ψ(2S) (1), DY (2), Combinatoric (3) |

The model class is defined in `utils/core_train_multiclass.py`
(`ParticleClassifierMulticlass`).
