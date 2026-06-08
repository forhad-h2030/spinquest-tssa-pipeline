# TSSA fit — RooFit-based J/ψ A_N extraction

Sanity check against the ML-based result in `tssa/`.  Completely independent
of the DNN classifier — uses only dimuon mass and px from the reconstructed
data, with no DNN probability scores.

---

## Method

The mass spectrum in each of the four spin states is fit simultaneously with
a shared J/ψ peak shape:

```
Model (per panel):
  J/ψ  : Gaussian(mean, sigma)          — free, shared across all 4 panels
  ψ(2S): Gaussian(mean × ratio, sigma × ratio)  — constrained via M_ψ'/M_J/ψ
  Bkg  : Exponential(tau)               — free per panel

Fit: simultaneous across UP-Left, UP-Right, DOWN-Left, DOWN-Right
```

Signal yields N_sig ± σ_fit from the fit (not simple counting) are then
used in the geometric-mean TSSA estimator:

```
A     = sqrt(N_uL * N_dR)
B     = sqrt(N_dL * N_uR)
A_raw = (A - B) / (A + B)
A_N   = A_raw / (eta * f * P)
```

Error propagation uses the fit parameter errors (σ_fit) rather than √N.

---

## Prerequisites

ROOT with PyROOT must be available:

```bash
# On local machine
conda activate root_env

# On Rivanna
module load root

# Check
python3 -c "import ROOT; print(ROOT.__version__)"
```

Run `post_processing/run.sh` first to produce `features_{up,down}.npz`.

---

## Usage

```bash
bash tssa_fit/run.sh
```

Or directly:

```bash
python3 tssa_fit/extract_an.py \
    --feat-up   post_processing/output/features_up.npz \
    --feat-down post_processing/output/features_down.npz \
    --out-dir   figures/tssa_fit
```

---

## Output

| File | Description |
|------|-------------|
| `figures/tssa_fit/fit_4panel.png` | 2×2 RooFit mass plots (data + fit components) |
| `figures/tssa_fit/fit_4panel_hist.png` | 2×2 raw mass histograms (ROOT stats boxes) |
| `figures/tssa_fit/an_result.txt` | N_sig per panel, A_raw, A_N ± dA_N |
| `figures/tssa_fit/fit_params.json` | Fit parameters (mean, sigma, yields, τ) consumed by `tssa/overlay.py` |

Figures are automatically copied to `analysis_note_spinquest/figures/` at the end of `run.sh`.

---

## Comparison with ML method

| | `tssa/asymmetry.py` | `tssa_fit/extract_an.py` |
|---|---|---|
| Signal selection | DNN score threshold | None (all reconstructed candidates) |
| Background | Reduced by DNN cut | Modeled by exponential in fit |
| N_sig | Count in mass window | RooFit Gaussian integral |
| Error on N | √N (Poisson) | Fit parameter error (σ_fit) |
| ROOT required | No | Yes |

The two A_N values should agree within uncertainties.  A significant
discrepancy would indicate either a bias in the DNN selection or a
problem with the fit model.
