# TSSA — Transverse Single-Spin Asymmetry extraction

Computes the J/ψ TSSA (A_N) from DNN-classified spin-up and spin-down
experimental data using the geometric-mean estimator.

Requires `post_processing/output/pred_{up,down}.npz` to exist
(run `post_processing/run.sh` first).

---

## Method

Events are split into four spin states by target polarization and dimuon px sign:

| State | Condition |
|-------|-----------|
| UP-Left    | spin=up,   px_dimu > 0 |
| UP-Right   | spin=up,   px_dimu ≤ 0 |
| DOWN-Left  | spin=down, px_dimu > 0 |
| DOWN-Right | spin=down, px_dimu ≤ 0 |

The geometric-mean estimator cancels luminosity asymmetries:

```
A     = sqrt(N_uL * N_dR)
B     = sqrt(N_dL * N_uR)
A_raw = (A - B) / (A + B)
A_N   = A_raw / (eta * f * P)
```

Statistical uncertainty — Poisson sqrt(N) on each count propagated through the
estimator (bootstrap-verified):

```
sigma_AN_stat = (1/(eta*f*P)) * A*B/(A+B)^2 * sqrt(1/N_uL + 1/N_uR + 1/N_dL + 1/N_dR)
```

A separate **model** uncertainty is the spread of A_N over the three ensemble
seeds (ddof=1). Both are reported separately and not combined.

Physics parameters (set in `config/pipeline.sh`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| eta | 0.6 | Spin-transfer efficiency |
| f   | 0.18 | Dilution factor |
| P   | 0.70 | Mean target polarization |

---

## Usage

```bash
bash tssa/run.sh
```

Or directly:

```bash
python3 tssa/asymmetry.py \
    --pred-up   post_processing/output/pred_up.npz \
    --pred-down post_processing/output/pred_down.npz \
    --out-dir   figures/tssa \
    --threshold 0.635
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `asymmetry.py` | DNN-based A_N extraction (4-pad mass + A_N panel, threshold scan, pT dependence) |
| `working_point.py` | Canonical working-point calculator: per-seed counts → central A_N + stat (Poisson) + model (seed spread). Run with no args for t=0.635 (or pass thresholds) |
| `plot_an_vs_purity.py` | A_N vs DNN threshold overlaid with MC J/ψ purity; marks the working point |
| `plot_an_compare.py` | DNN vs fit-based A_N comparison overlay (fit value is threshold-independent) |
| `plot_false_asymmetry_data.py` | False-asymmetry null test on real data (spin-label scrambling) |
| `plot_closure_null_asymmetry.py` | Closure null test on MC (random spin), 3-seed ensemble |
| `overlay.py` | Combined DNN + RooFit overlay figure — requires `fit_params.json` from `tssa_fit/run.sh` |

The nominal working point is **t = 0.635** (90% J/ψ purity within the ±3σ
window [2.71, 3.93] GeV, centered on the reconstructed simulated J/ψ peak
μ≈3.32; data-driven proportions); `working_point.py` reports
`A_N = -0.244 ± 0.537 (stat) ± 0.203 (model)`. The classifier-side purity and
score scripts (`plot_jpsi_purity_mass.py`, `plot_exp_jpsi_classification.py`)
live under `classifier/scripts/`.

---

## Output figures

| File | Description | Script |
|------|-------------|--------|
| `an_mass_4pad_p0.635.png` (→ note `an_mass_4spin.png`) | 2×2 mass panels (4 spin states) + A_N panel at the working point | `asymmetry.py` |
| `an_vs_threshold.png`, `an_vs_pt.png` | A_N vs threshold / vs mean pT | `asymmetry.py` |
| `an_vs_purity.png` | A_N vs threshold with MC purity overlay | `plot_an_vs_purity.py` |
| `an_compare_dnn_fit.png` | DNN vs fit-based A_N comparison | `plot_an_compare.py` |
| `false_asymmetry_data.png` | False-asymmetry null test (real data) | `plot_false_asymmetry_data.py` |
| `closure_null_asymmetry.png` | Closure null test (MC) | `plot_closure_null_asymmetry.py` |

Figures are automatically copied to `analysis_note_spinquest/figures/` at the end of `run.sh`.

---

## Run order

`tssa_fit/run.sh` must run before `tssa/run.sh` so that `fit_params.json` exists for the overlay.

```bash
bash tssa_fit/run.sh          # RooFit fit → fit_params.json
bash tssa/run.sh              # DNN A_N + overlay
```

---

## DNN threshold

The `--threshold` parameter sets the minimum J/ψ softmax score required for
an event to be included in the A_N count.  The threshold scan in
`an_vs_threshold.png` shows how A_N and its uncertainty vary with this cut.
A tighter cut increases purity but reduces statistics. The nominal working
point is **t = 0.635** (90% J/ψ purity), derived from the MC stress test in
`classifier/scripts/plot_jpsi_purity_mass.py` (`TARGET_PURITY`); the same value
is pinned in `plot_an_vs_purity.py`, `plot_closure_null_asymmetry.py`, and
`plot_false_asymmetry_data.py`.
