# Data format: converter output branches and DNN input features

This documents the ROOT branches written by the ktracker→flat-tree converter
(top) and the 18 DNN input features derived from them (bottom). The input ROOT
files themselves are gitignored; paths are set in `config/pipeline.sh`.

## Output branches saved (selected dimuon: μ⁺/μ⁻ and associated tracks)

These branches correspond to the **two muon tracks (μ⁺ and μ⁻)** of the **selected reconstructed dimuon**
candidate in the event.

 ### 1) Target-evaluated muon momenta (from the dimuon object)

These quantities are the **μ⁺ and μ⁻ three-momenta evaluated at the target plane**, as stored in the `SRecDimuon` object.

In this analysis, the target-level muon momenta are accessed directly via:
- `dim->p_pos_target`  &nbsp;— μ⁺ momentum at the target  
- `dim->p_neg_target`  &nbsp;— μ⁻ momentum at the target  

The **dimuon momentum at the target** is then constructed by summing the two:
```cpp
dd.mom_target = dim->p_pos_target + dim->p_neg_target;
```

| Branch name | Type | Meaning |
|---|---:|---|
| `rec_dimu_mu_pos_px` | `double` | μ⁺ target momentum \(p_x\) (from `p_pos_target.Px()`) |
| `rec_dimu_mu_pos_py` | `double` | μ⁺ target momentum \(p_y\) (from `p_pos_target.Py()`) |
| `rec_dimu_mu_pos_pz` | `double` | μ⁺ target momentum \(p_z\) (from `p_pos_target.Pz()`) |
| `rec_dimu_mu_neg_px` | `double` | μ⁻ target momentum \(p_x\) (from `p_neg_target.Px()`) |
| `rec_dimu_mu_neg_py` | `double` | μ⁻ target momentum \(p_y\) (from `p_neg_target.Py()`) |
| `rec_dimu_mu_neg_pz` | `double` | μ⁻ target momentum \(p_z\) (from `p_neg_target.Pz()`) |
---

### 2) Track state at Station 1 (from the associated reconstructed tracks)

These quantities come from the reconstructed track objects associated with the dimuon (μ⁺ and μ⁻),
evaluated at **Station 1**.

| Branch name | Type | Meaning |
|---|---:|---|
| `rec_track_pos_x_st1`  | `double` | μ⁺ track x-position at Station 1 |
| `rec_track_neg_x_st1`  | `double` | μ⁻ track x-position at Station 1 |
| `rec_track_pos_px_st1` | `double` | μ⁺ track \(p_x\) at Station 1 |
| `rec_track_neg_px_st1` | `double` | μ⁻ track \(p_x\) at Station 1 |

---

### 3) Track vertex position (from the associated reconstructed tracks)

These are the reconstructed vertex coordinates of the μ⁺ and μ⁻ tracks.

| Branch name | Type | Meaning |
|---|---:|---|
| `rec_track_pos_vx` | `double` | μ⁺ track vertex x |
| `rec_track_pos_vy` | `double` | μ⁺ track vertex y |
| `rec_track_pos_vz` | `double` | μ⁺ track vertex z |
| `rec_track_neg_vx` | `double` | μ⁻ track vertex x |
| `rec_track_neg_vy` | `double` | μ⁻ track vertex y |
| `rec_track_neg_vz` | `double` | μ⁻ track vertex z |
-----

# DNN Input Features (Notation & Definitions)

The **18 physics features** used as inputs to all DNN classifiers in the
pipeline — the first-stage binary classifiers (DNN-1/2/3: J/ψ vs non-J/ψ,
ψ(2S) vs non-ψ(2S), DY vs combinatoric) **and** the final four-class classifier
(DNN-4). The canonical order is defined in `utils/features.py` (`FEATURE_NAMES`);
the features are computed and written to `features_*.npz` by
`post_processing/extract.py` (uproot). The `#` column below is the feature's
position in the network input vector.

| # | Feature name | Definition | Computation |
|---|-------------|------------|-------------|
| 1 | `rec_dimu_y` | Dimuon rapidity | `dimu.Rapidity()` |
| 2 | `rec_dimu_eta` | Dimuon pseudorapidity | `dimu.Eta()` |
| 3 | `rec_dimu_E` | Dimuon energy | `dimu.E()` |
| 4 | `rec_dimu_pz` | Dimuon  Pz | `dimu.Pz()` |
| 5 | `rec_dimu_M` | Dimuon invariant mass | `dimu.M()` |
| 6 | `rec_mu_theta_pos` | μ⁺ x-bending angle (xz plane) | `arctan(px⁺ / pz⁺)` |
| 7 | `rec_mu_theta_neg` | μ⁻ x-bending angle (xz plane) | `arctan(px⁻ / pz⁻)` |
| 8 | `rec_mu_open_angle` | Opening angle between mu+ and mu- | `arccos( dot(p_pos, p_neg) / (norm(p_pos) * norm(p_neg)) )` |
| 9 | `rec_mu_dpt` | pT diff | `pT⁺ − pT⁻` |
|10 | `rec_dimu_mT` | transverse mass | `sqrt(M² + pT²)` |
|11 | `rec_mu_Epos` | μ⁺ energy | `mu_pos.E()` |
|12 | `rec_mu_Eneg` | μ⁻ energy | `mu_neg.E()` |
|13 | `rec_track_pos_x_st1` | μ⁺ x at St1 | ROOT branch |
|14 | `rec_track_neg_x_st1` | μ⁻ x at St1 | ROOT branch |
|15 | `rec_track_pos_px_st1` | μ⁺ Px at St1 | ROOT branch |
|16 | `rec_track_neg_px_st1` | μ⁻ Px at St1 | ROOT branch |
|17 | `rec_dz_vtx` |vertex-z diff | `z⁺ − z⁻` |
|18 | `rec_mu_deltaR` | μ⁺μ⁻ angular separation | `sqrt((Δη)² + (Δφ)²)`, Δφ wrapped to [−π,π) |
---
