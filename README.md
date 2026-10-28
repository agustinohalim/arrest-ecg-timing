# arrest-ecg-timing

Analysis code for the manuscript *Heart Rate Outperforms a CNN: Baselines and Recording-Time Matching in
ECG Prediction of Cardiac Arrest* (Halim, Lukman, Andika, Macnamara, Erwin; submitted). Versions 1.0.0 and
1.1.0 accompanied earlier versions of the manuscript.

On the Nightingale Open Science dataset arrest-ntuh-ecg (emergency department of National Taiwan
University Hospital), the scripts replicate a published case–control design for predicting cardiac
arrest from earlier ECGs, match cases and controls on the recording gap (days from ECG to the index
event), and compare a one-dimensional CNN with simple signal features and with the ECG machine's own
reading (MUSE 12SL statements and measurements).

**No data are included.** The dataset is available from Nightingale Open Science on application
(https://docs.ngsci.org/datasets/arrest-ntuh-ecg/; DOI 10.48815/N5WC7D) and may not be redistributed.
The scripts print aggregate results only; the logs in `logs/` are those outputs.

## Requirements

Python 3.12 with NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, PyTorch 2.14.0 (CPU), and
matplotlib 3.11.1 for the figures. Point the scripts to the data with the environment variable
`NGSCI_DATA` (the folder that contains `arrest-ntuh-ecg/v1`); the default is `C:/Data/Nightingale`.
Some scripts write out-of-fold predictions to `arrest-ntuh-ecg/turunan/` next to the data, never into
this repository.

## Scripts and where their results appear

Comments and log text are in Indonesian; variable names follow the manuscript's terms
(*jarak* = gap, *kasus* = cases, *kontrol* = controls, *pembacaan mesin* = machine reading,
*ciri sinyal sederhana* = simple signal features, *benih* = seed).

| Script | Log(s) | Manuscript |
|---|---|---|
| `01_uji_arrest.py` | `uji_arrest.log` | Data properties that leak the label: MUSE version, gap by group, date shifts (Section 3.1, Supplementary S1, Table S1) |
| `02_cnn_arrest.py` | `cnn_arrest_N.log`, `cnn_arrest_MA.log`, `cnn_arrest_MA2.log` | Design B, gap-band matching, five seeds (Supplementary Table S2); positive control |
| `03_cnn_kunjungan.py` | `cnn_kunjungan.log`, `cnn_kunjungan_bersih.log` | Design C, same-visit ECGs (Table 4); `bersih` is the final case definition |
| `04_replikasi.py` | `replikasi.log`, `replikasi_jarak_sama.log` | Design A replication and the same-window variant of design B (Table 1) |
| `05_metrik_tambahan.py` | `metrik_tambahan.log` | Calibration, precision–recall, operating points for design C (Section 3.5; Supplementary S6) |
| `06_revisi.py` | `revisi.log` | Design C: machine reading, retraining by lead time, export-version check (Table 5; Supplementary S5) |
| `07_sebab.py` | `sebab.log` | Design P with band matching (Supplementary S2) |
| `08_gambar.py` | — | Figures 1 and 2 (numbers copied from the logs; reads no data) |
| `09_uji_sebab.py` | `uji_sebab.log`, `triase_acak.log` | Design P with nearest-gap matching (Table 2) |
| `10_telaah_lanjut.py` | `telaah_lanjut.log` | Matched-set bootstrap, within-set permutation, initial rhythm, location, design C without acute infarct (Table 3, Supplementary S11) |
| `11_uji_silang.py` | `uji_silang.log` | Cause within PEA or asystole, unknown cause, VT/VF permutation, paired CNN vs machine reading, export version for the machine reading (Table 2–3, Supplementary S12) |
| `12_telaah_je.py` | `telaah_je.log` | Version 1.1.0. Recorded cardiac history of cases (`comorbidities.csv`) by cause, cause contrast within history strata, 12SL measurements against statement flags, within-set c-statistic (Sections 3.2–3.3) |
| `13_telaah_jiki.py` | `telaah_jiki.log` | Version 1.2.0. Interaction of cardiac history and recorded cause (difference in differences); CNN per training run, machine reading with trees and LR, CNN + machine reading (LR), paired over all matching seeds (Sections 4.4–4.5, Table 2) |
| `14_mesin_lr.py` | `mesin_lr.log` | Machine reading, measurements, and statements with logistic regression in design P; within-set permutation; cause contrast; export-version check (Sections 4.3, 4.5, 4.7) |
| `15_ukuran_per_ciri.py` | `ukuran_per_ciri.log` | Single-measurement AUROCs and leave-one-out LR in design P; single-measurement export-version check (Sections 4.3, 4.7) |
| `16_rancangan_ab_lr.py` | `rancangan_ab_lr.log` | Designs A and B with heart rate and machine reading (Table 1) |
| `17_ci_pembanding.py` | `ci_pembanding.log` | Confidence intervals and 1,000-permutation null for heart rate and measurements; A − B differences (Sections 4.2–4.3, Table 2) |
| `18_rancangan_c_lr.py` | `rancangan_c_lr.log` | Design C with heart rate and LR comparators, paired differences (Section 4.6, Table 3) |

Section and table numbers for scripts 13–18 refer to the current manuscript (version 1.2.0 of this
repository); those for scripts 01–12 refer to the earlier version (1.1.0), whose Tables 1–2 and design
C table correspond to Tables 1–3 here. `08_gambar.py` also draws Figure 2 of the current manuscript
(`Gambar2_auroc_jiki`).

Run each from `analysis/`, for example:

```bash
python -u 09_uji_sebab.py > ../logs/uji_sebab.log
```

CNN training on CPU is not fully deterministic, so a rerun reproduces CNN values to about the second
decimal; tree models and matching are seeded and reproduce exactly.

## Licence

Code: MIT (see `LICENSE`). Data: not included; governed by the Nightingale Open Science terms of
service.
