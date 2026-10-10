# arrest-ecg-timing

Analysis code for the manuscript *Timing bias and machine-read findings in ECG prediction of cardiac
arrest: a matched case–control replication* (Halim, Lukman, Andika, Macnamara, Erwin; submitted).

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
| `05_metrik_tambahan.py` | `metrik_tambahan.log` | Calibration, precision–recall, operating points for design C (Supplementary S6) |
| `06_revisi.py` | `revisi.log` | Design C: machine reading, retraining by lead time, export-version check (Supplementary S5, Table S3) |
| `07_sebab.py` | `sebab.log` | Design P with band matching (Supplementary S2) |
| `08_gambar.py` | — | Figures 1 and 2 (numbers copied from the logs; reads no data) |
| `09_uji_sebab.py` | `uji_sebab.log`, `triase_acak.log` | Design P with nearest-gap matching (Table 2) |
| `10_telaah_lanjut.py` | `telaah_lanjut.log` | Matched-set bootstrap, within-set permutation, initial rhythm, location, design C without acute infarct (Table 3, Supplementary S11) |
| `11_uji_silang.py` | `uji_silang.log` | Cause within PEA or asystole, unknown cause, VT/VF permutation, paired CNN vs machine reading, export version for the machine reading (Table 2–3, Supplementary S12) |

Run each from `analysis/`, for example:

```bash
python -u 09_uji_sebab.py > ../logs/uji_sebab.log
```

CNN training on CPU is not fully deterministic, so a rerun reproduces CNN values to about the second
decimal; tree models and matching are seeded and reproduce exactly.

## Licence

Code: MIT (see `LICENSE`). Data: not included; governed by the Nightingale Open Science terms of
service.
