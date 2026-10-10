"""Nightingale arrest-ntuh-ecg — uji kelayakan: EKG kunjungan IGD memprediksi henti jantung pada kunjungan yang sama?

Soal klinis: pasien datang ke IGD dan direkam EKG 12 sadapan. Dapatkah EKG itu menandai pasien yang
akan henti jantung dalam 24 jam berikutnya, sebelum kejadian?

Definisi kasus (bawaan, 'bersih', 7 Oktober 2026): EKG [0]pre pasien IHCA (henti jantung di IGD atau
ruang tunggu) yang direkam setelah kedatangan di IGD (hospitalization_datetime_offset) dan sebelum
awal CPR (cpr_start_datetime_offset), paling lama 24 jam sebelum CPR. Jam dihitung ke awal CPR.
Definisi pertama ('--lama', 6 Oktober): EKG [0]pre 0-24 jam sebelum ROSC; ternyata 17 dari 220 EKG
itu direkam pada atau sesudah awal CPR (12 di antaranya OHCA), jadi bukan prediksi.

  kontrol  EKG indeks 'control' pasien IGD tanpa henti jantung pada tanggal yang sama (9.976).
           Dokumentasi Nightingale tidak menyebut EKG mana di dalam kunjungan yang dipilih, dan
           kontrol tidak punya waktu kedatangan; waktu EKG kontrol di dalam kunjungan tidak dapat
           disamakan.

Pembanding: usia + jenis kelamin (LR); laju ventrikel saja (HGB); 16 ukuran 12SL MUSE (HGB) dan
+ usia/jk; ciri sinyal sederhana dari 02 (HGB); CNN 12 sadapan dari 02 (3 ulangan); CNN + usia/jk
(LR atas logit CNN out-of-fold, divalidasi silang lagi).

Pemeriksaan pengganggu, tanpa melatih ulang (prediksi out-of-fold yang sama, subset kasus lawan
semua kontrol), CNN dan usia+jk berdampingan:
  jam sebelum CPR    <= 1 jam, 1-4 jam, 4-24 jam, > 1 jam
  mirip triase       EKG <= 1 jam setelah datang dan > 1 jam sebelum CPR
  tahun              kasus 2015-2019 saja (tahun kontrol)
Versi MUSE: semua kasus MUSE 7, kontrol MUSE 9/10. Rancangan M di 02 (EKG pre kasus MUSE 7 lawan
kontrol MUSE 9/10, jarak disamakan) memberi CNN 0,539 = usia+jk, jadi CNN yang sama tidak membaca
versi dari sinyal; itu kontrol negatifnya.

Data tetap di C:/Data/Nightingale (atau NGSCI_DATA); hanya agregat yang dicetak. Prediksi out-of-fold
per EKG disimpan di samping data (arrest-ntuh-ecg/turunan/), bukan di repo.
Pakai: python -u 03_cnn_kunjungan.py > cnn_kunjungan_bersih.log   (--lama: definisi pertama)
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

_spek = importlib.util.spec_from_file_location('c02', os.path.join(os.path.dirname(__file__), '02_cnn_arrest.py'))
c02 = importlib.util.module_from_spec(_spek)
_spek.loader.exec_module(c02)

UKURAN = ['ventricular_rate', 'atrial_rate', 'pr_interval', 'qrs_duration', 'qt_interval',
          'qt_corrected', 'p_axis', 'r_axis', 't_axis', 'qrs_count', 'q_onset', 'q_offset',
          'p_onset', 'p_offset', 't_offset', 'qtc_frederica']
TAHUN_KONTROL = ['2015-16', '2017', '2018', '2019']
LAMA = '--lama' in sys.argv
rng = np.random.default_rng(20261007)


def waktu_bebas(s):
    """rosc-outcomes.csv mencampur '2123/2/7 20:40' dan '2123-02-07 20:40:00'; c02.waktu tidak
    mengurai bentuk pertama. Sel kosong dibaca pandas 3 sebagai NaN (float), bukan 'nan'."""
    out = []
    for v in pd.Series(s):
        if not isinstance(v, str) or not v.strip():
            out.append('NaT')
            continue
        d, _, t = v.strip().replace('/', '-').partition(' ')
        y, m, dd = d.split('-')
        hh, mm = (t or '00:00').split(':')[:2]
        out.append(f'{y}-{int(m):02d}-{int(dd):02d}T{int(hh):02d}:{int(mm):02d}')
    return np.array(out, dtype='datetime64[s]')


def jam(a, b):
    return (waktu_bebas(a) - waktu_bebas(b)).astype('timedelta64[s]').astype(float) / 3600


def ukuran():
    berkas = [os.path.join(c02.R, 'study-group', 'ecg-metadata', 'resting-ecg-measurements.csv')]
    berkas += sorted(glob.glob(os.path.join(c02.R, 'control-group', '*', '*', 'ecg-metadata',
                                            'resting-ecg-measurements.csv')))
    return pd.concat([pd.read_csv(b)[['ecg_id'] + UKURAN] for b in berkas]).drop_duplicates('ecg_id')


def kasus(D):
    k = D[(D.y == 1) & (D.ecg_timetag == '[0]pre')].copy()
    if LAMA:
        k['jam'] = k['jarak'] * 24
        k['datang'] = np.nan
        return k[(k.jam > 0) & (k.jam <= 24)]
    o = pd.read_csv(os.path.join(c02.R, 'study-group', 'rosc-outcomes.csv')).drop_duplicates('patient_ngsci_id')
    k = k.merge(o[['patient_ngsci_id', 'location_of_cardiac_arrest', 'hospitalization_datetime_offset',
                   'cpr_start_datetime_offset']], on='patient_ngsci_id', how='left')
    k['jam'] = jam(k.cpr_start_datetime_offset, k.acquisition_datetime_offset)
    k['datang'] = jam(k.acquisition_datetime_offset, k.hospitalization_datetime_offset)
    print('Kasus [0]pre per lokasi:', k.location_of_cardiac_arrest.value_counts(dropna=False).to_dict())
    return k[(k.location_of_cardiac_arrest == 'IHCA') & (k.datang >= 0) & (k.jam > 0) & (k.jam <= 24)]


def auc_sub(y, p, g, m):
    """AUC dan selang bootstrap pasien untuk subset m (kasus terpilih + semua kontrol)."""
    rng_lama, c02.rng = c02.rng, rng
    try:
        return c02.auc_ci(y[m], p[m], g[m])
    finally:
        c02.rng = rng_lama


def teks(h):
    return f'{h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}]'


def cetak(nama, hasil, y):
    print(f'  {nama:<38s} AUC {teks(hasil)}   n={int(y.sum())}/{int((1 - y).sum())}', flush=True)


def main():
    D = c02.muat()
    kas = kasus(D)
    kon = D[(D.y == 0) & (D.ecg_timetag == 'control')].copy()
    kon['datang'] = np.nan
    kon['jam'] = np.nan
    K = pd.concat([kas, kon], ignore_index=True).merge(ukuran(), on='ecg_id', how='left')
    jenis = 'EKG [0]pre 0-24 jam sebelum ROSC' if LAMA else 'IHCA, EKG setelah datang dan 0-24 jam sebelum CPR'
    print(f'Kasus ({jenis}): {len(kas)} EKG, {kas.patient_ngsci_id.nunique()} pasien; kontrol EKG indeks: {len(kon)}',
          flush=True)
    print('Jam sebelum kejadian (kasus):', pd.cut(kas.jam, [0, 1, 4, 24]).value_counts().sort_index().to_dict())
    if not LAMA:
        print(f'Jam setelah datang (kasus): median {kas.datang.median():.2f}, p75 {kas.datang.quantile(0.75):.2f}')
    print('Tahun kasus:', kas.tahun.value_counts().sort_index().to_dict(), flush=True)
    print(f'Ukuran 12SL hilang: {K[UKURAN].isna().all(axis=1).mean():.3f}', flush=True)

    X = c02.sinyal(K)
    y, g = K['y'].to_numpy().astype(np.float32), K['pid'].to_numpy()
    demo = K[['usia', 'pria']].to_numpy(float)
    ukr = K[UKURAN].to_numpy(float)

    print('\nK EKG kunjungan IGD: kasus lawan kontrol EKG indeks', flush=True)
    for nama, F, cara in [('usia + jenis kelamin (LR)', demo, 'lr'),
                          ('laju ventrikel saja (HGB)', K[['ventricular_rate']].to_numpy(float), 'hgb'),
                          ('ukuran 12SL (HGB)', ukr, 'hgb'),
                          ('ukuran 12SL + usia/jk (HGB)', np.column_stack([ukr, demo]), 'hgb'),
                          ('ciri sinyal sederhana (HGB)', c02.ciri(X), 'hgb')]:
        c02.lapor(nama, F, K, 3, cara)

    p_cnn = c02.oof(X, y, g, 3, 'cnn')
    semua = np.ones(len(y), bool)
    cetak('CNN 12 sadapan (3 ulangan)', auc_sub(y, p_cnn, g, semua), y)
    logit = np.log(np.clip(p_cnn, 1e-6, 1 - 1e-6) / np.clip(1 - p_cnn, 1e-6, 1))
    Z = np.column_stack([logit, np.nan_to_num(demo, nan=np.nanmean(demo[:, 0]))])
    p2 = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(Z, y, g):
        p2[te] = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    cetak('CNN + usia/jk (LR atas logit OOF)', auc_sub(y, p2, g, semua), y)

    turunan = os.path.join(os.path.dirname(c02.R), 'turunan')
    os.makedirs(turunan, exist_ok=True)
    pd.DataFrame({'ecg_id': K.ecg_id, 'y': y, 'jam': K.jam, 'datang': K.datang, 'p_cnn': p_cnn}).to_csv(
        os.path.join(turunan, f'oof_kunjungan_{"lama" if LAMA else "bersih"}.csv'), index=False)

    p_demo = c02.oof(demo, y, g, 1, 'lr')
    print('\nPengganggu: prediksi out-of-fold yang sama, subset kasus lawan semua kontrol', flush=True)
    print(f'  {"subset kasus":<38s} {"CNN":<24s} {"usia + jk":<24s} n', flush=True)
    kontrol = y == 0
    j = K['jam'].to_numpy()
    subset = [('<= 1 jam sebelum kejadian', j <= 1), ('1-4 jam', (j > 1) & (j <= 4)),
              ('4-24 jam', (j > 4) & (j <= 24)), ('> 1 jam', j > 1),
              ('kasus 2015-2019 saja', K['tahun'].isin(TAHUN_KONTROL).to_numpy())]
    if not LAMA:
        subset.append(('mirip triase (<=1 j datang, >1 j CPR)', (K['datang'].to_numpy() <= 1) & (j > 1)))
    for nama, m in subset:
        mm = kontrol | ((y == 1) & m)
        print(f'  {nama:<38s} {teks(auc_sub(y, p_cnn, g, mm)):<24s} {teks(auc_sub(y, p_demo, g, mm)):<24s} '
              f'{int(y[mm].sum())}', flush=True)

    print('\nTitik operasi CNN (semua kasus)', flush=True)
    for spes in (0.90, 0.95):
        ambang = np.quantile(p_cnn[kontrol], spes)
        print(f'  spesifisitas {spes:.2f}: sensitivitas {np.mean(p_cnn[y == 1] > ambang):.3f}', flush=True)


if __name__ == '__main__':
    main()
