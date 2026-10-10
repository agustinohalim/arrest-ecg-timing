"""Nightingale arrest-ntuh-ecg — analisis tambahan untuk revisi Naskah_Arrest.md (telaah panel 7 Oktober 2026,
Riset/EKG/Telaah/2026-10-07/report.md).

Bagian, dipilih lewat argumen (tanpa argumen: semua, urut):
  --tanggal     konsistensi pergeseran tanggal kasus: EKG [1]ROSC dan [2]24hr terhadap ROSC; urutan
                datang <= CPR <= ROSC (temuan methods M7)
  --pernyataan  pembanding klinis: bendera pernyataan diagnosis MUSE (kata kunci) + usia/jk, dan
                + 16 ukuran 12SL; CNN + semuanya (LR atas logit OOF). Rancangan C bersih
                (temuan D8, X8 devil's advocate)
  --muse        kontrol negatif langsung: CNN dan ciri sederhana membedakan EKG indeks kontrol
                MUSE 10 (2015-2016) dari MUSE 9 (2017-2019). Versi dan tahun berimpit penuh,
                jadi ini menguji jejak versi ekspor dan tahun sekaligus (temuan X3 / M6)
  --latih-ulang rancangan C dilatih ulang dengan kasus terbatas: > 15 menit, > 1 jam, mirip triase
                (temuan M4 / X4)
  --beda        selisih AUROC berpasangan (bootstrap pasien) CNN lawan usia+jk dan lawan
                pernyataan+12SL+usia/jk, rancangan C penuh dan subset dilatih ulang (temuan M3)

Kasus dan kontrol rancangan C diambil dari 03_cnn_kunjungan.py (definisi bersih). Data tetap di
C:/Data/Nightingale; hanya agregat yang dicetak. OOF tambahan ditulis ke arrest-ntuh-ecg/turunan/.
Pakai: python -u 06_revisi.py > revisi.log
"""
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c03 = _muat_modul('c03', '03_cnn_kunjungan.py')
TURUNAN = os.path.join(os.path.dirname(c02.R), 'turunan')
rng = np.random.default_rng(20261009)

# kata kunci dicocokkan pada teks pernyataan satu EKG yang digabung, huruf kecil, tanpa spasi:
# MUSE 7 menggabung potongan pernyataan ('fibrillationwith'), MUSE 9/10 memecahnya per baris
BENDERA = {
    'normal_ecg': r'(?<!ab)normalecg', 'abnormal_ecg': r'abnormalecg', 'borderline': r'borderline',
    'sinus_takikardia': r'sinustachy', 'sinus_bradikardia': r'sinusbrady',
    'fibrilasi_atrium': r'atrialfib', 'flutter_atrium': r'atrialflutter',
    'irama_tak_tentu': r'undeterminedrhythm|junctional|ventriculartachy|idioventricular',
    'pvc': r'prematureventricular', 'blok_av': r'avblock|avdissociation',
    'rbbb': r'rightbundle', 'lbbb': r'leftbundle', 'blok_lain': r'fascicular|intraventricular|bifascicular',
    'qt_panjang': r'prolongedqt', 'voltase_rendah': r'lowvoltage', 'lvh': r'ventricularhypertrophy',
    'infark': r'infarct', 'mi_akut': r'acutemi|injury|stelevation',
    'iskemia': r'ischemia', 'st_t': r'stabnormality|twaveabnormality|st&t|repolarization',
    'deviasi_aksis': r'axisdeviation|rightwardaxis', 'pacu': r'pacemaker|paced|pacing',
}


def jam(a, b):
    """c03.jam tanpa NaT: selisih dengan waktu kosong menjadi NaN, bukan bilangan negatif raksasa."""
    d = c03.waktu_bebas(a) - c03.waktu_bebas(b)
    return np.where(np.isnat(d), np.nan, d.astype('timedelta64[s]').astype(float) / 3600)


def teks(h):
    return f'{h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}]'


def auc_ci(y, p, g):
    rng_lama, c02.rng = c02.rng, rng
    try:
        return c02.auc_ci(y, p, g)
    finally:
        c02.rng = rng_lama


def beda_ci(y, pa, pb, g, n=1000):
    """AUROC(pa) - AUROC(pb) dengan bootstrap pasien berpasangan."""
    ug = np.unique(g)
    pos = {k: np.where(g == k)[0] for k in ug}
    bs = []
    for _ in range(n):
        ix = np.concatenate([pos[k] for k in rng.choice(ug, len(ug))])
        if 0 < y[ix].sum() < len(ix):
            bs.append(roc_auc_score(y[ix], pa[ix]) - roc_auc_score(y[ix], pb[ix]))
    bs = np.array(bs)
    return (roc_auc_score(y, pa) - roc_auc_score(y, pb), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
            np.mean(bs <= 0))


def rancangan_c():
    D = c02.muat()
    kas = c03.kasus(D)
    kon = D[(D.y == 0) & (D.ecg_timetag == 'control')].copy()
    kon['datang'] = np.nan
    kon['jam'] = np.nan
    K = pd.concat([kas, kon], ignore_index=True).merge(c03.ukuran(), on='ecg_id', how='left')
    return K.merge(pernyataan(), on='ecg_id', how='left').fillna({b: 0 for b in BENDERA})


def pernyataan():
    berkas = [os.path.join(c02.R, 'study-group', 'ecg-metadata', 'diagnosis.csv')]
    berkas += sorted(glob.glob(os.path.join(c02.R, 'control-group', '*', '*', 'ecg-metadata', 'diagnosis.csv')))
    d = pd.concat([pd.read_csv(b)[['ecg_id', 'stmt_text']] for b in berkas], ignore_index=True)
    d['stmt_text'] = d.stmt_text.fillna('').astype(str)
    t = d.groupby('ecg_id', sort=False)['stmt_text'].agg(''.join).str.lower().str.replace(r'\s+', '', regex=True)
    out = pd.DataFrame({b: t.str.contains(pola, regex=True).astype(float) for b, pola in BENDERA.items()})
    return out.reset_index()


def oof_lr_logit(Z, y, g):
    """LR atas [logit CNN, ciri lain], divalidasi silang lagi seperti 03."""
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(Z, y, g):
        p[te] = c02.make_pipeline(c02.SimpleImputer(), c02.StandardScaler(),
                                  LogisticRegression(max_iter=4000)).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return p


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def bagian_tanggal():
    print('\n== Konsistensi pergeseran tanggal kasus', flush=True)
    s = pd.read_csv(os.path.join(c02.R, 'study-group', 'ecg.csv'))
    k = pd.read_csv(os.path.join(c02.R, 'study-group', 'cohort.csv')).drop_duplicates('patient_ngsci_id')
    o = pd.read_csv(os.path.join(c02.R, 'study-group', 'rosc-outcomes.csv')).drop_duplicates('patient_ngsci_id')
    s = s.merge(k[['patient_ngsci_id', 'rosc_datetime_offset']], on='patient_ngsci_id')
    s['jam_rosc'] = jam(s.acquisition_datetime_offset, s.rosc_datetime_offset)
    for tag in ['[0]pre', '[1]ROSC', '[2]24hr']:
        j = s.loc[s.ecg_timetag == tag, 'jam_rosc'].dropna()
        q = j.quantile([0.05, 0.5, 0.95]).round(2).tolist()
        print(f'  {tag:<8s} n={len(j):<5d} jam EKG - ROSC: p5/median/p95 {q}; '
              f'negatif {np.mean(j < 0):.3f}; > 48 jam {np.mean(j > 48):.3f}', flush=True)
    o = o.merge(k[['patient_ngsci_id', 'rosc_datetime_offset']], on='patient_ngsci_id', how='left')
    cpr_rosc = jam(o.rosc_datetime_offset, o.cpr_start_datetime_offset)
    datang_cpr = jam(o.cpr_start_datetime_offset, o.hospitalization_datetime_offset)
    for nama, v, m in [('ROSC - awal CPR (jam)', cpr_rosc, np.ones(len(o), bool)),
                       ('awal CPR - datang (jam), IHCA', datang_cpr, (o.location_of_cardiac_arrest == 'IHCA').to_numpy())]:
        v = pd.Series(v[m]).dropna()
        print(f'  {nama:<32s} n={len(v):<4d} p5/median/p95 {v.quantile([0.05, 0.5, 0.95]).round(2).tolist()}; '
              f'negatif {np.mean(v < 0):.3f}', flush=True)


def bagian_pernyataan(K, simpan):
    print('\n== Pembanding klinis, rancangan C bersih', flush=True)
    y, g = K.y.to_numpy().astype(np.float32), K.pid.to_numpy()
    B = K[list(BENDERA)].to_numpy(float)
    demo = K[['usia', 'pria']].to_numpy(float)
    ukr = K[c03.UKURAN].to_numpy(float)
    kas, kon = K.y == 1, K.y == 0
    print('  prevalensi bendera (kasus / kontrol):', flush=True)
    for b in BENDERA:
        print(f'    {b:<18s} {K.loc[kas, b].mean():.3f} / {K.loc[kon, b].mean():.3f}', flush=True)
    hasil = {}
    for nama, F, cara in [('pernyataan MUSE (HGB)', B, 'hgb'),
                          ('pernyataan + usia/jk (HGB)', np.column_stack([B, demo]), 'hgb'),
                          ('pernyataan + 12SL + usia/jk (HGB)', np.column_stack([B, ukr, demo]), 'hgb')]:
        p = c02.oof(F, y, g, 3, cara)
        hasil[nama] = p
        print(f'  {nama:<40s} AUC {teks(auc_ci(y, p, g))}', flush=True)
    o = pd.read_csv(os.path.join(TURUNAN, 'oof_kunjungan_bersih.csv'))[['ecg_id', 'p_cnn']]
    p_cnn = K[['ecg_id']].merge(o, on='ecg_id', how='left').p_cnn.to_numpy()
    assert not np.isnan(p_cnn).any()
    Z = np.column_stack([logit(p_cnn), B, ukr, demo])
    p_semua = oof_lr_logit(Z, y, g)
    print(f'  {"CNN + pernyataan + 12SL + usia/jk (LR)":<40s} AUC {teks(auc_ci(y, p_semua, g))}', flush=True)
    simpan.update({'p_cnn': p_cnn, 'p_klinis': hasil['pernyataan + 12SL + usia/jk (HGB)'], 'p_cnn_klinis': p_semua})


def bagian_muse(D):
    print('\n== Kontrol negatif langsung: EKG indeks kontrol MUSE 10 (2015-16) lawan MUSE 9 (2017-19)', flush=True)
    K = D[(D.y == 0) & (D.ecg_timetag == 'control')].reset_index(drop=True).copy()
    K['y'] = K.tahun.isin(['2015', '2016']).astype(float)
    K['pid'] = K.patient_ngsci_id.astype(str)
    X = c02.sinyal(K)
    print(f'  MUSE 10: {int(K.y.sum())}, MUSE 9: {int((1 - K.y).sum())}', flush=True)
    for nama, F, cara, ulang in [('usia + jenis kelamin (LR)', K[['usia', 'pria']].to_numpy(float), 'lr', 1),
                                 ('ciri sinyal sederhana (HGB)', c02.ciri(X), 'hgb', 1),
                                 ('CNN 12 sadapan', X, 'cnn', 3)]:
        c02.lapor(nama, F, K, ulang, cara)
    del X


def bagian_latih_ulang(K, simpan):
    print('\n== Rancangan C dilatih ulang dengan kasus terbatas (kontrol semua)', flush=True)
    X_all = c02.sinyal(K)
    j, dt = K.jam.to_numpy(), K.datang.to_numpy()
    kon = (K.y == 0).to_numpy()
    for nama, m in [('> 15 menit sebelum CPR', j > 0.25), ('> 1 jam sebelum CPR', j > 1),
                    ('mirip triase (<=1 j datang, >1 j CPR)', (dt <= 1) & (j > 1))]:
        mm = kon | ((K.y == 1).to_numpy() & m)
        S = K[mm].reset_index(drop=True)
        X = X_all[mm]
        y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
        p_cnn = c02.oof(X, y, g, 3, 'cnn')
        p_demo = c02.oof(S[['usia', 'pria']].to_numpy(float), y, g, 1, 'lr')
        print(f'  {nama:<40s} CNN {teks(auc_ci(y, p_cnn, g))}  usia+jk {teks(auc_ci(y, p_demo, g))}  '
              f'n={int(y.sum())}/{int((1 - y).sum())}', flush=True)
        simpan[nama] = (y, g, p_cnn, p_demo)
    del X_all


def bagian_beda(K, simpan):
    print('\n== Selisih AUROC berpasangan (bootstrap pasien, 1.000): selisih [IK 95%], P(selisih <= 0)', flush=True)
    y, g = K.y.to_numpy().astype(np.float32), K.pid.to_numpy()
    p_demo = c02.oof(K[['usia', 'pria']].to_numpy(float), y, g, 1, 'lr')
    baris = [('rancangan C: CNN - usia+jk', y, g, simpan['p_cnn'], p_demo),
             ('rancangan C: CNN - pernyataan+12SL+usia/jk', y, g, simpan['p_cnn'], simpan['p_klinis']),
             ('rancangan C: CNN+klinis - pernyataan+12SL+usia/jk', y, g, simpan['p_cnn_klinis'], simpan['p_klinis'])]
    for nama in ['> 15 menit sebelum CPR', '> 1 jam sebelum CPR', 'mirip triase (<=1 j datang, >1 j CPR)']:
        if nama in simpan:
            yy, gg, pc, pd_ = simpan[nama]
            baris.append((f'dilatih ulang {nama}: CNN - usia+jk', yy, gg, pc, pd_))
    for nama, yy, gg, pa, pb in baris:
        d, lo, hi, p0 = beda_ci(yy, pa, pb, gg)
        print(f'  {nama:<62s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  P<=0 {p0:.3f}', flush=True)


def main():
    pilih = [a.lstrip('-') for a in sys.argv[1:]] or ['tanggal', 'pernyataan', 'muse', 'latih-ulang', 'beda']
    if 'tanggal' in pilih:
        bagian_tanggal()
    simpan = {}
    if set(pilih) & {'pernyataan', 'latih-ulang', 'beda'}:
        K = rancangan_c()
        print(f'\nRancangan C: {int(K.y.sum())} kasus, {int((1 - K.y).sum())} kontrol', flush=True)
        if 'pernyataan' in pilih or 'beda' in pilih:
            bagian_pernyataan(K, simpan)
    if 'muse' in pilih:
        bagian_muse(c02.muat())
    if 'latih-ulang' in pilih or 'beda' in pilih:
        bagian_latih_ulang(K, simpan)
    if 'beda' in pilih:
        bagian_beda(K, simpan)


if __name__ == '__main__':
    main()
