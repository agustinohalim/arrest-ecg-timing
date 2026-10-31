"""Nightingale arrest-ntuh-ecg — jejak versi MUSE pada ukuran laju, diukur langsung (telaah panel JIKI 30 Oktober 2026
(b), `../Telaah/2026-10-30b-Arrest-JIKI/report.md`, kritis "export-software confound is sized on the wrong contrast";
mayor "the measurement advantage is essentially heart rate"; mayor "comparator ... not run in design A").

Versi berimpit dengan label (semua kasus MUSE 7, kontrol 9/10), jadi jejak 7 lawan 9/10 pada ukuran 12SL tidak bisa
diukur dengan kontras di dalam satu kelompok. Untuk laju bisa: laju dihitung ulang dari gelombang dengan satu detektor
tetap yang sama untuk semua EKG, lalu d = ventricular_rate 12SL - laju sinyal. Bila 12SL menghitung laju berbeda antarversi,
d berbeda antara kasus dan kontrol; kalau tidak, keunggulan laju tidak berasal dari perangkat lunak yang menghitungnya.
Ini tidak menjawab QRS/QTc dan tidak menyingkirkan perubahan pada gelombang ekspor itu sendiri.

  lengan laju (Python sistem):
    1. kolom muse-info dan resting-ecg-measurements (nama saja), hitungan versi per kelompok
    2. semua EKG pre (kasus; kontrol 2015-2017): sepakat 12SL lawan sinyal, d per versi, AUROC d kasus lawan kontrol
    3. per sampel A 1-30, A 30-365, P, 3 benih, CNN 2 ulangan: AUROC laju 12SL dan laju sinyal, selisihnya, AUROC d;
       tambahan di atas CNN dari laju 12SL, laju sinyal, dan 13 ukuran tanpa keluarga laju; Spearman logit CNN-laju
    4. jejak MUSE 10 lawan 9 di dalam kontrol, per ukuran
  lengan rocket (venv-aeon): MiniRocket di rancangan A, ukuran LR - MiniRocket berpasangan.

CPU. Hanya agregat. Pakai:
  python -u 26_laju_sinyal.py --lengan laju > laju_sinyal.log
  ~/venv-aeon/Scripts/python.exe -u 26_laju_sinyal.py --lengan rocket > rocket_a.log
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c19 = _muat_modul('c19', '19_berpasangan.py')
c02, c03, c06, c09, c10 = c19.c02, c19.c03, c19.c06, c19.c09, c19.c10
rng = c19.rng
FS = c02.FS // 2
KELUARGA_LAJU = ['ventricular_rate', 'atrial_rate', 'qrs_count']


def laju_sinyal(X):
    """Puncak energi turunan 12 sadapan (halus 80 ms), jarak minimum 200 ms; 60 / median RR."""
    n = int(0.08 * FS)
    e = np.abs(np.diff(X, axis=2)).sum(axis=1)
    hasil = np.full(len(X), np.nan)
    for i, s in enumerate(e):
        s = np.convolve(s, np.ones(n) / n, mode='same')
        p, _ = find_peaks(s, height=0.4 * np.percentile(s, 99), distance=int(0.2 * FS))
        if len(p) >= 3:
            hasil[i] = 60 * FS / np.median(np.diff(p))
    return hasil


def isi(v):
    return np.where(np.isnan(v), np.nanmedian(v), v)


def versi_muse():
    v = pd.concat([pd.read_csv(f, usecols=['ecg_id', 'muse_version']) for f in
                   glob.glob(os.path.join(c02.R, '**', 'muse-info.csv'), recursive=True)], ignore_index=True)
    return v.drop_duplicates('ecg_id').set_index('ecg_id').muse_version.astype(str).str.split('.').str[0]


def bagian_kolom():
    print('== 1. Kolom metadata (nama saja)', flush=True)
    for pola in ('muse-info.csv', 'resting-ecg-measurements.csv'):
        f = sorted(glob.glob(os.path.join(c02.R, '**', pola), recursive=True))
        print(f'  {pola}: {len(f)} berkas; kolom {list(pd.read_csv(f[0], nrows=0).columns)}', flush=True)


def bagian_sepakat(pre, muse):
    print('\n== 2. Semua EKG pre: 12SL lawan sinyal', flush=True)
    T = pre[(pre.y == 1) | pre.tahun.isin(c02.SAH)].reset_index(drop=True).copy()
    T['versi'] = T.ecg_id.map(muse)
    T['sendiri'] = laju_sinyal(c02.sinyal(T))
    T['d'] = T.ventricular_rate - T.sendiri
    print(f'  EKG {len(T)}, laju sinyal hilang {int(T.sendiri.isna().sum())}, 12SL hilang '
          f'{int(T.ventricular_rate.isna().sum())}', flush=True)
    A = T.dropna(subset=['d'])
    print(f'  Pearson 12SL-sinyal {np.corrcoef(A.ventricular_rate, A.sendiri)[0, 1]:.3f}; |d|<=5: '
          f'{np.mean(A.d.abs() <= 5):.3f}', flush=True)
    for v, B in A.groupby('versi'):
        print(f'  MUSE {v:<3s} n {len(B):>5d}  y rerata {B.y.mean():.3f}  laju 12SL median {B.ventricular_rate.median():.1f}'
              f'  sinyal {B.sendiri.median():.1f}  d median {B.d.median():+.2f} rerata {B.d.mean():+.2f} sd {B.d.std():.2f}'
              f'  |d|<=5 {np.mean(B.d.abs() <= 5):.3f}', flush=True)
    for nama, S in (('semua', A), ('|d|<=10', A[A.d.abs() <= 10])):
        y, g, d = S.y.to_numpy(), S.pid.to_numpy(), S.d.to_numpy()
        b = c19.boot(y, g, lambda ix: roc_auc_score(y[ix], d[ix]))
        print(f'  AUROC d kasus lawan kontrol, {nama:<8s} {roc_auc_score(y, d):.3f} {c10.ci(b)} (n {len(S)})', flush=True)


def satu_sampel(S, y, g, u, h):
    X = c02.sinyal(S)
    ps = c19.cnn_ulang(X, y, g)
    v, s = isi(S.ventricular_rate.to_numpy(float)), isi(laju_sinyal(X))
    del X
    sisa = S[[k for k in c03.UKURAN if k not in KELUARGA_LAJU]].to_numpy(float)
    d = v - s
    pv = [c06.oof_lr_logit(np.column_stack([c06.logit(p), v]), y, g) for p in ps]
    ps2 = [c06.oof_lr_logit(np.column_stack([c06.logit(p), s]), y, g) for p in ps]
    pu = [c06.oof_lr_logit(np.column_stack([c06.logit(p), sisa]), y, g) for p in ps]

    def beda(ix):
        k = rng.integers(len(ps))
        a = roc_auc_score(y[ix], ps[k][ix])
        lv, ls = roc_auc_score(y[ix], v[ix]), roc_auc_score(y[ix], s[ix])
        return (lv - ls, roc_auc_score(y[ix], d[ix]), roc_auc_score(y[ix], pv[k][ix]) - a,
                roc_auc_score(y[ix], ps2[k][ix]) - a, roc_auc_score(y[ix], pu[k][ix]) - a)

    us, anggota = c19.unit(u)
    h['boot'].append(np.array([beda(np.concatenate([anggota[j] for j in rng.choice(us, len(us))]))
                               for _ in range(c19.N_BOOT)]))
    h['v'].append(roc_auc_score(y, v))
    h['s'].append(roc_auc_score(y, s))
    h['d'].append(roc_auc_score(y, d))
    h['rho'].append(np.mean([spearmanr(c06.logit(p), v).statistic for p in ps]))
    h['rho_y0'].append(np.mean([spearmanr(c06.logit(p)[y == 0], v[y == 0]).statistic for p in ps]))


def lapor(nama, h):
    b = np.concatenate(h['boot'])
    print(f'  {nama:<10s} AUROC laju 12SL {np.mean(h["v"]):.3f}, laju sinyal {np.mean(h["s"]):.3f}, d {np.mean(h["d"]):.3f}; '
          f'Spearman logit CNN-laju {np.mean(h["rho"]):+.3f} (kontrol {np.mean(h["rho_y0"]):+.3f})', flush=True)
    for k, lab in enumerate(('laju 12SL - laju sinyal', 'AUROC d', '(CNN + laju 12SL) - CNN', '(CNN + laju sinyal) - CNN',
                             '(CNN + 13 ukuran tanpa laju) - CNN')):
        print(f'    {lab:<36s} {b[:, k].mean():+.3f} {c10.ci(b[:, k])}', flush=True)


def baru():
    return {'boot': [], 'v': [], 's': [], 'd': [], 'rho': [], 'rho_y0': []}


def bagian_sampel(pre, kas, lama, jendela):
    print('\n== 3. Per sampel, 3 benih, CNN 2 ulangan', flush=True)
    for w, (m, _) in jendela.items():
        h = baru()
        for b in c19.BENIH:
            S = c19.susun_ab(pre, kas[m], lama, b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            satu_sampel(S, y, g, g, h)
            print(f'  A {w} benih {b} selesai', flush=True)
        lapor(f'A {w}', h)
    pre_p, calon_p, kp, _ = c09.data_p()
    h = baru()
    for b in c19.BENIH:
        S = c10.susun(pre_p, kp, calon_p, b)
        y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
        satu_sampel(S, y, g, s, h)
        print(f'  P benih {b} selesai', flush=True)
    lapor('P', h)


def bagian_jejak(pre, muse):
    print('\n== 4. Jejak MUSE 10 lawan 9, EKG pre kontrol 2015-2017, per ukuran (AUROC terarah)', flush=True)
    K = pre[(pre.y == 0) & pre.tahun.isin(c02.SAH)].copy()
    K['versi'] = K.ecg_id.map(muse)
    K = K[K.versi.isin(['9', '10'])]
    y = (K.versi == '10').to_numpy().astype(int)
    print(f'  MUSE 10: {y.sum()}, MUSE 9: {(1 - y).sum()}', flush=True)
    for u in c03.UKURAN:
        v = K[u].to_numpy(float)
        ok = ~np.isnan(v)
        a = roc_auc_score(y[ok], v[ok])
        print(f'  {u:<18s} {max(a, 1 - a):.3f}', flush=True)


def lengan_rocket(pre, kas, lama, jendela):
    c20 = _muat_modul('c20', '20_pembanding_kuat.py')
    print('== MiniRocket rancangan A, ukuran LR - MiniRocket, 3 benih', flush=True)
    for w, (m, _) in jendela.items():
        auc, boot = [], []
        for b in c19.BENIH:
            S = c19.susun_ab(pre, kas[m], lama, b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            X = c02.sinyal(S)
            r = c20.rocket(X, y, g, b)
            del X
            pu = c06.oof_lr_logit(S[c03.UKURAN].to_numpy(float), y, g)
            auc.append(roc_auc_score(y, r))
            boot.append(c19.boot(y, g, lambda ix: roc_auc_score(y[ix], pu[ix]) - roc_auc_score(y[ix], r[ix])))
            print(f'  A {w} benih {b} selesai', flush=True)
        d = np.concatenate(boot)
        print(f'  A {w}: MiniRocket {np.mean(auc):.3f}; ukuran LR - MiniRocket {d.mean():+.3f} {c10.ci(d)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengan', choices=['laju', 'rocket'], required=True)
    a = ap.parse_args()
    pre, kas, lama, jendela = c19.ab_data()
    if a.lengan == 'rocket':
        lengan_rocket(pre, kas, lama, jendela)
        return
    muse = versi_muse()
    bagian_kolom()
    bagian_sepakat(pre, muse)
    bagian_jejak(pre, muse)
    bagian_sampel(pre, kas, lama, jendela)


if __name__ == '__main__':
    main()
