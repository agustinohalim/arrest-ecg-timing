"""Nightingale arrest-ntuh-ecg — analisis sesudah telaah panel 16 Oktober 2026
(Riset/EKG/Telaah/2026-10-16-Arrest-JE/report.md), untuk versi Naskah_Arrest_JElectrocardiol.md.

Keberatan yang diuji (semuanya rancangan P, pencocokan sama dengan 10/11, tanpa CNN):
  1. "data tidak memuat riwayat jantung" keliru untuk kasus: study-group/comorbidities.csv memuat
     cad, hf, arrhythmia (dan lainnya) per kasus. Tabulasi riwayat jantung (cad/hf/arrhythmia) menurut
     sebab tercatat, pada kasus rancangan P. Kontrol tidak punya berkas komorbiditas.
  2. lingkaran pengodean: kontras sebab jantung - bukan jantung (unknown keluar) DI DALAM kasus dengan
     riwayat jantung dan DI DALAM kasus tanpa riwayat jantung. Bila label sebab tetap memisahkan pada
     kasus tanpa riwayat tercatat, riwayat yang diketahui bukan satu-satunya jalur lingkaran.
  3. pernyataan diagnosis bertanda USERINSERT seluruhnya (mungkin dikonfirmasi dokter) -> pembacaan
     mesin dipecah: 16 pengukuran 12SL saja lawan 22 bendera pernyataan saja.
  4. AUROC gabungan pada rancangan berpasangan -> c-statistik di dalam set: peluang kasus mendapat skor
     lebih tinggi daripada kontrol dari set yang sama (seri = 0,5), rerata atas benih.

Hanya agregat yang dicetak. Pakai: python -u 12_telaah_je.py > telaah_je.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c03 = _muat_modul('c03', '03_cnn_kunjungan.py')
c06 = _muat_modul('c06', '06_revisi.py')
c09 = _muat_modul('c09', '09_uji_sebab.py')
c10 = _muat_modul('c10', '10_telaah_lanjut.py')
BENIH = range(3)
c10.rng = np.random.default_rng(20261016)
MESIN, CIRI = c10.MESIN, c10.CIRI
UKUR = 'pengukuran 12SL saja (HGB)'
NYATA = 'bendera pernyataan saja (HGB)'
RIWAYAT = ['cad', 'hf', 'arrhythmia']


def c_dalam_set(y, p, s):
    """Peluang kasus > kontrol di dalam set yang sama (seri 0,5)."""
    nilai = []
    for k in np.unique(s):
        ix = np.where(s == k)[0]
        kas, kon = p[ix][y[ix] == 1], p[ix][y[ix] == 0]
        if len(kas) and len(kon):
            nilai += [float(np.mean((a > kon) + 0.5 * (a == kon))) for a in kas]
    return float(np.mean(nilai))


def jalankan(pre, kas, calon, nama, pecah=False):
    hasil = {}
    for b in BENIH:
        S = c10.susun(pre, kas, calon, b)
        y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
        X = c02.sinyal(S)
        F = c10.fitur(S, X)
        F = {k: F[k] for k in (MESIN, CIRI)}
        if pecah:
            F[UKUR] = (S[c03.UKURAN].to_numpy(float), 'hgb')
            F[NYATA] = (S[list(c06.BENDERA)].to_numpy(float), 'hgb')
        for f, (Fx, cara) in F.items():
            hasil[(f, b)] = (y, c02.oof(Fx, y, g, 1, cara), s)
        del X
    for f in sorted({k[0] for k in hasil}):
        a = [roc_auc_score(*hasil[(f, b)][:2]) for b in BENIH]
        cs = [c_dalam_set(*hasil[(f, b)]) for b in BENIH]
        bs = c10.boot_set(*hasil[(f, 0)])
        print(f'  {nama:<40s} n={int(kas.shape[0]):<4d} {f:<30s} AUROC {np.mean(a):.3f} '
              f'[{min(a):.3f}, {max(a):.3f}] benih 0 {a[0]:.3f} {c10.ci(bs)} | c dalam set {np.mean(cs):.3f} '
              f'[{min(cs):.3f}, {max(cs):.3f}]', flush=True)
    return hasil


def selisih(h1, h2, f):
    d, m = [], []
    for b in BENIH:
        y1, p1, s1 = h1[(f, b)]
        y2, p2, s2 = h2[(f, b)]
        m.append(roc_auc_score(y1, p1) - roc_auc_score(y2, p2))
        d.append(c10.boot_set(y1, p1, s1) - c10.boot_set(y2, p2, s2))
    d = np.concatenate(d)
    return f'{np.mean(m):+.3f} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}'


def main():
    pre, calon, kp, _ = c09.data_p()
    kom = pd.read_csv(os.path.join(c02.R, 'study-group', 'comorbidities.csv'))
    kom = kom.groupby('patient_ngsci_id')[RIWAYAT].max()
    kp = kp.copy()
    for r in RIWAYAT:
        kp[r] = kp.patient_ngsci_id.map(kom[r])
    kp['riwayat'] = kp[RIWAYAT].max(axis=1)
    sebab = kp.cause_of_cardiac_arrest
    jantung = (sebab == 'cardiac event').to_numpy()
    bukan = (~jantung & sebab.notna() & (sebab != 'unknown')).to_numpy()
    riw = (kp.riwayat == 1).to_numpy()

    print('== 1. Riwayat jantung (comorbidities.csv) pada kasus rancangan P menurut sebab tercatat', flush=True)
    print(f'  kasus {len(kp)}, riwayat terisi {int(kp.riwayat.notna().sum())}', flush=True)
    for nama, m in [('sebab jantung', jantung), ('bukan jantung (tanpa unknown)', bukan)]:
        sub = kp[m]
        teks = ', '.join(f'{r} {int(sub[r].sum())}/{len(sub)} ({100 * sub[r].mean():.1f} %)'
                         for r in RIWAYAT + ['riwayat'])
        print(f'  {nama:<32s} {teks}', flush=True)

    print('\n== 2. Kontras sebab di dalam strata riwayat jantung', flush=True)
    h = {}
    for nama, m in [('jantung, dengan riwayat', jantung & riw), ('jantung, tanpa riwayat', jantung & ~riw),
                    ('bukan jantung, dengan riwayat', bukan & riw), ('bukan jantung, tanpa riwayat', bukan & ~riw)]:
        h[nama] = jalankan(pre, kp[m], calon, nama)
    for f in (MESIN, CIRI):
        print(f'  jantung - bukan jantung, dengan riwayat | {f:<30s} '
              f'{selisih(h["jantung, dengan riwayat"], h["bukan jantung, dengan riwayat"], f)}', flush=True)
        print(f'  jantung - bukan jantung, tanpa riwayat  | {f:<30s} '
              f'{selisih(h["jantung, tanpa riwayat"], h["bukan jantung, tanpa riwayat"], f)}', flush=True)

    print('\n== 3 dan 4. Pengukuran lawan pernyataan; c-statistik di dalam set', flush=True)
    hs = {}
    for nama, m in [('semua sebab', np.ones(len(kp), bool)), ('sebab jantung', jantung),
                    ('bukan jantung (tanpa unknown)', bukan)]:
        hs[nama] = jalankan(pre, kp[m], calon, nama, pecah=True)
    for f in (UKUR, NYATA):
        print(f'  jantung - bukan jantung | {f:<30s} '
              f'{selisih(hs["sebab jantung"], hs["bukan jantung (tanpa unknown)"], f)}', flush=True)


if __name__ == '__main__':
    main()
