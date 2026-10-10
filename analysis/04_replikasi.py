"""Nightingale arrest-ntuh-ecg — replikasi rancangan Oberdier dkk. (2025) dan apa yang terjadi bila waktunya dibersihkan.

Oberdier MT dkk. (2025), Eur Heart J Digit Health 6(2):170-179, doi:10.1093/ehjdh/ztae088, memakai
dataset yang sama (dibaca di PMC11914729, 7 Oktober 2026):
  kasus    EKG dalam 1 hari sebelum henti jantung (model dasar); jendela lain 1 hari-1 bulan dan
           1 bulan-1 tahun
  kontrol  pasien IGD pada hari yang sama tanpa henti jantung, EKG direkam 1 bulan-2 tahun sebelum
           kunjungannya; usia dan jenis kelamin diseimbangkan
  hasil    AUC 0,774 (< 1 hari), 0,654 (1 hari-1 bulan), 0,574 (1 bulan-1 tahun)
Akibatnya kasus < 1 hari kebanyakan EKG kunjungan akut yang sama, sedangkan kontrolnya EKG kunjungan
lampau — rancu waktu (Hasil_Uji_Arrest.md, temuan 3).

Rancangan di sini (kontrol pre berjarak 30-730 hari, hanya 2015-2017 karena jarak kontrol 2018-2019
cacat; usia per 10 tahun x jenis kelamin diseimbangkan, 5 kontrol per kasus, 3 benih pencocokan):
  R0  kasus [0]pre 0-1 hari sebelum ROSC (seperti model dasar Oberdier)
  R0b R0, hanya kasus bersih (IHCA, EKG setelah datang dan sebelum CPR)
  R1  kasus 1-30 hari
  R2  kasus 30-365 hari
Dengan --jarak-sama, rancangan diganti versi yang jaraknya disamakan (kontrol pre dari jendela
jarak yang sama dengan kasus, usia x jk tetap diseimbangkan):
  R1m kasus 1-30 hari lawan kontrol 1-30 hari
  R2m kasus 30-365 hari lawan kontrol 30-365 hari
Pembandingnya sudah ada: EKG kunjungan lampau dengan jarak disamakan (02, CNN 0,539) dan EKG kunjungan
yang sama lawan EKG indeks kontrol, kasus bersih (03, CNN 0,791).

Model: usia + jk (LR; mendekati 0,5 bila penyeimbangan berhasil), ciri sinyal sederhana (HGB), CNN
12 sadapan (2 ulangan). Data tetap di C:/Data/Nightingale; hanya agregat yang dicetak.
Pakai: python -u 04_replikasi.py > replikasi.log   (--jarak-sama > replikasi_jarak_sama.log)
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c03 = _muat_modul('c03', '03_cnn_kunjungan.py')
SAH = ['2015', '2016', '2017']
BENIH = range(3)
RASIO = 5


def tandai_kasus(D):
    """Lokasi henti jantung, jam setelah datang, dan jam sebelum CPR untuk EKG kasus."""
    o = pd.read_csv(os.path.join(c02.R, 'study-group', 'rosc-outcomes.csv')).drop_duplicates('patient_ngsci_id')
    D = D.merge(o[['patient_ngsci_id', 'location_of_cardiac_arrest', 'hospitalization_datetime_offset',
                   'cpr_start_datetime_offset']], on='patient_ngsci_id', how='left')
    k = D.y == 1
    D['jam_cpr'] = np.nan
    D['datang'] = np.nan
    D.loc[k, 'jam_cpr'] = c03.jam(D.loc[k, 'cpr_start_datetime_offset'], D.loc[k, 'acquisition_datetime_offset'])
    D.loc[k, 'datang'] = c03.jam(D.loc[k, 'acquisition_datetime_offset'], D.loc[k, 'hospitalization_datetime_offset'])
    D['bersih'] = k & (D.location_of_cardiac_arrest == 'IHCA') & (D.datang >= 0) & (D.jam_cpr > 0)
    return D


def cocokkan(kas, calon, benih):
    """Kontrol diambil per strata usia 10 tahun x jenis kelamin, RASIO per kasus."""
    rc = np.random.default_rng(40000 + benih)
    sk = (kas.usia // 10).fillna(-1).astype(int).astype(str) + kas.pria.astype(str)
    sc = (calon.usia // 10).fillna(-1).astype(int).astype(str) + calon.pria.astype(str)
    pilih = []
    for s, n in sk.value_counts().items():
        ix = calon.index[sc == s].to_numpy()
        pilih.append(rc.choice(ix, min(len(ix), RASIO * n), replace=False))
    return np.concatenate(pilih)


def main():
    D = tandai_kasus(c02.muat())
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    kontrol = pre[(pre.y == 0) & pre.tahun.isin(SAH) & (pre.jarak >= 30) & (pre.jarak <= 730)]
    print(f'Kontrol pre 30-730 hari, 2015-2017: {len(kontrol)}', flush=True)
    kasus_pre = pre[pre.y == 1]
    rancangan = [
        ('R0  kasus 0-1 hari (seperti Oberdier)', (kasus_pre.jarak > 0) & (kasus_pre.jarak <= 1)),
        ('R0b kasus 0-1 hari, hanya bersih', (kasus_pre.jarak > 0) & (kasus_pre.jarak <= 1) & kasus_pre.bersih),
        ('R1  kasus 1-30 hari', (kasus_pre.jarak > 1) & (kasus_pre.jarak <= 30)),
        ('R2  kasus 30-365 hari', (kasus_pre.jarak > 30) & (kasus_pre.jarak <= 365)),
    ]
    r0 = kasus_pre[(kasus_pre.jarak > 0) & (kasus_pre.jarak <= 1)]
    print(f'Isi kasus R0 ({len(r0)}): OHCA {int((r0.location_of_cardiac_arrest == "OHCA").sum())}, '
          f'EKG pada/sesudah awal CPR {int((r0.jam_cpr <= 0).sum())}, '
          f'sebelum kedatangan {int((r0.datang < 0).sum())}, bersih {int(r0.bersih.sum())}', flush=True)

    if '--jarak-sama' in sys.argv:
        k = pre[(pre.y == 0) & pre.tahun.isin(SAH)]
        rancangan = [('R1m kasus 1-30 hari, kontrol 1-30 hari', (kasus_pre.jarak > 1) & (kasus_pre.jarak <= 30),
                      k[(k.jarak > 1) & (k.jarak <= 30)]),
                     ('R2m kasus 30-365 hari, kontrol 30-365 hari', (kasus_pre.jarak > 30) & (kasus_pre.jarak <= 365),
                      k[(k.jarak > 30) & (k.jarak <= 365)])]
        for judul, _, kk in rancangan:
            print(f'{judul}: calon kontrol {len(kk)}', flush=True)
    else:
        rancangan = [(j, m, kontrol) for j, m in rancangan]

    rekap = {}
    for judul, m, calon in rancangan:
        kas = kasus_pre[m]
        for b in BENIH:
            ix = np.concatenate([kas.index.to_numpy(), cocokkan(kas, calon, b)])
            S = pre.loc[np.sort(ix)].reset_index(drop=True)
            X = c02.sinyal(S)
            print(f'\n{judul}, benih {b}: median usia kasus {S[S.y == 1].usia.median():.0f}, '
                  f'kontrol {S[S.y == 0].usia.median():.0f}', flush=True)
            for nama, F, cara, ulang in [('usia + jenis kelamin (LR)', S[['usia', 'pria']].to_numpy(float), 'lr', 1),
                                         ('ciri sinyal sederhana (HGB)', c02.ciri(X), 'hgb', 1),
                                         ('CNN 12 sadapan', X, 'cnn', 2)]:
                rekap.setdefault((judul, nama), []).append(c02.lapor(nama, F, S, ulang, cara))
            del X

    print(f'\nRekap atas {len(BENIH)} benih pencocokan: rerata AUC [min, maks]', flush=True)
    for (judul, nama), v in rekap.items():
        print(f'  {judul:<40s} {nama:<28s} {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]', flush=True)


if __name__ == '__main__':
    main()
