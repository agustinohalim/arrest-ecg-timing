"""Nightingale arrest-ntuh-ecg — apakah EKG kunjungan lampau memprediksi henti jantung ARITMIK bila jarak
rekam disamakan?

Pemicu: Obermeyer dkk. (2026), Nature 655:210-218, doi:10.1038/s41586-026-10674-6, menerapkan model
Swedia tanpa penyetelan (zero-shot) ke dataset ini: EKG terakhir sebelum kunjungan IGD (EKG 2 hari
sebelum kunjungan dibuang), henti aritmik lawan kontrol AUC 0,767, henti non-aritmik 0,582. Di
04_replikasi.py --jarak-sama semua sebab digabung dan CNN dilatih dari awal: ~0,52. Pertanyaannya:
apakah nol itu karena sebab dicampur, atau karena dataset ini terlalu kecil untuk melatih dari awal?

Rancangan P (EKG kunjungan lampau, jarak disamakan):
  kasus    [0]pre 2-730 hari sebelum ROSC (seperti Obermeyer: 2 hari terakhir dibuang)
  kontrol  pre pasien kontrol 2015-2017, 2-730 hari sebelum EKG indeks (jarak 2018-2019 cacat)
  cocok    strata pita jarak (2-30, 30-180, 180-365, 365-730 hari) x usia 10 tahun x jenis kelamin,
           5 kontrol per kasus, 3 benih pencocokan
  subset   semua sebab; sebab jantung (cause_of_cardiac_arrest = 'cardiac event'); irama awal VT/VF;
           sebab bukan jantung (plasebo)
Ditambah rancangan C bersih (EKG kunjungan yang sama) diskor ulang per sebab dari OOF 03.

Model: usia + jk (LR), ciri sinyal sederhana (HGB), CNN 12 sadapan (2 ulangan) — sama dengan 04.
Data tetap di C:/Data/Nightingale; hanya agregat yang dicetak.
Pakai: python -u 07_sebab.py > sebab.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c03 = _muat_modul('c03', '03_cnn_kunjungan.py')
c06 = _muat_modul('c06', '06_revisi.py')
SAH = ['2015', '2016', '2017']
PITA = [2, 30, 180, 365, 730]
BENIH = range(3)
RASIO = 5


def sebab():
    o = pd.read_csv(os.path.join(c02.R, 'study-group', 'rosc-outcomes.csv')).drop_duplicates('patient_ngsci_id')
    return o[['patient_ngsci_id', 'cause_of_cardiac_arrest', 'initial_rhythm']]


def cocokkan(kas, calon, benih):
    rc = np.random.default_rng(50000 + benih)

    def strata(d):
        return (pd.cut(d.jarak, PITA, labels=False).astype(str) + ':' +
                (d.usia // 10).fillna(-1).astype(int).astype(str) + d.pria.astype(str))

    sk, sc = strata(kas), strata(calon)
    pilih, kurang = [], 0
    for s, n in sk.value_counts().items():
        ix = calon.index[sc == s].to_numpy()
        kurang += max(0, RASIO * n - len(ix))
        pilih.append(rc.choice(ix, min(len(ix), RASIO * n), replace=False))
    return np.concatenate(pilih), kurang


def rancangan_p():
    D = c02.muat().merge(sebab(), on='patient_ngsci_id', how='left')
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    calon = pre[(pre.y == 0) & pre.tahun.isin(SAH) & (pre.jarak > 2) & (pre.jarak <= 730)]
    kp = pre[(pre.y == 1) & (pre.jarak > 2) & (pre.jarak <= 730)]
    print(f'Rancangan P: kasus [0]pre 2-730 hari {len(kp)}, calon kontrol {len(calon)}', flush=True)
    print('  sebab kasus:', kp.cause_of_cardiac_arrest.value_counts().to_dict(), flush=True)
    print('  irama awal kasus:', kp.initial_rhythm.value_counts().to_dict(), flush=True)
    jantung = kp.cause_of_cardiac_arrest == 'cardiac event'
    subset = [('semua sebab', np.ones(len(kp), bool)), ('sebab jantung', jantung.to_numpy()),
              ('irama awal VT/VF', (kp.initial_rhythm == 'VT/VF').to_numpy()),
              ('sebab bukan jantung (plasebo)', (~jantung & kp.cause_of_cardiac_arrest.notna()).to_numpy())]
    rekap = {}
    for judul, m in subset:
        kas = kp[m]
        for b in BENIH:
            ix_k, kurang = cocokkan(kas, calon, b)
            S = pre.loc[np.sort(np.concatenate([kas.index.to_numpy(), ix_k]))].reset_index(drop=True)
            X = c02.sinyal(S)
            print(f'\nP {judul}, benih {b}: kontrol kurang {kurang}; median jarak kasus '
                  f'{S[S.y == 1].jarak.median():.0f} hari, kontrol {S[S.y == 0].jarak.median():.0f}', flush=True)
            for nama, F, cara, ulang in [('usia + jenis kelamin (LR)', S[['usia', 'pria']].to_numpy(float), 'lr', 1),
                                         ('ciri sinyal sederhana (HGB)', c02.ciri(X), 'hgb', 1),
                                         ('CNN 12 sadapan', X, 'cnn', 2)]:
                rekap.setdefault((judul, nama), []).append(c02.lapor(nama, F, S, ulang, cara))
            del X
    print(f'\nRekap rancangan P atas {len(BENIH)} benih: rerata AUC [min, maks]', flush=True)
    for (judul, nama), v in rekap.items():
        print(f'  {judul:<30s} {nama:<28s} {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]', flush=True)


def rancangan_c_per_sebab():
    print('\n== Rancangan C bersih diskor ulang per sebab (OOF 03, kasus terpilih lawan semua kontrol)', flush=True)
    K = c06.rancangan_c().merge(sebab(), on='patient_ngsci_id', how='left')
    o = pd.read_csv(os.path.join(c06.TURUNAN, 'oof_kunjungan_bersih.csv'))[['ecg_id', 'p_cnn']]
    K = K.merge(o, on='ecg_id', how='left')
    y, g, p = K.y.to_numpy().astype(np.float32), K.pid.to_numpy(), K.p_cnn.to_numpy()
    kontrol = y == 0
    jantung = (K.cause_of_cardiac_arrest == 'cardiac event').to_numpy()
    for nama, m in [('sebab jantung', jantung), ('irama awal VT/VF', (K.initial_rhythm == 'VT/VF').to_numpy()),
                    ('sebab bukan jantung', ~jantung & K.cause_of_cardiac_arrest.notna().to_numpy())]:
        mm = kontrol | ((y == 1) & m)
        print(f'  {nama:<26s} CNN {c06.teks(c06.auc_ci(y[mm], p[mm], g[mm]))}  n={int(y[mm].sum())}', flush=True)


def main():
    rancangan_c_per_sebab()
    rancangan_p()


if __name__ == '__main__':
    main()
