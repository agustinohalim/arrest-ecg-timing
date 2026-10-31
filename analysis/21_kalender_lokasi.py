"""Nightingale arrest-ntuh-ecg — tiga pemeriksaan murah untuk telaah ulang panel JIKI 28 Oktober 2026 (c)
(Riset/EKG/Telaah/2026-10-28c-Arrest-JIKI/report.md).

  1. Kalender: kasus dibatasi ke EKG tahun 2015-2017, tahun yang sama dengan kontrol rancangan jarak.
     Rancangan P (semua sebab) dan B (1-30, 30-365) dicocokkan ulang dengan benih 0-2; AUROC laju
     ventrikel mentah, LR pengukuran 12SL, dan jarak saja, dengan IK bootstrap set (P) atau pasien (B)
     digabung atas benih; dibandingkan dengan sampel penuh pada lari yang sama.
  2. Rancangan B: median jarak kasus dan kontrol dan AUROC jarak saja per jendela.
  3. Lokasi henti jantung (OHCA/IHCA) per rancangan: A/B per jendela, P, dan C.

CPU. Hanya agregat. Pakai: python -u 21_kalender_lokasi.py > kalender_lokasi.log
"""
import importlib.util
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c19 = _muat_modul('c19', '19_berpasangan.py')
c02, c03, c04, c06, c09, c10 = c19.c02, c19.c03, c19.c04, c19.c06, c19.c09, c19.c10
c19.rng = c10.rng = np.random.default_rng(20261102)
SAH = ['2015', '2016', '2017']


def nilai(S, u):
    y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
    p = {'laju': c19.laju(S), 'ukuran LR': c06.oof_lr_logit(S[c03.UKURAN].to_numpy(float), y, g),
         'jarak saja': -S.jarak.to_numpy(float)}
    return {k: (roc_auc_score(y, v), c19.boot(y, u, lambda ix, v=v: roc_auc_score(y[ix], v[ix])))
            for k, v in p.items()}


def lapor(nama, hasil, n):
    teks = []
    for k in hasil[0]:
        a = np.mean([h[k][0] for h in hasil])
        teks.append(f'{k} {a:.3f} {c10.ci(np.concatenate([h[k][1] for h in hasil]))}')
    print(f'  {nama:<34s} kasus {n:<4d} ' + '; '.join(teks), flush=True)


def lokasi(k):
    return k.location_of_cardiac_arrest.fillna('kosong').value_counts().to_dict()


def main():
    print('== 1. Kalender: kasus 2015-2017 saja, lawan sampel penuh (benih 0-2)', flush=True)
    pre, calon, kp, _ = c09.data_p()
    for nama, kk in [('P semua sebab, penuh', kp), ('P semua sebab, kasus 2015-2017', kp[kp.tahun.isin(SAH)])]:
        hasil = []
        for b in c19.BENIH:
            S = c10.susun(pre, kk, calon, b)
            hasil.append(nilai(S, S.set.to_numpy()))
        lapor(nama, hasil, len(kk))
    preb, kas, _, jendela = c19.ab_data()
    for w, (m, sama) in jendela.items():
        for label, kk in [('penuh', kas[m]), ('kasus 2015-2017', kas[m & kas.tahun.isin(SAH)])]:
            hasil = []
            for b in c19.BENIH:
                S = c19.susun_ab(preb, kk, sama, b)
                hasil.append(nilai(S, S.pid.to_numpy()))
            lapor(f'B {w}, {label}', hasil, len(kk))

    print('\n== 2. Rancangan B: median jarak (hari) dan jarak saja, benih 0', flush=True)
    for w, (m, sama) in jendela.items():
        S = c19.susun_ab(preb, kas[m], sama, 0)
        k, c = S[S.y == 1].jarak, S[S.y == 0].jarak
        print(f'  B {w:<7s} kasus median {k.median():.0f} (IQR {k.quantile(.25):.0f}-{k.quantile(.75):.0f}), '
              f'kontrol median {c.median():.0f} (IQR {c.quantile(.25):.0f}-{c.quantile(.75):.0f})', flush=True)

    print('\n== 3. Lokasi henti jantung per rancangan', flush=True)
    print(f'  P (2-730 hari)            {lokasi(kp)}', flush=True)
    kas_l = kas  # 04.tandai_kasus sudah membawa lokasi
    for w, (m, _) in jendela.items():
        print(f'  A/B {w:<7s}             {lokasi(kas_l[m])}', flush=True)
    print(f'  A/B 0-1 hari              {lokasi(kas_l[(kas.jarak >= 0) & (kas.jarak <= 1)])}', flush=True)
    print('  C: hanya IHCA menurut definisi (03_cnn_kunjungan.py)', flush=True)


if __name__ == '__main__':
    main()
