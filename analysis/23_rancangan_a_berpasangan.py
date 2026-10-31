"""Nightingale arrest-ntuh-ecg — pembanding - CNN berpasangan pada rancangan A (rancangan Oberdier dkk. 2025:
kasus 1-30 dan 30-365 hari, kontrol 30-730 hari, pencocokan 04, benih 0-2).

Untuk bingkai revisi 3 (Artikel_Arrest_Rencana.md, 29 Oktober 2026): klaim Discussion Oberdier "it is
reasonable to expect that the DL models captured their influence" diuji pada rancangan mereka sendiri,
bukan hanya pada rancangan berjarak sama. Fungsi dan cara bootstrap sama dengan 19 bagian 3 (bootstrap
pasien + satu ulangan latih CNN acak per tarikan, digabung atas benih).

CPU. Hanya agregat. Pakai: python -u 23_rancangan_a_berpasangan.py > rancangan_a_berpasangan.log
"""
import importlib.util
import os

import numpy as np


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c19 = _muat_modul('c19', '19_berpasangan.py')


def main():
    pre, kas, lama, jendela = c19.ab_data()
    print('== Rancangan A: pembanding - CNN berpasangan (bootstrap pasien + ulangan latih), 3 benih', flush=True)
    for w, (m, _) in jendela.items():
        hasil = {'auc': {}, 'beda': {}}
        for b in c19.BENIH:
            S = c19.susun_ab(pre, kas[m], lama, b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            c19.kumpul(hasil, S, y, g, g)
            print(f'  A {w} benih {b} selesai', flush=True)
        c19.lapor_cnn(f'A {w}', hasil)


if __name__ == '__main__':
    main()
