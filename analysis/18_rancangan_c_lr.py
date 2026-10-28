"""Nightingale arrest-ntuh-ecg — rancangan C (EKG kunjungan yang sama) dengan pembanding LR (versi JIKI).

Naskah JE: CNN 0,791 lawan pembacaan mesin + usia/jk (HGB) 0,747, selisih +0,045. Di rancangan P, LR atas
masukan mesin jauh di atas HGB (14_mesin_lr.py), jadi pembanding rancangan C diuji ulang: laju ventrikel
satu peubah, pengukuran 12SL (LR), pembacaan mesin + usia/jk (LR, c06.oof_lr_logit), dan selisih CNN
(OOF tersimpan, oof_kunjungan_bersih.csv) terhadapnya dengan bootstrap pasien berpasangan (c06.beda_ci).
Hanya agregat. Pakai: python -u 18_rancangan_c_lr.py > rancangan_c_lr.log
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


def main():
    K = c06.rancangan_c()
    y, g = K.y.to_numpy().astype(np.float32), K.pid.to_numpy()
    print(f'Rancangan C: {int(y.sum())} kasus / {int((1 - y).sum())} kontrol', flush=True)
    B = K[list(c06.BENDERA)].to_numpy(float)
    U = K[c03.UKURAN].to_numpy(float)
    demo = K[['usia', 'pria']].to_numpy(float)
    o = pd.read_csv(os.path.join(c06.TURUNAN, 'oof_kunjungan_bersih.csv'))[['ecg_id', 'p_cnn']]
    p_cnn = K[['ecg_id']].merge(o, on='ecg_id', how='left').p_cnn.to_numpy()
    assert not np.isnan(p_cnn).any()
    v = K.ventricular_rate.to_numpy(float)
    p = {'laju ventrikel saja': np.where(np.isnan(v), np.nanmedian(v), v),
         'pengukuran 12SL (LR)': c06.oof_lr_logit(U, y, g),
         'pembacaan mesin + usia/jk (LR)': c06.oof_lr_logit(np.column_stack([B, U, demo]), y, g),
         'pembacaan mesin + usia/jk (HGB)': c02.oof(np.column_stack([B, U, demo]), y, g, 3, 'hgb'),
         'CNN (OOF tersimpan)': p_cnn,
         'CNN + mesin + usia/jk (LR)': c06.oof_lr_logit(np.column_stack([c06.logit(p_cnn), B, U, demo]), y, g)}
    for f, q in p.items():
        print(f'  {f:<34s} AUC {c06.teks(c06.auc_ci(y, q, g))}', flush=True)
    for a, b in (('CNN (OOF tersimpan)', 'pembacaan mesin + usia/jk (LR)'),
                 ('CNN (OOF tersimpan)', 'pembacaan mesin + usia/jk (HGB)'),
                 ('CNN + mesin + usia/jk (LR)', 'pembacaan mesin + usia/jk (LR)')):
        print(f'  {a} - {b}: {c06.teks(c06.beda_ci(y, p[a], p[b], g))}', flush=True)


if __name__ == '__main__':
    main()
