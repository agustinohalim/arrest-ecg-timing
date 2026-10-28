"""Nightingale arrest-ntuh-ecg — pengukuran 12SL mana yang membawa sinyal LR semua sebab di rancangan P.

Pemicu: 14_mesin_lr.py — pengukuran 12SL saja dengan LR memberi 0,607 (p 0,010) untuk semua sebab, padahal
HGB 0,521. Per ukuran: AUROC satu-peubah dalam rancangan P (benih 0, tanpa pelatihan: nilai mentah, arah
dibalik bila < 0,5), beda median kasus - kontrol, dan AUROC satu-peubah yang sama untuk kontrol MUSE 10 lawan
MUSE 9 (jejak versi). Lalu LR dengan satu ukuran dibuang (leave-one-out) untuk melihat ketergantungan.
Hanya agregat. Pakai: python -u 15_ukuran_per_ciri.py > ukuran_per_ciri.log
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


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c03 = _muat_modul('c03', '03_cnn_kunjungan.py')
c06 = _muat_modul('c06', '06_revisi.py')
c09 = _muat_modul('c09', '09_uji_sebab.py')
c10 = _muat_modul('c10', '10_telaah_lanjut.py')
U = c03.UKURAN


def auc1(y, x):
    ok = ~np.isnan(x)
    a = roc_auc_score(y[ok], x[ok])
    return max(a, 1 - a), ('+' if a >= 0.5 else '-'), 1 - ok.mean()


def main():
    pre, calon, kp, _ = c09.data_p()
    S = c10.susun(pre, kp, calon, 0)
    y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
    K = c06.rancangan_c()
    K = K[K.y == 0].reset_index(drop=True)
    yv = K.tahun.isin(['2015', '2016']).to_numpy().astype(np.float32)
    print('== Per ukuran: rancangan P semua sebab (benih 0) | kontrol MUSE 10 lawan MUSE 9', flush=True)
    print(f'  {"ukuran":<16s} {"AUC P":>6s} arah hilang  med kasus  med kontrol | {"AUC versi":>9s} arah', flush=True)
    for u in U:
        x = S[u].to_numpy(float)
        a, d, h = auc1(y, x)
        av, dv, _ = auc1(yv, K[u].to_numpy(float))
        print(f'  {u:<16s} {a:6.3f} {d:>4s} {h:6.1%} {np.nanmedian(x[y == 1]):10.1f} {np.nanmedian(x[y == 0]):11.1f}'
              f' | {av:9.3f} {dv:>4s}', flush=True)
    print('\n== LR pengukuran 12SL, satu ukuran dibuang (rancangan P semua sebab, benih 0)', flush=True)
    Z = S[U].to_numpy(float)
    print(f'  semua 16: {roc_auc_score(y, c06.oof_lr_logit(Z, y, g)):.3f}', flush=True)
    for i, u in enumerate(U):
        p = c06.oof_lr_logit(np.delete(Z, i, axis=1), y, g)
        print(f'  tanpa {u:<16s} {roc_auc_score(y, p):.3f}', flush=True)


if __name__ == '__main__':
    main()
