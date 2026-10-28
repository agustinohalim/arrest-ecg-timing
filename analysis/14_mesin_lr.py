"""Nightingale arrest-ntuh-ecg — pembacaan mesin dengan regresi logistik (LR), rancangan P.

Pemicu: di 13_telaah_jiki.py pembacaan mesin semua sebab memberi 0,521 dengan HGB (angka naskah) tetapi
0,591 dengan LR (c06.oof_lr_logit, masukan sama). Klaim "semua sebab dekat peluang" bergantung pada model.
Dua penjelasan diuji:
  (a) sinyal nyata yang tidak ditangkap HGB tanpa setelan;
  (b) jejak perangkat lunak ekspor (kasus MUSE 7, kontrol MUSE 9/10) pada bendera pernyataan.
Uji:
  1. LR pada pembacaan mesin, pengukuran 12SL saja, bendera pernyataan saja; semua sebab, sebab jantung,
     bukan jantung (tanpa unknown); tiga benih pencocokan; c-statistik dalam set.
  2. nol permutasi label di dalam set (100 kali, benih 0) untuk ketiga masukan LR, semua sebab.
  3. selisih sebab jantung - bukan jantung dengan LR, bootstrap set digabung atas benih.
  4. kontrol MUSE 10 (2015-16) lawan MUSE 9 (2017-19) dengan LR, seperti 11 bagian 5.

Tanpa CNN. Hanya agregat yang dicetak. Pakai: python -u 14_mesin_lr.py > mesin_lr.log
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
c12 = _muat_modul('c12', '12_telaah_je.py')
BENIH = range(3)
rng = np.random.default_rng(20261029)
c10.rng = rng
MESIN, UKUR, NYATA = 'pembacaan mesin (LR)', 'pengukuran 12SL saja (LR)', 'bendera pernyataan saja (LR)'


def masukan(S):
    B = S[list(c06.BENDERA)].to_numpy(float)
    U = S[c03.UKURAN].to_numpy(float)
    return {MESIN: np.column_stack([B, U]), UKUR: U, NYATA: B}


def jalankan(pre, kas, calon, nama, perm=False):
    hasil = {}
    for b in BENIH:
        S = c10.susun(pre, kas, calon, b)
        y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
        for f, Z in masukan(S).items():
            hasil[(f, b)] = (y, c06.oof_lr_logit(Z, y, g), s)
            if perm and b == 0:
                amat = roc_auc_score(*hasil[(f, 0)][:2])
                nol = np.array([roc_auc_score(yp, c06.oof_lr_logit(Z, yp, g))
                                for yp in (c10.acak_dalam_set(y, s) for _ in range(c10.N_PERM))])
                print(f'    nol permutasi dalam set {nama} {f:<30s} rerata {nol.mean():.3f}, p95 '
                      f'{np.percentile(nol, 95):.3f}, p {(1 + np.sum(nol >= amat)) / (1 + c10.N_PERM):.3f}',
                      flush=True)
    for f in (MESIN, UKUR, NYATA):
        a = [roc_auc_score(*hasil[(f, b)][:2]) for b in BENIH]
        cs = [c12.c_dalam_set(*hasil[(f, b)]) for b in BENIH]
        bs = c10.boot_set(*hasil[(f, 0)])
        print(f'  {nama:<32s} n={len(kas):<4d} {f:<30s} AUROC {np.mean(a):.3f} [{min(a):.3f}, {max(a):.3f}] '
              f'benih 0 {a[0]:.3f} {c10.ci(bs)} | c dalam set {np.mean(cs):.3f}', flush=True)
    return hasil


def main():
    pre, calon, kp, _ = c09.data_p()
    sebab = kp.cause_of_cardiac_arrest
    jantung = (sebab == 'cardiac event').to_numpy()
    bukan = (~jantung & sebab.notna() & (sebab != 'unknown')).to_numpy()

    print('== 1 dan 2. Pembacaan mesin dengan LR, rancangan P', flush=True)
    h = {}
    for nama, m, perm in [('semua sebab', np.ones(len(kp), bool), True), ('sebab jantung', jantung, False),
                          ('bukan jantung (tanpa unknown)', bukan, False)]:
        h[nama] = jalankan(pre, kp[m], calon, nama, perm)

    print('\n== 3. Sebab jantung - bukan jantung, LR', flush=True)
    for f in (MESIN, UKUR, NYATA):
        print(f'  {f:<30s} {c12.selisih(h["sebab jantung"], h["bukan jantung (tanpa unknown)"], f)}', flush=True)

    print('\n== 4. Kontrol MUSE 10 (2015-16) lawan MUSE 9 (2017-19), LR', flush=True)
    K = c06.rancangan_c()
    K = K[K.y == 0].reset_index(drop=True)
    yv = K.tahun.isin(['2015', '2016']).to_numpy().astype(np.float32)
    gv = K.pid.to_numpy()
    print(f'  MUSE 10: {int(yv.sum())}, MUSE 9: {int((1 - yv).sum())}', flush=True)
    for f, Z in masukan(K).items():
        p = c06.oof_lr_logit(Z, yv, gv)
        print(f'  {f:<30s} AUC {c06.teks(c06.auc_ci(yv, p, gv))}', flush=True)
    p = c06.oof_lr_logit(K[['usia', 'pria']].to_numpy(float), yv, gv)
    print(f'  {"usia + jenis kelamin (LR)":<30s} AUC {c06.teks(c06.auc_ci(yv, p, gv))}', flush=True)


if __name__ == '__main__':
    main()
