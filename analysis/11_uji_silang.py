"""Nightingale arrest-ntuh-ecg — analisis sesudah telaah panel ketiga 10 Oktober 2026
(Riset/EKG/Telaah/2026-10-10b/report.md), untuk versi Naskah_Arrest_Resuscitation.md.

Keberatan yang diuji:
  1. [kritis] pemisahan irama mengulang label sebab (32/39 VT/VF berlabel jantung) -> uji silang:
     sebab jantung lawan bukan jantung DI DALAM irama PEA/asistol (label sebab tanpa beda irama).
  2. sebab 'unknown' masuk kelompok bukan jantung -> kontras jantung - bukan jantung dengan unknown
     dikeluarkan, dan dengan unknown dihitung sebagai jantung (konvensi Utstein OHCA).
  3. VT/VF tanpa nol sendiri -> permutasi dalam set (100) untuk VT/VF.
  4. 'CNN tidak melampaui pembacaan mesin' di rancangan P tanpa uji berpasangan -> selisih berpasangan
     CNN - pembacaan mesin, bootstrap set cocokan (benih 0), semua sebab dan sebab jantung.
  5. jejak era/versi MUSE hanya dicek pada CNN -> pembacaan mesin memisahkan EKG indeks kontrol MUSE 10
     lawan MUSE 9?

Pencocokan dan bootstrap/permutasi sama dengan 10_telaah_lanjut.py. Hanya agregat yang dicetak.
Pakai: python -u 11_uji_silang.py > uji_silang.log
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
BENIH = range(3)
c10.rng = np.random.default_rng(20261012)
MESIN, CIRI = c10.MESIN, c10.CIRI


def jalankan(pre, kas, calon, nama, perm=False, cnn=False):
    hasil = {}
    for b in BENIH:
        S = c10.susun(pre, kas, calon, b)
        y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
        X = c02.sinyal(S)
        F = c10.fitur(S, X)
        F = {k: F[k] for k in (MESIN, CIRI)}
        if cnn and b == 0:
            F['CNN'] = (X, 'cnn')
        for f, (Fx, cara) in F.items():
            p = c02.oof(Fx, y, g, 2 if cara == 'cnn' else 1, cara)
            hasil[(f, b)] = (y, p, s)
        if perm and b == 0:
            for f in (MESIN, CIRI):
                Fx, cara = F[f]
                amat = roc_auc_score(*hasil[(f, 0)][:2])
                nol = np.array([roc_auc_score(yp, c02.oof(Fx, yp, g, 1, cara))
                                for yp in (c10.acak_dalam_set(y, s) for _ in range(c10.N_PERM))])
                print(f'    nol permutasi dalam set {nama} {f:<28s} rerata {nol.mean():.3f}, p95 '
                      f'{np.percentile(nol, 95):.3f}, p {(1 + np.sum(nol >= amat)) / (1 + c10.N_PERM):.3f}', flush=True)
        del X
    for f in (MESIN, CIRI):
        a = [roc_auc_score(*hasil[(f, b)][:2]) for b in BENIH]
        bs = c10.boot_set(*hasil[(f, 0)])
        print(f'  {nama:<44s} n={int(kas.shape[0]):<4d} {f:<28s} {np.mean(a):.3f} [{min(a):.3f}, {max(a):.3f}]'
              f'  benih 0 {a[0]:.3f} {c10.ci(bs)}', flush=True)
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


def berpasangan(h, nama):
    y, pc, s = h[('CNN', 0)]
    _, pm, _ = h[(MESIN, 0)]
    us = np.unique(s)
    anggota = {k: np.where(s == k)[0] for k in us}
    d = []
    for _ in range(c10.N_BOOT):
        ix = np.concatenate([anggota[k] for k in c10.rng.choice(us, len(us))])
        d.append(roc_auc_score(y[ix], pc[ix]) - roc_auc_score(y[ix], pm[ix]))
    d = np.array(d)
    print(f'  {nama}: CNN {roc_auc_score(y, pc):.3f} - pembacaan mesin {roc_auc_score(y, pm):.3f} = '
          f'{roc_auc_score(y, pc) - roc_auc_score(y, pm):+.3f} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}', flush=True)


def main():
    pre, calon, kp, _ = c09.data_p()
    sebab, ritme = kp.cause_of_cardiac_arrest, kp.initial_rhythm
    jantung = (sebab == 'cardiac event').to_numpy()
    unknown = (sebab == 'unknown').to_numpy()
    bukan = (~jantung & sebab.notna()).to_numpy()
    pa = ritme.isin(['PEA', 'asystole']).to_numpy()
    vt = (ritme == 'VT/VF').to_numpy()

    print('== 1. Uji silang di dalam PEA/asistol', flush=True)
    h_jpa = jalankan(pre, kp[jantung & pa], calon, 'jantung, PEA/asistol', perm=True)
    h_bpa = jalankan(pre, kp[bukan & ~unknown & pa], calon, 'bukan jantung (tanpa unknown), PEA/asistol', perm=True)
    for f in (MESIN, CIRI):
        print(f'  selisih jantung - bukan jantung di PEA/asistol | {f:<28s} {selisih(h_jpa, h_bpa, f)}', flush=True)

    print('\n== 2. Sebab unknown', flush=True)
    h_j = jalankan(pre, kp[jantung], calon, 'sebab jantung')
    h_b_tanpa = jalankan(pre, kp[bukan & ~unknown], calon, 'bukan jantung tanpa unknown')
    h_j_plus = jalankan(pre, kp[jantung | unknown], calon, 'jantung + unknown')
    h_b_tanpa2 = h_b_tanpa
    for f in (MESIN, CIRI):
        print(f'  jantung - bukan jantung (unknown keluar)   | {f:<28s} {selisih(h_j, h_b_tanpa, f)}', flush=True)
        print(f'  jantung+unknown - bukan jantung            | {f:<28s} {selisih(h_j_plus, h_b_tanpa2, f)}', flush=True)

    print('\n== 3. VT/VF: nol permutasi dalam set', flush=True)
    jalankan(pre, kp[vt], calon, 'irama VT/VF', perm=True)

    print('\n== 4. CNN - pembacaan mesin berpasangan (benih 0, bootstrap set)', flush=True)
    for nama, m in [('semua sebab', np.ones(len(kp), bool)), ('sebab jantung', jantung)]:
        h = jalankan(pre, kp[m], calon, nama, cnn=True)
        berpasangan(h, nama)

    print('\n== 5. Pembacaan mesin: EKG indeks kontrol MUSE 10 (2015-16) lawan MUSE 9 (2017-19)', flush=True)
    K = c06.rancangan_c()
    K = K[K.y == 0].reset_index(drop=True)
    yv = K.tahun.isin(['2015', '2016']).to_numpy().astype(np.float32)
    gv = K.pid.to_numpy()
    Z = K[list(c06.BENDERA) + c03.UKURAN].to_numpy(float)
    p = c02.oof(Z, yv, gv, 1, 'hgb')
    print(f'  MUSE 10: {int(yv.sum())}, MUSE 9: {int((1 - yv).sum())}; pembacaan mesin AUC '
          f'{c06.teks(c06.auc_ci(yv, p, gv))}', flush=True)
    p = c02.oof(K[['usia', 'pria']].to_numpy(float), yv, gv, 1, 'lr')
    print(f'  usia + jenis kelamin AUC {c06.teks(c06.auc_ci(yv, p, gv))}', flush=True)


if __name__ == '__main__':
    main()
