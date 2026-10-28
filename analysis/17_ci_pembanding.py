"""Nightingale arrest-ntuh-ecg — interval kepercayaan untuk pembanding sederhana (versi JIKI).

Rancangan P (pencocokan 09/10, benih 0-2): laju ventrikel satu peubah, pengukuran 12SL (LR); bootstrap set
cocokan digabung atas benih, dan nol permutasi dalam set untuk laju (benih 0, 100 kali; tanpa pelatihan).
Rancangan A/B (pencocokan 04, benih 0): laju dan pengukuran 12SL (LR) dengan bootstrap pasien (c06.auc_ci),
serta selisih A - B per masukan dengan bootstrap independen.
Hanya agregat. Pakai: python -u 17_ci_pembanding.py > ci_pembanding.log
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
c04 = _muat_modul('c04', '04_replikasi.py')
c06 = _muat_modul('c06', '06_revisi.py')
c09 = _muat_modul('c09', '09_uji_sebab.py')
c10 = _muat_modul('c10', '10_telaah_lanjut.py')
rng = np.random.default_rng(20261030)
c10.rng = rng


def laju(S):
    v = S.ventricular_rate.to_numpy(float)
    return np.where(np.isnan(v), np.nanmedian(v), v)


def boot_pasien(y, p, g, n=1000):
    ug = np.unique(g)
    pos = {k: np.where(g == k)[0] for k in ug}
    out = np.empty(n)
    for b in range(n):
        ix = np.concatenate([pos[k] for k in rng.choice(ug, len(ug))])
        out[b] = roc_auc_score(y[ix], p[ix])
    return out


def main():
    print('== Rancangan P semua sebab', flush=True)
    pre, calon, kp, _ = c09.data_p()
    sebab = kp.cause_of_cardiac_arrest
    jantung = (sebab == 'cardiac event').to_numpy()
    bukan = (~jantung & sebab.notna() & (sebab != 'unknown')).to_numpy()
    for nama, m in [('semua sebab', np.ones(len(kp), bool)), ('sebab jantung', jantung), ('bukan jantung', bukan)]:
        bs = {'laju ventrikel saja': [], 'pengukuran 12SL (LR)': []}
        titik = {k: [] for k in bs}
        for b in range(3):
            S = c10.susun(pre, kp[m], calon, b)
            y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
            for f, p in (('laju ventrikel saja', laju(S)),
                         ('pengukuran 12SL (LR)', c06.oof_lr_logit(S[c03.UKURAN].to_numpy(float), y, g))):
                titik[f].append(roc_auc_score(y, p))
                bs[f].append(c10.boot_set(y, p, s))
            if b == 0 and nama == 'semua sebab':
                p = laju(S)
                amat = roc_auc_score(y, p)
                nol = np.array([roc_auc_score(c10.acak_dalam_set(y, s), p) for _ in range(1000)])
                print(f'  nol permutasi dalam set, laju (1000): rerata {nol.mean():.3f}, p95 '
                      f'{np.percentile(nol, 95):.3f}, p {(1 + np.sum(nol >= amat)) / 1001:.4f}', flush=True)
        for f in bs:
            print(f'  {nama:<14s} {f:<24s} {np.mean(titik[f]):.3f} {c10.ci(np.concatenate(bs[f]))}', flush=True)

    print('\n== Rancangan A/B, benih 0, bootstrap pasien', flush=True)
    D = c04.tandai_kasus(c02.muat())
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    pre = pre.merge(c03.ukuran(), on='ecg_id', how='left')
    kas = pre[pre.y == 1]
    k = pre[(pre.y == 0) & pre.tahun.isin(c04.SAH)]
    lama = k[(k.jarak >= 30) & (k.jarak <= 730)]
    for jendela, m, sama in [('1-30', (kas.jarak > 1) & (kas.jarak <= 30), k[(k.jarak > 1) & (k.jarak <= 30)]),
                             ('30-365', (kas.jarak > 30) & (kas.jarak <= 365), k[(k.jarak > 30) & (k.jarak <= 365)])]:
        bsr = {}
        for r, calon in (('A', lama), ('B', sama)):
            kk = kas[m]
            ix = np.concatenate([kk.index.to_numpy(), c04.cocokkan(kk, calon, 0)])
            S = pre.loc[np.sort(ix)].reset_index(drop=True)
            y, g = S.y.to_numpy().astype(np.float32), S.patient_ngsci_id.astype(str).to_numpy()
            for f, p in (('laju ventrikel saja', laju(S)),
                         ('pengukuran 12SL (LR)', c06.oof_lr_logit(S[c03.UKURAN].to_numpy(float), y, g))):
                bsr[(r, f)] = boot_pasien(y, p, g)
                print(f'  {jendela:<7s} {r} {f:<24s} {roc_auc_score(y, p):.3f} {c10.ci(bsr[(r, f)])}', flush=True)
        for f in ('laju ventrikel saja', 'pengukuran 12SL (LR)'):
            d = bsr[('A', f)] - bsr[('B', f)]
            print(f'  {jendela:<7s} A - B {f:<24s} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}', flush=True)


if __name__ == '__main__':
    main()
