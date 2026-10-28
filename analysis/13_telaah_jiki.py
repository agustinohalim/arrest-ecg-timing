"""Nightingale arrest-ntuh-ecg — analisis untuk versi JIKI, menjawab isu mayor telaah panel 16 Oktober 2026
(Riset/EKG/Telaah/2026-10-16b-Arrest-JE/report.md) sesudah penolakan Journal of Electrocardiology (28 Okt 2026).

Keberatan yang diuji (rancangan P, pencocokan sama dengan 10/11/12, benih pencocokan 0-2):
  1. "sinyal sebab jantung hanya pada kasus berriwayat" belum diuji sebagai interaksi -> beda-dalam-beda
     (jantung - bukan jantung | dengan riwayat) - (jantung - bukan jantung | tanpa riwayat), bootstrap set
     cocokan per kelompok, digabung atas benih pencocokan.
  2. CNN - pembacaan mesin berpasangan hanya benih 0 -> semua benih pencocokan, bootstrap set berpasangan
     digabung; AUROC tiap ulangan latih CNN (2 ulangan per benih, seperti naskah) dilaporkan terpisah.
  3. rancangan P tanpa model gabungan -> CNN + pembacaan mesin (LR atas logit CNN OOF + 38 masukan mesin,
     sama dengan rancangan C di 06). Pembandingnya pembacaan mesin dengan LR dan validasi silang yang sama
     (c06.oof_lr_logit tanpa logit CNN), supaya selisihnya milik CNN, bukan pergantian HGB ke LR.

Hanya agregat yang dicetak. Pakai: python -u 13_telaah_jiki.py > telaah_jiki.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c06 = _muat_modul('c06', '06_revisi.py')
c09 = _muat_modul('c09', '09_uji_sebab.py')
c10 = _muat_modul('c10', '10_telaah_lanjut.py')
c12 = _muat_modul('c12', '12_telaah_je.py')
BENIH = range(3)
ULANG_CNN = 2
rng = np.random.default_rng(20261028)
c10.rng = rng
MESIN, CIRI = c10.MESIN, c10.CIRI
GABUNG = 'CNN + pembacaan mesin (LR)'
MESIN_LR = 'pembacaan mesin (LR)'


def boot_beda(y, pa, pb, s, n=c10.N_BOOT):
    """AUROC(pa) - AUROC(pb) dengan set cocokan yang sama diambil ulang untuk keduanya."""
    us = np.unique(s)
    anggota = {k: np.where(s == k)[0] for k in us}
    out = np.empty(n)
    for b in range(n):
        ix = np.concatenate([anggota[k] for k in rng.choice(us, len(us))])
        out[b] = roc_auc_score(y[ix], pa[ix]) - roc_auc_score(y[ix], pb[ix])
    return out


def cnn_per_ulang(X, y, g):
    """Sama dengan c02.oof(X, y, g, ULANG_CNN, 'cnn'), tetapi tiap ulangan juga dikembalikan."""
    ps = []
    for b in range(ULANG_CNN):
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g):
            p[te] = c02.latih_prediksi(X[tr], y[tr], X[te], b)
        ps.append(p)
    return np.mean(ps, axis=0), ps


def bagian_interaksi(pre, calon, kp):
    print('== 1. Beda-dalam-beda riwayat jantung (rancangan P, tanpa CNN)', flush=True)
    print('  kontrol tidak punya catatan komorbiditas: strata hanya memilah kasus, kontrol tetap set cocokannya',
          flush=True)
    sebab = kp.cause_of_cardiac_arrest
    jantung = (sebab == 'cardiac event').to_numpy()
    bukan = (~jantung & sebab.notna() & (sebab != 'unknown')).to_numpy()
    riw = (kp.riwayat == 1).to_numpy()
    h = {}
    for nama, m in [('jr', jantung & riw), ('jt', jantung & ~riw), ('br', bukan & riw), ('bt', bukan & ~riw)]:
        h[nama] = c12.jalankan(pre, kp[m], calon, nama)
    for f in (MESIN, CIRI):
        titik, d = [], []
        for b in BENIH:
            a = {k: roc_auc_score(*h[k][(f, b)][:2]) for k in h}
            titik.append((a['jr'] - a['br']) - (a['jt'] - a['bt']))
            bs = {k: c10.boot_set(*h[k][(f, b)]) for k in h}
            d.append((bs['jr'] - bs['br']) - (bs['jt'] - bs['bt']))
        d = np.concatenate(d)
        print(f'  interaksi riwayat x sebab | {f:<28s} {np.mean(titik):+.3f} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}'
              f'  (per benih {np.round(titik, 3).tolist()})', flush=True)


def bagian_cnn(pre, calon, kp):
    print('\n== 2 dan 3. CNN, pembacaan mesin, dan gabungan, berpasangan atas semua benih pencocokan', flush=True)
    sebab = kp.cause_of_cardiac_arrest
    jantung = (sebab == 'cardiac event').to_numpy()
    bukan = (~jantung & sebab.notna() & (sebab != 'unknown')).to_numpy()
    for nama, m in [('semua sebab', np.ones(len(kp), bool)), ('sebab jantung', jantung),
                    ('bukan jantung (tanpa unknown)', bukan)]:
        kas = kp[m]
        auc = {k: [] for k in ('CNN', MESIN, MESIN_LR, GABUNG)}
        ulang = []
        d_cm, d_gm, d_gl = [], [], []
        for b in BENIH:
            S = c10.susun(pre, kas, calon, b)
            y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
            X = c02.sinyal(S)
            B = c10.fitur(S, X)[MESIN][0]
            pc, ps = cnn_per_ulang(X, y, g)
            del X
            pm = c02.oof(B, y, g, 1, 'hgb')
            pg = c06.oof_lr_logit(np.column_stack([c06.logit(pc), B]), y, g)
            pl = c06.oof_lr_logit(B, y, g)
            for k, p in (('CNN', pc), (MESIN, pm), (MESIN_LR, pl), (GABUNG, pg)):
                auc[k].append(roc_auc_score(y, p))
            ulang += [roc_auc_score(y, p) for p in ps]
            d_cm.append(boot_beda(y, pc, pm, s))
            d_gm.append(boot_beda(y, pg, pm, s))
            d_gl.append(boot_beda(y, pg, pl, s))
            print(f'  {nama:<32s} benih {b}: CNN {auc["CNN"][-1]:.3f}, mesin {auc[MESIN][-1]:.3f}, '
                  f'mesin LR {auc[MESIN_LR][-1]:.3f}, gabungan {auc[GABUNG][-1]:.3f}', flush=True)
        for k in auc:
            print(f'  {nama:<32s} n={len(kas):<4d} {k:<28s} AUROC {np.mean(auc[k]):.3f} '
                  f'[{min(auc[k]):.3f}, {max(auc[k]):.3f}]', flush=True)
        print(f'  {nama:<32s} AUROC CNN per ulangan latih ({len(ulang)}): rerata {np.mean(ulang):.3f}, '
              f'SD {np.std(ulang, ddof=1):.3f}, [{min(ulang):.3f}, {max(ulang):.3f}]', flush=True)
        for lbl, d, a, bb in (('CNN - pembacaan mesin', d_cm, 'CNN', MESIN),
                              ('gabungan - pembacaan mesin', d_gm, GABUNG, MESIN),
                              ('gabungan - pembacaan mesin LR', d_gl, GABUNG, MESIN_LR)):
            d = np.concatenate(d)
            t = np.mean(auc[a]) - np.mean(auc[bb])
            print(f'  {nama:<32s} {lbl:<28s} {t:+.3f} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}', flush=True)


def main():
    pre, calon, kp, _ = c09.data_p()
    kom = pd.read_csv(os.path.join(c02.R, 'study-group', 'comorbidities.csv'))
    kom = kom.groupby('patient_ngsci_id')[c12.RIWAYAT].max()
    kp = kp.copy()
    for r in c12.RIWAYAT:
        kp[r] = kp.patient_ngsci_id.map(kom[r])
    kp['riwayat'] = kp[c12.RIWAYAT].max(axis=1)
    bagian_interaksi(pre, calon, kp)
    bagian_cnn(pre, calon, kp)


if __name__ == '__main__':
    main()
