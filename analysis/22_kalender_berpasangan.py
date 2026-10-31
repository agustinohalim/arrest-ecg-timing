"""Nightingale arrest-ntuh-ecg — subset kalender berpasangan, untuk telaah ulang panel JIKI 28 Oktober 2026 (d)
(Riset/EKG/Telaah/2026-10-28d-Arrest-JIKI/report.md, kritis "LR pengukuran 0,507 pada 47 kasus 2015-2017").

Pertanyaan: pada kasus yang setahun kalender dengan kontrol (EKG 2015-2017), apakah hanya LR pengukuran yang
jatuh, atau model gelombang ikut jatuh? Kalau semua jatuh, kalender tidak berpihak pada pembanding sederhana.

Sampel: rancangan P semua sebab dan B 30-365, kasus 2015-2017, benih pencocokan 0-2 (seperti 21).
Model pada sampel yang sama: laju ventrikel mentah; LR pengukuran C = 1 (seperti 06), 0,1, dan 0,01;
CNN 02 (2 ulangan latih per benih, CPU); MiniRocket (seperti 20). Dilaporkan AUROC dan selisih berpasangan
pembanding - CNN dan pembanding - MiniRocket, bootstrap set (P) atau pasien (B), digabung atas benih.

CPU; perlu aeon (~/venv-aeon). Hanya agregat.
Pakai: ~/venv-aeon/Scripts/python.exe -u 22_kalender_berpasangan.py > kalender_berpasangan.log
"""
import importlib.util
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c20 = _muat_modul('c20', '20_pembanding_kuat.py')
c19 = c20.c19
c02, c03, c09, c10 = c19.c02, c19.c03, c19.c09, c19.c10
c19.rng = c10.rng = np.random.default_rng(20261103)
SAH = ['2015', '2016', '2017']


def lr_c(U, y, g, c):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(U, y, g):
        p[te] = c02.make_pipeline(c02.SimpleImputer(), c02.StandardScaler(),
                                  LogisticRegression(C=c, max_iter=4000)).fit(U[tr], y[tr]).predict_proba(U[te])[:, 1]
    return p


def sampel():
    pre, calon, kp, _ = c09.data_p()
    k = kp[kp.tahun.isin(SAH)]
    for b in c19.BENIH:
        S = c10.susun(pre, k, calon, b)
        yield f'P semua sebab, kasus 2015-2017 ({len(k)})', S, S.set.to_numpy(), b
    preb, kas, _, jendela = c19.ab_data()
    m, sama = jendela['30-365']
    k = kas[m & kas.tahun.isin(SAH)]
    for b in c19.BENIH:
        S = c19.susun_ab(preb, k, sama, b)
        yield f'B 30-365, kasus 2015-2017 ({len(k)})', S, S.pid.to_numpy(), b


def main():
    print('== Subset kalender 2015-2017: pembanding lawan CNN dan MiniRocket, berpasangan, 3 benih', flush=True)
    hasil = {}
    for nama, S, u, b in sampel():
        y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
        U = S[c03.UKURAN].to_numpy(float)
        X = c02.sinyal(S)
        ps = c19.cnn_ulang(X, y, g)
        roc = c20.rocket(X, y, g, b)
        del X
        pb = {'laju': c19.laju(S), 'ukuran LR C=1': lr_c(U, y, g, 1.0), 'ukuran LR C=0,1': lr_c(U, y, g, 0.1),
              'ukuran LR C=0,01': lr_c(U, y, g, 0.01)}
        h = hasil.setdefault(nama, {'auc': {}, 'cnn': {}, 'roc': {}})
        h['auc'].setdefault('CNN', []).append(np.mean([roc_auc_score(y, p) for p in ps]))
        h['auc'].setdefault('MiniRocket', []).append(roc_auc_score(y, roc))
        for f, p in pb.items():
            h['auc'].setdefault(f, []).append(roc_auc_score(y, p))
            h['cnn'].setdefault(f, []).append(c19.beda_cnn(y, u, p, ps))
            h['roc'].setdefault(f, []).append(
                c19.boot(y, u, lambda ix, p=p: roc_auc_score(y[ix], p[ix]) - roc_auc_score(y[ix], roc[ix])))
        print(f'  {nama} benih {b} selesai', flush=True)
    for nama, h in hasil.items():
        auc = {k: np.mean(v) for k, v in h['auc'].items()}
        print(f'  {nama}: AUROC ' + ', '.join(f'{k} {v:.3f}' for k, v in auc.items()), flush=True)
        for lawan, kunci in (('CNN', 'cnn'), ('MiniRocket', 'roc')):
            for f, d in h[kunci].items():
                d = np.concatenate(d)
                print(f'    {f:<18s} - {lawan:<10s} {auc[f] - auc[lawan]:+.3f} {c10.ci(d)}  P(<=0) '
                      f'{np.mean(d <= 0):.3f}', flush=True)


if __name__ == '__main__':
    main()
