"""Nightingale arrest-ntuh-ecg — uji langsung "captured" dan selang CNN A -> B (telaah panel JIKI 30 Oktober 2026 (b),
`../Telaah/2026-10-30b-Arrest-JIKI/report.md`, kritis "increment test cannot tell 'not captured' from 'captured
with noise'", mayor "no uncertainty for the CNN's A-to-B drop").

Bagian 1, probe linear, rancangan A (kasus 1-30 dan 30-365 hari, kontrol 30-730 hari, pencocokan 04, benih 0-2,
CNN 2 ulangan/benih seperti 19): sesudah dilatih pada label henti jantung, ciri 128 dimensi (sesudah rerata global)
dipakai ridge untuk menebak tiap dari 16 ukuran 12SL; ridge dan penskalaan dipasang pada lipatan latih, R2 dihitung
dari prediksi OOF. Pembanding: (a) jaringan yang sama tanpa pelatihan (bobot acak, BatchNorm dikalibrasi pada lipatan
latih), batas bawah; (b) kontrol positif, jaringan yang sama dilatih langsung menebak ukuran itu (MSE, target baku),
benih 0 ulangan 0, tiga ukuran saja, batas atas arsitektur ini.

Bagian 2: AUROC CNN dan penurunan A - B per jendela. Bootstrap pasien bersama: satu tarikan atas gabungan pasien
kedua rancangan, jadi kasus yang sama tetap berpasangan; satu ulangan latih acak per tarikan, digabung atas benih.

CPU. Hanya agregat. Pakai: python -u 25_probe_cnn.py > probe_cnn.log
"""
import importlib.util
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c19 = _muat_modul('c19', '19_berpasangan.py')
c02, c03, c10 = c19.c02, c19.c03, c19.c10
rng = c19.rng
UTAMA = ['ventricular_rate', 'qrs_duration', 'qt_corrected']
ALFA = np.logspace(-2, 5, 8)


def ciri_jaringan(m, X, sd):
    m.eval()
    out = []
    with torch.no_grad():
        xe = torch.from_numpy(X)
        for i in range(0, len(xe), 512):
            out.append(m.fitur(xe[i:i + 512] / sd).mean(dim=2).numpy())
    return np.concatenate(out)


def kalibrasi_bn(m, X, sd):
    m.train()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        for i in range(0, len(xt), 64):
            m.fitur(xt[i:i + 64] / sd)


def latih(Xtr, ytr, benih, mse=False):
    """Salinan 02.latih_prediksi yang mengembalikan jaringannya; mse=True untuk kontrol positif."""
    torch.manual_seed(benih)
    sd = Xtr.std() + 1e-6
    xt, yt = torch.from_numpy(Xtr), torch.tensor(ytr, dtype=torch.float32)
    m = c02.Jaringan()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    if mse:
        loss = nn.MSELoss()
    else:
        pos = ytr.mean()
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32))
    for _ in range(c02.EPOCH):
        m.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), c02.BATCH):
            b = perm[i:i + c02.BATCH]
            opt.zero_grad()
            loss(m(xt[b] / sd), yt[b]).backward()
            opt.step()
    return m, sd


def r2_probe(Ztr, Zte, T, tr, te, pred, ada):
    for k in range(T.shape[1]):
        a, b = tr[ada[tr, k]], te[ada[te, k]]
        pred[b, k] = make_pipeline(StandardScaler(), RidgeCV(alphas=ALFA)).fit(Ztr[ada[tr, k]], T[a, k]).predict(
            Zte[ada[te, k]])


def r2_per_ukuran(T, pred, ada):
    return np.array([r2_score(T[ada[:, k], k], pred[ada[:, k], k]) for k in range(T.shape[1])])


def probe(X, y, g, T):
    """Dua ulangan seperti 19.cnn_ulang: mengembalikan prediksi CNN dan R2 ciri terlatih dan acak."""
    ada = ~np.isnan(T)
    ps, r2t, r2a = [], [], []
    for b in range(c19.ULANG_CNN):
        p, pt, pa = np.zeros(len(y)), np.full(T.shape, np.nan), np.full(T.shape, np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g):
            m, sd = latih(X[tr], y[tr], b)
            with torch.no_grad():
                for i in range(0, len(te), 512):
                    p[te[i:i + 512]] = torch.sigmoid(m(torch.from_numpy(X[te[i:i + 512]]) / sd)).numpy()
            r2_probe(ciri_jaringan(m, X[tr], sd), ciri_jaringan(m, X[te], sd), T, tr, te, pt, ada)
            torch.manual_seed(b)
            m0 = c02.Jaringan()
            kalibrasi_bn(m0, X[tr], sd)
            r2_probe(ciri_jaringan(m0, X[tr], sd), ciri_jaringan(m0, X[te], sd), T, tr, te, pa, ada)
        ps.append(p)
        r2t.append(r2_per_ukuran(T, pt, ada))
        r2a.append(r2_per_ukuran(T, pa, ada))
    return ps, np.array(r2t), np.array(r2a)


def kontrol_positif(X, y, g, T, nama):
    hasil = {}
    for u in UTAMA:
        k = nama.index(u)
        ada = ~np.isnan(T[:, k])
        pred = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g):
            tr, te = tr[ada[tr]], te[ada[te]]
            mu, s = T[tr, k].mean(), T[tr, k].std()
            m, sd = latih(X[tr], ((T[tr, k] - mu) / s).astype(np.float32), 0, mse=True)
            m.eval()
            with torch.no_grad():
                pred[te] = m(torch.from_numpy(X[te]) / sd).numpy() * s + mu
        hasil[u] = r2_score(T[ada, k], pred[ada])
    return hasil


def boot_bersama(a, b):
    """a, b = (y, g, ps) rancangan A dan B; kembalikan AUROC A dan B dari tarikan pasien yang sama."""
    ua, aa = c19.unit(a[1])
    ub, ab = c19.unit(b[1])
    us = np.union1d(ua, ub)
    hasil = []
    for _ in range(c19.N_BOOT):
        tarik = rng.choice(us, len(us))
        ia = np.concatenate([aa[k] for k in tarik if k in aa])
        ib = np.concatenate([ab[k] for k in tarik if k in ab])
        hasil.append((roc_auc_score(a[0][ia], a[2][rng.integers(len(a[2]))][ia]),
                      roc_auc_score(b[0][ib], b[2][rng.integers(len(b[2]))][ib])))
    return np.array(hasil)


def main():
    pre, kas, lama, jendela = c19.ab_data()
    nama = list(c03.UKURAN)
    drop = {}
    print('== 1. Probe linear rancangan A: R2 OOF menebak ukuran 12SL dari ciri CNN, 3 benih, 2 ulangan', flush=True)
    for w, (m, sama) in jendela.items():
        rt, ra, sa, auc = [], [], [], []
        for b in c19.BENIH:
            S = c19.susun_ab(pre, kas[m], lama, b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            X = c02.sinyal(S)
            T = S[nama].to_numpy(float)
            ps, r2t, r2a = probe(X, y, g, T)
            rt.append(r2t)
            ra.append(r2a)
            auc.append(np.mean([roc_auc_score(y, p) for p in ps]))
            sa.append((y, g, ps))
            if b == 0:
                kp = kontrol_positif(X, y, g, T, nama)
            del X
            print(f'  A {w} benih {b} selesai', flush=True)
        rt, ra = np.concatenate(rt), np.concatenate(ra)
        print(f'  A {w}: AUROC CNN {np.mean(auc):.3f}; R2 ciri terlatih rerata [min; maks antarulangan] | bobot acak',
              flush=True)
        for k, u in enumerate(nama):
            print(f'    {u:<18s} {rt[:, k].mean():+.3f} [{rt[:, k].min():+.3f}; {rt[:, k].max():+.3f}]  |  '
                  f'{ra[:, k].mean():+.3f}', flush=True)
        print('    kontrol positif (dilatih menebak ukuran): ' + ', '.join(f'{u} {v:+.3f}' for u, v in kp.items()),
              flush=True)
        bs = []
        for b in c19.BENIH:
            S = c19.susun_ab(pre, kas[m], sama, b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            X = c02.sinyal(S)
            ps = c19.cnn_ulang(X, y, g)
            del X
            bs.append(boot_bersama(sa[b], (y, g, ps)))
            print(f'  B {w} benih {b} selesai', flush=True)
        bs = np.concatenate(bs)
        print(f'  AUROC CNN {w}: A {bs[:, 0].mean():.3f} {c10.ci(bs[:, 0])}, B {bs[:, 1].mean():.3f} '
              f'{c10.ci(bs[:, 1])}, P(B<=0,5) {np.mean(bs[:, 1] <= 0.5):.3f}', flush=True)
        drop[w] = bs[:, 0] - bs[:, 1]
    print('\n== 2. Penurunan CNN A - B, 3 benih, bootstrap pasien bersama + ulangan latih', flush=True)
    for w, d in drop.items():
        print(f'  CNN {w:<7s} A - B {np.mean(d):+.3f} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}', flush=True)
    k = drop['1-30'] - drop['30-365']
    print(f'  CNN (A-B | 1-30) - (A-B | 30-365) {np.mean(k):+.3f} {c10.ci(k)}', flush=True)


if __name__ == '__main__':
    main()
