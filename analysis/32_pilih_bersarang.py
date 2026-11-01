"""Nightingale arrest-ntuh-ecg — pengukuran tunggal terbaik dipilih di dalam lipatan (telaah panel JITIKA 1 November 2026,
`../Telaah/2026-11-01-Arrest-JITIKA/report.md`, kritis "Heart rate was chosen on the evaluation data, then compared with
out-of-fold models").

Laju jantung dipilih dari 16 pengukuran 12SL pada sampel yang sama, lalu AUROC-nya dibandingkan dengan CNN out-of-fold. Di
sini pemilihan itu masuk ke validasi: di tiap lipatan latih, dari 16 pengukuran dipilih yang AUROC-nya (arah bebas) paling jauh
dari 0,5; arah, rerata, dan simpangan baku dipasang di lipatan latih; hanya pengukuran itu yang menilai lipatan uji. Lipatan
berkelompok per pasien. Skor OOF digabung lalu AUROC dihitung. Titik: rerata 3 benih × 5 pembagian. Selang: pada tiap tarikan
bootstrap (unit sama dengan skrip 28) pemilihan diulang penuh dengan pembagian acak per pasien, jadi selang memuat ketidakpastian
pemilihan. Pembanding: ansambel CNN skrip 28 (tidak dilatih ulang; lipatannya sendiri).

Juga dicetak: AUROC ke-16 pengukuran (arah bebas), seberapa sering laju terpilih.

Aturan bacaan, ditulis sebelum hasil ada, per sampel:
  terpilih - ansambel, batas bawah > 0   -> "melebihi" boleh dipakai
  terpilih - ansambel, selang memuat 0   -> hanya "tidak di bawah"; kata "exceeded" ditarik untuk sampel itu
  terpilih - ansambel, batas atas < 0    -> klaim laju di sampel itu ditarik; CNN di atas pengukuran tunggal terpilih
Laju mentah - terpilih dicetak sebagai ukuran optimisme pemilihan; tidak ada ambang.

CPU. Tanpa melatih CNN. Hanya agregat. Pakai: python -u 32_pilih_bersarang.py > pilih_bersarang.log
"""
import importlib.util
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c28 = _muat_modul('c28', '28_lari_tetap.py')
c19, c03, c06, c10, c26 = c28.c19, c28.c03, c28.c06, c28.c10, c28.c26
ULANG, N_BOOT = c28.ULANG, 1000
rng = np.random.default_rng(32)


def auroc_kolom(X, y):
    r = rankdata(X, axis=0)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def pilih_nilai(Xtr, ytr, Xte):
    """Skor lipatan uji dari satu pengukuran yang dipilih di lipatan latih; indeks terpilih."""
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    a = auroc_kolom(Xtr, ytr)
    j = int(np.argmax(np.abs(a - .5)))
    sd = Xtr[:, j].std() or 1.0
    return np.sign(a[j] - .5) * (Xte[:, j] - Xtr[:, j].mean()) / sd, j


def oof_bersarang(U, y, lipat):
    z, pil = np.zeros(len(y)), []
    for k in np.unique(lipat):
        tr, te = lipat != k, lipat == k
        z[te], j = pilih_nilai(U[tr], y[tr], U[te])
        pil.append(j)
    return z, pil


def lipat_pasien(g, y_lip, seed):
    lipat = np.zeros(len(g), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y_lip, g)):
        lipat[te] = k
    return lipat


def lipat_acak(g):
    gu, inv = np.unique(g, return_inverse=True)
    return rng.integers(5, size=len(gu))[inv]


def lapor(nama, h):
    nm = c03.UKURAN
    tunggal, titik, boot, hitung, tb = [], [], [], {}, 0
    for b in c19.BENIH:
        S, u = h[nama](b)
        y, g = S.y.to_numpy().astype(int), S.pid.to_numpy()
        U = S[nm].to_numpy(float)
        ens = np.mean([c06.logit(np.load(c28.jalur(nama, b, k))) for k in range(ULANG)], axis=0)
        s = c26.isi(np.load(c28.jalur(nama, b)))
        arah = np.sign(roc_auc_score(y, s) - .5)
        U0 = np.where(np.isnan(U), np.nanmedian(U, axis=0), U)
        tunggal.append(auroc_kolom(U0, y))
        tp, ae, al = [], roc_auc_score(y, ens), roc_auc_score(y, arah * s)
        for k in range(ULANG):
            lipat = lipat_pasien(g, y, k)
            z, pil = oof_bersarang(U, y, lipat)
            tp.append(roc_auc_score(y, z))
            for j in pil:
                hitung[nm[j]] = hitung.get(nm[j], 0) + 1
            tb += len(pil)
        titik.append((np.mean(tp), ae, al))
        us, anggota = c19.unit(u)
        for _ in range(N_BOOT):
            ix = np.concatenate([anggota[j] for j in rng.choice(us, len(us))])
            yi = y[ix]
            lipat = lipat_acak(g[ix])
            z, _p = oof_bersarang(U[ix], yi, lipat)
            boot.append((roc_auc_score(yi, z), roc_auc_score(yi, ens[ix]), roc_auc_score(yi, arah * s[ix])))
        print(f'  [{nama} benih {b} dihitung]', flush=True)
    t, d = np.mean(titik, axis=0), np.array(boot)
    print(f'\n== {nama}', flush=True)
    print(f'  terpilih di dalam lipatan {t[0]:.3f} {c10.ci(d[:, 0])}; ansambel {t[1]:.3f} {c10.ci(d[:, 1])}; '
          f'laju sinyal mentah {t[2]:.3f} {c10.ci(d[:, 2])}', flush=True)
    print(f'  terpilih - ansambel {t[0] - t[1]:+.3f} {c10.ci(d[:, 0] - d[:, 1])}', flush=True)
    print(f'  laju mentah - ansambel {t[2] - t[1]:+.3f} {c10.ci(d[:, 2] - d[:, 1])}', flush=True)
    print(f'  laju mentah - terpilih (optimisme) {t[2] - t[0]:+.3f} {c10.ci(d[:, 2] - d[:, 0])}', flush=True)
    print('  terpilih: ' + ', '.join(f'{k} {v / tb:.0%}' for k, v in sorted(hitung.items(), key=lambda x: -x[1])), flush=True)
    a = np.mean(tunggal, axis=0)
    print('  16 pengukuran, AUROC arah bebas: ' + '; '.join(f'{nm[j]} {max(a[j], 1 - a[j]):.3f}'
                                                            for j in np.argsort(-np.abs(a - .5))), flush=True)


def main():
    h = c28.susunan()
    for n in c28.SAMPEL:
        lapor(n, h)


if __name__ == '__main__':
    main()
