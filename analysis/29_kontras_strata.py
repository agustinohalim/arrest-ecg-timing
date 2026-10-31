"""Nightingale arrest-ntuh-ecg — selang untuk kontras yang menopang klaim "sinyal di luar laju" (telaah panel JIKI
31 Oktober 2026 (b), `../Telaah/2026-10-31b-Arrest-JIKI/report.md`, kritis metode M1, mayor M3 dan M5).

Tanpa melatih: memakai OOF skrip 28 (75 CNN, ansambel = rerata logit lima ulangan) dan laju sinyal yang tersimpan.

  1. Rancangan A dan B per jendela, bootstrap pasien bersama seperti 25 dan 28, per tarikan: AUROC ansambel, AUROC
     ansambel di dalam kuintil laju, AUROC laju sinyal, AUROC laju di dalam kuintil (lantai), porsi tersisa, serta
     penurunan A - B masing-masing dan selisih penurunan (berstrata - laju). Titik potong kuintil ditetapkan pada satu
     sampel utuh per benih, tidak diulang di dalam tarikan.
  2. Rancangan P dan B: yang sama tanpa penurunan.
  3. Per sampel A 1-30, A 30-365, P: AUROC |d| kasus lawan kontrol (d = laju 12SL - laju sinyal), dan AUROC laju 12SL
     sesudah EKG dengan |d| > 5 dibuang.

Aturan bacaan, ditulis sebelum hasil ada:
  penurunan berstrata, selang mengandung 0        -> tidak boleh ditulis sinyal di luar laju hilang; hanya "tidak terdeteksi"
  penurunan berstrata, batas bawah > 0            -> sinyal di luar laju turun sesudah pencocokan
  selisih penurunan, selang mengandung 0          -> tidak bisa dikatakan sinyal di luar laju turun lebih dari laju; semua
                                                     sinyal mengecil bersama
  AUROC |d|, selang tidak mengandung 0,5          -> besar ketidaksepakatan berbeda menurut kelompok; "clears the vendor's
                                                     rate algorithm" tidak boleh dipertahankan, hanya kesamaan AUROC
  AUROC laju sesudah buang |d| > 5, selang pasangan -> bila berubah kurang dari 0,02, EKG yang tidak sepakat tidak
                                                     menggerakkan hasil laju

CPU. Hanya agregat. Pakai: python -u 29_kontras_strata.py > kontras_strata.log
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


c28 = _muat_modul('c28', '28_lari_tetap.py')
c19, c06, c10, c26 = c28.c19, c28.c06, c28.c10, c28.c26
rng = c19.rng
BARIS = ('ansambel', 'ansambel berstrata', 'laju', 'laju berstrata', 'porsi tersisa')


def muat(nama, h, b):
    S, u = h[nama](b)
    y = S.y.to_numpy().astype(int)
    e = np.mean([c06.logit(np.load(c28.jalur(nama, b, k))) for k in range(c28.ULANG)], axis=0)
    mentah = np.load(c28.jalur(nama, b))
    s = c26.isi(mentah)
    kl = np.searchsorted(np.quantile(s, [.2, .4, .6, .8]), s)
    return {'S': S, 'u': u, 'y': y, 'e': e, 's': s, 'kl': kl, 'mentah': mentah}


def ukur(d, ix):
    y, e, s, kl = d['y'][ix], d['e'][ix], d['s'][ix], d['kl'][ix]
    a, st = roc_auc_score(y, e), c28.auroc_strata(y, e, kl)
    return a, st, roc_auc_score(y, s), c28.auroc_strata(y, s, kl), (st - .5) / (a - .5)


def titik(d):
    return ukur(d, np.arange(len(d['y'])))


def tulis(nama, t, b):
    print(f'== {nama}', flush=True)
    for j, lab in enumerate(BARIS):
        print(f'  {lab:<20s} {t[:, j].mean():.3f} {c10.ci(b[:, j])}', flush=True)


def bagian_jendela(w, h):
    tA, tB, boot = [], [], []
    for b in c19.BENIH:
        A, B = muat(f'A {w}', h, b), muat(f'B {w}', h, b)
        tA.append(titik(A))
        tB.append(titik(B))
        ua, aa = c19.unit(A['u'])
        ub, ab = c19.unit(B['u'])
        us = np.union1d(ua, ub)
        for _ in range(c19.N_BOOT):
            tarik = rng.choice(us, len(us))
            ia = np.concatenate([aa[k] for k in tarik if k in aa])
            ib = np.concatenate([ab[k] for k in tarik if k in ab])
            boot.append((ukur(A, ia), ukur(B, ib)))
        print(f'  [{w} benih {b} dihitung]', flush=True)
    tA, tB, boot = np.array(tA), np.array(tB), np.array(boot)
    tulis(f'A {w}', tA, boot[:, 0])
    tulis(f'B {w}', tB, boot[:, 1])
    selisih = boot[:, 0] - boot[:, 1]
    t = tA - tB
    print(f'== Penurunan A - B, {w}', flush=True)
    for j, lab in enumerate(BARIS[:4]):
        print(f'  {lab:<20s} {t[:, j].mean():+.3f} {c10.ci(selisih[:, j])}', flush=True)
    print(f'  penurunan berstrata - penurunan laju {(t[:, 1] - t[:, 2]).mean():+.3f} '
          f'{c10.ci(selisih[:, 1] - selisih[:, 2])}', flush=True)


def bagian_p(h):
    t, boot = [], []
    for b in c19.BENIH:
        P = muat('P', h, b)
        t.append(titik(P))
        us, anggota = c19.unit(P['u'])
        boot += [ukur(P, np.concatenate([anggota[j] for j in rng.choice(us, len(us))])) for _ in range(c19.N_BOOT)]
    tulis('P', np.array(t), np.array(boot))


def bagian_d(h):
    print('== |d| kasus lawan kontrol; laju 12SL sesudah |d| > 5 dibuang', flush=True)
    for nama in ('A 1-30', 'A 30-365', 'P'):
        t, boot, buang = [], [], []
        for b in c19.BENIH:
            D = muat(nama, h, b)
            v = D['S'].ventricular_rate.to_numpy(float)
            ok = ~(np.isnan(v) | np.isnan(D['mentah']))
            y, u, v = D['y'][ok], D['u'][ok], v[ok]
            ad = np.abs(v - D['mentah'][ok])
            sepakat = ad <= 5
            t.append((roc_auc_score(y, ad), roc_auc_score(y, v), roc_auc_score(y[sepakat], v[sepakat]),
                      1 - sepakat[y == 1].mean(), 1 - sepakat[y == 0].mean()))

            def f(ix):
                k = sepakat[ix]
                a, c = roc_auc_score(y[ix], v[ix]), roc_auc_score(y[ix][k], v[ix][k])
                return roc_auc_score(y[ix], ad[ix]), c - a

            us, anggota = c19.unit(u)
            boot += [f(np.concatenate([anggota[j] for j in rng.choice(us, len(us))])) for _ in range(c19.N_BOOT)]
        t, boot = np.array(t), np.array(boot)
        print(f'  {nama:<9s} AUROC |d| {t[:, 0].mean():.3f} {c10.ci(boot[:, 0])}; laju 12SL {t[:, 1].mean():.3f}, sesudah '
              f'buang {t[:, 2].mean():.3f}, beda {(t[:, 2] - t[:, 1]).mean():+.3f} {c10.ci(boot[:, 1])}; '
              f'|d| > 5: kasus {t[:, 3].mean():.1%}, kontrol {t[:, 4].mean():.1%}', flush=True)


def main():
    h = c28.susunan()
    for w in ('1-30', '30-365'):
        bagian_jendela(w, h)
    bagian_p(h)
    bagian_d(h)


if __name__ == '__main__':
    main()
