"""Nightingale arrest-ntuh-ecg — satu himpunan lari CNN tetap untuk semua tambahan, penurunan, dan pembanding (telaah panel
JIKI 30 Oktober 2026 (c), `../Telaah/2026-10-30c-Arrest-JIKI/report.md`, kritis metode X1 "increment cannot separate 'not
used' from 'used noisily'" dan X2 "increments are measured against the weakest reported CNN run").

Tabel 4 mencampur AUROC CNN skrip 24 dengan tambahan skrip 26, dan `laju_sinyal.log` tidak mencetak AUROC CNN lari itu.
Di sini semua dihitung dari prediksi OOF yang sama: 5 sampel (A 1-30, A 30-365, P, B 1-30, B 30-365) × 3 benih × 5 ulangan.

  --tahap latih: latih CNN, simpan OOF per ulangan di luar repo (ARREST_OOF, bawaan C:/Data/Nightingale/_oof_arrest28).
                 Bisa diulang: ulangan yang sudah ada dilewati, jadi lari yang dihentikan sistem bisa diteruskan.
  --tahap lapor: hitung agregat dari OOF yang ada, tanpa melatih.

Dicetak per sampel: AUROC CNN per ulangan dan ansambel (rerata logit); tambahan laju sinyal, 13 ukuran, 16 ukuran di atas
CNN (satu ulangan acak per tarikan, jadi ragam antarulangan ikut masuk selang) dan di atas ansambel; tambahan terbalik
(laju + CNN) - laju saja; AUROC CNN di dalam kuintil laju (rerata tertimbang n1·n0) dan porsi kelebihan atas 0,5 yang
tersisa; ukuran LR - CNN; penurunan A - B.

Aturan bacaan, ditulis sebelum hasil ada, per sampel, memakai ansambel:
  tambahan laju, batas bawah <= 0                       -> tidak ada bukti laju menambah; klaim apa pun tentang laju diam
  tambahan laju, batas bawah > 0, porsi tersisa < 0,5   -> sinyal CNN sebagian besar laju yang terekam bising ("used noisily")
  tambahan laju, batas bawah > 0, porsi tersisa >= 0,5  -> laju tidak terserap dan sinyal CNN bukan laju ("not exploited")
Porsi tidak dibaca bila AUROC ansambel <= 0,55: kelebihan atas 0,5 terlalu kecil untuk dibagi.

CPU. Hanya agregat. Pakai:
  python -u 28_lari_tetap.py --tahap latih > lari_tetap_latih.log
  python -u 28_lari_tetap.py --tahap lapor > lari_tetap.log
"""
import argparse
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


c26 = _muat_modul('c26', '26_laju_sinyal.py')
c19, c02, c03, c06, c09, c10 = c26.c19, c26.c02, c26.c03, c26.c06, c26.c09, c26.c10
rng = c19.rng
ULANG = 5
OOF = os.environ.get('ARREST_OOF', 'C:/Data/Nightingale/_oof_arrest28')
SAMPEL = ['A 1-30', 'A 30-365', 'P', 'B 1-30', 'B 30-365']
LAB = ('CNN', '+ laju sinyal', '+ 13 ukuran', '+ 16 ukuran', '(laju + CNN) - laju', 'ukuran LR - CNN', 'CNN berstrata laju',
       'laju - CNN', '(laju + 13) - laju')
STRATA = LAB.index('CNN berstrata laju')


def susunan():
    """nama sampel -> fungsi benih -> (S, unit bootstrap)."""
    pre, kas, lama, jendela = c19.ab_data()
    pre_p, calon_p, kp, _ = c09.data_p()

    def ab(m, calon):
        def f(b):
            S = c19.susun_ab(pre, kas[m], calon, b)
            return S, S.pid.to_numpy()
        return f

    def p(b):
        S = c10.susun(pre_p, kp, calon_p, b)
        return S, S.set.to_numpy()

    h = {'P': p}
    for w, (m, sama) in jendela.items():
        h[f'A {w}'] = ab(m, lama)
        h[f'B {w}'] = ab(m, sama)
    return h


def jalur(nama, b, k=None):
    n = nama.replace(' ', '_')
    return os.path.join(OOF, f'{n}_b{b}_laju.npy' if k is None else f'{n}_b{b}_u{k}.npy')


def simpan(f, a):
    np.save(f + '.tmp.npy', a)
    os.replace(f + '.tmp.npy', f)


def latih(h):
    os.makedirs(OOF, exist_ok=True)
    for nama in SAMPEL:
        for b in c19.BENIH:
            kurang = [k for k in range(ULANG) if not os.path.exists(jalur(nama, b, k))]
            if not kurang and os.path.exists(jalur(nama, b)):
                continue
            S, _ = h[nama](b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            X = c02.sinyal(S)
            simpan(jalur(nama, b), c26.laju_sinyal(X))
            for k in kurang:
                p = np.zeros(len(y))
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=k).split(np.zeros(len(y)), y, g):
                    p[te] = c02.latih_prediksi(X[tr], y[tr], X[te], k)
                simpan(jalur(nama, b, k), p)
                print(f'  {nama} benih {b} ulangan {k} selesai, AUROC {roc_auc_score(y, p):.3f}', flush=True)
            del X


def auroc_strata(y, z, kl):
    """AUROC di dalam tiap strata, rerata tertimbang n1·n0."""
    t, w = 0.0, 0.0
    for k in np.unique(kl):
        m = kl == k
        n1, n0 = int(y[m].sum()), int(m.sum() - y[m].sum())
        if n1 and n0:
            a = (rankdata(z[m])[y[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            t += a * n1 * n0
            w += n1 * n0
    return t / w


def rakit(S, y, g, ps, s):
    """Semua skor satu benih; set skor = satu daftar per ulangan, ditambah ansambel di ujung."""
    lg = [c06.logit(p) for p in ps]
    lg.append(np.mean(lg, axis=0))
    U = S[c03.UKURAN].to_numpy(float)
    sisa = S[[k for k in c03.UKURAN if k not in c26.KELUARGA_LAJU]].to_numpy(float)
    return {'cnn': lg,
            'laju': [c06.oof_lr_logit(np.column_stack([a, s]), y, g) for a in lg],
            'u13': [c06.oof_lr_logit(np.column_stack([a, sisa]), y, g) for a in lg],
            'u16': [c06.oof_lr_logit(np.column_stack([a, U]), y, g) for a in lg],
            'lr': c06.oof_lr_logit(U, y, g),
            'laju0': c06.oof_lr_logit(s[:, None], y, g),
            'laju13': c06.oof_lr_logit(np.column_stack([s, sisa]), y, g)}


def nilai(y, r, kl, ix, k):
    """Baris LAB pada tarikan ix, skor ulangan k (ansambel bila k = -1)."""
    yi, a = y[ix], roc_auc_score(y[ix], r['cnn'][k][ix])
    laju, l0 = roc_auc_score(yi, r['laju'][k][ix]), roc_auc_score(yi, r['laju0'][ix])
    return (a, laju - a, roc_auc_score(yi, r['u13'][k][ix]) - a, roc_auc_score(yi, r['u16'][k][ix]) - a,
            laju - l0, roc_auc_score(yi, r['lr'][ix]) - a, auroc_strata(yi, r['cnn'][k][ix], kl[ix]),
            l0 - a, roc_auc_score(yi, r['laju13'][ix]) - l0)


def lapor_sampel(nama, h):
    per, ens, tiap, pa = [], [], [], {}
    titik, laju_st, laju0 = [], [], []
    for b in c19.BENIH:
        S, u = h[nama](b)
        y, g = S.y.to_numpy().astype(int), S.pid.to_numpy()
        ps = [np.load(jalur(nama, b, k)) for k in range(ULANG)]
        s = c26.isi(np.load(jalur(nama, b)))
        r = rakit(S, y, g, ps, s)
        kl = np.searchsorted(np.quantile(s, [.2, .4, .6, .8]), s)
        semua = np.arange(len(y))
        tiap.append([nilai(y, r, kl, semua, k) for k in range(ULANG)])
        titik.append(nilai(y, r, kl, semua, -1))
        laju_st.append(auroc_strata(y, s, kl))
        laju0.append(roc_auc_score(y, r['laju0']))
        pa[b] = (y, g, r['cnn'][:ULANG])
        us, anggota = c19.unit(u)
        tarik = [np.concatenate([anggota[j] for j in rng.choice(us, len(us))]) for _ in range(c19.N_BOOT)]
        per.append(np.array([nilai(y, r, kl, ix, rng.integers(ULANG)) for ix in tarik]))
        ens.append(np.array([nilai(y, r, kl, ix, -1) for ix in tarik]))
        print(f'  [{nama} benih {b} dihitung]', flush=True)
    tiap, titik, per, ens = np.array(tiap), np.array(titik), np.concatenate(per), np.concatenate(ens)
    print(f'\n== {nama}', flush=True)
    print('  AUROC CNN per ulangan (rerata benih): ' + ', '.join(f'{v:.3f}' for v in tiap[:, :, 0].mean(0)) +
          f'; per benih: ' + ', '.join(f'{v:.3f}' for v in tiap[:, :, 0].mean(1)), flush=True)
    print(f'  laju saja (LR) {np.mean(laju0):.3f}; laju berstrata (sisa di dalam kuintil) {np.mean(laju_st):.3f}', flush=True)
    for j, lab in enumerate(LAB):
        print(f'  {lab:<20s} satu lari {tiap[:, :, j].mean():+.3f} {c10.ci(per[:, j])} rentang ulangan '
              f'[{tiap[:, :, j].mean(0).min():+.3f}, {tiap[:, :, j].mean(0).max():+.3f}] | ansambel '
              f'{titik[:, j].mean():+.3f} {c10.ci(ens[:, j])}', flush=True)
    for nm, a, st in (('satu lari', tiap[:, :, 0].mean(), tiap[:, :, STRATA].mean()),
                      ('ansambel', titik[:, 0].mean(), titik[:, STRATA].mean())):
        print(f'  porsi kelebihan tersisa, {nm}: ({st:.3f} - 0,5) / ({a:.3f} - 0,5) = {(st - .5) / (a - .5):.2f}', flush=True)
    return pa


def boot_bersama(a, b):
    """Seperti 25: tarikan pasien yang sama untuk A dan B, satu ulangan acak per tarikan."""
    ua, aa = c19.unit(a[1])
    ub, ab = c19.unit(b[1])
    us = np.union1d(ua, ub)
    hasil = []
    for _ in range(c19.N_BOOT):
        tarik = rng.choice(us, len(us))
        ia = np.concatenate([aa[k] for k in tarik if k in aa])
        ib = np.concatenate([ab[k] for k in tarik if k in ab])
        hasil.append((roc_auc_score(a[0][ia], a[2][rng.integers(ULANG)][ia]),
                      roc_auc_score(b[0][ib], b[2][rng.integers(ULANG)][ib])))
    return np.array(hasil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tahap', choices=['latih', 'lapor'], required=True)
    a = ap.parse_args()
    h = susunan()
    if a.tahap == 'latih':
        latih(h)
        return
    ada = [n for n in SAMPEL if all(os.path.exists(jalur(n, b, k)) for b in c19.BENIH for k in range(ULANG))]
    print(f'sampel lengkap: {ada}', flush=True)
    pa = {n: lapor_sampel(n, h) for n in ada}
    print('\n== Penurunan CNN A - B, 3 benih, 5 ulangan, bootstrap pasien bersama', flush=True)
    for w in ('1-30', '30-365'):
        if f'A {w}' in pa and f'B {w}' in pa:
            d = np.concatenate([boot_bersama(pa[f'A {w}'][b], pa[f'B {w}'][b]) for b in c19.BENIH])
            print(f'  {w:<7s} A {d[:, 0].mean():.3f} {c10.ci(d[:, 0])}, B {d[:, 1].mean():.3f} {c10.ci(d[:, 1])}, '
                  f'A - B {(d[:, 0] - d[:, 1]).mean():+.3f} {c10.ci(d[:, 0] - d[:, 1])}', flush=True)


if __name__ == '__main__':
    main()
