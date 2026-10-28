"""Nightingale arrest-ntuh-ecg — uji berpasangan untuk telaah panel JIKI 28 Oktober 2026
(Riset/EKG/Telaah/2026-10-28-Arrest-JIKI/report.md).

Keberatan yang diuji:
  1. "matching cut 1-30 days but not 30-365" bersandar pada satu benih dan dua IK terpisah ->
     rancangan A/B (pencocokan 04, benih 0-2): selisih A - B per jendela, digabung atas benih, dan
     kontras langsung (A - B | 1-30) - (A - B | 30-365); laju ventrikel mentah dan pengukuran 12SL (LR);
     bootstrap pasien, tiap rancangan diambil ulang sendiri.
  2. judul "heart rate outperforms a CNN" tanpa uji berpasangan -> CNN dilatih ulang (2 ulangan per benih,
     seperti 13) pada rancangan P semua sebab dan bukan jantung (pencocokan 09/10, set cocokan) dan pada
     rancangan B (1-30, 30-365; pencocokan 04). Selisih laju - CNN, pengukuran LR - CNN, mesin LR - CNN
     pada sampel yang sama; tiap tarikan bootstrap mengambil ulang set (P) atau pasien (B) dan memilih
     satu ulangan latih CNN secara acak, sehingga ragam antarulangan ikut masuk; digabung atas benih.

CPU. Hanya agregat. Pakai: python -u 19_berpasangan.py > berpasangan.log
Hanya rancangan B 30-365: python -u 19_berpasangan.py --hanya-b-30-365 > berpasangan_b30365.log
"""
import importlib.util
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


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
BENIH = range(3)
ULANG_CNN = 2
N_BOOT = 1000
rng = np.random.default_rng(20261031)
c10.rng = rng
LAJU, UKUR, MESIN = 'laju ventrikel (mentah)', 'pengukuran 12SL (LR)', 'pembacaan mesin (LR)'


def laju(S):
    v = S.ventricular_rate.to_numpy(float)
    return np.where(np.isnan(v), np.nanmedian(v), v)


def pembanding(S, y, g):
    U = S[c03.UKURAN].to_numpy(float)
    M = np.column_stack([S[list(c06.BENDERA)].to_numpy(float), U])
    return {LAJU: laju(S), UKUR: c06.oof_lr_logit(U, y, g), MESIN: c06.oof_lr_logit(M, y, g)}


def cnn_ulang(X, y, g):
    ps = []
    for b in range(ULANG_CNN):
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g):
            p[te] = c02.latih_prediksi(X[tr], y[tr], X[te], b)
        ps.append(p)
    return ps


def unit(u):
    us = np.unique(u)
    return us, {k: np.where(u == k)[0] for k in us}


def boot(y, u, f, n=N_BOOT):
    """f(ix) dihitung pada tiap tarikan unit (set atau pasien) dengan pengembalian."""
    us, anggota = unit(u)
    return np.array([f(np.concatenate([anggota[k] for k in rng.choice(us, len(us))])) for _ in range(n)])


def beda_cnn(y, u, pa, ps):
    """AUROC(pa) - AUROC(CNN), CNN = satu ulangan latih acak per tarikan."""
    def f(ix):
        pc = ps[rng.integers(len(ps))]
        return roc_auc_score(y[ix], pa[ix]) - roc_auc_score(y[ix], pc[ix])
    return boot(y, u, f)


def ab_data():
    D = c04.tandai_kasus(c02.muat())
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    n0 = len(pre)
    pre = pre.merge(c03.ukuran(), on='ecg_id', how='left').merge(c06.pernyataan(), on='ecg_id', how='left')
    pre = pre.fillna({b: 0 for b in c06.BENDERA})
    assert len(pre) == n0
    kas = pre[pre.y == 1]
    k = pre[(pre.y == 0) & pre.tahun.isin(c04.SAH)]
    lama = k[(k.jarak >= 30) & (k.jarak <= 730)]
    jendela = {'1-30': ((kas.jarak > 1) & (kas.jarak <= 30), k[(k.jarak > 1) & (k.jarak <= 30)]),
               '30-365': ((kas.jarak > 30) & (kas.jarak <= 365), k[(k.jarak > 30) & (k.jarak <= 365)])}
    return pre, kas, lama, jendela


def susun_ab(pre, kk, calon, b):
    ix = np.concatenate([kk.index.to_numpy(), c04.cocokkan(kk, calon, b)])
    S = pre.loc[np.sort(ix)].reset_index(drop=True)
    S['pid'] = S.patient_ngsci_id.astype(str)
    return S


def bagian_ab(pre, kas, lama, jendela):
    print('== 1. Rancangan A/B: selisih A - B per jendela dan kontras antarjendela, 3 benih, bootstrap pasien',
          flush=True)
    bs = {}
    for w, (m, sama) in jendela.items():
        for r, calon in (('A', lama), ('B', sama)):
            for b in BENIH:
                S = susun_ab(pre, kas[m], calon, b)
                y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
                for f, p in pembanding(S, y, g).items():
                    if f == MESIN:
                        continue
                    bs.setdefault((w, r, f), []).append(boot(y, g, lambda ix: roc_auc_score(y[ix], p[ix])))
    for f in (LAJU, UKUR):
        d = {}
        for w in jendela:
            d[w] = np.concatenate([a - b for a, b in zip(bs[(w, 'A', f)], bs[(w, 'B', f)])])
            print(f'  {f:<26s} {w:<7s} A - B {np.mean(d[w]):+.3f} {c10.ci(d[w])}  P(<=0) {np.mean(d[w] <= 0):.3f}',
                  flush=True)
        k = d['1-30'] - d['30-365']
        print(f'  {f:<26s} (A-B | 1-30) - (A-B | 30-365) {np.mean(k):+.3f} {c10.ci(k)}  P(<=0) '
              f'{np.mean(k <= 0):.3f}', flush=True)


def lapor_cnn(nama, hasil):
    auc = {k: np.mean(v) for k, v in hasil['auc'].items()}
    print(f'  {nama:<30s} AUROC: ' + ', '.join(f'{k} {v:.3f}' for k, v in auc.items()), flush=True)
    for f in (LAJU, UKUR, MESIN):
        d = np.concatenate(hasil['beda'][f])
        print(f'  {nama:<30s} {f + " - CNN":<34s} {auc[f] - auc["CNN"]:+.3f} {c10.ci(d)}  P(<=0) '
              f'{np.mean(d <= 0):.3f}', flush=True)


def kumpul(hasil, S, y, u, g):
    X = c02.sinyal(S)
    ps = cnn_ulang(X, y, g)
    del X
    pb = pembanding(S, y, g)
    hasil['auc'].setdefault('CNN', []).append(np.mean([roc_auc_score(y, p) for p in ps]))
    for f, p in pb.items():
        hasil['auc'].setdefault(f, []).append(roc_auc_score(y, p))
        hasil['beda'].setdefault(f, []).append(beda_cnn(y, u, p, ps))


def bagian_cnn_p():
    print('\n== 2. Rancangan P: pembanding - CNN berpasangan (bootstrap set + ulangan latih), 3 benih', flush=True)
    pre, calon, kp, _ = c09.data_p()
    sebab = kp.cause_of_cardiac_arrest
    bukan = ((sebab != 'cardiac event') & sebab.notna() & (sebab != 'unknown')).to_numpy()
    for nama, m in [('P semua sebab', np.ones(len(kp), bool)), ('P bukan jantung (tanpa unknown)', bukan)]:
        hasil = {'auc': {}, 'beda': {}}
        for b in BENIH:
            S = c10.susun(pre, kp[m], calon, b)
            y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
            kumpul(hasil, S, y, s, g)
            print(f'  {nama} benih {b} selesai', flush=True)
        lapor_cnn(nama, hasil)


def bagian_cnn_b(pre, kas, jendela, hanya=None):
    print('\n== 3. Rancangan B: pembanding - CNN berpasangan (bootstrap pasien + ulangan latih), 3 benih',
          flush=True)
    for w, (m, sama) in jendela.items():
        if hanya and w != hanya:
            continue
        hasil = {'auc': {}, 'beda': {}}
        for b in BENIH:
            S = susun_ab(pre, kas[m], sama, b)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            kumpul(hasil, S, y, g, g)
            print(f'  B {w} benih {b} selesai', flush=True)
        lapor_cnn(f'B {w}', hasil)


def main():
    pre, kas, lama, jendela = ab_data()
    if '--hanya-b-30-365' in sys.argv:
        # Lari 28 Okt dihentikan sistem (memori habis) sesudah benih 0 jendela ini; bagian lain sudah
        # di berpasangan.log. Keadaan rng berbeda dari lari penuh, jadi angka bootstrap tidak identik.
        bagian_cnn_b(pre, kas, jendela, '30-365')
        return
    bagian_ab(pre, kas, lama, jendela)
    bagian_cnn_p()
    bagian_cnn_b(pre, kas, jendela)


if __name__ == '__main__':
    main()
