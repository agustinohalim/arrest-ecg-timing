"""Nightingale arrest-ntuh-ecg — analisis sesudah telaah ulang panel 10 Oktober 2026
(Riset/EKG/Telaah/2026-10-10/report.md), untuk versi Naskah_Arrest_Resuscitation.md.

Keberatan yang diuji:
  1. [kritis] sebab jantung dicatat sesudah resusitasi dan bisa memakai riwayat jantung yang juga
     tampak di EKG lama (melingkar) -> subset menurut irama awal saja, tanpa label sebab:
     shockable (VT/VF) lawan non-shockable (PEA, asistol), pada kasus kunjungan terdahulu 2-730 hari.
     Ditambah tabulasi silang sebab x irama.
  2. nol permutasi memecah set yang dicocokkan -> permutasi label di DALAM tiap set cocokan
     (satu kasus + kontrolnya), sehingga setiap set tetap berisi satu "kasus".
  3. bootstrap mengabaikan set cocokan -> bootstrap set cocokan (kasus dan kontrolnya bersama).
  4. permutasi CNN hanya pada sebab jantung -> permutasi dalam set untuk semua sebab, sebab jantung,
     dan sebab bukan jantung (5 kali masing-masing).
  5. lokasi henti bercampur -> selisih sebab jantung - bukan jantung di IHCA dan OHCA terpisah
     (penilaian ulang OOF pada set yang kasusnya di lokasi itu).
  6. pembacaan mesin tanpa usia/jk di rancangan P -> model pembacaan mesin + usia + jk.
  7. rancangan C didorong EKG infark akut yang sudah dikenali -> AUROC CNN dan pembacaan mesin
     tanpa EKG berbendera infark akut/cedera (kasus dan kontrol), dan selisihnya.

Pencocokan sama persis dengan 09 (cocok_dekat; benih 0-2), dengan pencatatan set.
Data tetap di C:/Data/Nightingale; hanya agregat yang dicetak.
Pakai: python -u 10_telaah_lanjut.py > telaah_lanjut.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
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
BENIH = range(3)
RASIO = c09.RASIO
N_PERM = 100
N_PERM_CNN = 5
N_BOOT = 1000
rng = np.random.default_rng(20261011)
MESIN = 'pembacaan mesin (HGB)'
MESIN_DEMO = 'pembacaan mesin + usia/jk (HGB)'
CIRI = 'ciri sinyal sederhana (HGB)'
CNN = 'CNN 12 sadapan'


def cocok_set(kas, calon, benih):
    """Sama dengan c09.cocok_dekat, tetapi juga mengembalikan indeks kasus pemilik tiap kontrol."""
    rc = np.random.default_rng(60000 + benih)
    sk = (kas.usia // 10).fillna(-1).astype(int).astype(str) + kas.pria.astype(str)
    sc = (calon.usia // 10).fillna(-1).astype(int).astype(str) + calon.pria.astype(str)
    lc = np.log(calon.jarak.to_numpy())
    pilih, pemilik = [], []
    for s in sk.unique():
        sisa = list(np.where((sc == s).to_numpy())[0])
        ik = kas.index[(sk == s).to_numpy()].to_numpy()
        for i in rc.permutation(ik):
            if not sisa:
                continue
            d = np.abs(lc[sisa] - np.log(kas.at[i, 'jarak']))
            ambil = [sisa[j] for j in np.argsort(d)[:RASIO]]
            pilih += [calon.index[j] for j in ambil]
            pemilik += [i] * len(ambil)
            sisa = [j for j in sisa if j not in set(ambil)]
    return np.array(pilih), np.array(pemilik)


def susun(pre, kas, calon, benih):
    ix_k, pemilik = cocok_set(kas, calon, benih)
    acuan, _ = c09.cocok_dekat(kas, calon, benih)
    assert np.array_equal(np.sort(ix_k), np.sort(acuan)), 'pencocokan berbeda dari 09'
    urut = np.sort(np.concatenate([kas.index.to_numpy(), ix_k]))
    peta = dict(zip(ix_k, pemilik))
    peta.update({i: i for i in kas.index})
    S = pre.loc[urut].reset_index(drop=True)
    S['set'] = [peta[i] for i in urut]
    return S


def fitur(S, X):
    B = S[list(c06.BENDERA) + c03.UKURAN].to_numpy(float)
    return {MESIN: (B, 'hgb'),
            MESIN_DEMO: (np.column_stack([B, S[['usia', 'pria']].to_numpy(float)]), 'hgb'),
            CIRI: (c02.ciri(X), 'hgb')}


def boot_set(y, p, s, n=N_BOOT):
    """Bootstrap set cocokan: set diambil ulang dengan pengembalian, kasus dan kontrolnya bersama."""
    us = np.unique(s)
    anggota = {k: np.where(s == k)[0] for k in us}
    out = np.empty(n)
    for b in range(n):
        ix = np.concatenate([anggota[k] for k in rng.choice(us, len(us))])
        out[b] = roc_auc_score(y[ix], p[ix])
    return out


def acak_dalam_set(y, s):
    """Permutasi label di dalam tiap set: label kasus dipindah ke anggota acak set yang sama."""
    yp = np.zeros_like(y)
    for k in np.unique(s):
        ix = np.where(s == k)[0]
        yp[rng.choice(ix)] = 1
    return yp


def ci(a):
    return f'[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]'


def main():
    pre, calon, kp, _ = c09.data_p()
    ritme = kp.initial_rhythm
    sebab = kp.cause_of_cardiac_arrest
    jantung = (sebab == 'cardiac event').to_numpy()
    subset = {
        'semua sebab': np.ones(len(kp), bool),
        'sebab jantung': jantung,
        'sebab bukan jantung': (~jantung & sebab.notna()).to_numpy(),
        'irama shockable (VT/VF)': (ritme == 'VT/VF').to_numpy(),
        'irama non-shockable (PEA, asistol)': ritme.isin(['PEA', 'asystole']).to_numpy(),
    }

    print('== Kasus kunjungan terdahulu 2-730 hari: sebab x irama awal', flush=True)
    print(pd.crosstab(sebab.fillna('kosong'), ritme.fillna('kosong'), margins=True).to_string(), flush=True)
    print('\n== Sebab x lokasi', flush=True)
    print(pd.crosstab(sebab.fillna('kosong'), kp.location_of_cardiac_arrest.fillna('kosong'), margins=True).to_string(),
          flush=True)

    simpan = {}
    rekap = {}
    for nama, m in subset.items():
        kas = kp[m]
        for b in BENIH:
            S = susun(pre, kas, calon, b)
            y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
            X = c02.sinyal(S)
            F = fitur(S, X)
            pakai_cnn = b == 0 or nama.startswith('irama')
            if pakai_cnn:
                F[CNN] = (X, 'cnn')
            print(f'\n  {nama}, benih {b}: {int(y.sum())} kasus / {int((1 - y).sum())} kontrol; '
                  f'{len(np.unique(s))} set', flush=True)
            for f, (Fx, cara) in F.items():
                p = c02.oof(Fx, y, g, 2 if cara == 'cnn' else 1, cara)
                a = roc_auc_score(y, p)
                bs = boot_set(y, p, s)
                rekap.setdefault((nama, f), []).append(a)
                simpan[(nama, f, b)] = (y, p, s, S.location_of_cardiac_arrest.to_numpy(), S.set.to_numpy())
                print(f'    {f:<34s} AUC {a:.3f}  bootstrap set {ci(bs)}', flush=True)
            if b == 0 and not nama.startswith('irama'):
                for f in [MESIN, CIRI]:
                    Fx, cara = F[f]
                    nol = np.array([roc_auc_score(yp, c02.oof(Fx, yp, g, 1, cara))
                                    for yp in (acak_dalam_set(y, s) for _ in range(N_PERM))])
                    amat = rekap[(nama, f)][-1]
                    print(f'    nol permutasi dalam set {f:<26s} rerata {nol.mean():.3f}, p95 '
                          f'{np.percentile(nol, 95):.3f}, maks {nol.max():.3f}, p(nol >= amatan) '
                          f'{(1 + np.sum(nol >= amat)) / (1 + N_PERM):.3f}', flush=True)
                nol = [roc_auc_score(yp, c02.oof(X, yp, g, 1, 'cnn'))
                       for yp in (acak_dalam_set(y, s) for _ in range(N_PERM_CNN))]
                print(f'    nol permutasi dalam set CNN ({N_PERM_CNN} kali)     {np.round(nol, 3).tolist()}', flush=True)
            del X

    print('\n== Rekap atas benih: rerata [min, maks] (CNN: benih 0 saja, kecuali subset irama)', flush=True)
    for (nama, f), v in rekap.items():
        print(f'  {nama:<36s} {f:<34s} {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]', flush=True)

    def selisih(a, bnama, f, benih, lokasi=None):
        semua = []
        for b in benih:
            parts = []
            for nm in (a, bnama):
                y, p, s, lok, _ = simpan[(nm, f, b)]
                if lokasi is not None:
                    set_lok = np.unique(s[(y == 1) & (lok == lokasi)])
                    keep = np.isin(s, set_lok)
                    y, p, s = y[keep], p[keep], s[keep]
                parts.append((roc_auc_score(y, p), boot_set(y, p, s), int(y.sum())))
            semua.append((parts[0][0] - parts[1][0], parts[0][1] - parts[1][1], parts[0][2], parts[1][2],
                          parts[0][0], parts[1][0]))
        d = np.concatenate([x[1] for x in semua])
        return (np.mean([x[0] for x in semua]), d, semua[0][2], semua[0][3],
                np.mean([x[4] for x in semua]), np.mean([x[5] for x in semua]))

    print('\n== Selisih AUROC, bootstrap set cocokan digabung atas benih', flush=True)
    for a, bnama in [('sebab jantung', 'sebab bukan jantung'),
                     ('irama shockable (VT/VF)', 'irama non-shockable (PEA, asistol)')]:
        for f in [MESIN, MESIN_DEMO, CIRI, CNN]:
            benih = [0] if (f == CNN and not a.startswith('irama')) else list(BENIH)
            m, d, na, nb, aa, ab = selisih(a, bnama, f, benih)
            print(f'  {a} - {bnama} | {f:<34s} {aa:.3f} - {ab:.3f} = {m:+.3f} {ci(d)}  P(<=0) {np.mean(d <= 0):.3f}'
                  f'  (benih {benih})', flush=True)

    print('\n== Sebab jantung - bukan jantung menurut lokasi henti (penilaian ulang OOF)', flush=True)
    for lokasi in ['IHCA', 'OHCA']:
        for f in [MESIN, CIRI]:
            m, d, na, nb, aa, ab = selisih('sebab jantung', 'sebab bukan jantung', f, list(BENIH), lokasi)
            print(f'  {lokasi} | {f:<34s} kasus {na} / {nb}: {aa:.3f} - {ab:.3f} = {m:+.3f} {ci(d)}'
                  f'  P(<=0) {np.mean(d <= 0):.3f}', flush=True)

    print('\n== Rancangan C tanpa EKG berbendera infark akut/cedera (penilaian ulang OOF)', flush=True)
    K = c06.rancangan_c()
    y, g = K.y.to_numpy().astype(np.float32), K.pid.to_numpy()
    B = K[list(c06.BENDERA)].to_numpy(float)
    Z = np.column_stack([B, K[c03.UKURAN].to_numpy(float), K[['usia', 'pria']].to_numpy(float)])
    p_mesin = c02.oof(Z, y, g, 3, 'hgb')
    o = pd.read_csv(os.path.join(c06.TURUNAN, 'oof_kunjungan_bersih.csv'))[['ecg_id', 'p_cnn']]
    p_cnn = K[['ecg_id']].merge(o, on='ecg_id', how='left').p_cnn.to_numpy()
    assert not np.isnan(p_cnn).any()
    for nama, keep in [('semua', np.ones(len(K), bool)), ('tanpa infark akut/cedera', (K.mi_akut == 0).to_numpy())]:
        yy, gg = y[keep], g[keep]
        print(f'  {nama}: {int(yy.sum())} kasus / {int((1 - yy).sum())} kontrol', flush=True)
        for f, p in [('CNN', p_cnn[keep]), ('pembacaan mesin + usia/jk', p_mesin[keep])]:
            print(f'    {f:<28s} AUC {c06.teks(c06.auc_ci(yy, p, gg))}', flush=True)
        print(f'    CNN - pembacaan mesin       {c06.teks(c06.beda_ci(yy, p_cnn[keep], p_mesin[keep], gg))}', flush=True)


if __name__ == '__main__':
    main()
