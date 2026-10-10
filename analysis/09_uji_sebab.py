"""Nightingale arrest-ntuh-ecg — uji ketahanan sinyal sebab jantung di rancangan P (telaah ulang panel
8 Oktober 2026, Riset/EKG/Telaah/2026-10-08/report.md).

Keberatan yang diuji:
  1. sisa ketidakcocokan jarak: dalam pita, EKG kasus sebab jantung 11-15 hari lebih baru daripada
     kontrolnya, plasebo hanya 3-8 hari. -> AUROC jarak saja per subset, untuk pencocokan pita (07)
     dan pencocokan tetangga terdekat pada log jarak (baru, 'dekat').
  2. selisih jantung lawan bukan jantung tidak pernah diuji -> bootstrap selisih AUROC per benih
     (subset saling bebas di sisi kasus) dan gabungan benih (memasukkan variasi pencocokan).
  3. tidak ada nol empiris untuk rancangan seimbang -> permutasi label di dalam set yang sama.
  4. label sebab jantung mungkin = penyakit jantung yang sudah tampak di EKG -> pembanding
     pembacaan mesin (pernyataan MUSE + 12SL) pada EKG kunjungan lampau.
  5. komposisi: lokasi henti jantung dan rincian sebab per subset.
  --triase-acak: kendali ukuran sampel untuk latih ulang mirip triase (51 kasus acak dari > 1 jam).

Pencocokan 'dekat': per strata usia 10 tahun x jenis kelamin, kasus diurutkan acak (benih), setiap
kasus mengambil 5 kontrol tersisa dengan |log jarak kasus - log jarak kontrol| terkecil, tanpa
pengembalian. Kasus [0]pre 2-730 hari; kontrol pre 2015-2017 2-730 hari (sama dengan 07).

Data tetap di C:/Data/Nightingale; hanya agregat yang dicetak. OOF disimpan ke turunan/ untuk uji selisih.
Pakai: python -u 09_uji_sebab.py > uji_sebab.log ;  python -u 09_uji_sebab.py --triase-acak > triase_acak.log
"""
import importlib.util
import os
import sys

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
c07 = _muat_modul('c07', '07_sebab.py')
BENIH = range(3)
RASIO = 5
N_PERM = 50
rng = np.random.default_rng(20261010)


def data_p():
    o = pd.read_csv(os.path.join(c02.R, 'study-group', 'rosc-outcomes.csv')).drop_duplicates('patient_ngsci_id')
    D = c02.muat().merge(o[['patient_ngsci_id', 'cause_of_cardiac_arrest', 'initial_rhythm',
                            'location_of_cardiac_arrest']], on='patient_ngsci_id', how='left')
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    pre = pre.merge(c03.ukuran(), on='ecg_id', how='left').merge(c06.pernyataan(), on='ecg_id', how='left')
    pre = pre.fillna({b: 0 for b in c06.BENDERA})
    calon = pre[(pre.y == 0) & pre.tahun.isin(c07.SAH) & (pre.jarak > 2) & (pre.jarak <= 730)]
    kp = pre[(pre.y == 1) & (pre.jarak > 2) & (pre.jarak <= 730)]
    jantung = kp.cause_of_cardiac_arrest == 'cardiac event'
    subset = {'semua sebab': np.ones(len(kp), bool), 'sebab jantung': jantung.to_numpy(),
              'irama awal VT/VF': (kp.initial_rhythm == 'VT/VF').to_numpy(),
              'sebab bukan jantung': (~jantung & kp.cause_of_cardiac_arrest.notna()).to_numpy()}
    return pre, calon, kp, subset


def cocok_dekat(kas, calon, benih):
    rc = np.random.default_rng(60000 + benih)
    sk = (kas.usia // 10).fillna(-1).astype(int).astype(str) + kas.pria.astype(str)
    sc = (calon.usia // 10).fillna(-1).astype(int).astype(str) + calon.pria.astype(str)
    lc = np.log(calon.jarak.to_numpy())
    pilih, kurang = [], 0
    for s in sk.unique():
        sisa = list(np.where((sc == s).to_numpy())[0])
        ik = kas.index[(sk == s).to_numpy()].to_numpy()
        for i in rc.permutation(ik):
            if not sisa:
                kurang += RASIO
                continue
            d = np.abs(lc[sisa] - np.log(kas.at[i, 'jarak']))
            ambil = [sisa[j] for j in np.argsort(d)[:RASIO]]
            kurang += RASIO - len(ambil)
            pilih += [calon.index[j] for j in ambil]
            sisa = [j for j in sisa if j not in set(ambil)]
    return np.array(pilih), kurang


def fitur(S, X=None):
    out = {'jarak saja (HGB)': (S[['jarak']].to_numpy(float), 'hgb'),
           'usia + jenis kelamin (LR)': (S[['usia', 'pria']].to_numpy(float), 'lr'),
           'pembacaan mesin: pernyataan + 12SL (HGB)': (S[list(c06.BENDERA) + c03.UKURAN].to_numpy(float), 'hgb')}
    if X is not None:
        out['ciri sinyal sederhana (HGB)'] = (c02.ciri(X), 'hgb')
    return out


def auc_ci(y, p, g):
    return c06.auc_ci(y, p, g)


def boot_auc(y, p, n=1000):
    """Bootstrap terpisah untuk kasus dan kontrol (subset saling bebas di sisi kasus)."""
    ik, ic = np.where(y == 1)[0], np.where(y == 0)[0]
    out = np.empty(n)
    for b in range(n):
        ix = np.concatenate([rng.choice(ik, len(ik)), rng.choice(ic, len(ic))])
        out[b] = roc_auc_score(y[ix], p[ix])
    return out


def main_utama():
    pre, calon, kp, subset = data_p()
    print('== Komposisi kasus rancangan P (2-730 hari)', flush=True)
    for nama, m in subset.items():
        k = kp[m]
        print(f'  {nama:<22s} n={len(k):<4d} lokasi {k.location_of_cardiac_arrest.value_counts().to_dict()}', flush=True)
        print(f'  {"":<22s} sebab {k.cause_of_cardiac_arrest.value_counts().to_dict()}', flush=True)
        print(f'  {"":<22s} irama {k.initial_rhythm.value_counts().to_dict()}', flush=True)

    print('\n== Pencocokan pita (seperti 07): jarak saja dan pembacaan mesin, tanpa CNN', flush=True)
    for nama, m in subset.items():
        kas = kp[m]
        v = {}
        for b in BENIH:
            ix_k, _ = c07.cocokkan(kas, calon, b)
            S = pre.loc[np.sort(np.concatenate([kas.index.to_numpy(), ix_k]))].reset_index(drop=True)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            for f, (F, cara) in fitur(S).items():
                v.setdefault(f, []).append(roc_auc_score(y, c02.oof(F, y, g, 1, cara)))
        for f, a in v.items():
            print(f'  {nama:<22s} {f:<42s} {np.mean(a):.3f} [{np.min(a):.3f}, {np.max(a):.3f}]', flush=True)

    print('\n== Pencocokan dekat (log jarak, usia x jk, 5:1): semua model', flush=True)
    simpan = {}
    rekap = {}
    for nama, m in subset.items():
        kas = kp[m]
        for b in BENIH:
            ix_k, kurang = cocok_dekat(kas, calon, b)
            S = pre.loc[np.sort(np.concatenate([kas.index.to_numpy(), ix_k]))].reset_index(drop=True)
            y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
            print(f'\n  {nama}, benih {b}: kontrol kurang {kurang}; median jarak kasus {S[S.y == 1].jarak.median():.0f}'
                  f' / kontrol {S[S.y == 0].jarak.median():.0f} hari; median usia {S[S.y == 1].usia.median():.0f}'
                  f' / {S[S.y == 0].usia.median():.0f}; tahun kasus {S[S.y == 1].tahun.value_counts().to_dict()}',
                  flush=True)
            X = c02.sinyal(S)
            F = fitur(S, X)
            F['CNN 12 sadapan'] = (X, 'cnn')
            for f, (Fx, cara) in F.items():
                p = c02.oof(Fx, y, g, 2 if cara == 'cnn' else 1, cara)
                h = auc_ci(y, p, g)
                rekap.setdefault((nama, f), []).append(h[0])
                simpan[(nama, f, b)] = (y, p)
                print(f'    {f:<42s} AUC {c06.teks(h)}', flush=True)
            # nol empiris: permutasi label, ciri cepat saja
            for f in ['ciri sinyal sederhana (HGB)', 'pembacaan mesin: pernyataan + 12SL (HGB)']:
                if b > 0:
                    continue
                Fx, cara = F[f]
                nol = []
                for _ in range(N_PERM):
                    yp = rng.permutation(y)
                    nol.append(roc_auc_score(yp, c02.oof(Fx, yp, g, 1, cara)))
                nol = np.array(nol)
                print(f'    nol permutasi {f:<28s} rerata {nol.mean():.3f}, p95 {np.percentile(nol, 95):.3f}, '
                      f'p(nol >= amatan) {np.mean(nol >= rekap[(nama, f)][-1]):.2f}', flush=True)
            if nama == 'sebab jantung' and b == 0:
                nol = []
                for r in range(5):
                    yp = rng.permutation(y)
                    nol.append(roc_auc_score(yp, c02.oof(X, yp, g, 1, 'cnn')))
                print(f'    nol permutasi CNN (5 kali)            {np.round(nol, 3).tolist()}', flush=True)
            del X
    print('\nRekap pencocokan dekat atas 3 benih: rerata [min, maks]', flush=True)
    for (nama, f), v in rekap.items():
        print(f'  {nama:<22s} {f:<42s} {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]', flush=True)

    print('\n== Selisih AUROC sebab jantung - sebab bukan jantung (pencocokan dekat)', flush=True)
    print('   per benih: bootstrap kasus dan kontrol terpisah; gabungan: bootstrap dari ketiga benih', flush=True)
    for f in ['jarak saja (HGB)', 'ciri sinyal sederhana (HGB)', 'pembacaan mesin: pernyataan + 12SL (HGB)',
              'CNN 12 sadapan']:
        semua = []
        for b in BENIH:
            ya, pa = simpan[('sebab jantung', f, b)]
            yb, pb = simpan[('sebab bukan jantung', f, b)]
            d = boot_auc(ya, pa) - boot_auc(yb, pb)
            semua.append(d)
            print(f'  {f:<42s} benih {b}: {roc_auc_score(ya, pa) - roc_auc_score(yb, pb):+.3f} '
                  f'[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  P(<=0) {np.mean(d <= 0):.3f}', flush=True)
        d = np.concatenate(semua)
        print(f'  {f:<42s} gabungan: {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]'
              f'  P(<=0) {np.mean(d <= 0):.3f}', flush=True)


def main_triase_acak():
    """Kendali ukuran sampel: CNN dilatih ulang dengan 51 kasus acak dari kasus > 1 jam, lawan semua kontrol."""
    K = c06.rancangan_c()
    j = K.jam.to_numpy()
    ix_k = np.where((K.y == 1).to_numpy() & (j > 1))[0]
    ix_c = np.where((K.y == 0).to_numpy())[0]
    X_all = c02.sinyal(K)
    print(f'Kasus > 1 jam: {len(ix_k)}; diambil 51 secara acak, 3 kali; CNN 1 ulangan', flush=True)
    for r in range(3):
        pilih = np.sort(np.concatenate([np.random.default_rng(70000 + r).choice(ix_k, 51, replace=False), ix_c]))
        S = K.iloc[pilih].reset_index(drop=True)
        y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
        p = c02.oof(X_all[pilih], y, g, 1, 'cnn')
        print(f'  acak {r}: CNN {c06.teks(auc_ci(y, p, g))}  n={int(y.sum())}/{int((1 - y).sum())}', flush=True)


if __name__ == '__main__':
    if '--triase-acak' in sys.argv:
        main_triase_acak()
    else:
        main_utama()
