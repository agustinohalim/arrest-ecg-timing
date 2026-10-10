"""Nightingale arrest-ntuh-ecg — CNN 1D pada sinyal mentah 12 sadapan, rancangan naif lawan jarak disamakan.

Menjawab keberatan atas 01_uji_arrest.py: ciri sederhana mungkin terlalu dangkal untuk menangkap
tanda risiko pada EKG pre. Di sini model belajar langsung dari sinyal.

  N  naif           kasus [0]pre lawan kontrol pre, semua tahun (605 / 6.410)
  M  jarak disamakan kasus pre berjarak > 1 hari ke henti jantung lawan kontrol pre 2015-2017,
                    kontrol diambil per pita jarak (1-30, 30-180, 180-365, 365-730, > 730 hari),
                    lima kontrol per kasus; pencocokan diulang dengan lima benih, masing-masing
                    dengan pembangkit acak sendiri. Pada sampel M yang sama juga dihitung ciri sinyal
                    sederhana + HistGradientBoosting dan jarak saja, supaya CNN dan ciri sebanding.
  A  pembanding atas kasus [1]ROSC lawan kontrol control (EKG pasca-henti jantung; bukan prediksi)

Masukan: 12 × 2.500 (10 detik, diturunkan dari 500 ke 250 Hz dengan rerata dua sampel), dikurangi
rerata per sadapan, dibagi simpangan baku global data latih lipatan.
Arsitektur: empat blok Conv1d(kernel 7) + BatchNorm + ReLU + MaxPool(4), kanal 32-64-64-128, rerata
global, dropout 0,3, satu lapis linear; sekitar 110 ribu parameter. Adam 1e-3, 20 epoch, batch 64,
bobot kelas seimbang. StratifiedGroupKFold 5 lipatan (grup = pasien); N dan M 3 ulangan, A 1 ulangan.
AUC out-of-fold dari rerata ulangan; selang 95 % dari 1.000 bootstrap pasien. Berjalan di CPU.

Data tetap di C:/Data/Nightingale (atau NGSCI_DATA); hanya agregat yang dicetak.
Pakai: python 02_cnn_arrest.py > cnn_arrest.log   (--tanpa-naif melewati N)
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sys
from torch import nn

R = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'arrest-ntuh-ecg', 'v1')
FS = 500
EPOCH, BATCH, BOOT = 20, 64, 1000
BENIH_COCOK = range(5)
SAH = ['2015', '2016', '2017']
PITA = [(0.5, 5), (5, 15), (15, 40), (40, 100)]
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
rng = np.random.default_rng(20261006)


def waktu(s):
    return np.array(pd.Series(s).astype(str).str.replace(' ', 'T').replace('nan', 'NaT'), dtype='datetime64[s]')


def hari(a, b):
    return (waktu(a) - waktu(b)).astype('timedelta64[s]').astype(float) / 86400


def muat():
    """Satu baris per EKG: id, pasien, label, tag, tahun, berkas npy, indeks, usia, jenis kelamin, jarak."""
    s = pd.read_csv(os.path.join(R, 'study-group', 'ecg-waveforms', 'waveform-npy.csv'))
    s['npy'] = os.path.join(R, 'study-group', 'ecg-waveforms', 'waveform-rhythm.npy')
    s = s.merge(pd.read_csv(os.path.join(R, 'study-group', 'ecg.csv'))[
        ['ecg_id', 'patient_ngsci_id', 'acquisition_datetime_offset']], on='ecg_id')
    k = pd.read_csv(os.path.join(R, 'study-group', 'cohort.csv')).drop_duplicates('patient_ngsci_id')
    s = s.merge(k[['patient_ngsci_id', 'sex', 'age', 'rosc_datetime_offset']], on='patient_ngsci_id')
    s['jarak'] = hari(s['rosc_datetime_offset'], s['acquisition_datetime_offset'])
    s['y'] = 1
    c = []
    for d in sorted(glob.glob(os.path.join(R, 'control-group', '*', '*'))):
        w = pd.read_csv(os.path.join(d, 'ecg-waveforms', 'waveform-npy.csv'))
        w['npy'] = os.path.join(d, 'ecg-waveforms', 'waveform-rhythm.npy')
        c.append(w)
    c = pd.concat(c, ignore_index=True)
    ck = pd.read_csv(os.path.join(R, 'control-group', 'ecg-cohort.csv'))
    c = c.merge(ck[['ecg_id', 'patient_ngsci_id', 'acquisition_datetime_offset', 'age', 'sex']], on='ecg_id')
    idx = ck[ck.ecg_timetag == 'control'].groupby('patient_ngsci_id')['acquisition_datetime_offset'].first()
    c['jarak'] = hari(c['patient_ngsci_id'].map(idx), c['acquisition_datetime_offset'])
    c['y'] = 0
    d = pd.concat([s, c], ignore_index=True)
    d['tahun'] = d['year'].astype(str)
    # cacat data: kontrol 2018-2019 tidak bergeser seragam antara EKG pre dan indeks
    d.loc[(d.y == 0) & ~d.tahun.isin(SAH), 'jarak'] = np.nan
    d['usia'] = pd.to_numeric(d['age'].astype(str).str.replace('+', '', regex=False), errors='coerce')
    d['pria'] = (d['sex'].astype(str).str.lower() == 'male').astype(float)
    d['pid'] = d['y'].astype(str) + ':' + d['patient_ngsci_id'].astype(str)
    return d


def sinyal(d):
    """12 × 2.500 float32, 250 Hz, rerata per sadapan dibuang."""
    X = np.empty((len(d), 12, 2500), np.float32)
    for path, g in d.groupby('npy'):
        a = np.load(path, mmap_mode='r')
        for pos, i in zip(g.index, g['npy_index'].astype(int)):
            x = np.asarray(a[i], dtype=np.float32).reshape(12, 2500, 2).mean(axis=2)
            X[d.index.get_loc(pos)] = x - x.mean(axis=1, keepdims=True)
    return X


def ciri(X):
    """Ciri sederhana yang sama dengan 01_uji_arrest.py, dihitung dari sinyal 250 Hz."""
    f = np.fft.rfftfreq(X.shape[2], 2 / FS)
    p = np.abs(np.fft.rfft(X, axis=2)) ** 2
    tot = p[:, :, (f >= 0.5) & (f < 100)].sum(axis=2) + 1e-9
    pita = [p[:, :, (f >= lo) & (f < hi)].sum(axis=2) / tot for lo, hi in PITA]
    return np.concatenate([X.std(axis=2), np.abs(np.diff(X, axis=2)).mean(axis=2)] + pita, axis=1)


class Jaringan(nn.Module):
    def __init__(self):
        super().__init__()
        blok, cin = [], 12
        for cout in (32, 64, 64, 128):
            blok += [nn.Conv1d(cin, cout, 7, padding=3), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(4)]
            cin = cout
        self.fitur = nn.Sequential(*blok)
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.kepala(self.fitur(x).mean(dim=2)).squeeze(1)


def latih_prediksi(Xtr, ytr, Xte, benih):
    torch.manual_seed(benih)
    sd = Xtr.std() + 1e-6
    # dibagi sd per batch, bukan seluruh larik sekaligus: satu salinan lebih sedikit di memori
    xt, yt = torch.from_numpy(Xtr), torch.tensor(ytr, dtype=torch.float32)
    m = Jaringan()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = ytr.mean()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32))
    for _ in range(EPOCH):
        m.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss(m(xt[b] / sd), yt[b]).backward()
            opt.step()
    m.eval()
    out = []
    with torch.no_grad():
        xe = torch.from_numpy(Xte)
        for i in range(0, len(xe), 512):
            out.append(torch.sigmoid(m(xe[i:i + 512] / sd)).numpy())
    return np.concatenate(out)


def oof(X, y, g, ulang, cara):
    ps = []
    for b in range(ulang):
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g):
            if cara == 'cnn':
                p[te] = latih_prediksi(X[tr], y[tr], X[te], b)
            else:
                m = (make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)) if cara == 'lr'
                     else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=b))
                p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ps.append(p)
    return np.mean(ps, axis=0)


def auc_ci(y, p, g):
    ug = np.unique(g)
    pos = {k: np.where(g == k)[0] for k in ug}
    bs = []
    for _ in range(BOOT):
        ix = np.concatenate([pos[k] for k in rng.choice(ug, len(ug))])
        if 0 < y[ix].sum() < len(ix):
            bs.append(roc_auc_score(y[ix], p[ix]))
    return roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def lapor(nama, X, d, ulang, cara):
    t0 = time.time()
    y, g = d['y'].to_numpy().astype(np.float32), d['pid'].to_numpy()
    a, lo, hi = auc_ci(y, oof(X, y, g, ulang, cara), g)
    print(f'  {nama:<38s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]   n={int(y.sum())}/{int((1 - y).sum())}'
          f'   ({(time.time() - t0) / 60:.0f} menit)', flush=True)
    return a


def main():
    D = muat()
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    print(f'EKG pre: kasus {int(pre.y.sum())}, kontrol {int((1 - pre.y).sum())}', flush=True)
    Xpre = sinyal(pre)

    if '--tanpa-naif' not in sys.argv:
        print('\nN naif, semua tahun', flush=True)
        lapor('CNN 12 sadapan', Xpre, pre, 3, 'cnn')

    # Pencocokan memakai pembangkit acak sendiri per benih, terpisah dari bootstrap, supaya sampel M
    # tidak bergantung pada apa yang dijalankan sebelumnya. Diulang dengan BENIH_COCOK benih.
    batas = [-np.inf, 1, 30, 180, 365, 730, np.inf]
    pre['pita'] = pd.cut(pre['jarak'], batas, labels=False)
    kas = pre[(pre.y == 1) & (pre.jarak > 1)]
    rekap = {}
    for s in BENIH_COCOK:
        rc = np.random.default_rng(30000 + s)
        pilih = [kas.index.to_numpy()]
        for b, nb in kas['pita'].value_counts().sort_index().items():
            calon = pre[(pre.y == 0) & (pre['pita'] == b)].index.to_numpy()
            pilih.append(rc.choice(calon, min(len(calon), 5 * nb), replace=False))
        ix = np.sort(np.concatenate(pilih))
        M = pre.loc[ix].reset_index(drop=True)
        XM = Xpre[ix]
        print(f'\nM jarak disamakan, benih pencocokan {s} (kasus > 1 hari, kontrol 2015-17 per pita, 1:5); '
              f'median jarak kasus {M[M.y == 1].jarak.median():.0f} hari, kontrol {M[M.y == 0].jarak.median():.0f} hari',
              flush=True)
        for nama, X, cara in [('jarak saja (HGB)', M[['jarak']].to_numpy(float), 'hgb'),
                              ('usia + jenis kelamin (LR)', M[['usia', 'pria']].to_numpy(float), 'lr'),
                              ('ciri sinyal sederhana (HGB)', ciri(XM), 'hgb'),
                              ('CNN 12 sadapan', XM, 'cnn')]:
            rekap.setdefault(nama, []).append(lapor(nama, X, M, 3, cara))
        del XM
    print(f'\nM rekap atas {len(BENIH_COCOK)} benih pencocokan: rerata AUC [min, maks]', flush=True)
    for nama, v in rekap.items():
        print(f'  {nama:<38s} {np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]', flush=True)

    print('\nA pembanding atas: [1]ROSC lawan control', flush=True)
    del Xpre
    A = D[D.ecg_timetag.isin(['[1]ROSC', 'control'])].reset_index(drop=True)
    XA = sinyal(A)
    lapor('CNN 12 sadapan', XA, A, 1, 'cnn')


if __name__ == '__main__':
    main()
