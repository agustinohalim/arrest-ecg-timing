"""Nightingale arrest-ntuh-ecg — metrik tambahan dari prediksi out-of-fold 03 (definisi bersih), tanpa melatih ulang.

Masukan: C:/Data/Nightingale/arrest-ntuh-ecg/turunan/oof_kunjungan_bersih.csv (ditulis 03).
Setiap kasus dan kontrol satu EKG per pasien, jadi bootstrap per EKG sama dengan bootstrap per pasien.

  AUPRC dengan selang bootstrap, dibanding prevalensi kohort
  kalibrasi: Brier, intersep dan lereng kalibrasi (CNN dilatih dengan bobot kelas, jadi tidak
             diharapkan terkalibrasi); sesudah kalibrasi ulang Platt di dalam validasi silang
  kepekaan: buang EKG <= 15 menit sebelum CPR; buang <= 15 menit dan hanya kasus 2015-2019
  titik operasi: sensitivitas, PPV, dan jumlah yang perlu diperiksa pada spesifisitas 0,90 dan 0,95

Data tetap di C:/Data/Nightingale; hanya agregat yang dicetak.
Pakai: python -u 05_metrik_tambahan.py > metrik_tambahan.log
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'arrest-ntuh-ecg')
TAHUN_KONTROL = ['2015-16', '2017', '2018', '2019']
rng = np.random.default_rng(20261008)


def ci(f, y, p, n=1000):
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(f(y[i], p[i]))
    return f(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def teks(h):
    return f'{h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}]'


def kalibrasi(y, p):
    """Intersep dan lereng: regresi logistik label atas logit prediksi."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    m = LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y)
    return m.intercept_[0], m.coef_[0, 0]


def platt(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None]
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(z, y):
        out[te] = LogisticRegression(C=1e6, max_iter=2000).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
    return out


def main():
    d = pd.read_csv(os.path.join(R, 'turunan', 'oof_kunjungan_bersih.csv'))
    th = pd.read_csv(os.path.join(R, 'v1', 'study-group', 'ecg.csv'))[['ecg_id', 'year']]
    d = d.merge(th, on='ecg_id', how='left')
    y, p = d.y.to_numpy().astype(int), d.p_cnn.to_numpy()
    print(f'EKG: {len(d)}, kasus {y.sum()}, prevalensi {y.mean():.4f}')

    print('\nDiskriminasi (semua kasus bersih)')
    print(f'  AUROC  {teks(ci(roc_auc_score, y, p))}')
    print(f'  AUPRC  {teks(ci(average_precision_score, y, p))}   (prevalensi {y.mean():.4f})')

    print('\nKalibrasi')
    a, b = kalibrasi(y, p)
    print(f'  CNN mentah        Brier {brier_score_loss(y, p):.4f}  intersep {a:.2f}  lereng {b:.2f}  '
          f'rerata prediksi {p.mean():.3f}')
    pc = platt(y, p)
    a, b = kalibrasi(y, pc)
    print(f'  sesudah Platt CV  Brier {brier_score_loss(y, pc):.4f}  intersep {a:.2f}  lereng {b:.2f}  '
          f'rerata prediksi {pc.mean():.4f}  (Brier prevalensi saja {brier_score_loss(y, np.full(len(y), y.mean())):.4f})')
    ke = pd.qcut(pc, 10, labels=False, duplicates='drop')
    print('  desil prediksi Platt: rerata prediksi / proporsi kasus')
    for k in sorted(np.unique(ke)):
        m = ke == k
        print(f'    {k}: {pc[m].mean():.4f} / {y[m].mean():.4f}  (n={m.sum()})')

    print('\nKepekaan (kasus terpilih lawan semua kontrol)')
    kontrol = y == 0
    for nama, m in [('buang <= 15 menit sebelum CPR', d.jam > 0.25),
                    ('buang <= 15 menit, kasus 2015-2019', (d.jam > 0.25) & d.year.isin(TAHUN_KONTROL))]:
        mm = kontrol | ((y == 1) & m.to_numpy())
        print(f'  {nama:<40s} AUROC {teks(ci(roc_auc_score, y[mm], p[mm]))}  '
              f'AUPRC {teks(ci(average_precision_score, y[mm], p[mm]))}  n={y[mm].sum()}')

    print('\nTitik operasi CNN (prevalensi kohort; PPV bergantung prevalensi)')
    for spes in (0.90, 0.95):
        amb = np.quantile(p[kontrol], spes)
        pos = p > amb
        tp, fp = (pos & (y == 1)).sum(), (pos & (y == 0)).sum()
        print(f'  spesifisitas {spes:.2f}: sensitivitas {tp / y.sum():.3f}  PPV {tp / (tp + fp):.3f}  '
              f'yang ditandai per 1.000 EKG {1000 * pos.mean():.0f}')


if __name__ == '__main__':
    main()
