"""Nightingale arrest-ntuh-ecg — apakah henti jantung dapat dibedakan dari kontrol oleh hal selain EKG-nya?

Data: Nightingale Open Science, arrest-ntuh-ecg v1 (NTUH, Taiwan). Disimpan di luar repo
(C:/Data/Nightingale, atau peubah lingkungan NGSCI_DATA). Skrip ini hanya mencetak agregat;
tidak ada baris data pasien yang ditulis ke repo.

Rancangan dataset (docs.ngsci.org/datasets/arrest-ntuh-ecg):
  kasus    974 pasien henti jantung + ROSC di IGD, 2011-2019; EKG [0]pre, [1]ROSC, [2]24hr
  kontrol  9.976 pasien IGD tanpa henti jantung pada tanggal yang sama, 2015-2019; EKG control, pre

Uji:
  T1 probe metadata      versi MUSE (mayor) dan format label tahun, tanpa sinyal
  T2 jarak waktu         EKG pre -> kunjungan indeks, kasus lawan kontrol
  T3 EKG pre             kasus [0]pre lawan kontrol pre (soal prediksi yang sebenarnya)
                         a usia+jenis kelamin  b ukuran 12SL dari MUSE  c b+a
                         d ciri sinyal hitungan sendiri (tidak lewat algoritme MUSE)  e d+a
                         semua tahun, lalu hanya 2017-2019 (tahun kasus dan kontrol tumpang tindih
                         dengan label tahun tunggal)
  T4 probe versi         di dalam kontrol saja: MUSE 10 (2015-16) lawan MUSE 9 (2017-19), EKG pre;
                         ukuran 12SL lawan ciri sinyal sendiri. Bila ukuran 12SL membedakan versi,
                         ukuran itu membawa jejak versi, dan di T3 versi = label.
  T5 EKG indeks          kasus [1]ROSC lawan kontrol control (pembanding atas; pasca-henti jantung)

Validasi: 5 lipatan berkelompok menurut pasien; AUC out-of-fold; selang 95 % dari 1.000 bootstrap
pasien. Pakai: python 01_uji_arrest.py > uji_arrest.log
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

R = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'arrest-ntuh-ecg', 'v1')
FS = 500
UKURAN = ['ventricular_rate', 'atrial_rate', 'pr_interval', 'qrs_duration', 'qt_interval',
          'qt_corrected', 'p_axis', 'r_axis', 't_axis', 'qrs_count', 'q_onset', 'q_offset',
          'p_onset', 'p_offset', 't_offset', 'qtc_frederica']
PITA = [(0.5, 5), (5, 15), (15, 40), (40, 100)]
rng = np.random.default_rng(20261005)


def waktu(s):
    """Tanggal tergeser (tahun 2111-2220): pd.to_datetime(errors='coerce') diam-diam gagal pada
    sebagian format, jadi diurai dengan numpy."""
    return np.array(pd.Series(s).astype(str).str.replace(' ', 'T').replace('nan', 'NaT'),
                    dtype='datetime64[s]')


def hari(a, b):
    return (waktu(a) - waktu(b)).astype('timedelta64[s]').astype(float) / 86400


def usia(s):
    return pd.to_numeric(s.astype(str).str.replace('+', '', regex=False), errors='coerce')


def muat_kelompok(folder_meta, folder_wave, label_tag):
    """Gabungkan metadata, ukuran 12SL, dan indeks npy untuk satu kelompok berkas."""
    muse = pd.read_csv(os.path.join(folder_meta, 'muse-info.csv'))
    ukur = pd.read_csv(os.path.join(folder_meta, 'resting-ecg-measurements.csv'))
    idx = pd.read_csv(os.path.join(folder_wave, 'waveform-npy.csv'))
    d = idx.merge(muse[['ecg_id', 'muse_version']], on='ecg_id', how='left')
    d = d.merge(ukur[['ecg_id'] + UKURAN], on='ecg_id', how='left')
    d['npy'] = os.path.join(folder_wave, 'waveform-rhythm.npy')
    d['sumber'] = label_tag
    return d


def ciri_sinyal(d):
    """Ciri dari sinyal 12 sadapan 10 s yang dihitung sendiri: simpangan baku, beda mutlak rata-rata,
    dan daya relatif empat pita per sadapan, ditambah laju denyut kasar dari autokorelasi sadapan II."""
    out = np.full((len(d), 12 * (2 + len(PITA)) + 1), np.nan)
    f = np.fft.rfftfreq(5000, 1 / FS)
    for path, grup in d.groupby('npy'):
        a = np.load(path, mmap_mode='r')
        for pos, i in zip(grup.index, grup['npy_index'].astype(int)):
            x = np.asarray(a[i], dtype=float)
            x = x - x.mean(axis=1, keepdims=True)
            p = np.abs(np.fft.rfft(x, axis=1)) ** 2
            tot = p[:, (f >= 0.5) & (f < 100)].sum(axis=1) + 1e-9
            pita = [p[:, (f >= lo) & (f < hi)].sum(axis=1) / tot for lo, hi in PITA]
            c = np.column_stack([x.std(axis=1), np.abs(np.diff(x, axis=1)).mean(axis=1)] + pita)
            ii = x[1] - np.convolve(x[1], np.ones(100) / 100, mode='same')
            ac = np.correlate(ii, ii, mode='full')[len(ii) - 1:]
            lag = np.argmax(ac[int(0.3 * FS):int(2.0 * FS)]) + int(0.3 * FS)
            out[d.index.get_loc(pos)] = np.concatenate([c.ravel(), [60 * FS / lag]])
    return out


def auc_ci(y, p, g):
    auc = roc_auc_score(y, p)
    ug = np.unique(g)
    pos = {k: np.where(g == k)[0] for k in ug}
    bs = []
    for _ in range(1000):
        ix = np.concatenate([pos[k] for k in rng.choice(ug, len(ug))])
        if len(np.unique(y[ix])) == 2:
            bs.append(roc_auc_score(y[ix], p[ix]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, lo, hi


def cv_auc(X, y, g, model='hgb'):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        if model == 'lr':
            m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return auc_ci(y, p, g)


def lapor(nama, hasil, n_pos, n_neg):
    auc, lo, hi = hasil
    print(f'  {nama:<42s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]   n={n_pos}/{n_neg}')


# ---------- muat ----------
S = muat_kelompok(os.path.join(R, 'study-group', 'ecg-metadata'),
                  os.path.join(R, 'study-group', 'ecg-waveforms'), 'kasus')
S = S.merge(pd.read_csv(os.path.join(R, 'study-group', 'ecg.csv'))[
    ['ecg_id', 'patient_ngsci_id', 'acquisition_datetime_offset']], on='ecg_id', how='left')
kohort = pd.read_csv(os.path.join(R, 'study-group', 'cohort.csv')).drop_duplicates('patient_ngsci_id')
S = S.merge(kohort[['patient_ngsci_id', 'sex', 'age', 'rosc_datetime_offset']], on='patient_ngsci_id', how='left')

C = pd.concat([muat_kelompok(os.path.join(d, 'ecg-metadata'), os.path.join(d, 'ecg-waveforms'), 'kontrol')
               for d in sorted(glob.glob(os.path.join(R, 'control-group', '*', '*')))], ignore_index=True)
ck = pd.read_csv(os.path.join(R, 'control-group', 'ecg-cohort.csv'))
C = C.merge(ck[['ecg_id', 'patient_ngsci_id', 'acquisition_datetime_offset', 'age', 'sex']], on='ecg_id', how='left')

D = pd.concat([S, C], ignore_index=True)
D['y'] = (D['sumber'] == 'kasus').astype(int)
D['usia'] = usia(D['age'])
D['pria'] = (D['sex'].astype(str).str.lower() == 'male').astype(float)
D['muse_mayor'] = D['muse_version'].astype(str).str.split('.').str[0]
D['tahun'] = D['year'].astype(str)
D['pid'] = D['patient_ngsci_id'].astype(str)
print('EKG per kelompok:')
print(D.groupby(['sumber', 'ecg_timetag']).size().to_string())

# ---------- T1 ----------
print('\nT1 probe metadata (tanpa sinyal)')
print(pd.crosstab([D['sumber'], D['tahun']], D['muse_mayor']).to_string())
pre = D[D['ecg_timetag'].isin(['[0]pre', 'pre'])].copy()
X = pd.get_dummies(pre['muse_mayor']).to_numpy(float)
lapor('versi MUSE mayor saja, EKG pre', cv_auc(X, pre['y'].to_numpy(), pre['pid'].to_numpy(), 'lr'),
      pre['y'].sum(), (1 - pre['y']).sum())
print('  label tahun kasus:', sorted(D.loc[D.y == 1, 'tahun'].unique()),
      '| kontrol:', sorted(D.loc[D.y == 0, 'tahun'].unique()))

# ---------- T2 ----------
print('\nT2 jarak EKG pre ke kunjungan indeks (hari)')
idx_kasus = S.groupby('patient_ngsci_id')['rosc_datetime_offset'].first()
idx_kontrol = C[C['ecg_timetag'] == 'control'].groupby('patient_ngsci_id')['acquisition_datetime_offset'].first()
# Cacat data: pada kontrol 2018-2019 tanggal EKG pre dan EKG indeks tidak digeser dengan geseran
# yang sama (jarak sampai +-20.000 hari, separuh negatif). Jarak kontrol hanya sahih 2015-2017.
SAH = ['2015', '2016', '2017']
for nama, sub, ind in [('kasus', pre[pre.y == 1], idx_kasus),
                       ('kontrol 2015-17', pre[(pre.y == 0) & pre.tahun.isin(SAH)], idx_kontrol),
                       ('kontrol 2018-19', pre[(pre.y == 0) & ~pre.tahun.isin(SAH)], idx_kontrol)]:
    gap = pd.Series(hari(sub['patient_ngsci_id'].map(ind), sub['acquisition_datetime_offset']))
    q = gap.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist()
    print(f'  {nama:<8s} n={gap.notna().sum():5d}  p10/p25/median/p75/p90 = {q}  '
          f'<=1 hari {np.mean(gap <= 1):.3f}  >365 hari {np.mean(gap > 365):.3f}')
pre['jarak'] = np.nan
for v, ind in [(1, idx_kasus), (0, idx_kontrol)]:
    m = pre.y == v
    pre.loc[m, 'jarak'] = hari(pre.loc[m, 'patient_ngsci_id'].map(ind), pre.loc[m, 'acquisition_datetime_offset'])
pre.loc[(pre.y == 0) & ~pre.tahun.isin(SAH), 'jarak'] = np.nan

# ---------- T3 ----------
print('\nT3 EKG pre: kasus lawan kontrol')
pre = pre.reset_index(drop=True)
sig = ciri_sinyal(pre)
for judul, mask in [('semua tahun', np.ones(len(pre), bool)),
                    ('2017-2019 saja', pre['tahun'].isin(['2017', '2018', '2019']).to_numpy())]:
    p = pre[mask]
    y, g = p['y'].to_numpy(), p['pid'].to_numpy()
    npos, nneg = y.sum(), (1 - y).sum()
    print(f' {judul}')
    demo = p[['usia', 'pria']].to_numpy(float)
    ukr = p[UKURAN].to_numpy(float)
    s = sig[mask]
    lapor('a usia + jenis kelamin (LR)', cv_auc(demo, y, g, 'lr'), npos, nneg)
    lapor('b ukuran 12SL MUSE', cv_auc(ukr, y, g), npos, nneg)
    lapor('c ukuran 12SL + usia/jk', cv_auc(np.column_stack([ukr, demo]), y, g), npos, nneg)
    lapor('d ciri sinyal sendiri', cv_auc(s, y, g), npos, nneg)
    lapor('e ciri sinyal + usia/jk', cv_auc(np.column_stack([s, demo]), y, g), npos, nneg)
    v = p['jarak'].notna().to_numpy()
    if (1 - y[v]).sum():
        lapor('f jarak pre->indeks saja (kontrol 2015-17)', cv_auc(p.loc[v, ['jarak']].to_numpy(float), y[v], g[v]),
              y[v].sum(), (1 - y[v]).sum())

# ---------- T6 ----------
print('\nT6 EKG pre, jarak disamakan')
BIN = [1, 30, 180, 365, 730, np.inf]
pre['bin'] = pd.cut(pre['jarak'], [-np.inf] + BIN, labels=False)
for judul, minimal in [('jarak > 1 hari, kontrol 2015-17 dicocokkan per pita jarak', 1)]:
    kas = pre[(pre.y == 1) & (pre['jarak'] > minimal)]
    pilih = [kas.index]
    for b, nb in kas['bin'].value_counts().items():
        calon = pre[(pre.y == 0) & (pre['bin'] == b)].index
        pilih.append(rng.choice(calon, min(len(calon), 5 * nb), replace=False))
    mask = np.zeros(len(pre), bool); mask[np.concatenate(pilih)] = True
    p = pre[mask]
    y, g = p['y'].to_numpy(), p['pid'].to_numpy()
    print(f' {judul}; median jarak kasus {kas.jarak.median():.0f} hari, kontrol {p[p.y == 0].jarak.median():.0f} hari')
    demo = p[['usia', 'pria']].to_numpy(float)
    lapor('jarak saja (harus dekat 0,5)', cv_auc(p[['jarak']].to_numpy(float), y, g), y.sum(), (1 - y).sum())
    lapor('a usia + jenis kelamin (LR)', cv_auc(demo, y, g, 'lr'), y.sum(), (1 - y).sum())
    lapor('b ukuran 12SL MUSE', cv_auc(p[UKURAN].to_numpy(float), y, g), y.sum(), (1 - y).sum())
    lapor('d ciri sinyal sendiri', cv_auc(sig[mask], y, g), y.sum(), (1 - y).sum())
    lapor('e ciri sinyal + usia/jk', cv_auc(np.column_stack([sig[mask], demo]), y, g), y.sum(), (1 - y).sum())
kas1 = pre[(pre.y == 1)]
print(f'  kasus pre <= 1 hari: {np.mean(kas1.jarak <= 1):.3f}; <= 0 hari (setelah ROSC?): {np.mean(kas1.jarak < 0):.3f}')

# ---------- T4 ----------
print('\nT4 probe versi di dalam kontrol: MUSE 10 (2015-16) lawan MUSE 9 (2017-19), EKG pre')
kp = pre[(pre.y == 0) & pre['muse_mayor'].isin(['9', '10'])]
yv = (kp['muse_mayor'] == '10').astype(int).to_numpy()
gv = kp['pid'].to_numpy()
mk = ((pre.y == 0) & pre['muse_mayor'].isin(['9', '10'])).to_numpy()
lapor('usia + jenis kelamin (LR)', cv_auc(kp[['usia', 'pria']].to_numpy(float), yv, gv, 'lr'), yv.sum(), (1 - yv).sum())
lapor('ukuran 12SL MUSE', cv_auc(kp[UKURAN].to_numpy(float), yv, gv), yv.sum(), (1 - yv).sum())
lapor('ciri sinyal sendiri', cv_auc(sig[mk], yv, gv), yv.sum(), (1 - yv).sum())

# ---------- T5 ----------
print('\nT5 EKG indeks: kasus [1]ROSC lawan kontrol control')
ix = D[D['ecg_timetag'].isin(['[1]ROSC', 'control'])].reset_index(drop=True)
y, g = ix['y'].to_numpy(), ix['pid'].to_numpy()
lapor('ukuran 12SL MUSE', cv_auc(ix[UKURAN].to_numpy(float), y, g), y.sum(), (1 - y).sum())
lapor('ciri sinyal sendiri', cv_auc(ciri_sinyal(ix), y, g), y.sum(), (1 - y).sum())
