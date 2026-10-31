"""Nightingale arrest-ntuh-ecg — QRS, QT, dan QTc dihitung ulang dari gelombang, untuk menguji apakah sinyal sebab jantung di
rancangan P (skrip 30: 16 ukuran 0,704 lawan laju 0,552) bergantung pada program ukur 12SL yang berimpit dengan versi MUSE
(telaah panel JIKI 31 Oktober 2026 (d) dan (e), kritis "measurements not cleared of the export software").

Seperti laju di skrip 26: satu prosedur tetap yang sama untuk semua EKG, 500 Hz asli. Urutan sadapan baku I, II, III,
aVR, ... (diperiksa dari polaritas QRS: indeks 3 negatif pada 87 % EKG).

Percobaan pertama (log interval_sinyal_dwt*.log) memakai NeuroKit2 ecg_delineate metode dwt pada sadapan II per detak, dan
gagal sebagai ukuran: QRS median 162 ms lawan 12SL 88 ms, Spearman 0,11; QTc Spearman 0,17. Bagian 2 percobaan itu tidak
dibaca. Prosedur yang dipakai sekarang, denyut median 12 sadapan:
  puncak R dari sadapan II (nk.ecg_clean, nk.ecg_peaks); tiap sadapan disaring Butterworth 0,5-40 Hz orde 2; denyut median
  jendela R - 250 ms sampai R + min(600 ms, 0,95 RR); kecepatan spasial v = |turunan| 12 sadapan, dihaluskan 10 ms; onset dan
  offset QRS = titik pertama di kiri dan kanan puncak v (dalam R +- 80 ms) dengan v < AMB x puncak; garis dasar = rerata
  20 ms sebelum onset; magnitudo vektor M dihaluskan 18 ms; puncak T = maks M dalam [offset + 60 ms, R + min(450 ms,
  0,65 RR)]; akhir T = potong garis singgung kemiringan turun terbesar dalam 150 ms sesudah puncak T dengan persentil 5 M di
  rentang itu; QT = akhir T - onset; QTc Bazett = QT / sqrt(RR dalam detik).
AMB disetel hanya dari kesepakatan dengan 12SL pada 300 EKG acak, tanpa label: 0,05 (QRS Spearman 0,70, QT 0,84); 0,10,
0,15, dan 0,20 lebih buruk. Dibekukan sebelum seluruh EKG diukur.

Gerbang validitas, ditulis sebelum seluruh EKG diukur: ukuran hitung ulang yang Spearman-nya dengan 12SL < 0,70 pada semua
EKG pre dinyatakan tidak tervalidasi dan dikeluarkan dari model "ulang" di bagian 2; model itu lalu memakai sisanya, dan
yang dikeluarkan disebut dalam laporan.

  --tahap daftar (Python sistem): daftar EKG pre (kasus; kontrol 2015-2017) ke luar repo (ARREST_UKUR, bawaan
                 C:/Data/Nightingale/_ukur_ulang/daftar.csv)
  --tahap ukur   (~/venv-nk, neurokit2 0.2.13, pandas < 3): ukuran per EKG ke ukur.csv di folder yang sama
  --tahap lapor  (Python sistem): kesepakatan dengan 12SL per versi MUSE; AUROC selisih kasus lawan kontrol; per sebab di
                 rancangan P (set dan bootstrap seperti 30): 4 ukuran tercetak (laju, QRS, QT, QTc), 4 ukuran hitung ulang,
                 laju hitung ulang saja; selisih berpasangan.

Aturan bacaan, ditulis sebelum hasil ada:
  sebab jantung, ulang - laju, batas bawah > 0              -> sinyal sebab jantung pada interval bertahan dari gelombang;
                                                               bebas dari program ukur 12SL, belum dari gelombang ekspor
  selang mengandung 0 atau di bawah 0, sedangkan            -> sinyal sebab jantung bergantung pada ukuran tercetak; tidak
  tercetak - laju > 0                                          boleh disebut bebas dari perangkat lunak
  ulang - tercetak pada sebab jantung                       -> dilaporkan sebagai besar kehilangan, tanpa aturan
("ulang" = ukuran hitung ulang yang lolos gerbang; "tercetak" = padanan 12SL-nya.) Analisis post hoc; hasilnya arah.

  --tahap atenuasi (Python sistem; ditambah sesudah telaah panel 31 Okt (f), tiga peran: hilangnya keunggulan bisa karena
                 ukuran hitung ulang lebih kasar). Ukuran tercetak yang lolos gerbang diberi derau Gaussian bebas, SD per
                 ukuran dikalibrasi pada semua EKG pre sehingga Spearman(tercetak + derau, tercetak) = Spearman(ulang,
                 tercetak); 50 tarikan derau x 3 benih di rancangan P sebab jantung; dicetak rerata dan persentil 2,5-97,5
                 tarikan untuk AUROC dan (berderau - laju ulang). Aturan bacaan, ditulis sebelum hasil ada:
    rerata (berderau - laju) <= (ulang - laju) + 0,02    -> derau setara cukup menjelaskan hilangnya keunggulan; tidak
                                                            ada bukti jejak perangkat lunak
    rerata (berderau - laju) >= (tercetak - laju) - 0,02 -> derau setara tidak menjelaskannya; bacaan perangkat lunak lebih
                                                            kuat, tetap tanpa bukti langsung
    di antaranya                                         -> sebagian; tidak dipisahkan
  Derau bebas tidak meniru galat yang berbeda antara kasus dan kontrol; batas ini disebut.

CPU. Hanya agregat. Pakai:
  python -u 31_interval_sinyal.py --tahap daftar
  ~/venv-nk/Scripts/python.exe -u 31_interval_sinyal.py --tahap ukur > interval_sinyal_ukur.log
  python -u 31_interval_sinyal.py --tahap lapor > interval_sinyal.log
"""
import argparse
import csv
import importlib.util
import os

import numpy as np

UKUR = os.environ.get('ARREST_UKUR', 'C:/Data/Nightingale/_ukur_ulang')
DAFTAR, HASIL = os.path.join(UKUR, 'daftar.csv'), os.path.join(UKUR, 'ukur.csv')
FS = 500
AMB = 0.05
CETAK = ['ventricular_rate', 'qrs_duration', 'qt_interval', 'qt_corrected']
ULANG = ['laju_u', 'qrs_u', 'qt_u', 'qtc_u']


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


def daftar():
    c26 = _muat_modul('c26', '26_laju_sinyal.py')
    c09, c02 = c26.c09, c26.c02
    pre, _, _, _ = c09.data_p()
    T = pre[(pre.y == 1) | pre.tahun.isin(c02.SAH)].drop_duplicates('ecg_id')
    os.makedirs(UKUR, exist_ok=True)
    T[['ecg_id', 'npy', 'npy_index']].to_csv(DAFTAR, index=False)
    print(f'{len(T)} EKG', flush=True)


def satu(X, nk):
    """Denyut median 12 sadapan; laju, QRS, QT (ms), QTc Bazett."""
    c = nk.ecg_clean(X[1], sampling_rate=FS)
    _, r = nk.ecg_peaks(c, sampling_rate=FS)
    R = np.asarray(r['ECG_R_Peaks'])
    rr = np.median(np.diff(R))
    a, b = int(0.25 * FS), int(min(0.6 * FS, 0.95 * rr))
    R = R[(R - a >= 0) & (R + b < X.shape[1])]
    Y = np.array([nk.signal_filter(x, sampling_rate=FS, lowcut=0.5, highcut=40, method='butterworth', order=2) for x in X])
    B = np.median(np.stack([Y[:, p - a:p + b] for p in R]), axis=0)
    v = np.convolve(np.sqrt((np.diff(B, axis=1) ** 2).sum(0)), np.ones(5) / 5, mode='same')
    w = int(0.08 * FS)
    on = off = a - w + np.argmax(v[a - w:a + w])
    amb = AMB * v[on]
    while on > 0 and v[on] > amb:
        on -= 1
    while off < len(v) - 1 and v[off] > amb:
        off += 1
    base = B[:, max(on - 10, 0):on + 1].mean(1, keepdims=True)
    M = np.convolve(np.sqrt(((B - base) ** 2).sum(0)), np.ones(9) / 9, mode='same')
    m0, m1 = off + int(0.06 * FS), min(len(M) - 1, a + int(min(0.45 * FS, 0.65 * rr)))
    tp = m0 + np.argmax(M[m0:m1])
    m2 = min(len(M) - 1, tp + int(0.15 * FS))
    d = np.diff(M[tp:m2])
    k = np.argmin(d)
    te = tp + k + (M[tp + k] - np.percentile(M[tp:m2 + 1], 5)) / -d[k]
    qrs, qt, rr = (off - on) * 1000 / FS, (te - on) * 1000 / FS, rr / FS
    if not (np.isfinite(qrs) and np.isfinite(qt) and np.isfinite(rr)):
        raise ValueError('tak lengkap')
    return 60 / rr, qrs, qt, qt / np.sqrt(rr)


def ukur():
    import warnings
    import neurokit2 as nk
    warnings.filterwarnings('ignore')
    baris = list(csv.DictReader(open(DAFTAR)))
    berkas, keluar, gagal = {}, [], 0
    for i, b in enumerate(baris):
        a = berkas.setdefault(b['npy'], np.load(b['npy'], mmap_mode='r'))
        x = np.asarray(a[int(float(b['npy_index']))], float)
        try:
            hasil = satu(x, nk)
        except Exception:
            hasil = None
        gagal += hasil is None
        keluar.append([b['ecg_id']] + (list(hasil) if hasil else [''] * 4))
        if i % 500 == 0:
            print(f'  {i}/{len(baris)}', flush=True)
    with open(HASIL, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ecg_id'] + ULANG)
        w.writerows(keluar)
    print(f'selesai: {len(baris)} EKG, gagal {gagal}', flush=True)


def lapor():
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    c30 = _muat_modul('c30', '30_sebab_p.py')
    c26, c19, c10, c09, c06 = c30.c26, c30.c19, c30.c10, c30.c09, c30.c06
    rng = c19.rng
    U = pd.read_csv(HASIL)
    pre, calon, kp, _ = c09.data_p()
    muse = c26.versi_muse()

    print('== 1. Kesepakatan dengan 12SL, EKG pre (kasus; kontrol 2015-2017)', flush=True)
    T = pre[(pre.y == 1) | pre.tahun.isin(c26.c02.SAH)].drop_duplicates('ecg_id').merge(U, on='ecg_id')
    T['versi'] = T.ecg_id.map(muse)
    print(f'  EKG {len(T)}, gagal didelineasi {int(T.qrs_u.isna().sum())}', flush=True)
    lolos = []
    for c, u in zip(CETAK, ULANG):
        A = T.dropna(subset=[c, u])
        d = A[c] - A[u]
        rho = spearmanr(A[c], A[u]).statistic
        if rho >= 0.70:
            lolos.append(u)
        print(f'  {c:<17s} Spearman {rho:.3f} ({"lolos" if rho >= 0.70 else "TIDAK LOLOS"} gerbang); median cetak '
              f'{A[c].median():.0f}, ulang {A[u].median():.0f}; selisih median {d.median():+.1f}; |selisih| > 40 '
              f'{np.mean(d.abs() > 40):.3f}', flush=True)
        for v, B in A.groupby('versi'):
            print(f'    MUSE {v:<3s} n {len(B):>5d} selisih median {(B[c] - B[u]).median():+.1f}, IQR '
                  f'{(B[c] - B[u]).quantile(.75) - (B[c] - B[u]).quantile(.25):.1f}', flush=True)
        y, g = A.y.to_numpy(), A.pid.to_numpy()
        dd = d.to_numpy()
        b = c19.boot(y, g, lambda ix: roc_auc_score(y[ix], dd[ix]))
        print(f'    AUROC selisih kasus lawan kontrol {roc_auc_score(y, dd):.3f} {c10.ci(b)}', flush=True)

    cetak = [c for c, u in zip(CETAK, ULANG) if u in lolos]
    print(f'\n== 2. Rancangan P per sebab: tercetak {cetak}, ulang {lolos}, laju hitung ulang; bootstrap set, 3 benih',
          flush=True)
    LAB = ('laju ulang', 'tercetak', 'ulang', 'tercetak - laju', 'ulang - laju', 'ulang - tercetak')
    h = {'jantung': lambda v: ['cardiac event'],
         'bukan jantung': lambda v: [x for x in v if x not in ('cardiac event', 'unknown')]}
    titik, boot = {n: [] for n in h}, {n: [] for n in h}
    for bn in c19.BENIH:
        S = c10.susun(pre, kp, calon, bn).merge(U, on='ecg_id', how='left')
        kasus = S[S.y == 1]
        set_sebab = dict(zip(kasus.set, kasus.cause_of_cardiac_arrest))
        ragam = set(kasus.cause_of_cardiac_arrest.dropna())
        for n, f in h.items():
            m = S.set.map(set_sebab).isin(f(ragam)).to_numpy()
            Q = S[m].reset_index(drop=True)
            y, g, u = Q.y.to_numpy().astype(int), Q.pid.to_numpy(), Q.set.to_numpy()
            laju = Q.laju_u.fillna(Q.laju_u.median()).to_numpy()
            pc = c06.oof_lr_logit(Q[cetak].to_numpy(float), y, g)
            pu = c06.oof_lr_logit(Q[lolos].to_numpy(float), y, g)

            def ukur_ix(ix):
                yi = y[ix]
                a0, ac, au = roc_auc_score(yi, laju[ix]), roc_auc_score(yi, pc[ix]), roc_auc_score(yi, pu[ix])
                return a0, ac, au, ac - a0, au - a0, au - ac

            titik[n].append(ukur_ix(np.arange(len(y))))
            us, anggota = c19.unit(u)
            boot[n].append(np.array([ukur_ix(np.concatenate([anggota[j] for j in rng.choice(us, len(us))]))
                                     for _ in range(c19.N_BOOT)]))
        print(f'  [benih {bn} dihitung]', flush=True)
    for n in h:
        t, q = np.array(titik[n]), np.concatenate(boot[n])
        print(f'== P, sebab {n}', flush=True)
        for j, lab in enumerate(LAB):
            print(f'  {lab:<22s} {t[:, j].mean():+.3f} {c10.ci(q[:, j])}', flush=True)
    t = np.array(titik['jantung']) - np.array(titik['bukan jantung'])
    q = np.concatenate(boot['jantung']) - np.concatenate(boot['bukan jantung'])
    print('== Jantung - bukan jantung', flush=True)
    for j, lab in enumerate(LAB[:3]):
        print(f'  {lab:<22s} {t[:, j].mean():+.3f} {c10.ci(q[:, j])}', flush=True)


def atenuasi(n_derau=50):
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    c30 = _muat_modul('c30', '30_sebab_p.py')
    c26, c19, c10, c09, c06 = c30.c26, c30.c19, c30.c10, c30.c09, c30.c06
    g_rng = np.random.default_rng(20261101)
    U = pd.read_csv(HASIL)
    pre, calon, kp, _ = c09.data_p()
    T = pre[(pre.y == 1) | pre.tahun.isin(c26.c02.SAH)].drop_duplicates('ecg_id').merge(U, on='ecg_id')
    pasang = [(c, u) for c, u in zip(CETAK, ULANG) if u != 'qrs_u']
    sd = {}
    print('== Kalibrasi derau (semua EKG pre)', flush=True)
    for c, u in pasang:
        A = T.dropna(subset=[c, u])
        sasaran = spearmanr(A[c], A[u]).statistic
        x = A[c].to_numpy(float)
        z = g_rng.standard_normal((5, len(x)))
        lo, hi = 0.0, 10 * x.std()
        for _ in range(40):
            m = (lo + hi) / 2
            r = np.mean([spearmanr(x, x + m * zi).statistic for zi in z])
            lo, hi = (m, hi) if r > sasaran else (lo, m)
        sd[c] = (lo + hi) / 2
        print(f'  {c:<17s} Spearman sasaran {sasaran:.3f}; SD derau {sd[c]:.1f}', flush=True)
    cetak = [c for c, _ in pasang]
    hasil = []
    for bn in c19.BENIH:
        S = c10.susun(pre, kp, calon, bn).merge(U, on='ecg_id', how='left')
        kasus = S[S.y == 1]
        set_sebab = dict(zip(kasus.set, kasus.cause_of_cardiac_arrest))
        Q = S[S.set.map(set_sebab).isin(['cardiac event']).to_numpy()].reset_index(drop=True)
        y, g = Q.y.to_numpy().astype(int), Q.pid.to_numpy()
        laju = Q.laju_u.fillna(Q.laju_u.median()).to_numpy()
        a0 = roc_auc_score(y, laju)
        X = Q[cetak].to_numpy(float)
        for _ in range(n_derau):
            Xn = X + g_rng.standard_normal(X.shape) * np.array([sd[c] for c in cetak])
            a = roc_auc_score(y, c06.oof_lr_logit(Xn, y, g))
            hasil.append((a, a - a0))
        print(f'  [benih {bn} dihitung]', flush=True)
    h = np.array(hasil)
    print('== P, sebab jantung: tercetak + derau setara hitung ulang', flush=True)
    for j, lab in enumerate(('tercetak berderau', 'berderau - laju ulang')):
        print(f'  {lab:<22s} {h[:, j].mean():+.3f} [{np.percentile(h[:, j], 2.5):+.3f}, {np.percentile(h[:, j], 97.5):+.3f}] '
              f'({len(h)} tarikan)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tahap', choices=['daftar', 'ukur', 'lapor', 'atenuasi'], required=True)
    {'daftar': daftar, 'ukur': ukur, 'lapor': lapor, 'atenuasi': atenuasi}[ap.parse_args().tahap]()


if __name__ == '__main__':
    main()
