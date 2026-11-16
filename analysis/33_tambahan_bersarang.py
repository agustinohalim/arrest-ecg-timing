"""Nightingale arrest-ntuh-ecg — tambahan CNN dan laju jantung dengan validasi silang bersarang (penelaah JITIKA, putaran 1,
16 November 2026: Reviewer A butir 3, Reviewer B butir 1).

Masalah di skrip 28: tahap 1 (CNN) memakai StratifiedGroupKFold(random_state=k), tahap 2 (`06_revisi.oof_lr_logit`) memakai
random_state=11. Logit pasien latih tahap 2 bisa berasal dari CNN yang dilatih memakai pasien uji tahap 2 beserta labelnya.
Di sini kedua tahap berbagi satu pembagian luar, dan logit baris latih tahap 2 berasal dari CNN dalam yang tidak pernah
melihat lipatan uji luar:

  lipatan luar   StratifiedGroupKFold(5, shuffle, random_state=0) per pasien, sama untuk kelima lari (perlu untuk ansambel);
  logit uji      CNN lari k dilatih pada 4/5 luar, memprediksi 1/5 luar (benih torch k). Lari 0 = lari 0 skrip 28 (pembagian
                 dan benih sama), jadi dipakai ulang; lari 1-4 dilatih ulang pada pembagian luar ini;
  logit latih    di tiap lipatan luar f, bagian latihnya dibagi lagi StratifiedGroupKFold(5, random_state=0); CNN lari k
                 dilatih pada 4/5 dalam, memprediksi 1/5 dalam -> logit OOF dalam untuk semua baris latih luar f;
  tahap 2        regresi logistik (imputasi, skala, LR seperti 06) dipasang pada [logit dalam, ciri] baris latih luar f,
                 menilai [logit uji, ciri] lipatan f. Ansambel = rerata logit lima lari, di tahap dalam maupun luar.
  tanpa CNN      laju saja, 16 ukuran, laju + 13: dipasang pada lipatan luar yang sama (satu tahap, tidak berubah sifatnya).

CNN dalam dilatih pada 64 % data, CNN luar pada 80 %, jadi skala logit dalam dan uji bisa sedikit berbeda; ini sifat biasa
penumpukan bersarang dan, bila berpengaruh, membuat tambahan lebih kecil, bukan lebih besar.
Selang: bootstrap per unit (pasien untuk A/B, set untuk P) seperti skrip 28, 3 benih pencocokan, satu lari acak per tarikan
untuk kolom satu lari. AUROC di dalam kuintil laju tidak memakai tahap 2; dicetak ulang hanya karena lipatan luarnya baru.

Aturan bacaan, ditulis 16 Nov 2026 sebelum hasil ada:
  1. Angka bersarang menggantikan semua kolom tambahan Tabel 4 dan paragraf "A single run overstates the increment",
     apa pun arahnya; angka skrip 28 tidak dipakai lagi untuk tambahan.
  2. A 1-30, ansambel ditambahkan ke laju (laju + ansambel) - laju:
       batas bawah > 0   -> "the ensemble added X to heart rate" tetap, dengan angka baru;
       selang memuat 0   -> klaim tambahan di atas laju ditarik dari Abstract, Results, Discussion, Conclusion; "signal
                            beyond rate" hanya bersandar pada AUROC di dalam kuintil laju dan ditulis begitu;
       batas atas < 0    -> sama dengan di atas, ditambah: laju tidak tertinggal dari CNN di sampel itu.
  3. Sampel lain dan arah sebaliknya (laju ditambahkan ke ansambel; 13 ukuran): setiap sel yang berpindah sisi nol
     (selang memuat 0 <-> tidak) mengubah kalimat yang memakainya; yang tidak berpindah hanya diganti angkanya.
  4. Bila skala logit dalam/uji tampak merusak (AUROC tahap 2 laju + CNN di bawah laju saja di semua sampel), dicatat di
     Hasil sebagai kemungkinan, tidak dipakai untuk menolak angka.

Pakai (CPU, hanya agregat; logit di luar repo, ARREST_OOF33, bawaan C:/Data/Nightingale/_oof_arrest33):
  python -u 33_tambahan_bersarang.py --tahap latih [--sampel "A 1-30"] > tambahan_bersarang_latih.log
  python -u 33_tambahan_bersarang.py --tahap lapor > tambahan_bersarang.log
Tahap latih bisa diteruskan: lari yang berkasnya sudah ada dilewati.
"""
import argparse
import importlib.util
import os
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c28 = _muat_modul('c28', '28_lari_tetap.py')
c26, c19, c02, c03, c06, c10 = c28.c26, c28.c19, c28.c02, c28.c03, c28.c06, c28.c10
rng = c19.rng
ULANG, SAMPEL, LAB = c28.ULANG, c28.SAMPEL, c28.LAB
OOF = os.environ.get('ARREST_OOF33', 'C:/Data/Nightingale/_oof_arrest33')
RS_LUAR = RS_DALAM = 0


def luar(y, g):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=RS_LUAR).split(np.zeros(len(y)), y, g))


def jalur(nama, b, k, jenis):
    return os.path.join(OOF, f'{nama.replace(" ", "_")}_b{b}_u{k}_{jenis}.npy')


def siapkan(h, sampel):
    """Python sistem: simpan X, y, pid per sampel dan benih, untuk pelatihan di venv-gpu.

    Di venv-gpu (pandas 2.3.3) susunan sampel B 30-365 berbeda dari Python sistem (pandas 3.0.5) (diuji 16 Nov 2026,
    sidik pid/y/unit), jadi venv-gpu tidak boleh menyusun sampel sendiri; ia hanya membaca larik ini."""
    os.makedirs(OOF, exist_ok=True)
    for nama in sampel:
        for b in c19.BENIH:
            if os.path.exists(jalur(nama, b, 'X', 'larik')):
                continue
            S, _ = h[nama](b)
            c28.simpan(jalur(nama, b, 'y', 'larik'), S.y.to_numpy().astype(np.float32))
            c28.simpan(jalur(nama, b, 'g', 'larik'), S.pid.to_numpy())
            c28.simpan(jalur(nama, b, 'X', 'larik'), c02.sinyal(S))
            print(f'  {nama} benih {b} disiapkan', flush=True)


def latih(h, sampel):
    os.makedirs(OOF, exist_ok=True)
    for nama in sampel:
        for b in c19.BENIH:
            kurang = [k for k in range(ULANG)
                      if not (os.path.exists(jalur(nama, b, k, 'uji')) and os.path.exists(jalur(nama, b, k, 'dalam')))]
            if not kurang:
                continue
            if h is None:       # venv-gpu: baca larik yang disiapkan Python sistem
                y = np.load(jalur(nama, b, 'y', 'larik'))
                g = np.load(jalur(nama, b, 'g', 'larik'), allow_pickle=True)
                X = np.load(jalur(nama, b, 'X', 'larik'))
            else:
                S, _ = h[nama](b)
                y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
                X = c02.sinyal(S)
            lip = luar(y, g)
            for k in kurang:
                lama = c28.jalur(nama, b, k)
                if k == RS_LUAR and os.path.exists(lama) and c02.PERANGKAT.type == 'cpu':   # GPU: lari 0 dilatih ulang
                    shutil.copyfile(lama, jalur(nama, b, k, 'uji'))
                    uji = np.load(lama)
                elif os.path.exists(jalur(nama, b, k, 'uji')):     # lari terhenti sesudah logit uji tersimpan
                    uji = np.load(jalur(nama, b, k, 'uji'))
                else:
                    uji = np.zeros(len(y))
                    for tr, te in lip:
                        uji[te] = c02.latih_prediksi(X[tr], y[tr], X[te], k)
                    c28.simpan(jalur(nama, b, k, 'uji'), uji)
                dalam = np.full((len(lip), len(y)), np.nan)
                for f, (tr, _) in enumerate(lip):
                    for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=RS_DALAM).split(
                            np.zeros(len(tr)), y[tr], g[tr]):
                        dalam[f, tr[ite]] = c02.latih_prediksi(X[tr[itr]], y[tr[itr]], X[tr[ite]], k)
                c28.simpan(jalur(nama, b, k, 'dalam'), dalam)
                print(f'  {nama} benih {b} ulangan {k} selesai, AUROC uji {roc_auc_score(y, uji):.3f}', flush=True)
            del X


def _lr():
    return c02.make_pipeline(c02.SimpleImputer(), c02.StandardScaler(), LogisticRegression(max_iter=4000))


def tahap2(uji, dalam, F, y, lip):
    """uji: logit uji (n,); dalam: logit dalam (lipatan, n); F: ciri lain (n, j) atau None."""
    p = np.zeros(len(y))
    for f, (tr, te) in enumerate(lip):
        Ztr = dalam[f, tr][:, None] if F is None else np.column_stack([dalam[f, tr], F[tr]])
        Zte = uji[te][:, None] if F is None else np.column_stack([uji[te], F[te]])
        p[te] = _lr().fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
    return p


def satu_tahap(Z, y, lip):
    p = np.zeros(len(y))
    for tr, te in lip:
        p[te] = _lr().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return p


def rakit(S, y, lip, uji, dalam, s):
    """Seperti c28.rakit, tetapi tahap 2 bersarang. uji: daftar per lari; dalam: daftar per lari (lipatan, n)."""
    lu = [c06.logit(p) for p in uji]
    ld = [c06.logit(np.nan_to_num(d, nan=.5)) for d in dalam]
    lu.append(np.mean(lu, axis=0))
    ld.append(np.mean(ld, axis=0))
    U = S[c03.UKURAN].to_numpy(float)
    sisa = S[[k for k in c03.UKURAN if k not in c26.KELUARGA_LAJU]].to_numpy(float)
    return {'cnn': lu,
            'laju': [tahap2(a, d, s[:, None], y, lip) for a, d in zip(lu, ld)],
            'u13': [tahap2(a, d, sisa, y, lip) for a, d in zip(lu, ld)],
            'u16': [tahap2(a, d, U, y, lip) for a, d in zip(lu, ld)],
            'lr': satu_tahap(U, y, lip),
            'laju0': satu_tahap(s[:, None], y, lip),
            'laju13': satu_tahap(np.column_stack([s, sisa]), y, lip)}


def lapor_sampel(nama, h):
    per, ens, tiap, titik, laju0, laju_st = [], [], [], [], [], []
    for b in c19.BENIH:
        S, u = h[nama](b)
        y, g = S.y.to_numpy().astype(int), S.pid.to_numpy()
        lip = luar(y, g)
        if os.path.exists(jalur(nama, b, 'y', 'larik')):     # dilatih dari larik siapan: pastikan barisnya sama
            assert np.array_equal(np.load(jalur(nama, b, 'y', 'larik')), y.astype(np.float32)), (nama, b, 'y')
            assert np.array_equal(np.load(jalur(nama, b, 'g', 'larik'), allow_pickle=True), g), (nama, b, 'pid')
        uji = [np.load(jalur(nama, b, k, 'uji')) for k in range(ULANG)]
        dalam = [np.load(jalur(nama, b, k, 'dalam')) for k in range(ULANG)]
        s = c26.isi(np.load(c28.jalur(nama, b)))
        r = rakit(S, y, lip, uji, dalam, s)
        kl = np.searchsorted(np.quantile(s, [.2, .4, .6, .8]), s)
        semua = np.arange(len(y))
        tiap.append([c28.nilai(y, r, kl, semua, k) for k in range(ULANG)])
        titik.append(c28.nilai(y, r, kl, semua, -1))
        laju0.append(roc_auc_score(y, r['laju0']))
        laju_st.append(c28.auroc_strata(y, s, kl))
        us, anggota = c19.unit(u)
        tarik = [np.concatenate([anggota[j] for j in rng.choice(us, len(us))]) for _ in range(c19.N_BOOT)]
        per.append(np.array([c28.nilai(y, r, kl, ix, rng.integers(ULANG)) for ix in tarik]))
        ens.append(np.array([c28.nilai(y, r, kl, ix, -1) for ix in tarik]))
        print(f'  [{nama} benih {b} dihitung]', flush=True)
    tiap, titik, per, ens = np.array(tiap), np.array(titik), np.concatenate(per), np.concatenate(ens)
    print(f'\n== {nama} (bersarang, lipatan luar random_state={RS_LUAR})', flush=True)
    print('  AUROC CNN per ulangan (rerata benih): ' + ', '.join(f'{v:.3f}' for v in tiap[:, :, 0].mean(0)), flush=True)
    print(f'  laju saja (LR) {np.mean(laju0):.3f}; laju berstrata {np.mean(laju_st):.3f}', flush=True)
    for j, lab in enumerate(LAB):
        print(f'  {lab:<20s} satu lari {tiap[:, :, j].mean():+.3f} {c10.ci(per[:, j])} rentang ulangan '
              f'[{tiap[:, :, j].mean(0).min():+.3f}, {tiap[:, :, j].mean(0).max():+.3f}] | ansambel '
              f'{titik[:, j].mean():+.3f} {c10.ci(ens[:, j])}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tahap', choices=['siapkan', 'latih', 'lapor'], required=True)
    ap.add_argument('--sampel', action='append', choices=SAMPEL)
    ap.add_argument('--dari-larik', action='store_true', help='venv-gpu: latih dari larik siapan, tanpa menyusun sampel')
    a = ap.parse_args()
    sampel = a.sampel or SAMPEL
    if a.tahap == 'latih' and a.dari_larik:
        print(f'perangkat: {c02.PERANGKAT}', flush=True)
        latih(None, sampel)
        return
    h = c28.susunan()
    if a.tahap == 'siapkan':
        siapkan(h, sampel)
        return
    if a.tahap == 'latih':
        latih(h, sampel)
        return
    ada = [n for n in sampel if all(os.path.exists(jalur(n, b, k, j)) for b in c19.BENIH for k in range(ULANG)
                                    for j in ('uji', 'dalam'))]
    print(f'sampel lengkap: {ada}', flush=True)
    for n in ada:
        lapor_sampel(n, h)


if __name__ == '__main__':
    main()
