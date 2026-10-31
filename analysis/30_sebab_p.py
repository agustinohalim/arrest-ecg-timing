"""Nightingale arrest-ntuh-ecg — laju, ukuran, dan ansambel CNN per sebab di rancangan P (telaah panel JIKI 31 Oktober
2026 (c), `../Telaah/2026-10-31c-Arrest-JIKI/report.md`, kritis "'mostly heart rate' reverses in cardiac-cause arrests").

Tanpa melatih: OOF skrip 28 (75 CNN dilatih pada semua sebab, ansambel = rerata logit lima ulangan), dinilai pada set
cocokan yang kasusnya sebab jantung atau bukan jantung (unknown keluar). Regresi logistik ukuran dan tambahan dipasang di
dalam tiap subset, dengan lipatan kelompok pasien. Bootstrap set, 3 benih.

Dicetak per subset: AUROC laju sinyal, ukuran LR, ansambel; ukuran - laju; ansambel - laju; (laju + ansambel) - laju;
(laju + ukuran) - laju; ukuran - ansambel. Lalu selisih subset jantung - bukan jantung untuk laju, ukuran, dan ansambel.

Aturan bacaan, ditulis sebelum hasil ada:
  ukuran - laju pada sebab jantung, batas bawah > 0        -> di sebab jantung sinyal ukuran bukan laju; "mostly heart
                                                              rate" hanya berlaku pada gabungan
  ansambel - laju pada sebab jantung, batas bawah > 0      -> CNN melampaui laju pada sebab jantung
  (laju + ansambel) - laju pada sebab jantung, batas bawah > 0 -> CNN menambah di atas laju pada sebab jantung
  selang mengandung 0                                      -> tidak terpisah; tidak boleh dibaca sebagai sama
Subset dan analisis ini post hoc, jadi hasilnya arah, bukan temuan terkonfirmasi.

CPU. Hanya agregat. Pakai: python -u 30_sebab_p.py > sebab_p.log
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
c19, c06, c10, c26, c03, c09 = c28.c19, c28.c06, c28.c10, c28.c26, c28.c03, c28.c09
rng = c19.rng
LAB = ('laju', 'ukuran LR', 'ansambel', 'ukuran - laju', 'ansambel - laju', '(laju + ansambel) - laju', '(laju + ukuran) - laju',
       'ukuran - ansambel')


def satu_subset(S, set_sebab, sebab, b):
    """Skor satu benih pada set yang kasusnya bersebab `sebab`."""
    m = S.set.map(set_sebab).isin(sebab).to_numpy()
    T = S[m].reset_index(drop=True)
    y, g, u = T.y.to_numpy().astype(int), T.pid.to_numpy(), T.set.to_numpy()
    e = np.mean([c06.logit(np.load(c28.jalur('P', b, k))) for k in range(c28.ULANG)], axis=0)[m]
    s = c26.isi(np.load(c28.jalur('P', b)))[m]
    U = T[c03.UKURAN].to_numpy(float)
    return {'y': y, 'u': u, 'laju': s, 'ukur': c06.oof_lr_logit(U, y, g), 'ens': e,
            'r0': c06.oof_lr_logit(s[:, None], y, g),
            're': c06.oof_lr_logit(np.column_stack([s, e]), y, g),
            'ru': c06.oof_lr_logit(np.column_stack([s, U]), y, g)}


def ukur(d, ix):
    y = d['y'][ix]
    a = {k: roc_auc_score(y, d[k][ix]) for k in ('laju', 'ukur', 'ens', 'r0', 're', 'ru')}
    return (a['laju'], a['ukur'], a['ens'], a['ukur'] - a['laju'], a['ens'] - a['laju'], a['re'] - a['r0'], a['ru'] - a['r0'],
            a['ukur'] - a['ens'])


def tarik(d):
    us, anggota = c19.unit(d['u'])
    return np.array([ukur(d, np.concatenate([anggota[j] for j in rng.choice(us, len(us))])) for _ in range(c19.N_BOOT)])


def main():
    pre, calon, kp, _ = c09.data_p()
    h = {'jantung': lambda v: ['cardiac event'],
         'bukan jantung': lambda v: [x for x in v if x not in ('cardiac event', 'unknown')]}
    titik, boot = {n: [] for n in h}, {n: [] for n in h}
    for b in c19.BENIH:
        S = c10.susun(pre, kp, calon, b)
        kasus = S[S.y == 1]
        set_sebab = dict(zip(kasus.set, kasus.cause_of_cardiac_arrest))
        ragam = set(kasus.cause_of_cardiac_arrest.dropna())
        for n, f in h.items():
            d = satu_subset(S, set_sebab, f(ragam), b)
            titik[n].append(ukur(d, np.arange(len(d['y']))))
            boot[n].append(tarik(d))
            print(f'  [{n} benih {b}: {int(d["y"].sum())} kasus, {int((1 - d["y"]).sum())} kontrol]', flush=True)
    for n in h:
        t, q = np.array(titik[n]), np.concatenate(boot[n])
        print(f'== P, sebab {n}', flush=True)
        for j, lab in enumerate(LAB):
            print(f'  {lab:<26s} {t[:, j].mean():+.3f} {c10.ci(q[:, j])}', flush=True)
    t = np.array(titik['jantung']) - np.array(titik['bukan jantung'])
    q = np.concatenate(boot['jantung']) - np.concatenate(boot['bukan jantung'])
    print('== Jantung - bukan jantung', flush=True)
    for j, lab in enumerate(LAB[:3]):
        print(f'  {lab:<26s} {t[:, j].mean():+.3f} {c10.ci(q[:, j])}', flush=True)


if __name__ == '__main__':
    main()
