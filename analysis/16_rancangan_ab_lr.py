"""Nightingale arrest-ntuh-ecg — rancangan A (replikasi) dan B (jarak disamakan) dengan pembacaan mesin.

Pemicu: 14/15 — dalam rancangan P, pengukuran 12SL dengan LR memberi 0,607 (laju jantung), padahal ciri
sinyal HGB dan CNN dekat 0,52. Rancangan A dan B (04_replikasi.py) hanya diuji dengan usia + jk, ciri
sinyal HGB, dan CNN, sehingga klaim "pencocokan jarak menghapus sinyal 1-30 hari" belum diuji dengan
pembacaan mesin. Di sini: rancangan dan pencocokan sama persis dengan 04 (c04.cocokkan, benih 0-2),
masukan baru: laju ventrikel satu peubah (tanpa pelatihan), pengukuran 12SL (LR), pembacaan mesin (LR),
pembacaan mesin (HGB), usia + jk (LR). Tanpa CNN. Hanya agregat.
Pakai: python -u 16_rancangan_ab_lr.py > rancangan_ab_lr.log
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


c02 = _muat_modul('c02', '02_cnn_arrest.py')
c03 = _muat_modul('c03', '03_cnn_kunjungan.py')
c04 = _muat_modul('c04', '04_replikasi.py')
c06 = _muat_modul('c06', '06_revisi.py')
BENIH = c04.BENIH


def model(S):
    y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy() if 'pid' in S else S.patient_ngsci_id.to_numpy()
    U = S[c03.UKURAN].to_numpy(float)
    M = np.column_stack([S[list(c06.BENDERA)].to_numpy(float), U])
    vr = S.ventricular_rate.to_numpy(float)
    ok = ~np.isnan(vr)
    return {
        'laju ventrikel saja (mentah)': roc_auc_score(y[ok], vr[ok]),
        'usia + jenis kelamin (LR)': roc_auc_score(y, c02.oof(S[['usia', 'pria']].to_numpy(float), y, g, 1, 'lr')),
        'pengukuran 12SL (LR)': roc_auc_score(y, c06.oof_lr_logit(U, y, g)),
        'pembacaan mesin (LR)': roc_auc_score(y, c06.oof_lr_logit(M, y, g)),
        'pembacaan mesin (HGB)': roc_auc_score(y, c02.oof(M, y, g, 1, 'hgb')),
    }


def main():
    D = c04.tandai_kasus(c02.muat())
    pre = D[D.ecg_timetag.isin(['[0]pre', 'pre'])].reset_index(drop=True)
    n0 = len(pre)
    pre = pre.merge(c03.ukuran(), on='ecg_id', how='left').merge(c06.pernyataan(), on='ecg_id', how='left')
    pre = pre.fillna({b: 0 for b in c06.BENDERA})
    assert len(pre) == n0
    kas = pre[pre.y == 1]
    k = pre[(pre.y == 0) & pre.tahun.isin(c04.SAH)]
    lama = k[(k.jarak >= 30) & (k.jarak <= 730)]
    rancangan = [
        ('A  0-1 hari lawan kontrol 30-730', (kas.jarak > 0) & (kas.jarak <= 1), lama),
        ('A  1-30 hari lawan kontrol 30-730', (kas.jarak > 1) & (kas.jarak <= 30), lama),
        ('B  1-30 hari lawan kontrol 1-30', (kas.jarak > 1) & (kas.jarak <= 30), k[(k.jarak > 1) & (k.jarak <= 30)]),
        ('A  30-365 hari lawan kontrol 30-730', (kas.jarak > 30) & (kas.jarak <= 365), lama),
        ('B  30-365 hari lawan kontrol 30-365', (kas.jarak > 30) & (kas.jarak <= 365),
         k[(k.jarak > 30) & (k.jarak <= 365)]),
    ]
    print('Rerata AUROC atas 3 benih pencocokan [min, maks]; laju mentah: AUROC > 0,5 = kasus lebih cepat', flush=True)
    for judul, m, calon in rancangan:
        kk = kas[m]
        hasil = {}
        for b in BENIH:
            ix = np.concatenate([kk.index.to_numpy(), c04.cocokkan(kk, calon, b)])
            S = pre.loc[np.sort(ix)].reset_index(drop=True)
            S['pid'] = S.patient_ngsci_id.astype(str)
            for f, a in model(S).items():
                hasil.setdefault(f, []).append(a)
            if b == 0:
                vr = S.groupby('y').ventricular_rate.median()
                n = (int(S.y.sum()), int((1 - S.y).sum()))
        print(f'\n{judul}: kasus {n[0]} / kontrol {n[1]}; median laju kasus {vr[1]:.0f}, kontrol {vr[0]:.0f} '
              f'(benih 0)', flush=True)
        for f, v in hasil.items():
            print(f'  {f:<30s} {np.mean(v):.3f} [{min(v):.3f}, {max(v):.3f}]', flush=True)


if __name__ == '__main__':
    main()
