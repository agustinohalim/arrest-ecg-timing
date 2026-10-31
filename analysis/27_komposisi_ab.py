"""Nightingale arrest-ntuh-ecg — komposisi kontrol rancangan A lawan B dan selang pembanding (telaah panel JIKI 30 Oktober
2026 (b), `../Telaah/2026-10-30b-Arrest-JIKI/report.md`, mayor "design A-to-B contrast changes the control population"
dan "add CIs to every AUROC in Table 1").

Bagian 1: untuk tiap jendela, kontrol A (30-730 hari) lawan B (jendela sama dengan kasus), benih 0-2: tumpang tindih
pasien, usia, pria, tahun, porsi MUSE 10, laju median. Tanpa sinyal.
Bagian 2: AUROC laju mentah dan ukuran LR pada A dan B dengan selang bootstrap pasien, digabung atas benih.

CPU. Hanya agregat. Pakai: python -u 27_komposisi_ab.py > komposisi_ab.log
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


c26 = _muat_modul('c26', '26_laju_sinyal.py')
c19, c10 = c26.c19, c26.c10


def rangkum(K, muse):
    v = K.ecg_id.map(muse)
    return (f'n {len(K):>4d}  usia {K.usia.mean():.1f}  pria {K.pria.mean():.3f}  tahun '
            f'{K.tahun.astype(str).str[:4].astype(int).mean():.2f}  MUSE 10 {(v == "10").mean():.3f}  '
            f'laju median {K.ventricular_rate.median():.1f}')


def main():
    pre, kas, lama, jendela = c19.ab_data()
    muse = c26.versi_muse()
    print('== 1. Kontrol A lawan B, per benih', flush=True)
    print(f'  EKG pre per pasien kontrol, maks {lama.groupby("patient_ngsci_id").size().max()}', flush=True)
    for w, (m, sama) in jendela.items():
        ca, cb = set(lama.patient_ngsci_id), set(sama.patient_ngsci_id)
        print(f'  {w}: calon A {len(ca)}, calon B {len(cb)}, calon di keduanya {len(ca & cb)}', flush=True)
        for b in c19.BENIH:
            Sa = c19.susun_ab(pre, kas[m], lama, b)
            Sb = c19.susun_ab(pre, kas[m], sama, b)
            Ka, Kb = Sa[Sa.y == 0], Sb[Sb.y == 0]
            irisan = len(set(Ka.pid) & set(Kb.pid))
            print(f'    benih {b} pasien kontrol di A dan B {irisan} ({irisan / len(Kb):.3f} dari B)', flush=True)
            print(f'      A {rangkum(Ka, muse)}', flush=True)
            print(f'      B {rangkum(Kb, muse)}', flush=True)

    print('\n== 2. AUROC dengan selang, 3 benih, bootstrap pasien digabung', flush=True)
    for w, (m, sama) in jendela.items():
        for r, calon in (('A', lama), ('B', sama)):
            bs = {}
            for b in c19.BENIH:
                S = c19.susun_ab(pre, kas[m], calon, b)
                y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
                for f, p in c19.pembanding(S, y, g).items():
                    if f != c19.MESIN:
                        bs.setdefault(f, []).append(c19.boot(y, g, lambda ix: roc_auc_score(y[ix], p[ix])))
            for f, v in bs.items():
                v = np.concatenate(v)
                print(f'  {r} {w:<7s} {f:<26s} {v.mean():.3f} {c10.ci(v)}', flush=True)


if __name__ == '__main__':
    main()
