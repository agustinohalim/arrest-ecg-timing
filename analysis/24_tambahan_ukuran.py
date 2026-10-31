"""Nightingale arrest-ntuh-ecg — apakah ukuran 12SL menambah di atas CNN? (telaah panel JIKI 30 Oktober 2026,
`../Telaah/2026-10-30-Arrest-JIKI/report.md`, kritis "the test adds a condition the quoted statement does not
make": klaim Oberdier "DL models captured their influence" adalah klaim cakupan, jadi uji langsungnya adalah
tambahan ukuran di atas CNN, bukan ukuran lawan CNN).

Per sampel (rancangan A dan B, 1-30 dan 30-365 hari, pencocokan 04; rancangan P semua sebab, pencocokan 09/10;
benih 0-2), CNN dilatih ulang 2 kali seperti 19. Gabungan = LR atas [logit CNN OOF, 16 ukuran 12SL], divalidasi
silang lagi (06.oof_lr_logit). Selisih gabungan - CNN berpasangan: bootstrap pasien (A, B) atau set (P), satu
ulangan latih acak per tarikan, digabung atas benih. Juga: ukuran LR - CNN pada lari yang sama (pembanding 19/23).

CPU. Hanya agregat. Pakai: python -u 24_tambahan_ukuran.py > tambahan_ukuran.log
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


c19 = _muat_modul('c19', '19_berpasangan.py')
c02, c03, c06, c09, c10 = c19.c02, c19.c03, c19.c06, c19.c09, c19.c10
rng = c19.rng


def satu_sampel(S, y, g, u, hasil):
    X = c02.sinyal(S)
    ps = c19.cnn_ulang(X, y, g)
    del X
    U = S[c03.UKURAN].to_numpy(float)
    pu = c06.oof_lr_logit(U, y, g)
    pg = [c06.oof_lr_logit(np.column_stack([c06.logit(p), U]), y, g) for p in ps]

    def beda(ix):
        k = rng.integers(len(ps))
        a = roc_auc_score(y[ix], ps[k][ix])
        return roc_auc_score(y[ix], pg[k][ix]) - a, roc_auc_score(y[ix], pu[ix]) - a

    us, anggota = c19.unit(u)
    d = np.array([beda(np.concatenate([anggota[j] for j in rng.choice(us, len(us))])) for _ in range(c19.N_BOOT)])
    hasil['cnn'].append(np.mean([roc_auc_score(y, p) for p in ps]))
    hasil['gab'].append(np.mean([roc_auc_score(y, p) for p in pg]))
    hasil['ukur'].append(roc_auc_score(y, pu))
    hasil['d_gab'].append(d[:, 0])
    hasil['d_ukur'].append(d[:, 1])


def lapor(nama, h):
    print(f'  {nama:<14s} AUROC CNN {np.mean(h["cnn"]):.3f}, ukuran LR {np.mean(h["ukur"]):.3f}, '
          f'CNN + ukuran (LR) {np.mean(h["gab"]):.3f}', flush=True)
    for k, lab in (('d_gab', '(CNN + ukuran) - CNN'), ('d_ukur', 'ukuran LR - CNN')):
        d = np.concatenate(h[k])
        print(f'  {nama:<14s} {lab:<22s} {np.mean(d):+.3f} {c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}', flush=True)


def baru():
    return {'cnn': [], 'gab': [], 'ukur': [], 'd_gab': [], 'd_ukur': []}


def main():
    print('== Tambahan ukuran 12SL di atas CNN, 3 benih, CNN 2 ulangan/benih', flush=True)
    pre, kas, lama, jendela = c19.ab_data()
    for w, (m, sama) in jendela.items():
        for r, calon in (('A', lama), ('B', sama)):
            h = baru()
            for b in c19.BENIH:
                S = c19.susun_ab(pre, kas[m], calon, b)
                y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
                satu_sampel(S, y, g, g, h)
                print(f'  {r} {w} benih {b} selesai', flush=True)
            lapor(f'{r} {w}', h)
    pre_p, calon_p, kp, _ = c09.data_p()
    h = baru()
    for b in c19.BENIH:
        S = c10.susun(pre_p, kp, calon_p, b)
        y, g, s = S.y.to_numpy().astype(np.float32), S.pid.to_numpy(), S.set.to_numpy()
        satu_sampel(S, y, g, s, h)
        print(f'  P benih {b} selesai', flush=True)
    lapor('P semua sebab', h)


if __name__ == '__main__':
    main()
