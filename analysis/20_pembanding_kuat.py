"""Nightingale arrest-ntuh-ecg — pembanding dengan upaya wajar, untuk telaah ulang panel JIKI 28 Oktober 2026
(Riset/EKG/Telaah/2026-10-28b-Arrest-JIKI/report.md, temuan kritis "CNN is a straw man" dan mayor "trees
untuned").

Tiga lengan, masing-masing dijalankan terpisah (--lengan):
  rocket  MiniRocket multivariat (aeon, 10.000 kernel) atas 12 × 2.500 sinyal yang sama dengan CNN,
          StandardScaler + RidgeClassifierCV (alpha 1e-3..1e3, LOO bawaan), dipasang per lipatan luar.
          Pembanding baku deret waktu (Dempster dkk.). CPU; perlu venv dengan aeon (~/venv-aeon).
  pohon   HistGradientBoosting pada pembacaan mesin (bendera + ukuran 12SL, 38 masukan) dengan setelan
          dipilih di dalam lipatan latih: GridSearchCV atas learning_rate {0,03; 0,1}, max_depth {2; 3; tanpa},
          max_iter {100; 300}, min_samples_leaf {20; 50}, StratifiedGroupKFold 3 lipatan dalam, skor AUROC.
  cnn     CNN 02 disetel: tiap lipatan luar dibagi lagi (StratifiedGroupKFold 4, satu lipatan jadi validasi);
          kisi lr {1e-3; 3e-4} × lebar {1×; 2× kanal}, augmentasi (geser melingkar ≤ 1 dtk, skala amplitudo
          0,8–1,2), AdamW weight decay 1e-4, sampai 60 epoch; dipilih (lr, lebar, epoch) dengan AUROC
          validasi tertinggi, lalu dilatih ulang pada seluruh lipatan latih dengan setelan itu. GPU bila ada.

Sampel: rancangan P semua sebab dan bukan jantung (pencocokan 09/10), rancangan B 1-30 dan 30-365 (04);
benih pencocokan 0-2; lipatan luar StratifiedGroupKFold 5 (grup = pasien) seperti 19. Dilaporkan AUROC
lengan dan selisih berpasangan laju ventrikel − lengan dan pengukuran 12SL (LR) − lengan, bootstrap set
(P) atau pasien (B), digabung atas benih.

Hanya agregat. Pakai:
  ~/venv-aeon/Scripts/python.exe -u 20_pembanding_kuat.py --lengan rocket > pembanding_kuat_rocket.log
  python -u 20_pembanding_kuat.py --lengan pohon > pembanding_kuat_pohon.log
  ~/venv-gpu/Scripts/python.exe -u 20_pembanding_kuat.py --lengan cnn > pembanding_kuat_cnn.log
"""
import argparse
import importlib.util
import itertools
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _muat_modul(nama, berkas):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(__file__), berkas))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


c19 = _muat_modul('c19', '19_berpasangan.py')
c02, c03, c06, c09, c10 = c19.c02, c19.c03, c19.c06, c19.c09, c19.c10
c19.rng = c10.rng = np.random.default_rng(20261101)
LAJU, UKUR = c19.LAJU, c19.UKUR
EPOCH_MAKS = 60


# ---------- lengan

def rocket(X, y, g, b):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g):
        t = MiniRocket(n_kernels=10000, random_state=b)
        Ftr = t.fit_transform(X[tr])
        m = c02.make_pipeline(c02.StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
        p[te] = m.fit(Ftr, y[tr]).decision_function(t.transform(X[te]))
    return p


def pohon(S, y, g, b):
    from sklearn.model_selection import GridSearchCV
    B = S[list(c06.BENDERA) + c03.UKURAN].to_numpy(float)
    kisi = {'learning_rate': [0.03, 0.1], 'max_depth': [2, 3, None], 'max_iter': [100, 300],
            'min_samples_leaf': [20, 50]}
    p, pilih = np.zeros(len(y)), []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(B, y, g):
        dalam = list(StratifiedGroupKFold(3, shuffle=True, random_state=b).split(B[tr], y[tr], g[tr]))
        cari = GridSearchCV(c02.HistGradientBoostingClassifier(random_state=b), kisi, scoring='roc_auc',
                            cv=dalam, n_jobs=2).fit(B[tr], y[tr])
        pilih.append(tuple(sorted(cari.best_params_.items())))
        p[te] = cari.predict_proba(B[te])[:, 1]
    return p, pilih


class JaringanLebar(c02.nn.Module):
    def __init__(self, lebar):
        super().__init__()
        nn = c02.nn
        blok, cin = [], 12
        for cout in (32, 64, 64, 128):
            blok += [nn.Conv1d(cin, cout * lebar, 7, padding=3), nn.BatchNorm1d(cout * lebar), nn.ReLU(),
                     nn.MaxPool1d(4)]
            cin = cout * lebar
        self.fitur = nn.Sequential(*blok)
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(cin, 1))

    def forward(self, x):
        return self.kepala(self.fitur(x).mean(dim=2)).squeeze(1)


def _latih(xt, yt, lr, lebar, benih, dev, n_epoch, xv=None, yv=None):
    """Melatih; bila ada validasi, kembalikan AUROC validasi per epoch."""
    torch, nn = c02.torch, c02.nn
    torch.manual_seed(benih)
    gen = torch.Generator(device='cpu').manual_seed(benih)
    m = JaringanLebar(lebar).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(yt.mean())
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    auc = []
    for _ in range(n_epoch):
        m.train()
        perm = torch.randperm(len(yt), generator=gen)
        for i in range(0, len(yt), c02.BATCH):
            ix = perm[i:i + c02.BATCH].to(dev)
            xb = xt[ix]
            geser = int(torch.randint(-250, 251, (1,), generator=gen))
            skala = (0.8 + 0.4 * torch.rand(len(ix), 1, 1, generator=gen)).to(dev)
            xb = torch.roll(xb, geser, dims=2) * skala
            opt.zero_grad()
            loss(m(xb), yt[ix]).backward()
            opt.step()
        if xv is not None:
            auc.append(roc_auc_score(yv, _prediksi(m, xv)))
    return m, auc


def _prediksi(m, x):
    torch = c02.torch
    m.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(m(x[i:i + 512])) for i in range(0, len(x), 512)]).cpu().numpy()


def cnn(X, y, g, b):
    torch = c02.torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    p, pilih = np.zeros(len(y)), []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g):
        sd = X[tr].std() + 1e-6
        dalam_tr, dalam_va = next(StratifiedGroupKFold(4, shuffle=True, random_state=b).split(
            np.zeros(len(tr)), y[tr], g[tr]))
        ke = lambda ix: torch.from_numpy(X[ix] / sd).to(dev)
        xt, yt = ke(tr[dalam_tr]), torch.tensor(y[tr][dalam_tr], dtype=torch.float32, device=dev)
        xv = ke(tr[dalam_va])
        terbaik = (-1, None)
        for lr in (1e-3, 3e-4):
            for lebar in (1, 2):
                _, auc = _latih(xt, yt, lr, lebar, b, dev, EPOCH_MAKS, xv, y[tr][dalam_va])
                e = int(np.argmax(auc))
                if auc[e] > terbaik[0]:
                    terbaik = (auc[e], (lr, lebar, e + 1))
        lr, lebar, n_ep = terbaik[1]
        pilih.append((lr, lebar, n_ep, round(terbaik[0], 3)))
        del xt, xv
        xa = ke(tr)
        m, _ = _latih(xa, torch.tensor(y[tr], dtype=torch.float32, device=dev), lr, lebar, b, dev, n_ep)
        p[te] = _prediksi(m, ke(te))
        del xa, m
        if dev == 'cuda':
            torch.cuda.empty_cache()
    return p, pilih


# ---------- sampel

def sampel():
    """(nama, benih, S, unit bootstrap) untuk keempat sampel 19."""
    pre, calon, kp, _ = c09.data_p()
    sebab = kp.cause_of_cardiac_arrest
    bukan = ((sebab != 'cardiac event') & sebab.notna() & (sebab != 'unknown')).to_numpy()
    for nama, m in [('P semua sebab', np.ones(len(kp), bool)), ('P bukan jantung (tanpa unknown)', bukan)]:
        for b in c19.BENIH:
            S = c10.susun(pre, kp[m], calon, b)
            yield nama, b, S, S.set.to_numpy()
    pre, kas, _, jendela = c19.ab_data()
    for w, (m, sama) in jendela.items():
        for b in c19.BENIH:
            S = c19.susun_ab(pre, kas[m], sama, b)
            yield f'B {w}', b, S, S.pid.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengan', choices=['rocket', 'pohon', 'cnn'], required=True)
    ap.add_argument('--coba', action='store_true', help='hanya sampel dan benih pertama (uji asap)')
    a = ap.parse_args()
    lengan = a.lengan
    nama_lengan = {'rocket': 'MiniRocket', 'pohon': 'mesin (pohon disetel)', 'cnn': 'CNN disetel'}[lengan]
    print(f'== Lengan {nama_lengan}: AUROC dan pembanding - lengan berpasangan, 3 benih', flush=True)
    hasil = {}
    for nama, b, S, u in (itertools.islice(sampel(), 1) if a.coba else sampel()):
        y, g = S.y.to_numpy().astype(np.float32), S.pid.to_numpy()
        pilih = None
        if lengan == 'pohon':
            pa, pilih = pohon(S, y, g, b)
        else:
            X = c02.sinyal(S)
            pa, pilih = (rocket(X, y, g, b), None) if lengan == 'rocket' else cnn(X, y, g, b)
            del X
        pb = c19.pembanding(S, y, g)
        h = hasil.setdefault(nama, {'auc': [], 'laju': [], 'ukur': [], 'd_laju': [], 'd_ukur': []})
        h['auc'].append(roc_auc_score(y, pa))
        h['laju'].append(roc_auc_score(y, pb[LAJU]))
        h['ukur'].append(roc_auc_score(y, pb[UKUR]))
        for k, f in (('d_laju', LAJU), ('d_ukur', UKUR)):
            h[k].append(c19.boot(y, u, lambda ix, q=pb[f]: roc_auc_score(y[ix], q[ix]) - roc_auc_score(y[ix], pa[ix])))
        print(f'  {nama} benih {b}: {nama_lengan} {h["auc"][-1]:.3f}'
              + (f'  setelan {pilih}' if pilih else ''), flush=True)
    for nama, h in hasil.items():
        print(f'  {nama:<32s} AUROC {nama_lengan} {np.mean(h["auc"]):.3f} (rentang {min(h["auc"]):.3f}-'
              f'{max(h["auc"]):.3f}); laju {np.mean(h["laju"]):.3f}; ukuran LR {np.mean(h["ukur"]):.3f}', flush=True)
        for k, f in (('d_laju', LAJU), ('d_ukur', UKUR)):
            d = np.concatenate(h[k])
            print(f'  {nama:<32s} {f + " - " + nama_lengan:<50s} {np.mean(h["laju" if k == "d_laju" else "ukur"]) - np.mean(h["auc"]):+.3f} '
                  f'{c10.ci(d)}  P(<=0) {np.mean(d <= 0):.3f}', flush=True)


if __name__ == '__main__':
    main()
