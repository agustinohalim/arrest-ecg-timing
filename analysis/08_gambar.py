"""Gambar Naskah_Arrest.md. Angka disalin dari log (replikasi.log, replikasi_jarak_sama.log, sebab.log,
cnn_kunjungan_bersih.log, revisi.log) lewat Hasil_Uji_Arrest.md; tidak membaca data Nightingale.

  Gambar 1  alur kohort: kelompok kasus dan kontrol ke rancangan A, B, P, C
  Gambar 2  AUROC CNN dan ciri sederhana per rancangan dan sebab: titik = rerata benih pencocokan
            (A, B, P) atau OOF tunggal (C); garis = IK bootstrap pasien benih 0 (A, B, P) atau IK (C)

Pakai: python 08_gambar.py  -> Gambar/Gambar1_alur.png, Gambar/Gambar2_auroc.png (300 dpi)
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

KELUAR = os.path.join(os.path.dirname(__file__), 'Gambar')
os.makedirs(KELUAR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
JINGGA, BIRU = '#c55a11', '#1f4e79'


def kotak(ax, x, y, w, h, teks, warna='#f2f2f2'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.01,rounding_size=0.01',
                                fc=warna, ec='#444444', lw=0.7))
    ax.text(x, y, teks, ha='center', va='center', fontsize=7, linespacing=1.3)


def panah(ax, x0, y0, x1, y1, warna='#444444'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=warna, shrinkA=0, shrinkB=0))


def garis(ax, xs, ys, warna):
    ax.plot(xs, ys, color=warna, lw=0.9)


def gambar1():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.20, 1)
    ax.axis('off')
    kotak(ax, 0.27, 0.93, 0.44, 0.09, 'Case group: 974 patients with ROSC, 2011–2019\n'
          '1,686 ECGs: 605 pre, 838 at ROSC, 243 at 24 h', '#fde3d4')
    kotak(ax, 0.73, 0.93, 0.40, 0.09, 'Control group: 9,976 ED patients, 2015–2019\n'
          '16,386 ECGs: 9,976 index, 6,410 pre', '#d9e7f5')
    kotak(ax, 0.60, 0.76, 0.34, 0.08, 'Control pre ECGs, 2015–2017 only\n'
          '(2018–2019 date shifts inconsistent)', '#d9e7f5')
    panah(ax, 0.60, 0.885, 0.60, 0.80, BIRU)

    # Bingkai revisi 3 (30 Okt 2026): rancangan C dan subkelompok sebab tidak lagi di naskah.
    y, w, h = 0.36, 0.28, 0.24
    xs = [0.17, 0.50, 0.83]
    teks = [
        'Design A (replication)\nCases: pre 0–1, 1–30,\n30–365 days (220/120/175)\n'
        'Controls: pre 30–730 days\n(2,226 candidates)\nAge × sex matched, 5:1',
        'Design B (matched window)\nCases and controls from\nthe same window\n'
        '1–30 days: 120 / 548\n30–365 days: 175 / 862\nAge × sex matched, 5:1',
        'Design P (nearest gap)\nCases: 2–730 days (320)\n'
        'Out-of-hospital 185,\nin-hospital 116, other 19\n'
        'Nearest gap, age, sex; 5:1',
    ]
    for x, t in zip(xs, teks):
        kotak(ax, x, y, w, h, t, '#eeeeee')
    atas = y + h / 2

    # kasus (jingga): turun dari kotak kasus ke rel, lalu ke ketiga rancangan (sisi kiri kotak)
    garis(ax, [0.10, 0.10], [0.885, 0.62], JINGGA)
    garis(ax, [0.10, xs[2] - 0.05], [0.62, 0.62], JINGGA)
    for x in xs:
        panah(ax, x - 0.05, 0.62, x - 0.05, atas, JINGGA)
    ax.text(0.11, 0.70, 'case ECGs', color=JINGGA, fontsize=6.5)

    # kontrol pre 2015-2017 (biru): ke A, B, P (sisi kanan kotak)
    garis(ax, [0.60, 0.60], [0.72, 0.56], BIRU)
    garis(ax, [xs[0] + 0.05, xs[2] + 0.05], [0.56, 0.56], BIRU)
    for x in xs:
        panah(ax, x + 0.05, 0.56, x + 0.05, atas, BIRU)
    ax.text(0.59, 0.66, 'control pre ECGs', color=BIRU, fontsize=6.5, ha='right')

    for eks in ('png', 'pdf'):
        fig.savefig(os.path.join(KELUAR, f'Gambar1_alur.{eks}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# (label, CNN titik, bawah, atas, ciri titik, bawah, atas, mesin titik, bawah, atas | None)
# A, B: titik = rerata benih, garis = IK bootstrap pasien benih 0. P: titik = rerata benih (uji_sebab.log;
# PEA/asistol dari telaah_lanjut.log), garis = IK bootstrap set cocokan benih 0 (telaah_lanjut.log, 10 Okt).
BARIS = [
    ('A  1–30 d vs 30–730 d (published windows)', 0.643, 0.579, 0.699, 0.626, 0.599, 0.703, None),
    ('B  1–30 d vs 1–30 d', 0.519, 0.447, 0.563, 0.469, 0.393, 0.510, None),
    ('A  30–365 d vs 30–730 d (published windows)', 0.574, 0.517, 0.609, 0.540, 0.488, 0.581, None),
    ('B  30–365 d vs 30–365 d', 0.516, 0.486, 0.579, 0.534, 0.461, 0.557, None),
    ('P  all causes (320)', 0.546, 0.498, 0.564, 0.519, 0.472, 0.537, (0.525, 0.485, 0.558)),
    ('P  cardiac cause (104)', 0.630, 0.579, 0.693, 0.601, 0.554, 0.670, (0.640, 0.602, 0.705)),
    ('P  non-cardiac cause (216)', 0.572, 0.529, 0.609, 0.473, 0.441, 0.525, (0.509, 0.478, 0.558)),
    ('P  initial VT/VF (39)', 0.616, 0.513, 0.717, 0.702, 0.621, 0.784, (0.645, 0.549, 0.729)),
    ('P  initial PEA or asystole (272)', 0.544, 0.503, 0.577, 0.460, 0.423, 0.500, (0.523, 0.493, 0.564)),
    ('C  same visit, all (178)', 0.791, 0.757, 0.826, 0.719, 0.678, 0.758, (0.747, 0.714, 0.780)),
]
HIJAU = '#2e7d32'


def gambar2():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    n = len(BARIS)
    for i, (lab, c, cl, ch, f, fl, fh, m) in enumerate(BARIS):
        yy = n - 1 - i
        ax.plot([cl, ch], [yy + 0.2] * 2, color=BIRU, lw=1.2)
        ax.plot(c, yy + 0.2, 'o', color=BIRU, ms=4.5, label='CNN' if i == 0 else None)
        ax.plot([fl, fh], [yy] * 2, color=JINGGA, lw=1.2)
        ax.plot(f, yy, 's', color=JINGGA, ms=4, label='Simple signal features' if i == 0 else None)
        if m:
            ax.plot([m[1], m[2]], [yy - 0.2] * 2, color=HIJAU, lw=1.2)
            ax.plot(m[0], yy - 0.2, 'D', color=HIJAU, ms=3.8, label='Machine reading' if i == 4 else None)
    ax.axvline(0.5, ls='--', color='#888888', lw=0.8)
    for batas in (n - 4.5, n - 9.5):
        ax.axhline(batas, color='#cccccc', lw=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([b[0] for b in BARIS][::-1])
    ax.set_xlim(0.38, 0.86)
    ax.set_xlabel('AUROC')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for eks in ('png', 'pdf'):
        fig.savefig(os.path.join(KELUAR, f'Gambar2_auroc.{eks}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# Versi JIKI (28 Okt 2026): titik saja, tanpa garis IK (IK tiap model berasal dari skrip berbeda).
# (label, laju jantung, pengukuran LR, mesin LR, mesin pohon, CNN). A, B: rancangan_ab_lr.log dan
# replikasi*.log; P: ci_pembanding.log, mesin_lr.log, telaah_jiki.log; C: rancangan_c_lr.log (mesin dengan
# usia/jk). Telaah 28 Okt: pengukuran LR ditambah, baris berjarak sama ditandai.
BARIS_JIKI = [
    ('A  1–30 d, published windows', 0.615, 0.674, 0.669, 0.604, 0.643),
    ('B  1–30 d, gap matched', 0.531, 0.586, 0.584, 0.525, 0.519),
    ('A  30–365 d, published windows', 0.621, 0.609, 0.591, 0.538, 0.574),
    ('B  30–365 d, gap matched', 0.610, 0.591, 0.567, 0.536, 0.516),
    ('P  all causes (320), gap matched', 0.597, 0.607, 0.587, 0.525, 0.546),
    ('P  cardiac cause (104), gap matched', 0.547, 0.697, 0.661, 0.640, 0.630),
    ('P  non-cardiac (200), gap matched', 0.614, 0.601, 0.602, 0.515, 0.561),
    ('C  same visit (178)', 0.617, 0.734, 0.775, 0.747, 0.791),
]
UNGU = '#6a3d9a'


def gambar2_jiki():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    n = len(BARIS_JIKI)
    gaya = [('Heart rate (raw)', JINGGA, 's', 0.30), ('Measurements (LR)', '#b8860b', '^', 0.15),
            ('Machine reading (LR)', HIJAU, 'D', 0.0), ('Machine reading (trees)', UNGU, 'v', -0.15),
            ('CNN', BIRU, 'o', -0.30)]
    for i, (lab, *nilai) in enumerate(BARIS_JIKI):
        yy = n - 1 - i
        ax.plot([min(nilai), max(nilai)], [yy, yy], color='#dddddd', lw=0.8, zorder=1)
        for k, (nama, warna, tanda, geser) in enumerate(gaya):
            ax.plot(nilai[k], yy + geser, tanda, color=warna, ms=4.5, zorder=2,
                    label=nama if i == 0 else None)
    ax.axvline(0.5, ls='--', color='#888888', lw=0.8)
    for batas in (n - 4.5, n - 7.5):
        ax.axhline(batas, color='#cccccc', lw=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([b[0] for b in BARIS_JIKI][::-1])
    ax.set_xlim(0.46, 0.82)
    ax.set_xlabel('AUROC')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for eks in ('png', 'pdf'):
        fig.savefig(os.path.join(KELUAR, f'Gambar2_auroc_jiki.{eks}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# Revisi JIKI 30 Okt 2026 (b): AUROC dengan selang. CNN A/B dari probe_cnn.log; laju dan ukuran A/B dari
# komposisi_ab.log; P dari ci_pembanding.log; CNN P hanya titik (selang pembeda ada di tambahan_ukuran.log).
BARIS_JIKI3 = [
    # CNN: satu himpunan lari tetap, 28_lari_tetap.py (lari_tetap.log), satu lari acak per tarikan
    ('A  1–30 d', (0.628, 0.568, 0.685), (0.615, 0.558, 0.673), (0.674, 0.616, 0.731)),
    ('B  1–30 d', (0.528, 0.465, 0.590), (0.531, 0.475, 0.589), (0.586, 0.528, 0.644)),
    ('A  30–365 d', (0.550, 0.487, 0.610), (0.621, 0.574, 0.666), (0.609, 0.562, 0.653)),
    ('B  30–365 d', (0.523, 0.463, 0.579), (0.610, 0.562, 0.658), (0.591, 0.542, 0.641)),
    ('P  2–730 d', (0.539, 0.496, 0.582), (0.597, 0.563, 0.632), (0.607, 0.574, 0.640)),
]


def gambar3_jiki():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    gaya = [('CNN', BIRU, 'o', 0.22), ('Heart rate', JINGGA, 's', 0.0), ('Measurements (LR)', '#b8860b', '^', -0.22)]
    n = len(BARIS_JIKI3)
    for i, (lab, *nilai) in enumerate(BARIS_JIKI3):
        yy = n - 1 - i
        for (nama, warna, tanda, geser), (t, a, b) in zip(gaya, nilai):
            if a is not None:
                ax.plot([a, b], [yy + geser] * 2, color=warna, lw=1.0, zorder=1)
            ax.plot(t, yy + geser, tanda, color=warna, ms=4.5, zorder=2, label=nama if i == 0 else None)
    ax.axvline(0.5, ls='--', color='#888888', lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([b[0] for b in BARIS_JIKI3][::-1])
    ax.set_xlim(0.44, 0.76)
    ax.set_xlabel('AUROC with 95 % CI')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for eks in ('png', 'pdf'):
        fig.savefig(os.path.join(KELUAR, f'Gambar3_auroc_jiki.{eks}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    gambar1()
    gambar2()
    gambar2_jiki()
    gambar3_jiki()
    print('Gambar1_alur.png, Gambar2_auroc.png, Gambar2_auroc_jiki.png, Gambar3_auroc_jiki.png ->', KELUAR)
