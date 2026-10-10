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
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1)
    ax.axis('off')
    kotak(ax, 0.27, 0.93, 0.44, 0.09, 'Case group: 974 patients with ROSC, 2011–2019\n'
          '1,686 ECGs: 605 pre, 838 at ROSC, 243 at 24 h', '#fde3d4')
    kotak(ax, 0.73, 0.93, 0.40, 0.09, 'Control group: 9,976 ED patients, 2015–2019\n'
          '16,386 ECGs: 9,976 index, 6,410 pre', '#d9e7f5')
    kotak(ax, 0.60, 0.76, 0.34, 0.08, 'Control pre ECGs, 2015–2017 only\n'
          '(2018–2019 date shifts inconsistent)', '#d9e7f5')
    panah(ax, 0.60, 0.885, 0.60, 0.80, BIRU)

    y, w, h = 0.34, 0.225, 0.29
    xs = [0.125, 0.375, 0.625, 0.875]
    teks = [
        'Design A (replication)\nCases: pre 0–1, 1–30,\n30–365 days (220/120/175)\n'
        'Controls: pre 30–730 days\n(2,226 candidates)\nAge × sex matched, 5:1',
        'Design B (matched gap,\nall causes)\nSame window: 1–30 or\n30–365 days\n'
        'Gap band: all pre > 1 day,\n381 cases, 1,905 controls',
        'Design P (prior visit)\nCases: 2–730 days (320)\n'
        'Cause: cardiac 104,\nnon-cardiac 216\nRhythm: VT/VF 39,\nPEA/asystole 272, other 9\n'
        'Nearest gap, age, sex; 5:1',
        'Design C (same visit)\nIn-hospital arrest, ECG after\narrival, 0–24 h before CPR\n'
        '178 cases vs 9,976\ncontrol index ECGs',
    ]
    for x, t, c in zip(xs, teks, ['#eeeeee', '#eeeeee', '#fff2b3', '#eeeeee']):
        kotak(ax, x, y, w, h, t, c)
    atas = y + h / 2

    # kasus (jingga): turun dari kotak kasus ke rel, lalu ke keempat rancangan (sisi kiri kotak)
    garis(ax, [0.10, 0.10], [0.885, 0.62], JINGGA)
    garis(ax, [xs[0] - 0.04, xs[3] - 0.04], [0.62, 0.62], JINGGA)
    for x in xs:
        panah(ax, x - 0.04, 0.62, x - 0.04, atas, JINGGA)
    ax.text(0.11, 0.70, 'case ECGs', color=JINGGA, fontsize=6.5)

    # kontrol pre 2015-2017 (biru): ke A, B, P (sisi kanan kotak)
    garis(ax, [0.60, 0.60], [0.72, 0.56], BIRU)
    garis(ax, [xs[0] + 0.04, xs[2] + 0.04], [0.56, 0.56], BIRU)
    for x in xs[:3]:
        panah(ax, x + 0.04, 0.56, x + 0.04, atas, BIRU)
    ax.text(0.59, 0.66, 'control pre ECGs', color=BIRU, fontsize=6.5, ha='right')

    # indeks kontrol (biru): langsung ke C
    garis(ax, [0.91, 0.91], [0.885, 0.70], BIRU)
    panah(ax, 0.91, 0.70, 0.91, atas, BIRU)
    ax.text(0.92, 0.70, 'control\nindex ECGs', color=BIRU, fontsize=6.5, va='top')

    kotak(ax, 0.84, 0.04, 0.30, 0.16, 'Case pre ECGs not used in design C:\n'
          'arrest outside the hospital,\nECG at or after CPR start,\nECG before ED arrival,\n'
          'or more than 24 h before CPR', '#ffffff')
    panah(ax, 0.875, y - h / 2, 0.875, 0.12)
    fig.savefig(os.path.join(KELUAR, 'Gambar1_alur.png'), dpi=300, bbox_inches='tight')
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
    fig.savefig(os.path.join(KELUAR, 'Gambar2_auroc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    gambar1()
    gambar2()
    print('Gambar1_alur.png, Gambar2_auroc.png ->', KELUAR)
