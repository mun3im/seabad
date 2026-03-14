"""
Generate spectrogram thumbnail sheets for manual QA.
500 random positive clips → 13 PNG pages (5 cols × 8 rows = 40 per page)
Output sized for 4K screen (3840 × 2160).
"""

import os
import math
import random
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DIR  = '/Volumes/Evo/mybad/positive'
OUTPUT_DIR  = '/Volumes/Evo/mybad/qa_thumbnails'
N_SAMPLES   = 500
SEED        = 42

COLS, ROWS  = 5, 8          # 40 per page
DPI         = 100
FIG_W       = 3840 / DPI    # 38.4 in  → 3840 px wide
FIG_H       = 2160 / DPI    # 21.6 in  → 2160 px tall

# Spectrogram params (match pipeline)
SR          = 16_000
N_FFT       = 512
HOP_LENGTH  = 128
N_MELS      = 80
FMAX        = 8_000
# ─────────────────────────────────────────────────────────────────────────────

PER_PAGE = COLS * ROWS

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sample files ──────────────────────────────────────────────────────────────
all_files = sorted(Path(SOURCE_DIR).rglob('*.wav'))
print(f'Found {len(all_files):,} WAV files in {SOURCE_DIR}')

if len(all_files) < N_SAMPLES:
    raise SystemExit(f'Only {len(all_files)} files found — need {N_SAMPLES}.')

random.seed(SEED)
sampled = random.sample(all_files, N_SAMPLES)
sampled.sort()   # deterministic page order

n_pages = math.ceil(N_SAMPLES / PER_PAGE)
print(f'Generating {n_pages} pages × {PER_PAGE} clips  →  {OUTPUT_DIR}')

# ── Page loop ─────────────────────────────────────────────────────────────────
for page_idx in range(n_pages):
    batch = sampled[page_idx * PER_PAGE : (page_idx + 1) * PER_PAGE]

    fig, axes = plt.subplots(ROWS, COLS, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        hspace=0.008, wspace=0.008)

    for i, ax in enumerate(axes.flatten()):
        ax.set_facecolor('black')
        ax.axis('off')

        if i >= len(batch):
            continue

        fpath = batch[i]
        # Show species/subfolder + filename so context is visible
        label = f'{fpath.parent.name}/{fpath.stem}'

        try:
            y, _ = librosa.load(fpath, sr=SR, mono=True, duration=3.0)
            S = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmax=FMAX)
            S_db = librosa.power_to_db(S, ref=np.max)

            ax.imshow(S_db, aspect='auto', origin='lower',
                      cmap='magma', vmin=-80, vmax=0,
                      interpolation='nearest')

            ax.text(0.01, 0.98, label,
                    transform=ax.transAxes,
                    fontsize=5.5, color='white', va='top', ha='left',
                    fontfamily='monospace',
                    bbox=dict(facecolor='black', alpha=0.75,
                              pad=1.5, edgecolor='none'))

        except Exception as exc:
            ax.text(0.5, 0.5, f'ERROR\n{fpath.name[:30]}',
                    transform=ax.transAxes, ha='center', va='center',
                    color='red', fontsize=6)
            print(f'  [!] {fpath.name}: {exc}')

    # Page number — bottom-right corner
    global_start = page_idx * PER_PAGE + 1
    global_end   = min(global_start + PER_PAGE - 1, N_SAMPLES)
    fig.text(0.995, 0.005,
             f'Page {page_idx + 1}/{n_pages}  ·  clips {global_start}–{global_end}',
             ha='right', va='bottom', fontsize=7, color='#888888')

    out_path = os.path.join(OUTPUT_DIR, f'page_{page_idx + 1:03d}.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches=None, facecolor='black')
    plt.close(fig)
    print(f'  ✓ page_{page_idx + 1:03d}.png  ({len(batch)} clips)')

print(f'\nDone — {n_pages} pages saved to:\n  {OUTPUT_DIR}')
