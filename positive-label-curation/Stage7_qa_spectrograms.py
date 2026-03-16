"""
Stage7_qa_spectrograms.py

Generate spectrogram thumbnail sheets for manual QA.
500 random positive clips → 13 PNG pages (5 cols × 8 rows = 40 per page)
Output sized for 4K screen (3840 × 2160).

Usage:
  python Stage7_qa_spectrograms.py                    # Use defaults from config.py
  python Stage7_qa_spectrograms.py --n-samples 1000   # Custom sample size
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Import centralized config
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Please ensure config.py is in the same directory.")
    sys.exit(1)

# ── Config from config.py ─────────────────────────────────────────────────────
SOURCE_DIR  = config.STAGE7_SOURCE_DIR
OUTPUT_DIR  = config.STAGE7_OUTPUT_DIR
N_SAMPLES   = config.STAGE7_N_SAMPLES
SEED        = config.STAGE7_SEED

COLS, ROWS  = 5, 5          # 25 per page
DPI         = 100
FIG_W       = 3840 / DPI    # 38.4 in  → 3840 px wide
FIG_H       = 2160 / DPI    # 21.6 in  → 2160 px tall

# Spectrogram params (match pipeline)
SR          = config.TARGET_SAMPLE_RATE  # 16kHz from config
N_FFT       = 512
HOP_LENGTH  = 128
N_MELS      = 80
FMAX        = 8_000
# ─────────────────────────────────────────────────────────────────────────────

PER_PAGE = COLS * ROWS


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram thumbnail sheets for manual QA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        default=SOURCE_DIR,
        help=f"Source directory containing WAV files (default: {SOURCE_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for PNG pages (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help=f"Number of random clips to sample (default: {N_SAMPLES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    source_dir = args.source_dir
    output_dir = args.output_dir
    n_samples = args.n_samples
    seed = args.seed

    os.makedirs(output_dir, exist_ok=True)

    # ── Sample files ──────────────────────────────────────────────────────────────
    all_files = sorted(Path(source_dir).rglob('*.wav'))
    print(f'Found {len(all_files):,} WAV files in {source_dir}')

    if len(all_files) < n_samples:
        raise SystemExit(f'Only {len(all_files)} files found — need {n_samples}.')

    random.seed(seed)
    sampled = random.sample(all_files, n_samples)
    sampled.sort()   # deterministic page order

    n_pages = math.ceil(n_samples / PER_PAGE)
    print(f'Generating {n_pages} pages × {PER_PAGE} clips  →  {output_dir}')

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

                ax.text(0.01, 0.11, label,
                        transform=ax.transAxes,
                        fontsize=18, color='white', va='top', ha='left',
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
        global_end   = min(global_start + PER_PAGE - 1, n_samples)
        fig.text(0.995, 0.005,
                 f'Page {page_idx + 1}/{n_pages}  ·  clips {global_start}–{global_end}',
                 ha='right', va='bottom', fontsize=10, color='#888888')

        out_path = os.path.join(output_dir, f'page_{page_idx + 1:03d}.png')
        fig.savefig(out_path, dpi=DPI, bbox_inches=None, facecolor='black')
        plt.close(fig)
        print(f'  ✓ page_{page_idx + 1:03d}.png  ({len(batch)} clips)')

    print(f'\nDone — {n_pages} pages saved to:\n  {output_dir}')
