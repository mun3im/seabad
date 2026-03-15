#!/usr/bin/env python3
"""
Stage: Extract ALL non-bird sounds from ESC-50 dataset
- Exclude: 'rooster', 'hen', 'crow', 'chirping_birds'
- For each file, find the LOUDEST 3s segment using a sliding window
- Save as: esc-<original_name>-<start_ms:05d>.wav
- Only reject if clip is all-zero, missing, or too short
"""

import os
import sys
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

# ============================= CONFIGURATION =============================
CSV_PATH = '/Volumes/Evo/datasets/ESC50/meta/esc50.csv'
AUDIO_DIR = '/Volumes/Evo/datasets/ESC50/audio/'
NEG_DIR = '/Volumes/Evo/mybad_v4/negative/esc'

TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000 samples

# Bird categories in ESC-50 (to exclude)
BIRD_CATEGORIES = {
    'rooster',
    'hen',
    'crow',
    'chirping_birds'
}

# ============================= SETUP =============================
os.makedirs(NEG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ESC50_extraction_all_non_bird.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================= FUNCTIONS =============================

def extract_loudest_3s_clip(row, audio_dir):
    """
    Extract the loudest 3-second segment and return start time in milliseconds.
    Returns (clip: np.ndarray or None, start_ms: int or None, reason: str or None)
    """
    filename = row['filename']
    src_path = os.path.join(audio_dir, filename)

    if not os.path.isfile(src_path):
        return None, None, 'missing_file'

    try:
        y, sr = librosa.load(src_path, sr=TARGET_SR, mono=True)

        if len(y) < CLIP_SAMPLES:
            return None, None, 'too_short'

        # If exactly 3 seconds, use as-is
        if len(y) == CLIP_SAMPLES:
            if np.all(y == 0):
                return None, None, 'all_zero'
            return y, 0, None

        # Sliding window to find loudest 3s segment
        hop_samples = int(0.1 * TARGET_SR)  # 100ms hop
        best_rms = -1.0
        best_clip = None
        best_start_sample = 0

        for start in range(0, len(y) - CLIP_SAMPLES + 1, hop_samples):
            segment = y[start:start + CLIP_SAMPLES]
            rms = np.sqrt(np.mean(np.square(segment.astype(np.float64))))
            if rms > best_rms:
                best_rms = rms
                best_clip = segment.copy()
                best_start_sample = start

        # Fallback to center if no segment found (shouldn't happen)
        if best_clip is None:
            center = len(y) // 2
            best_start_sample = max(0, center - CLIP_SAMPLES // 2)
            best_clip = y[best_start_sample:best_start_sample + CLIP_SAMPLES]
            best_clip = np.resize(best_clip, CLIP_SAMPLES)  # ensure correct length

        if np.all(best_clip == 0):
            return None, None, 'all_zero'

        start_ms = int(round(1000.0 * best_start_sample / TARGET_SR))
        return best_clip, start_ms, None

    except Exception as e:
        logger.error(f"Error loading {filename}: {type(e).__name__}: {e}")
        return None, None, 'processing_error'

# ============================= MAIN =============================

def main():
    logger.info("=" * 70)
    logger.info("ESC-50 → EXTRACT ALL NON-BIRD SOUNDS (LOUDEST 3S SEGMENT)")
    logger.info("=" * 70)
    logger.info(f"Source:  {AUDIO_DIR}")
    logger.info(f"Output:  {NEG_DIR}")
    logger.info(f"Excluded bird categories: {sorted(BIRD_CATEGORIES)}")
    logger.info("Only all-zero or invalid files are rejected.")
    logger.info("")

    if not os.path.exists(CSV_PATH):
        logger.error(f"Metadata CSV not found: {CSV_PATH}")
        return 1

    df = pd.read_csv(CSV_PATH)
    df['category'] = df['category'].astype(str)

    candidates_df = df[~df['category'].isin(BIRD_CATEGORIES)].copy()
    total_original = len(df)
    total_candidates = len(candidates_df)
    logger.info(f"Total ESC-50 files: {total_original}")
    logger.info(f"Non-bird candidate files: {total_candidates}")

    # Process all candidates
    valid_clips = []
    rejection_log = []
    skip_counts = defaultdict(int)

    for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Processing clips"):
        clip, start_ms, reason = extract_loudest_3s_clip(row, AUDIO_DIR)

        if clip is not None:
            valid_clips.append({
                'clip': clip,
                'filename': row['filename'],
                'category': row['category'],
                'start_ms': start_ms
            })
        else:
            if reason:
                skip_counts[reason] += 1
                rejection_log.append((row['filename'], row['category'], reason))

    logger.info(f"Valid non-bird clips after filtering: {len(valid_clips)}")
    logger.info(f"Total rejected: {len(rejection_log)}")

    # Save all valid clips
    saved_count = 0
    for item in tqdm(valid_clips, desc="Saving clips"):
        base = os.path.splitext(item['filename'])[0]
        start_ms = item['start_ms']
        new_name = f"esc-{base}-{start_ms:04d}.wav"
        out_path = os.path.join(NEG_DIR, new_name)
        sf.write(out_path, item['clip'], TARGET_SR)
        saved_count += 1

    # ============================= REJECTION REPORT =============================
    if rejection_log:
        logger.info("\n" + "="*70)
        logger.info("REJECTION DETAILS")
        logger.info("="*70)
        logger.info(f"{'Filename':<20} {'Category':<20} {'Reason'}")
        logger.info("-" * 70)
        for filename, category, reason in rejection_log:
            logger.info(f"{filename:<20} {category:<20} {reason}")
        logger.info("="*70)
    else:
        logger.info("No files were rejected.")

    # ============================= SUMMARY =============================
    logger.info("")
    logger.info("=" * 70)
    logger.info("ESC-50 NON-BIRD EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total ESC-50 files:                     {total_original:5d}")
    logger.info(f"Non-bird candidates:                    {total_candidates:5d}")
    logger.info(f"Successfully saved:                     {saved_count:5d}")
    logger.info(f"Rejected (all-zero, missing, etc.):     {len(rejection_log):5d}")
    logger.info("")
    logger.info(f"Output directory: {NEG_DIR}")
    logger.info("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())