#!/usr/bin/env python3
"""
Stage 5: Extract negative samples from ESC-50
- Excludes bird categories: rooster, hen, crow, chirping_birds
- Extracts the loudest 3 s clip per file using a sliding window
- Only rejects files that are all-zero, missing, or too short
- Output: negative/esc/esc-<name>-<start_ms:04d>.wav
"""

import os
import sys
import logging
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from config import (
    ESC50_CSV_PATH, ESC50_AUDIO_DIR, STAGE5_NEG_DIR,
    TARGET_SR, extract_loudest_3s_clip,
)

BIRD_CATEGORIES = {'rooster', 'hen', 'crow', 'chirping_birds'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Stage5_esc50.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(STAGE5_NEG_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 5: ESC-50 → EXTRACT ALL NON-BIRD SOUNDS (LOUDEST 3S SEGMENT)")
    logger.info("=" * 70)
    logger.info(f"Source:                   {ESC50_AUDIO_DIR}")
    logger.info(f"Output:                   {STAGE5_NEG_DIR}")
    logger.info(f"Excluded bird categories: {sorted(BIRD_CATEGORIES)}")
    logger.info("")

    if not os.path.exists(ESC50_CSV_PATH):
        logger.error(f"Metadata CSV not found: {ESC50_CSV_PATH}")
        return 1

    df = pd.read_csv(ESC50_CSV_PATH)
    df['category'] = df['category'].astype(str)

    total_original  = len(df)
    candidates_df   = df[~df['category'].isin(BIRD_CATEGORIES)].copy()
    total_candidates = len(candidates_df)
    logger.info(f"Total ESC-50 files:        {total_original}")
    logger.info(f"Non-bird candidate files:  {total_candidates}")

    valid_clips   = []
    rejection_log = []
    skip_counts   = defaultdict(int)

    for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Processing ESC-50"):
        filename = row['filename']
        src_path = os.path.join(ESC50_AUDIO_DIR, filename)

        if not os.path.isfile(src_path):
            skip_counts['missing_file'] += 1
            rejection_log.append((filename, row['category'], 'missing_file'))
            continue

        try:
            y, _ = librosa.load(src_path, sr=TARGET_SR, mono=True)
            clip, start_ms, reason = extract_loudest_3s_clip(y)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            skip_counts['processing_error'] += 1
            rejection_log.append((filename, row['category'], 'processing_error'))
            continue

        if clip is not None:
            valid_clips.append({'clip': clip, 'filename': filename,
                                'category': row['category'], 'start_ms': start_ms})
        else:
            skip_counts[reason] += 1
            rejection_log.append((filename, row['category'], reason))

    logger.info(f"Valid non-bird clips: {len(valid_clips)}")
    logger.info(f"Total rejected:       {len(rejection_log)}")

    saved_count = 0
    for item in tqdm(valid_clips, desc="Saving clips"):
        base     = os.path.splitext(item['filename'])[0]
        new_name = f"esc-{base}-{item['start_ms']:04d}.wav"
        sf.write(os.path.join(STAGE5_NEG_DIR, new_name), item['clip'], TARGET_SR)
        saved_count += 1

    if rejection_log:
        logger.info("\n" + "=" * 70)
        logger.info("REJECTION DETAILS")
        logger.info("=" * 70)
        for filename, category, reason in rejection_log:
            logger.info(f"  {filename:<20} {category:<20} {reason}")
        logger.info("=" * 70)
    else:
        logger.info("No files were rejected.")

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 5 COMPLETE — ESC-50")
    logger.info("=" * 70)
    logger.info(f"  Total ESC-50 files:     {total_original:5d}")
    logger.info(f"  Non-bird candidates:    {total_candidates:5d}")
    logger.info(f"  Saved:                  {saved_count:5d}")
    logger.info(f"  Rejected:               {len(rejection_log):5d}")
    logger.info(f"  Output: {STAGE5_NEG_DIR}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
