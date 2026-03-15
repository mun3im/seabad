#!/usr/bin/env python3
"""
Stage 4: Extract negative samples from FSC22
- Excludes bird classes: 23 (BirdChirping), 24 (WingFlapping)
- Extracts the loudest 3 s clip per file using a sliding window
- Low-amplitude clips are kept; only all-zero / missing / too-short are rejected
- Output: negative/fsc/fsc-<name>-<start_ms:04d>.wav
"""

import os
import sys
import logging
import pandas as pd
import soundfile as sf
import librosa
from collections import defaultdict
from tqdm import tqdm

from config import (
    FSC22_AUDIO_DIR, FSC22_METADATA_PATH, STAGE4_NEG_DIR,
    TARGET_SR, extract_loudest_3s_clip,
)

BIRD_CLASSES = {23, 24}  # BirdChirping & WingFlapping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Stage4_fsc22.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(STAGE4_NEG_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 4: FSC22 → EXTRACT NON-BIRD SOUNDS (LOUDEST 3S SEGMENT)")
    logger.info("=" * 70)
    logger.info(f"Source:          {FSC22_AUDIO_DIR}")
    logger.info(f"Output:          {STAGE4_NEG_DIR}")
    logger.info(f"Excluded classes (bird): {sorted(BIRD_CLASSES)}")
    logger.info("")

    if not os.path.exists(FSC22_METADATA_PATH):
        logger.error(f"Metadata not found: {FSC22_METADATA_PATH}")
        return 1

    df = pd.read_csv(FSC22_METADATA_PATH)
    total_files  = len(df)
    non_bird_df  = df[~df['Class ID'].isin(BIRD_CLASSES)].copy()
    bird_count   = total_files - len(non_bird_df)
    logger.info(f"Total files in metadata:   {total_files}")
    logger.info(f"Non-bird files to process: {len(non_bird_df)}")
    logger.info(f"Skipped (bird classes):    {bird_count}")

    valid_clips   = []
    rejection_log = []
    skip_counts   = defaultdict(int)

    for _, row in tqdm(non_bird_df.iterrows(), total=len(non_bird_df), desc="Processing FSC22"):
        dataset_filename = row['Dataset File Name']
        filepath = os.path.join(FSC22_AUDIO_DIR, dataset_filename)

        if not os.path.isfile(filepath):
            skip_counts['missing_file'] += 1
            rejection_log.append((dataset_filename, 'missing_file'))
            continue

        try:
            y, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
            clip, start_ms, reason = extract_loudest_3s_clip(y)
        except Exception as e:
            logger.error(f"Error loading {dataset_filename}: {e}")
            skip_counts['processing_error'] += 1
            rejection_log.append((dataset_filename, 'processing_error'))
            continue

        if clip is not None:
            valid_clips.append({'clip': clip, 'filename': dataset_filename, 'start_ms': start_ms})
        else:
            skip_counts[reason] += 1
            rejection_log.append((dataset_filename, reason))

    logger.info(f"Valid non-bird clips: {len(valid_clips)}")
    logger.info(f"Total rejected:       {len(rejection_log)}")

    saved_count = 0
    for item in tqdm(valid_clips, desc="Saving clips"):
        base     = os.path.splitext(item['filename'])[0]
        new_name = f"fsc-{base}-{item['start_ms']:04d}.wav"
        sf.write(os.path.join(STAGE4_NEG_DIR, new_name), item['clip'], TARGET_SR)
        saved_count += 1

    if rejection_log:
        logger.info("\n" + "=" * 70)
        logger.info("REJECTION DETAILS")
        logger.info("=" * 70)
        for filename, reason in rejection_log:
            logger.info(f"  {filename:<30} {reason}")
        logger.info("=" * 70)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 4 COMPLETE — FSC22")
    logger.info("=" * 70)
    logger.info(f"  Total in metadata:      {total_files:5d}")
    logger.info(f"  Skipped (bird classes): {bird_count:5d}")
    logger.info(f"  Saved:                  {saved_count:5d}")
    logger.info(f"  Rejected:               {len(rejection_log):5d}")
    logger.info(f"  Output: {STAGE4_NEG_DIR}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
