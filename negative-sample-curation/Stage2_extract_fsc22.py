#!/usr/bin/env python3
"""
Stage 3: Extract negative samples from FSC22 dataset
- Skip bird classes (23: BirdChirping, 24: WingFlapping)
- For each non-bird file, extract the LOUDEST 3s segment using a sliding window
- Save as: fsc-<original_name>-<start_ms:05d>.wav
- Do NOT apply RMS threshold — low amplitude is acceptable
- Only reject if clip is all-zero, missing, or too short
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import logging
from tqdm import tqdm
from collections import defaultdict

# ============================= CONFIGURATION =============================
AUDIO_DIR = "/Volumes/Evo/datasets/FSC22/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
METADATA_PATH = "/Volumes/Evo/datasets/FSC22/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"
NEG_DIR = "/Volumes/Evo/mybad_v4/negative/fsc"  # Non-bird sounds only

TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000

# Bird-related classes (positive samples — EXCLUDE these)
BIRD_CLASSES = {23, 24}  # BirdChirping & WingFlapping

# ============================= SETUP =============================
os.makedirs(NEG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('FSC22_extraction.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================= FUNCTIONS =============================

def extract_loudest_3s_clip(filepath):
    """
    Extract the loudest 3-second segment from an audio file.
    Returns: (clip: np.ndarray or None, start_ms: int or None, reason: str or None)
    """
    if not os.path.isfile(filepath):
        return None, None, 'missing_file'

    try:
        y, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)

        if len(y) < CLIP_SAMPLES:
            return None, None, 'too_short'

        if len(y) == CLIP_SAMPLES:
            if np.all(y == 0):
                return None, None, 'all_zero'
            return y, 0, None

        # Sliding window (100ms hop)
        hop_samples = int(0.1 * TARGET_SR)
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

        if best_clip is None:
            # Fallback to center
            center = len(y) // 2
            best_start_sample = max(0, center - CLIP_SAMPLES // 2)
            best_clip = y[best_start_sample:best_start_sample + CLIP_SAMPLES]
            if len(best_clip) != CLIP_SAMPLES:
                best_clip = np.resize(best_clip, CLIP_SAMPLES)

        if np.all(best_clip == 0):
            return None, None, 'all_zero'

        start_ms = int(round(1000.0 * best_start_sample / TARGET_SR))
        return best_clip, start_ms, None

    except Exception as e:
        logger.error(f"Error loading {filepath}: {type(e).__name__}: {e}")
        return None, None, 'processing_error'

# ============================= MAIN =============================

def main():
    logger.info("=" * 70)
    logger.info("FSC22 → EXTRACT NON-BIRD SOUNDS (LOUDEST 3S SEGMENT)")
    logger.info("=" * 70)
    logger.info(f"Source:  {AUDIO_DIR}")
    logger.info(f"Output:  {NEG_DIR}")
    logger.info(f"Excluded classes (bird): {sorted(BIRD_CLASSES)}")
    logger.info("Low-amplitude clips are KEPT. Only all-zero/invalid files rejected.")
    logger.info("")

    if not os.path.exists(METADATA_PATH):
        logger.error(f"Metadata not found: {METADATA_PATH}")
        return 1

    df = pd.read_csv(METADATA_PATH)
    total_files = len(df)
    logger.info(f"Total files in metadata: {total_files}")

    # Filter out bird classes upfront
    non_bird_df = df[~df['Class ID'].isin(BIRD_CLASSES)].copy()
    bird_count = total_files - len(non_bird_df)
    logger.info(f"Non-bird files to process: {len(non_bird_df)}")
    logger.info(f"Skipped (bird classes): {bird_count}")

    # Process non-bird files
    valid_clips = []
    rejection_log = []
    skip_counts = defaultdict(int)

    for _, row in tqdm(non_bird_df.iterrows(), total=len(non_bird_df), desc="Processing FSC22"):
        dataset_filename = row['Dataset File Name']
        filepath = os.path.join(AUDIO_DIR, dataset_filename)

        clip, start_ms, reason = extract_loudest_3s_clip(filepath)

        if clip is not None:
            valid_clips.append({
                'clip': clip,
                'filename': dataset_filename,
                'start_ms': start_ms
            })
        else:
            if reason:
                skip_counts[reason] += 1
                rejection_log.append((dataset_filename, reason))

    logger.info(f"Valid non-bird clips after filtering: {len(valid_clips)}")
    logger.info(f"Total rejected: {len(rejection_log)}")

    # Save all valid clips with new naming scheme
    saved_count = 0
    for item in tqdm(valid_clips, desc="Saving clips"):
        base = os.path.splitext(item['filename'])[0]
        start_ms = item['start_ms']
        new_name = f"fsc-{base}-{start_ms:04d}.wav"
        out_path = os.path.join(NEG_DIR, new_name)
        sf.write(out_path, item['clip'], TARGET_SR)
        saved_count += 1

    # ============================= REJECTION REPORT =============================
    if rejection_log:
        logger.info("\n" + "="*70)
        logger.info("REJECTION DETAILS")
        logger.info("="*70)
        logger.info(f"{'Filename':<30} {'Reason'}")
        logger.info("-" * 70)
        for filename, reason in rejection_log:
            logger.info(f"{filename:<30} {reason}")
        logger.info("="*70)

    # ============================= SUMMARY =============================
    logger.info("")
    logger.info("=" * 70)
    logger.info("FSC22 EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total files in metadata:        {total_files:5d}")
    logger.info(f"Skipped (bird classes):         {bird_count:5d}")
    logger.info(f"Successfully saved:             {saved_count:5d}")
    logger.info(f"Rejected (all-zero, etc.):      {len(rejection_log):5d}")
    logger.info("")
    logger.info(f"Output directory: {NEG_DIR}")
    logger.info("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())