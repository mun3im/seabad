#!/usr/bin/env python3
"""
Stage 1: Extract negative samples from BirdVox-DCASE-20k
- Filters out files with bird sounds (hasbird == 1)
- Extracts the loudest 3 s clip per file using a sliding window
- Rejects files that are too short or all-zero
- Output: negative/bv/bv-<itemid>-<start_ms:05d>.wav
"""

import sys
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    DCASE_DATASETS, STAGE1_OUTPUT_DIR,
    TARGET_SR, CLIP_DURATION, MIN_DURATION,
    process_dcase_file,
)

DATASET_NAME = "BirdVox-DCASE-20k"
DATASET_INFO = DCASE_DATASETS[DATASET_NAME]
OUTPUT_DIR   = STAGE1_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Stage1_rejections.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 1: BirdVox-DCASE-20k → NEGATIVE SAMPLES")
    logger.info("=" * 70)
    logger.info(f"CSV:    {DATASET_INFO['csv']}")
    logger.info(f"WAV:    {DATASET_INFO['wav']}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Config: {CLIP_DURATION}s clips @ {TARGET_SR} Hz")
    logger.info("")

    df = pd.read_csv(DATASET_INFO["csv"])
    no_bird_df = df[df["hasbird"] == 0]
    total = len(no_bird_df)
    logger.info(f"Negative files (hasbird == 0): {total}")

    stats = {
        'negative_saved':   0,
        'too_short':        0,
        'all_zero':         0,
        'missing_file':     0,
        'no_id_field':      0,
        'processing_error': 0,
    }

    row_dicts = [row.to_dict() for _, row in no_bird_df.iterrows()]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_dcase_file, rd, DATASET_INFO, OUTPUT_DIR)
            for rd in row_dicts
        ]
        with tqdm(total=len(futures), desc="Processing BirdVox", unit="files") as pbar:
            for future in as_completed(futures):
                success, reason, _ = future.result()
                if success:
                    stats['negative_saved'] += 1
                else:
                    stats[reason] = stats.get(reason, 0) + 1
                pbar.update(1)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 1 COMPLETE — BirdVox-DCASE-20k")
    logger.info("=" * 70)
    logger.info(f"  Saved:                  {stats['negative_saved']:5d}")
    logger.info(f"  Skipped (too short):    {stats['too_short']:5d}")
    logger.info(f"  Skipped (all-zero):     {stats['all_zero']:5d}")
    logger.info(f"  Missing files:          {stats['missing_file']:5d}")
    logger.info(f"  Processing errors:      {stats['processing_error']:5d}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
