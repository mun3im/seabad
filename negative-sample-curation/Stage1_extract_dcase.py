#!/usr/bin/env python3
"""
Stage 1: Extract negative samples from DCASE datasets
- Processes BirdVox-DCASE-20k, Freefield1010, and Warblrb10k
- Extracts 3s clips from center of audio files
- Filters out bird sounds (hasbird == 1)
- Applies quality filtering (RMS threshold, minimum duration)
- Target sample rate: 16kHz
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import librosa
import numpy as np
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
import soundfile as sf

# ============================= CONFIGURATION =============================
# Dataset configuration
DATASETS = {
    "BirdVox-DCASE-20k": {
        "csv": Path("/Volumes/Evo/datasets/birdvox/BirdVoxDCASE20k_csvpublic.csv"),
        "wav": Path("/Volumes/Evo/datasets/birdvox/wav"),
        "subdir": "bv"
    },
    "Freefield1010": {
        "csv": Path("/Volumes/Evo/datasets/freefield1010/ff1010bird_metadata_2018.csv"),
        "wav": Path("/Volumes/Evo/datasets/freefield1010/wav"),
        "subdir": "ff"
    },
    "Warblrb10k": {
        "csv": Path("/Volumes/Evo/datasets/warblr/warblrb10k_public_metadata_2018.csv"),
        "wav": Path("/Volumes/Evo/datasets/warblr/wav"),
        "subdir": "wb"
    },
}

# Base output directory
BASE_OUTPUT_DIR = Path("/Volumes/Evo/mybad_v4/negative")

# Audio processing parameters
TARGET_SR = 16000
CLIP_DURATION = 3.0
MIN_DURATION = 3.0  # Skip files shorter than this
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000 samples

# ============================= SETUP =============================
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging (uniform format across all stages)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Stage1_rejections.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================= FUNCTIONS =============================

def compute_rms(y: np.ndarray) -> float:
    """Compute RMS of signal y (float numpy array)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


def process_audio_file(row_dict, dataset_info, output_dir):
    """
    Process a single audio file using loudest 3s window.
    Returns: (success, reason, out_path_or_filepath)
    """
    try:
        # Handle different ID columns
        if "itemid" in row_dict:
            file_id = str(row_dict["itemid"])
            filename = f"{file_id}.wav"
        elif "filename" in row_dict:
            filename = row_dict["filename"]
            file_id = os.path.splitext(filename)[0]
        else:
            return False, 'no_id_field', None

        filepath = dataset_info["wav"] / filename
        if not filepath.exists():
            return False, 'missing_file', str(filepath)

        y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)

        if len(y) < CLIP_SAMPLES:
            return False, 'too_short', str(filepath)

        # === LOUDEST 3S WINDOW ===
        if len(y) == CLIP_SAMPLES:
            clip = y
            start_sample = 0
        else:
            hop = int(0.1 * TARGET_SR)
            best_rms = -1.0
            best_clip = None
            best_start = 0
            for start in range(0, len(y) - CLIP_SAMPLES + 1, hop):
                seg = y[start:start + CLIP_SAMPLES]
                rms = np.sqrt(np.mean(np.square(seg.astype(np.float64))))
                if rms > best_rms:
                    best_rms = rms
                    best_clip = seg.copy()
                    best_start = start
            if best_clip is None:
                best_start = max(0, len(y)//2 - CLIP_SAMPLES//2)
                best_clip = y[best_start:best_start + CLIP_SAMPLES]
                best_clip = np.resize(best_clip, CLIP_SAMPLES)
            clip = best_clip
            start_sample = best_start

        if np.all(clip == 0):
            return False, 'all_zero', str(filepath)

        start_ms = int(round(1000.0 * start_sample / TARGET_SR))
        out_filename = f"{dataset_info['subdir']}-{file_id}-{start_ms:05d}.wav"
        out_path = output_dir / out_filename

        # Save with soundfile (no pydub)
        sf.write(str(out_path), clip, TARGET_SR)

        return True, None, str(out_path)

    except Exception as e:
        logger.error(f"Error processing {file_id}: {e}")
        return False, 'processing_error', None


def process_dataset(dataset_name, dataset_info):
    """Process all negative files in a dataset."""
    logger.info(f"Processing dataset: {dataset_name}")

    output_dir = BASE_OUTPUT_DIR / dataset_info["subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        df = pd.read_csv(dataset_info["csv"])
        no_bird_df = df[df.get("hasbird", 0) == 0]
        total_files = len(no_bird_df)
        logger.info(f"Found {total_files} negative files (hasbird == 0)")

        # Statistics
        stats = {
            'negative_saved': 0,
            'too_short': 0,
            'all_zero': 0,  # ← was 'too_quiet'
            'missing_file': 0,
            'processing_error': 0
        }

        # Convert rows to dicts for safer pickling in ProcessPool
        row_dicts = [row[1].to_dict() for row in no_bird_df.iterrows()]

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_audio_file, row_dict, dataset_info, output_dir)
                for row_dict in row_dicts
            ]

            with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="files") as pbar:
                for future in as_completed(futures):
                    success, reason, filepath = future.result()

                    if success:
                        stats['negative_saved'] += 1
                    else:
                        stats[reason] = stats.get(reason, 0) + 1

                    pbar.update(1)

        logger.info(f"Dataset {dataset_name} - Successfully processed: {stats['negative_saved']}/{total_files}")
        logger.info(f"Dataset {dataset_name} - Skipped/Failed: {total_files - stats['negative_saved']}/{total_files}")

        return stats

    except Exception as e:
        logger.exception(f"Could not process dataset {dataset_name}: {e}")
        return None


# ============================= MAIN PROCESSING =============================

def main():
    logger.info("=" * 70)
    logger.info("STAGE 1: DCASE → NEGATIVE SAMPLES EXTRACTION")
    logger.info("=" * 70)
    logger.info(f"Datasets: BirdVox-DCASE-20k, Freefield1010, Warblrb10k")
    logger.info(f"Output:   {BASE_OUTPUT_DIR}")
    logger.info(f"Config:   {CLIP_DURATION}s clips @ {TARGET_SR} Hz")
    logger.info("")

    all_stats = {}

    # Process each dataset
    for dataset_name, dataset_info in DATASETS.items():
        stats = process_dataset(dataset_name, dataset_info)
        if stats:
            all_stats[dataset_name] = stats
        logger.info("=" * 70)

    # ============================= SUMMARY =============================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 1 COMPLETE - DCASE EXTRACTION SUMMARY")
    logger.info("=" * 70)

    total_saved = 0
    total_short = 0
    total_quiet = 0
    total_missing = 0
    total_errors = 0
    total_all_zero = 0

    for dataset_name, stats in all_stats.items():
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Negative samples saved:             {stats['negative_saved']:5d}")
        logger.info(f"  Skipped (shorter than {MIN_DURATION}s):       {stats['too_short']:5d}")
        logger.info(f"  Skipped (all-zero audio):           {stats.get('all_zero', 0):5d}")
        logger.info(f"  Errors (missing files):             {stats['missing_file']:5d}")
        logger.info(f"  Errors (processing):                {stats['processing_error']:5d}")

        total_saved += stats['negative_saved']
        total_short += stats['too_short']
        total_all_zero += stats.get('all_zero', 0)
        total_missing += stats['missing_file']
        total_errors += stats['processing_error']

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"TOTAL NEGATIVE SAMPLES SAVED:       {total_saved:5d}")
    logger.info(f"Total skipped (too short):          {total_short:5d}")
    logger.info(f"Total skipped (all-zero):           {total_all_zero:5d}")
    logger.info(f"Total errors (missing):             {total_missing:5d}")
    logger.info(f"Total errors (processing):          {total_errors:5d}")
    logger.info("")
    logger.info(f"Output subdirectories under {BASE_OUTPUT_DIR}:")
    for dataset_name, dataset_info in DATASETS.items():
        logger.info(f"  - {dataset_name:20s} → {dataset_info['subdir']}/")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
