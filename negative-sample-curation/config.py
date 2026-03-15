#!/usr/bin/env python3
"""
config.py

Centralized configuration and shared utilities for the SEABAD negative-sample pipeline.
All Stage*.py scripts should import settings and helpers from this file.

Pipeline:
  Stage1 → BirdVox-DCASE-20k   → negative/bv/
  Stage2 → Freefield1010        → negative/ff/
  Stage3 → Warblrb10k           → negative/wb/
  Stage4 → FSC22                → negative/fsc/
  Stage5 → ESC-50               → negative/esc/
  Stage6 → DataSEC              → negative/datasec/
"""

import os
from pathlib import Path
import numpy as np

# ========== DATASET DIRECTORIES ==========
DATASET_ROOT = "/Volumes/Evo/SEABAD"
NEGATIVE_DIR  = os.path.join(DATASET_ROOT, "negative")

# ========== PER-STAGE OUTPUT DIRECTORIES ==========
STAGE1_OUTPUT_DIR = Path(NEGATIVE_DIR) / "bv"       # BirdVox-DCASE-20k
STAGE2_OUTPUT_DIR = Path(NEGATIVE_DIR) / "ff"       # Freefield1010
STAGE3_OUTPUT_DIR = Path(NEGATIVE_DIR) / "wb"       # Warblrb10k
STAGE4_NEG_DIR    = os.path.join(NEGATIVE_DIR, "fsc")     # FSC22
STAGE5_NEG_DIR    = os.path.join(NEGATIVE_DIR, "esc")     # ESC-50
STAGE6_NEG_DIR    = os.path.join(NEGATIVE_DIR, "datasec") # DataSEC

# ========== SOURCE DATASET PATHS ==========

# Stages 1–3: DCASE-family datasets
DCASE_DATASETS = {
    "BirdVox-DCASE-20k": {
        "csv":    Path("/Volumes/Evo/datasets/birdvox/BirdVoxDCASE20k_csvpublic.csv"),
        "wav":    Path("/Volumes/Evo/datasets/birdvox/wav"),
        "subdir": "bv",
    },
    "Freefield1010": {
        "csv":    Path("/Volumes/Evo/datasets/freefield1010/ff1010bird_metadata_2018.csv"),
        "wav":    Path("/Volumes/Evo/datasets/freefield1010/wav"),
        "subdir": "ff",
    },
    "Warblrb10k": {
        "csv":    Path("/Volumes/Evo/datasets/warblr/warblrb10k_public_metadata_2018.csv"),
        "wav":    Path("/Volumes/Evo/datasets/warblr/wav"),
        "subdir": "wb",
    },
}

# Stage 4 — FSC22
FSC22_AUDIO_DIR    = "/Volumes/Evo/datasets/FSC22/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
FSC22_METADATA_PATH = "/Volumes/Evo/datasets/FSC22/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"

# Stage 5 — ESC-50
ESC50_CSV_PATH  = "/Volumes/Evo/datasets/ESC50/meta/esc50.csv"
ESC50_AUDIO_DIR = "/Volumes/Evo/datasets/ESC50/audio/"

# Stage 6 — DataSEC
DATASEC_ROOT                 = "/Volumes/Evo/datasets/DataSEC"
DATASEC_TARGET_NEGATIVE_TOTAL = 3597

# ========== AUDIO PROCESSING ==========
TARGET_SR    = 16000                          # 16 kHz
CLIP_DURATION = 3.0                           # seconds
CLIP_SAMPLES  = int(TARGET_SR * CLIP_DURATION) # 48 000 samples
MIN_DURATION  = 3.0                           # reject files shorter than this

# ========== SHARED UTILITIES ==========

def extract_loudest_3s_clip(y: np.ndarray):
    """
    Find the loudest 3-second window in y using a 100 ms sliding hop.

    Returns (clip, start_ms, reason):
        success → (np.ndarray, int,  None)
        failure → (None,       None, reason_str)
    Rejection reasons: 'too_short', 'all_zero'
    """
    if len(y) < CLIP_SAMPLES:
        return None, None, 'too_short'

    if len(y) == CLIP_SAMPLES:
        if np.all(y == 0):
            return None, None, 'all_zero'
        return y.copy(), 0, None

    hop = int(0.1 * TARGET_SR)
    best_rms   = -1.0
    best_clip  = None
    best_start = 0

    for start in range(0, len(y) - CLIP_SAMPLES + 1, hop):
        seg = y[start:start + CLIP_SAMPLES]
        rms = float(np.sqrt(np.mean(np.square(seg.astype(np.float64)))))
        if rms > best_rms:
            best_rms   = rms
            best_clip  = seg.copy()
            best_start = start

    if best_clip is None:           # fallback: centre of file
        best_start = max(0, len(y) // 2 - CLIP_SAMPLES // 2)
        best_clip  = y[best_start:best_start + CLIP_SAMPLES]
        best_clip  = np.resize(best_clip, CLIP_SAMPLES)

    if np.all(best_clip == 0):
        return None, None, 'all_zero'

    start_ms = int(round(1000.0 * best_start / TARGET_SR))
    return best_clip, start_ms, None


def extract_loudest_3s_or_pad(y: np.ndarray, filename_for_log: str = ""):
    """
    Like extract_loudest_3s_clip but zero-pads files shorter than 3 s
    instead of rejecting them.

    Returns (clip, start_ms, is_short):
        success  → (np.ndarray, int,  bool)   is_short=True when zero-padded
        failure  → (None,       None, False)
    """
    orig_len = len(y)

    if orig_len == 0:
        return None, None, False

    if orig_len == CLIP_SAMPLES:
        if np.all(y == 0):
            return None, None, False
        return y.copy(), 0, False

    if orig_len > CLIP_SAMPLES:
        hop        = int(0.1 * TARGET_SR)
        best_rms   = -1.0
        best_clip  = None
        best_start = 0

        for start in range(0, orig_len - CLIP_SAMPLES + 1, hop):
            seg = y[start:start + CLIP_SAMPLES]
            rms = float(np.sqrt(np.mean(np.square(seg.astype(np.float64)))))
            if rms > best_rms:
                best_rms   = rms
                best_clip  = seg.copy()
                best_start = start

        if best_clip is not None and not np.all(best_clip == 0):
            return best_clip, int(round(1000.0 * best_start / TARGET_SR)), False

    # Short clip → zero-pad to 3 s
    padded = np.pad(y, (0, CLIP_SAMPLES - orig_len), mode='constant', constant_values=0)
    if np.all(padded == 0):
        return None, None, False
    return padded, 0, True


def process_dcase_file(row_dict: dict, dataset_info: dict, output_dir):
    """
    Process one audio file from a DCASE-family dataset (BirdVox / Freefield / Warblrb).
    Extracts the loudest 3 s clip and writes it to output_dir.

    Designed to run inside a ProcessPoolExecutor worker — all imports are local.

    Returns (success: bool, reason: str | None, path: str | None).
    """
    import librosa
    import soundfile as sf
    from pathlib import Path as _Path

    try:
        if "itemid" in row_dict:
            file_id  = str(row_dict["itemid"])
            filename = f"{file_id}.wav"
        elif "filename" in row_dict:
            filename = row_dict["filename"]
            file_id  = os.path.splitext(filename)[0]
        else:
            return False, 'no_id_field', None

        filepath = _Path(dataset_info["wav"]) / filename
        if not filepath.exists():
            return False, 'missing_file', str(filepath)

        y, _ = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
        clip, start_ms, reason = extract_loudest_3s_clip(y)
        if clip is None:
            return False, reason, str(filepath)

        out_filename = f"{dataset_info['subdir']}-{file_id}-{start_ms:05d}.wav"
        out_path     = _Path(output_dir) / out_filename
        sf.write(str(out_path), clip, TARGET_SR)
        return True, None, str(out_path)

    except Exception as e:
        return False, 'processing_error', str(e)
