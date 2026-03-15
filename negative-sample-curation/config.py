#!/usr/bin/env python3
"""
config.py

Centralized configuration for SEABAD negative-sample pipeline.
All Stage*.py scripts should import settings from this file.
"""

import os
from pathlib import Path

# ========== DATASET DIRECTORIES ==========
DATASET_ROOT = "/Volumes/Evo/SEABAD"
NEGATIVE_DIR = os.path.join(DATASET_ROOT, "negative")  # All Stage outputs land here

# ========== PER-STAGE OUTPUT SUBDIRECTORIES ==========
# Stage1: DCASE (BirdVox-DCASE-20k, Freefield1010, Warblrb10k)
STAGE1_BASE_OUTPUT_DIR = Path(NEGATIVE_DIR)
STAGE1_SUBDIRS = {
    "BirdVox-DCASE-20k": "bv",
    "Freefield1010":      "ff",
    "Warblrb10k":         "wb",
}

# Stage2: FSC22
STAGE2_NEG_DIR = os.path.join(NEGATIVE_DIR, "fsc")

# Stage3: ESC-50
STAGE3_NEG_DIR = os.path.join(NEGATIVE_DIR, "esc")

# Stage4: DataSEC
STAGE4_NEG_DIR = os.path.join(NEGATIVE_DIR, "datasec")

# ========== SOURCE DATASET PATHS ==========
# Stage1 — DCASE source datasets
DCASE_DATASETS = {
    "BirdVox-DCASE-20k": {
        "csv": Path("/Volumes/Evo/datasets/birdvox/BirdVoxDCASE20k_csvpublic.csv"),
        "wav": Path("/Volumes/Evo/datasets/birdvox/wav"),
    },
    "Freefield1010": {
        "csv": Path("/Volumes/Evo/datasets/freefield1010/ff1010bird_metadata_2018.csv"),
        "wav": Path("/Volumes/Evo/datasets/freefield1010/wav"),
    },
    "Warblrb10k": {
        "csv": Path("/Volumes/Evo/datasets/warblr/warblrb10k_public_metadata_2018.csv"),
        "wav": Path("/Volumes/Evo/datasets/warblr/wav"),
    },
}

# Stage2 — FSC22 source
FSC22_AUDIO_DIR = "/Volumes/Evo/datasets/FSC22/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
FSC22_METADATA_PATH = "/Volumes/Evo/datasets/FSC22/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"

# Stage3 — ESC-50 source
ESC50_CSV_PATH = "/Volumes/Evo/datasets/ESC50/meta/esc50.csv"
ESC50_AUDIO_DIR = "/Volumes/Evo/datasets/ESC50/audio/"

# Stage4 — DataSEC source
DATASEC_ROOT = "/Volumes/Evo/datasets/DataSEC"
DATASEC_TARGET_NEGATIVE_TOTAL = 3597

# ========== AUDIO PROCESSING ==========
TARGET_SR = 16000        # 16 kHz
CLIP_DURATION = 3.0      # seconds
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000 samples
MIN_DURATION = 3.0       # Skip files shorter than this
