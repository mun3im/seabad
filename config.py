#!/usr/bin/env python3
"""
config.py

Centralized configuration for SEABAD dataset pipeline.
All Stage*.py scripts should import settings from this file.
"""

import os

# ========== DATASET DIRECTORIES ==========
DATASET_ROOT = "/Volumes/Evo/SEABAD"
FLAC_OUTPUT_DIR = os.path.join(DATASET_ROOT, "asean-flacs")
METADATA_DIR = "metadata"  # Store all CSV files and reports here

# Stage5 & Stage6 directories
POSITIVE_STAGING_DIR = os.path.join(DATASET_ROOT, "positive_staging")  # Stage5 WAV extraction
POSITIVE_FINAL_DIR = os.path.join(DATASET_ROOT, "positive")  # Stage6 balanced final dataset

# ========== PIPELINE CSV FLOW ==========
# Stage1 (fetch) → Stage2 (analyze) → Stage3 (download) → Stage4 (dedupe)

# Stage1: Fetch Metadata from Xeno-Canto API
STAGE1_OUTPUT_CSV = os.path.join(METADATA_DIR, "Stage1out_xc_bird_metadata.csv")

# Stage2: Analyze Metadata (optional - reads Stage1 output, prints stats only)
STAGE2_INPUT_CSV = STAGE1_OUTPUT_CSV  # from Stage1

# Stage3: Download & Convert to FLAC (reads Stage1 output)
STAGE3_INPUT_CSV = STAGE1_OUTPUT_CSV  # from Stage1
STAGE3_OUTPUT_CSV = os.path.join(METADATA_DIR, "Stage3out_successful_conversions.csv")
STAGE3_FAILED_CSV = os.path.join(METADATA_DIR, "Stage3out_failed_downloads.csv")
STAGE3_LOG_CSV = os.path.join(METADATA_DIR, "Stage3_download_log.csv")

# Stage4: Deduplicate FLACs (reads Stage3 conversions)
STAGE4_INPUT_CSV = STAGE3_OUTPUT_CSV  # from Stage3
STAGE4_OUTPUT_CSV = os.path.join(METADATA_DIR, "Stage4out_unique_flacs.csv")
STAGE4_REMOVED_CSV = os.path.join(METADATA_DIR, "Stage4_removed_near_duplicates_metadata.csv")
STAGE4_REPORT_TXT = os.path.join(METADATA_DIR, "Stage4_report.txt")

# Stage5: Extract 3s WAV clips (reads Stage4 deduplicated)
STAGE5_INPUT_CSV = STAGE4_OUTPUT_CSV  # from Stage4
STAGE5_OUTPUT_CSV = os.path.join(METADATA_DIR, "Stage5out_unique_3sclips.csv")
STAGE5_WAV_DIR = POSITIVE_STAGING_DIR  # Extract to staging area

# Stage6: Balance species (reads Stage5, moves to final positive dir)
STAGE6_INPUT_CSV = STAGE5_OUTPUT_CSV  # from Stage5
STAGE6_OUTPUT_CSV = os.path.join(METADATA_DIR, "Stage6out_balanced_clips.csv")
STAGE6_STAGING_DIR = POSITIVE_STAGING_DIR  # Read from staging
STAGE6_FINAL_DIR = POSITIVE_FINAL_DIR  # Move to final positive dir
STAGE6_MAX_CLIPS = 25000  # Default balanced dataset size

# Audio conversion settings
TARGET_SAMPLE_RATE = 16000  # 16kHz
TARGET_CHANNELS = 1  # mono
TARGET_FORMAT = "flac"
TARGET_SUBTYPE = "PCM_16"

# Download settings
RATE_LIMIT_DELAY = 0.1  # seconds between requests
MAX_RETRIES = 4
REQUEST_TIMEOUT = 30
CHUNK_SIZE = 1024 * 64  # 64KB
MIN_BYTES_ACCEPTED = 1024  # 1KB minimum file size
USER_AGENT = "xc_downloader/1.0 (+https://your.email@example.com)"

# Species filtering
EXCLUDE_SPECIES = [
    "Identity unknown",
    "identity unknown",
    "Unknown",
    "unknown"
]  # Species to exclude from download/conversion

# ========== XENO-CANTO API ==========
MAX_YEAR = 2025  # Filter to include only recordings from before 2026
BASE_URL = "https://xeno-canto.org/api/3/recordings"
