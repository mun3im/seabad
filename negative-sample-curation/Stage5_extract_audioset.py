#!/usr/bin/env python3
"""
Extract 3,600 realistic non-bird environmental negatives from AudioSet:
- 1,800 urban (traffic, engine, etc.)
- 1,800 natural (forest, rain, wind — without birds/insects)
- Downloads metadata automatically if missing
- Uses loudest 3s window
- Output: as-<youtube_id>-<start_ms:05d>.wav
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import requests
from tqdm import tqdm
import logging
import subprocess
import tempfile
from pathlib import Path
import time

# ============================= CONFIGURATION =============================
METADATA_DIR = Path("/Volumes/Evo/datasets/audioset_meta")
OUTPUT_DIR = Path("/Volumes/Evo/mybad_v4/negative/audioset")
TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)
TARGET_COUNT = 1800  # Per category

METADATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'audioset_extraction.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================= AUDIOSET LABELS =============================

URBAN_LABELS = {
    '/m/07r04',      # Traffic noise
    '/m/01b_21',     # Engine
    '/m/04_sv',      # Motorcycle
    '/m/0k4j',       # Car
    '/m/01bjv',      # Bus
    '/m/07_j1_',     # Truck
    '/m/0199g',      # Bicycle
    '/m/02mk9',      # Horn
    '/m/0284vy3',    # Siren
    '/m/0ngn',       # Construction
    '/m/0347h',      # Jackhammer
    '/m/01j3sz',     # Train
    '/m/06ncr',      # Subway, metro, underground
    '/m/01940j',     # Alarm
    '/m/07rkbfh',    # Air brake
    '/m/01h8n0',     # Conversation (but we'll exclude speech anyway)
    '/m/04rmv',      # Gunshot (urban gunfire is common in some datasets)
    '/m/02p8v3',     # Water tap, faucet
    '/m/09f_7n',     # Stream (corrected MID if needed)
}

NATURAL_LABELS = {
    '/m/0jbk',       # Rain
    '/m/07pbtc8',    # Raindrop
    '/m/083vt',      # Wind
    '/m/02_djx',     # Wind chime (bonus)
    '/m/01jwgf',     # Ocean
    '/m/01h8n0',     # Waves, surf
    '/m/01hwkn',     # Thunderstorm
    '/m/032s66',     # Waterfall
    '/m/09f_7n',     # Stream (corrected)
    '/m/014zdl',     # River
    '/m/01j4z',      # Forest/woodland
    '/m/02hnl7',     # Drip
    # Removed bird/insect-related: '/m/030rvx', '/m/014z8_', '/m/0ch8v'
}

FORBIDDEN_LABELS = {
    '/m/07qfr',   # Bird
    '/m/018vs',   # Insect
    '/m/019nj4',  # Animal
    '/m/09x0r',   # Speech
    '/m/0dgw9r',  # Human voice
    '/m/07p6fty', # Bird vocalization
    '/m/01lsmm',  # Crowd
    '/m/02l0t',   # Domestic animals
    '/m/07qfr',   # Music is already excluded by label set, but safe to keep
}

# ============================= DOWNLOAD METADATA =============================

def download_audioset_csv(filename):
    url = f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{filename}"
    out_path = METADATA_DIR / filename
    if out_path.exists():
        logger.info(f"Metadata already exists: {filename}")
        return
    logger.info(f"Downloading {filename}...")
    r = requests.get(url)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        f.write(r.content)
    logger.info(f"Saved {filename}")

def load_audioset_csv(path):
    """Parse AudioSet CSV files robustly (handles quoted labels with commas)."""
    rows = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= 3:  # skip header comments
                continue
            line = line.strip()
            if not line:
                continue

            # Split into max 4 parts: YTID, start, end, labels
            parts = line.split(',', 3)
            if len(parts) != 4:
                logger.warning(f"Skipping malformed line {line_num} in {path}")
                continue

            ytid, start_str, end_str, labels_raw = parts
            ytid = ytid.strip()
            start_str = start_str.strip()
            end_str = end_str.strip()
            labels_raw = labels_raw.strip()

            # Remove surrounding double quotes
            if labels_raw.startswith('"') and labels_raw.endswith('"'):
                labels_raw = labels_raw[1:-1]

            try:
                start_sec = float(start_str)
                end_sec = float(end_str)
            except ValueError:
                logger.warning(f"Invalid time in line {line_num}: {start_str}, {end_str}")
                continue

            rows.append({
                'YTID': ytid,
                'start_seconds': start_sec,
                'end_seconds': end_sec,
                'labels': labels_raw
            })
    return pd.DataFrame(rows)

# ============================= FILTERING =============================

def classify_clip(label_str):
    if pd.isna(label_str):
        return None
    labels = set(lbl.strip() for lbl in str(label_str).split(','))
    if labels & FORBIDDEN_LABELS:
        return None
    if labels & URBAN_LABELS:
        return 'urban'
    if labels & NATURAL_LABELS:
        return 'natural'
    return None

# ============================= AUDIO PROCESSING =============================

def extract_loudest_3s(y, sr=TARGET_SR):
    if len(y) < CLIP_SAMPLES:
        # Pad with zeros if slightly shorter
        pad_len = CLIP_SAMPLES - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')
        return y, 0

    if len(y) == CLIP_SAMPLES:
        return y, 0

    hop = int(0.1 * sr)
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
        if len(best_clip) < CLIP_SAMPLES:
            pad_len = CLIP_SAMPLES - len(best_clip)
            best_clip = np.pad(best_clip, (0, pad_len), mode='constant')

    if np.all(best_clip == 0):
        return None, None

    start_ms = int(round(1000.0 * best_start / sr))
    return best_clip, start_ms

def download_and_process(ytid, start_sec, end_sec):
    url = f"https://www.youtube.com/watch?v={ytid}"
    with tempfile.TemporaryDirectory() as tmpdir:
        for attempt in range(3):  # Retry up to 3 times
            try:
                outtmpl = os.path.join(tmpdir, '%(id)s.%(ext)s')
                cmd = [
                    'yt-dlp',
                    '--quiet',
                    '--no-warnings',
                    '-x',
                    '--audio-format', 'wav',
                    '--audio-quality', '0',
                    '--output', outtmpl,
                    '--postprocessor-args', f'-ss {start_sec} -t 10',
                    url
                ]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    err_msg = result.stderr.decode('utf-8').strip()
                    logger.warning(f"Attempt {attempt+1} failed for {ytid}: {err_msg}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                wav_path = os.path.join(tmpdir, f"{ytid}.wav")
                if not os.path.exists(wav_path):
                    logger.warning(f"Attempt {attempt+1} failed for {ytid}: No WAV file produced")
                    time.sleep(2 ** attempt)
                    continue

                y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                return extract_loudest_3s(y)

            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {ytid}: {str(e)}")
                time.sleep(2 ** attempt)  # Backoff

        logger.error(f"All attempts failed for {ytid}")
        return None, None

# ============================= MAIN =============================

def main():
    # Download metadata
    for fname in ['balanced_train_segments.csv', 'eval_segments.csv']:
        download_audioset_csv(fname)

    # Load and parse
    dfs = []
    for csv_name in ['balanced_train_segments.csv', 'eval_segments.csv']:
        path = METADATA_DIR / csv_name
        logger.info(f"Parsing {csv_name}...")
        df = load_audioset_csv(path)
        dfs.append(df)
        logger.info(f"Loaded {len(df)} segments from {csv_name}")

    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total AudioSet segments loaded: {len(full_df)}")

    # Classify
    full_df['type'] = full_df['labels'].apply(classify_clip)
    urban_df = full_df[full_df['type'] == 'urban']
    natural_df = full_df[full_df['type'] == 'natural']

    logger.info(f"Urban candidates: {len(urban_df)}")
    logger.info(f"Natural candidates: {len(natural_df)}")

    # Sample up to TARGET_COUNT each
    urban_sample = urban_df.sample(n=min(TARGET_COUNT, len(urban_df)), random_state=42)
    natural_sample = natural_df.sample(n=min(TARGET_COUNT, len(natural_df)), random_state=42)
    final_df = pd.concat([urban_sample, natural_sample]).reset_index(drop=True)

    logger.info(f"Processing {len(final_df)} clips ({len(urban_sample)} urban + {len(natural_sample)} natural)")

    saved = 0
    for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Downloading"):
        ytid = row['YTID']
        start_sec = float(row['start_seconds'])
        end_sec = float(row['end_seconds'])

        # Skip if already exists (check any variant)
        existing = list(OUTPUT_DIR.glob(f"as-{ytid}-*.wav"))
        if existing:
            logger.info(f"Skipping existing: {ytid}")
            continue

        clip, start_ms = download_and_process(ytid, start_sec, end_sec)
        if clip is not None:
            out_name = f"as-{ytid}-{start_ms:05d}.wav"
            sf.write(OUTPUT_DIR / out_name, clip, TARGET_SR)
            saved += 1

    logger.info(f"✅ Done. Saved {saved} AudioSet negatives to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
