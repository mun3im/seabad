#!/usr/bin/env python3
"""
Stage 6: Extract negative samples from DataSEC
- Group 1 (EXCLUDED): bird/animal-related folders
- Group 2 (CORE):     non-bird, non-music, non-voice environmental sounds
- Group 3 (VOICES):   Voices
- Group 4 (MUSIC):    Music
- Target: exactly 3 597 samples
- Short clips (<3 s) are zero-padded and used as filler
- Output: negative/datasec/datasec-<folder>-<stem>-<onset_ms:05d>.wav
"""

import os
import glob
import sys
import logging
import random
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

from config import (
    DATASEC_ROOT, STAGE6_NEG_DIR, DATASEC_TARGET_NEGATIVE_TOTAL,
    TARGET_SR, extract_loudest_3s_or_pad,
)

BIRD_FOLDERS  = {'Birds', 'Chicken coop', 'Crows seagulls and magpies'}
MUSIC_FOLDER  = 'Music'
VOICE_FOLDER  = 'Voices'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Stage6_datasec.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def get_all_audio_files(root_dir):
    files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return sorted(files)


def main():
    os.makedirs(STAGE6_NEG_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 6: DataSEC → NEGATIVE SAMPLES (with short-file padding)")
    logger.info("=" * 70)
    logger.info(f"Source: {DATASEC_ROOT}")
    logger.info(f"Output: {STAGE6_NEG_DIR}")
    logger.info(f"Target: {DATASEC_TARGET_NEGATIVE_TOTAL} samples")
    logger.info("")

    all_files = get_all_audio_files(DATASEC_ROOT)
    logger.info(f"Total audio files found: {len(all_files)}")

    # Group by top-level folder
    folder_files = {}
    for f in all_files:
        top = os.path.relpath(f, DATASEC_ROOT).split(os.sep)[0]
        folder_files.setdefault(top, []).append(f)

    group1_excluded = []
    core_clips_long  = []
    core_clips_short = []
    voice_clips      = []
    music_clips      = []
    rejected_count   = 0

    for folder, files in folder_files.items():
        if folder in BIRD_FOLDERS:
            group1_excluded.extend(files)
            logger.info(f"Excluded bird/animal folder '{folder}': {len(files)} files")
            continue

        is_voice = (folder == VOICE_FOLDER)
        is_music = (folder == MUSIC_FOLDER)

        for audio_file in tqdm(files, desc=f"Processing {folder}", unit="file"):
            try:
                y, _ = librosa.load(audio_file, sr=TARGET_SR, mono=True)
                clip, onset_ms, is_short = extract_loudest_3s_or_pad(y, audio_file)

                if clip is None:
                    rejected_count += 1
                    continue

                basename     = Path(audio_file).stem
                folder_clean = folder.replace(' ', '_').replace('/', '_')
                out_name     = f"datasec-{folder_clean}-{basename}-{onset_ms:05d}.wav"
                if is_short:
                    out_name = out_name.replace(".wav", "_padded.wav")

                entry = (clip, out_name)

                if is_voice:
                    voice_clips.append(entry)
                elif is_music:
                    music_clips.append(entry)
                elif is_short:
                    core_clips_short.append(entry)
                else:
                    core_clips_long.append(entry)

            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                rejected_count += 1

    logger.info(f"\nGroup 1 excluded (bird/animal):        {len(group1_excluded)}")
    logger.info(f"Group 2 core environmental (>= 3s):    {len(core_clips_long)}")
    logger.info(f"Group 2 core environmental (padded):   {len(core_clips_short)}")
    logger.info(f"Group 3 Voices:                        {len(voice_clips)}")
    logger.info(f"Group 4 Music:                         {len(music_clips)}")
    logger.info(f"Rejected (silent/failed):              {rejected_count}")

    # Build final selection
    final_selection = []
    stats = {'core_long': 0, 'core_short': 0, 'voices': 0, 'music': 0}

    final_selection.extend(core_clips_long)
    stats['core_long'] = len(core_clips_long)

    final_selection.extend(core_clips_short)
    stats['core_short'] = len(core_clips_short)

    needed = DATASEC_TARGET_NEGATIVE_TOTAL - len(final_selection)
    if needed > 0:
        take_voices = min(len(voice_clips), needed)
        final_selection.extend(voice_clips[:take_voices])
        stats['voices'] = take_voices
        needed -= take_voices

    if needed > 0:
        take_music = min(len(music_clips), needed)
        if take_music > 0:
            random.seed(42)
            indices = random.sample(range(len(music_clips)), take_music)
            final_selection.extend([music_clips[i] for i in indices])
        stats['music'] = take_music

    if len(final_selection) < DATASEC_TARGET_NEGATIVE_TOTAL:
        logger.warning(f"Could only collect {len(final_selection)} / {DATASEC_TARGET_NEGATIVE_TOTAL} samples")
    elif len(final_selection) > DATASEC_TARGET_NEGATIVE_TOTAL:
        final_selection = final_selection[:DATASEC_TARGET_NEGATIVE_TOTAL]

    for clip, out_name in tqdm(final_selection, desc="Saving files", unit="file"):
        sf.write(os.path.join(STAGE6_NEG_DIR, out_name), clip, TARGET_SR)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 6 COMPLETE — DataSEC")
    logger.info("=" * 70)
    logger.info(f"  Group 1 excluded (bird/animal): {len(group1_excluded)}")
    logger.info(f"  Group 2 core >= 3s:             {stats['core_long']}")
    logger.info(f"  Group 2 core padded:            {stats['core_short']}")
    logger.info(f"  Group 3 Voices:                 {stats['voices']}")
    logger.info(f"  Group 4 Music:                  {stats['music']}")
    logger.info(f"  Total saved:                    {len(final_selection)}")
    logger.info(f"  Output: {STAGE6_NEG_DIR}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
