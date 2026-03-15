#!/usr/bin/env python3
"""
Final DATASEC extraction for bird gatekeeper negatives (updated).
- Group 1 (EXCLUDED): bird/animal-related sounds
- Group 2 (CORE): non-bird, non-music, non-voice environmental sounds
- Group 3 (VOICES): Voices
- Group 4 (MUSIC): Music
- Target: exactly 3,597 samples
- Short clips (<3s) are zero-padded to 3s and used as filler, sorted by duration
- With tqdm progress tracking
"""

import os
import glob
import librosa
import soundfile as sf
import numpy as np
import logging
from pathlib import Path
import random
from tqdm import tqdm

# ============================= CONFIG =============================
DATASEC_ROOT = '/Volumes/Evo/datasets/DataSEC'
NEG_DIR = '/Volumes/Evo/mybad_v4/negative/datasec'

# Group 1: EXCLUDE (bird/animal-related — expanded from DataSEC paper)
BIRD_FOLDERS = {
    'Birds',
    'Chicken coop',
    'Crows seagulls and magpies'
}

MUSIC_FOLDER = 'Music'
VOICE_FOLDER = 'Voices'

TARGET_NEGATIVE_TOTAL = 3597

TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)

os.makedirs(NEG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datasec_extraction_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================= UTILS =============================

def get_all_audio_files(root_dir):
    extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return sorted(files)


def extract_loudest_3s_or_pad(y, sr=TARGET_SR, filename_for_log=""):
    orig_len = len(y)

    if orig_len == 0:
        logger.debug(f"Empty file skipped: {filename_for_log}")
        return None, None, False

    # If exactly 3s and not silent → use as-is
    if orig_len == CLIP_SAMPLES and not np.all(y == 0):
        return y, 0, False

    # If longer → find loudest 3s
    if orig_len > CLIP_SAMPLES:
        hop = int(0.1 * sr)
        best_rms = -1.0
        best_clip = None
        best_start = 0

        for start in range(0, orig_len - CLIP_SAMPLES + 1, hop):
            seg = y[start:start + CLIP_SAMPLES]
            rms = np.sqrt(np.mean(np.square(seg.astype(np.float64))))
            if rms > best_rms:
                best_rms = rms
                best_clip = seg.copy()
                best_start = start

        if best_clip is not None and not np.all(best_clip == 0):
            start_ms = int(round(1000.0 * best_start / sr))
            return best_clip, start_ms, False

    # If shorter → zero-pad to 3s
    if orig_len < CLIP_SAMPLES:
        pad_len = CLIP_SAMPLES - orig_len
        padded = np.pad(y, (0, pad_len), mode='constant', constant_values=0)
        if np.all(padded == 0):
            logger.debug(f"Silent after padding: {filename_for_log}")
            return None, None, False
        return padded, 0, True  # start at 0 ms for padded, flag as short

    # Otherwise (e.g. silent long clip)
    return None, None, False


# ============================= MAIN =============================

def main():
    logger.info("Starting final DATASEC extraction (with short-file padding)...")
    all_files = get_all_audio_files(DATASEC_ROOT)
    logger.info(f"Total audio files found: {len(all_files)}")

    # Group files by top folder
    folder_files = {}
    for f in all_files:
        rel = os.path.relpath(f, DATASEC_ROOT)
        top = rel.split(os.sep)[0]
        folder_files.setdefault(top, []).append(f)

    # Containers
    group1_excluded = []  # bird/animal
    core_clips_long = []  # Group 2: >= 3s environmental
    core_clips_short = []  # Group 2: padded short environmental
    voice_clips = []  # Group 3: voices (all >= 3s, no shorts)
    music_clips = []  # Group 4: music (all >= 3s, no shorts)

    rejected_count = 0

    # Process files with tqdm
    for folder, files in folder_files.items():
        if folder in BIRD_FOLDERS:
            group1_excluded.extend(files)
            logger.info(f"Excluded bird/animal folder '{folder}': {len(files)} files")
            continue

        is_voice = (folder == VOICE_FOLDER)
        is_music = (folder == MUSIC_FOLDER)

        for audio_file in tqdm(files, desc=f"Processing {folder}", unit="file"):
            try:
                y, sr = librosa.load(audio_file, sr=TARGET_SR, mono=True)
                clip, start_ms, is_short = extract_loudest_3s_or_pad(y, sr, audio_file)

                if clip is None:
                    rejected_count += 1
                    continue

                basename = Path(audio_file).stem
                folder_clean = folder.replace(' ', '_').replace('/', '_')
                out_name = f"datasec-{folder_clean}-{basename}-{start_ms:05d}.wav"
                if is_short:
                    out_name = out_name.replace(".wav", "_padded.wav")

                entry = (clip, out_name)

                # Collect by group
                if is_voice:
                    voice_clips.append(entry)
                elif is_music:
                    music_clips.append(entry)
                else:
                    if is_short:
                        core_clips_short.append(entry)
                    else:
                        core_clips_long.append(entry)

            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                rejected_count += 1

    logger.info(f"\nGroup 1 (excluded bird/animal): {len(group1_excluded)} files")
    logger.info(f"Group 2 (core environmental >= 3s): {len(core_clips_long)}")
    logger.info(f"Group 2 (padded shorts): {len(core_clips_short)}")
    logger.info(f"Group 3 (Voices): {len(voice_clips)}")
    logger.info(f"Group 4 (Music): {len(music_clips)}")
    logger.info(f"Rejected (silent/failed load): {rejected_count}")

    # Build final selection
    final_selection = []
    stats = {'core_long': 0, 'core_short': 0, 'voices': 0, 'music': 0}

    # 1. Take ALL core long
    final_selection.extend(core_clips_long)
    stats['core_long'] = len(core_clips_long)

    # 2. Take ALL core short (padded)
    final_selection.extend(core_clips_short)
    stats['core_short'] = len(core_clips_short)

    # 3. If needed → add voices
    needed = TARGET_NEGATIVE_TOTAL - len(final_selection)
    if needed > 0:
        take_voices = min(len(voice_clips), needed)
        final_selection.extend(voice_clips[:take_voices])
        stats['voices'] = take_voices
        needed -= take_voices

    # 4. Last resort: music
    if needed > 0:
        take_music = min(len(music_clips), needed)
        if take_music > 0:
            random.seed(42)
            indices = random.sample(range(len(music_clips)), take_music)
            final_selection.extend([music_clips[i] for i in indices])
        stats['music'] = take_music
        needed -= take_music

    # Final check & save
    final_count = len(final_selection)
    if final_count < TARGET_NEGATIVE_TOTAL:
        logger.warning(f"Could only collect {final_count} / {TARGET_NEGATIVE_TOTAL} samples")
    elif final_count > TARGET_NEGATIVE_TOTAL:
        # Trim excess (unlikely)
        final_selection = final_selection[:TARGET_NEGATIVE_TOTAL]

    # Save files with progress bar
    for clip, out_name in tqdm(final_selection, desc="Saving files", unit="file"):
        sf.write(os.path.join(NEG_DIR, out_name), clip, TARGET_SR)

    # Calculate totals
    total_long = stats['core_long'] + stats['voices'] + stats['music']
    total_short = stats['core_short']

    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPOSITION:")
    logger.info(f"Group 1: excluded bird/bird-related: {len(group1_excluded)}")
    logger.info(f"Group 2: core environmental >= 3s  : {stats['core_long']}")
    logger.info(f"         padded shorts             : {stats['core_short']}")
    logger.info(f"Group 3: Voices                    : {stats['voices']}")
    logger.info(f"Group 4: Music                     : {stats['music']}")
    logger.info(f"")
    logger.info(f"total >= 3s                        : {total_long}")
    logger.info(f"total short                        : {total_short}")
    logger.info(f"Total                              : {len(final_selection)}")
    logger.info("=" * 60)
    logger.info(f"Saved to: {NEG_DIR}")

    return 0


if __name__ == "__main__":
    exit(main())