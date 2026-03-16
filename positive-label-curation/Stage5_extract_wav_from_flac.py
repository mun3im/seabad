#!/usr/bin/env python3
"""
Stage5_extract_3s_clips_from_flac.py

Extracts 3-second WAV clips from FLAC files with metadata enrichment from Stage4.

Input: Stage4out_unique_flacs.csv (default, from config.py)
Output: Stage5out_unique_3sclips.csv (default, from config.py)
  WAV clips written to: positive_staging/ directory (default, from config.py)
  - Contains all Stage4 metadata fields PLUS:
    - onset_ms: clip start time in milliseconds
    - clip_filename: format xc{id}_{quality}_{onset_ms}.wav

Key features:
 - Writes 3s chunks as WAV files (16 kHz, mono, PCM_16)
 - Applies clipping correction when needed
 - Guarantees at least one sample per file (picks best if none exceed threshold)
 - Dual progress bars for species and files
 - Two workflow modes: RMS-only (quick) or Balanced (for Stage6)

TWO WORKFLOWS:

1. RMS-ONLY WORKFLOW (Quick & Dirty):
   - Selects top N clips (default 25,000) by RMS energy; rest go to quarantine
   - Final dataset ready immediately after Stage5
   - Usage:
     python Stage5_extract_3s_clips_from_flac.py \
       --threshold 0.001 \
       --max-clips 25000
   (Uses defaults from config.py: FLAC_OUTPUT_DIR → POSITIVE_STAGING_DIR)

2. BALANCED WORKFLOW (Ecologically Diverse):
   - Keep ALL clips, no quarantining
   - Proceed to Stage6 for species-level balancing
   - Usage:
     python Stage5_extract_3s_clips_from_flac.py \
       --threshold 0.001 \
       --no-quarantine
   (Uses defaults from config.py)

     Then run Stage6:
     python Stage6_balance_species.py \
       --target-size 25000
   (Reads from POSITIVE_STAGING_DIR, moves to POSITIVE_FINAL_DIR)

"""

from pathlib import Path
import argparse
import os
import sys
import math
import time
import logging
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
from io import BytesIO
import pandas as pd
from tqdm import tqdm

import config

# ---------------- CONFIG / DEFAULTS ----------------
DEFAULT_SR = 16000  # final sampling rate (Hz)
WINDOW_SEC = 3.0    # 3-second windows
STEP_SEC = 0.1      # 100 ms step (fixed)
CLIPPING_CEILING = 0.99  # target peak after correction
SOFT_LIMIT_ALPHA = 5.0   # soft limiter shape parameter
MIN_SEPARATION_SEC = 1.5  # minimum temporal separation between chosen chunks
# ---------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("extract_clips")

# -------- helper utilities --------
def load_metadata_lookup(csv_path: Path) -> dict:
    """
    Load Stage4_unique_flacs.csv and return a dict mapping xc_id -> metadata dict.
    Keys in metadata dict match the CSV column names (id, en, rec, cnt, lat, lon, lic, q, length, smp).
    """
    if not csv_path or not csv_path.exists():
        logger.warning(f"Metadata CSV not found: {csv_path}. Proceeding without metadata enrichment.")
        return {}

    try:
        df = pd.read_csv(csv_path, dtype=str)  # Read all as strings to preserve formatting
        lookup = {}
        for _, row in df.iterrows():
            xc_id = str(row.get('id', '')).strip()
            if xc_id:
                lookup[xc_id] = row.to_dict()
        logger.info(f"Loaded metadata for {len(lookup)} recordings from {csv_path}")
        return lookup
    except Exception as e:
        logger.error(f"Failed to load metadata CSV {csv_path}: {e}")
        return {}


def extract_xc_id_from_name(name: str) -> Optional[str]:
    """Try to extract integer Xeno-canto id from filename like 'xc123456' or 'xc123456_extra'."""
    import re
    m = re.search(r'xc(\d+)', name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def extract_xc_quality(name: str) -> str:
    """
    Extract XC quality rating from filename.
    Format: xc{id}_{quality} where quality is A, B, C, D, or U
    Returns 'U' if not found.
    """
    import re
    m = re.search(r'xc\d+_([A-DU])', name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else 'U'


def rms_of_segment(y: np.ndarray) -> float:
    """Compute RMS of audio vector y (mono)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


def is_clipped(y: np.ndarray, threshold: float = 0.9999) -> bool:
    """Detect clipping by checking if any sample magnitude >= threshold."""
    return bool(np.any(np.abs(y) >= threshold))


def peak_scale_and_soft_limit(y: np.ndarray, ceiling: float = CLIPPING_CEILING, alpha: float = SOFT_LIMIT_ALPHA) -> np.ndarray:
    """
    1) Linear peak scaling so that max(abs(y)) <= ceiling
    2) Apply soft limiter y_out = (1/alpha) * tanh(alpha * y_scaled)
    Returns float32 array in [-1,1].
    """
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak == 0.0:
        return y
    # scale so peak <= ceiling (but if peak < ceiling we still might apply soft limiter lightly)
    scale = 1.0
    if peak > ceiling:
        scale = ceiling / peak
    y_scaled = y * scale
    # apply soft limiter
    y_limited = (1.0 / alpha) * np.tanh(alpha * y_scaled)
    # If the limiter introduces values >1 (shouldn't), clip
    y_limited = np.clip(y_limited, -1.0, 1.0)
    return y_limited


def ensure_mono(y: np.ndarray) -> np.ndarray:
    """Convert multichannel audio to mono by averaging channels if needed."""
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)

# -------- selection helpers --------
def sliding_windows_rms(y: np.ndarray, sr: int, window_sec: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (starts_samples, rms_values) for sliding windows over y.
    starts_samples: numpy array of start sample indices
    rms_values: numpy array of RMS floats corresponding to each start
    """
    win_len = int(round(window_sec * sr))
    step = int(round(step_sec * sr))
    if win_len <= 0 or step <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    starts = np.arange(0, max(1, len(y) - win_len + 1), step, dtype=int)
    rms_list = []
    for s in starts:
        seg = y[s: s + win_len]
        rms_list.append(rms_of_segment(seg))
    return starts, np.array(rms_list, dtype=float)


def choose_diverse_chunks(starts: np.ndarray, rms_vals: np.ndarray, sr: int, num_chunks: int,
                          min_separation_sec: float, threshold: float) -> List[Tuple[int,float]]:
    """
    Choose up to num_chunks windows preferring diversity across time and above threshold.
    Returns list of (start_sample, rms) tuples.
    """
    chosen = []
    if starts.size == 0:
        return chosen
    # candidate indices
    candidate_idx = np.flatnonzero(rms_vals >= threshold)
    # if no candidates above threshold, return empty (caller may fallback)
    if candidate_idx.size == 0:
        return chosen
    # create a list of (idx, rms) and sort by rms desc
    cand = sorted(((int(i), float(rms_vals[i])) for i in candidate_idx), key=lambda x: x[1], reverse=True)
    chosen_starts = []
    min_sep_samples = int(round(min_separation_sec * sr))
    for idx, rms in cand:
        s = int(starts[idx])
        # check separation
        too_close = False
        for cs in chosen_starts:
            if abs(cs - s) < min_sep_samples:
                too_close = True
                break
        if not too_close:
            chosen.append((s, rms))
            chosen_starts.append(s)
            if len(chosen) >= num_chunks:
                break
    return chosen


def choose_best_chunks_any(starts: np.ndarray, rms_vals: np.ndarray, num_chunks: int, sr: int, min_sep_sec: float) -> List[Tuple[int,float]]:
    """Pick best num_chunks by RMS but enforce min separation. Allows below-threshold picks."""
    if starts.size == 0:
        return []
    idxs = np.argsort(rms_vals)[::-1]  # sorted by rms desc
    chosen = []
    chosen_starts = []
    min_sep_samples = int(round(min_sep_sec * sr))
    for i in idxs:
        s = int(starts[i])
        rms = float(rms_vals[i])
        too_close = False
        for cs in chosen_starts:
            if abs(cs - s) < min_sep_samples:
                too_close = True
                break
        if not too_close:
            chosen.append((s, rms))
            chosen_starts.append(s)
            if len(chosen) >= num_chunks:
                break
    return chosen

# -------- main processing per file --------
def process_file(path: Path, species: str, out_root: Path, sr_out: int, threshold: float,
                 csv_records: List[dict], metadata_lookup: dict = None, dry_run: bool = False) -> Tuple[List[dict], int]:
    """
    Process a single FLAC file and return updated csv_records and number of clips saved.
    Always guarantees at least one clip per file (picks best if none exceed threshold).
    metadata_lookup: dict mapping xc_id -> metadata dict from Stage4 CSV
    """
    saved_count = 0
    # read file (librosa load uses soundfile backend by default)
    try:
        y, sr = librosa.load(str(path), sr=sr_out, mono=True)  # ensures resampled to sr_out and mono
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return csv_records, saved_count

    duration = librosa.get_duration(y=y, sr=sr)
    base_name = path.stem  # e.g. xc12345_A
    xc_id = extract_xc_id_from_name(base_name) or "unknown"
    quality = extract_xc_quality(base_name)
    file_duration = duration  # Store for CSV

    # Get metadata from lookup (if available)
    metadata = {}
    if metadata_lookup and xc_id != "unknown":
        metadata = metadata_lookup.get(xc_id, {})

    def make_record(onset_ms, clip_dur, rms, clipped, fname, fpath):
        """Helper to create CSV record with all fields including metadata from Stage4."""
        # Start with all Stage4 metadata fields
        record = {}
        if metadata:
            # Copy all fields from Stage4 CSV
            for key, value in metadata.items():
                record[key] = value
        else:
            # If no metadata, populate with minimal required fields
            # Normalize species name: replace underscores with spaces for consistency
            normalized_species = species.replace('_', ' ')
            record = {
                "id": xc_id,
                "en": normalized_species,
                "rec": "",
                "cnt": "",
                "lat": "",
                "lon": "",
                "lic": "",
                "q": quality,
                "length": "",
                "smp": ""
            }

        # Add Stage5-specific fields: onset_ms and clip_filename
        record["onset_ms"] = onset_ms
        record["clip_filename"] = fname

        return record

    # Determine expected number of chunks and offset rule (3s windows)
    if duration < 3.0:
        chunks_expected = 0
        offset_sec = 0.0
    elif duration <= 12.0:
        chunks_expected = 1
        offset_sec = 0.0
    else:
        # Long files: skip first 3s to avoid possible voice annotation
        chunks_expected = 1
        offset_sec = 3.0

    win_len_samples = int(round(WINDOW_SEC * sr))
    step_samples = int(round(STEP_SEC * sr))
    starts, rms_vals = sliding_windows_rms(y, sr, WINDOW_SEC, STEP_SEC)

    # choose diverse chunks above threshold
    candidates = choose_diverse_chunks(starts, rms_vals, sr, chunks_expected, MIN_SEPARATION_SEC, threshold)

    # if not enough candidates, pick best remaining (allow below threshold)
    # This guarantees at least one sample per file
    if len(candidates) < chunks_expected:
        fallback = choose_best_chunks_any(starts, rms_vals, chunks_expected, sr, MIN_SEPARATION_SEC)
        # merge unique starts, prioritize previously chosen ones
        chosen_dict = {s: rms for s, rms in candidates}
        for s, rms in fallback:
            if s not in chosen_dict and len(chosen_dict) < chunks_expected:
                chosen_dict[s] = rms
        candidates = sorted([(s, chosen_dict[s]) for s in chosen_dict], key=lambda x: x[0])

    # if still empty, skip this file (silently record in CSV only)
    if not candidates:
        # Record in CSV for tracking but don't print warning
        csv_records.append(make_record(0, 0.0, 0.0, False, "SKIPPED_TOO_SHORT", "SKIPPED_TOO_SHORT"))
        return csv_records, saved_count

    # Save chosen chunks as FLAC with onset_ms suffix
    for s_samples, rms_pre in candidates:
        start_sec = s_samples / sr
        onset_ms = int(round(start_sec * 1000.0))

        # Flatten output - all files in output root (no subdirectories)
        out_fname = f"{base_name}_{onset_ms}.wav"
        out_path = out_root / out_fname

        if dry_run:
            # For dry-run, keep the precomputed rms as best estimate
            csv_records.append(make_record(onset_ms, WINDOW_SEC, rms_pre, False, out_fname, str(out_path)))
            saved_count += 1
            continue

        # Extract chunk and apply clipping correction if needed
        s = s_samples
        e = s + win_len_samples
        seg = y[s:e]

        clipped = is_clipped(seg, threshold=0.9999)
        if clipped:
            seg = peak_scale_and_soft_limit(seg, ceiling=CLIPPING_CEILING, alpha=SOFT_LIMIT_ALPHA)

        # ensure dtype float32
        seg_to_write = seg.astype(np.float32)
        # recompute RMS of the final segment (this is what we'll log)
        final_rms = rms_of_segment(seg_to_write)

        try:
            # write wav (soundfile infers format from extension too)
            sf.write(str(out_path), seg_to_write, sr, format='WAV', subtype='PCM_16')
            csv_records.append(make_record(onset_ms, WINDOW_SEC, final_rms, clipped, out_fname, str(out_path)))
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to write {out_path}: {e}")

    return csv_records, saved_count


def select_top_clips_by_rms(csv_path: Path, output_root: Path, max_clips: int = 25000, no_quarantine: bool = False, dry_run: bool = False):
    """
    Select top N clips by RMS energy; rest go to quarantine.

    Two workflows:
    1. RMS-only workflow: Apply max_clips cap and quarantine excess clips
    2. Balanced workflow (--no-quarantine): Keep all clips for Stage6 species balancing
    """
    print(f"\n{'='*60}")
    print("Post-processing: RMS-based selection")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    total_clips = len(df)
    print(f"Total clips generated: {total_clips:,}")

    if no_quarantine:
        print(f"\n--no-quarantine flag set: Keeping all {total_clips:,} clips for Stage6 balancing")
        print(f"No quarantining will be performed.")
        print(f"Proceed to Stage6 for species-level balancing.")
        return

    # RMS-only workflow: apply max_clips cap
    print(f"\nApplying RMS-only workflow with max_clips={max_clips:,}")

    if total_clips <= max_clips:
        print(f"≤{max_clips:,} clips — no quarantining needed.")
        print(f"Final clip count: {total_clips:,}")
        return

    # Sort by RMS descending
    df = df.sort_values(by="rms_energy", ascending=False).reset_index(drop=True)

    # Split into keep and quarantine
    keep_df = df.head(max_clips)
    quarantine_df = df.iloc[max_clips:]

    quarantine_dir = output_root / "quarantine"
    if not dry_run:
        quarantine_dir.mkdir(exist_ok=True)

    # Move quarantined files with progress bar
    print(f"\nMoving {len(quarantine_df):,} clips to quarantine...")
    for _, row in tqdm(quarantine_df.iterrows(), total=len(quarantine_df), desc="Quarantining", unit="file"):
        src_path = Path(row["out_path"])
        dst_path = quarantine_dir / src_path.name
        if src_path.exists():
            if dry_run:
                pass  # Silent in dry-run
            else:
                try:
                    src_path.rename(dst_path)
                except Exception as e:
                    logger.error(f"Failed to move {src_path} to quarantine: {e}")
        else:
            logger.warning(f"Quarantine file not found (already moved?): {src_path}")

    # Save kept clips CSV (overwrite with only top clips)
    if not dry_run:
        keep_df.to_csv(csv_path, index=False)
        print(f"\nCSV updated to contain only top {max_clips:,} loudest clips")
    else:
        print(f"\n[DRY] Would truncate CSV to {max_clips:,} rows (top RMS)")

    print(f"Quarantined {len(quarantine_df):,} clips into {quarantine_dir}")
    print(f"Final clip count: {max_clips:,}")
    print(f"\nRMS-only workflow complete. Dataset ready for use.")

# -------- main loop over folders --------
def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Extract 3s WAV clips from FLAC files with smart selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOWS:
  RMS-only (quick):    Use --max-clips to cap dataset size immediately
  Balanced (diverse):  Use --no-quarantine, then proceed to Stage6

EXAMPLES:
  # RMS-only workflow (final dataset ready after Stage5)
  python Stage5_extract_3s_clips_from_flac.py --inroot flac/ --outroot clips/ \\
    --output-csv Stage5_unique_3sclips.csv --metadata-csv Stage4_unique_flacs.csv \\
    --max-clips 25000

  # Balanced workflow (keep all clips for Stage6)
  python Stage5_extract_3s_clips_from_flac.py --inroot flac/ --outroot clips/ \\
    --output-csv Stage5_unique_3sclips.csv --metadata-csv Stage4_unique_flacs.csv \\
    --no-quarantine
        """
    )

    # Required arguments
    parser.add_argument("--inroot", default=config.FLAC_OUTPUT_DIR, metavar="DIR",
                        help=f"Input directory containing species subfolders with FLAC files (default: {config.FLAC_OUTPUT_DIR})")
    parser.add_argument("--outroot", default=config.STAGE5_WAV_DIR, metavar="DIR",
                        help=f"Output directory for extracted 3s WAV clips (default: {config.STAGE5_WAV_DIR})")

    # Output options
    parser.add_argument("--output-csv", default=config.STAGE5_OUTPUT_CSV, metavar="FILE",
                        help=f"Output CSV file with clip metadata (default: {config.STAGE5_OUTPUT_CSV})")
    parser.add_argument("--metadata-csv", type=Path, default=config.STAGE5_INPUT_CSV, metavar="FILE",
                        help=f"Stage4 output CSV for enriching output with metadata (default: {config.STAGE5_INPUT_CSV})")

    # Processing options
    parser.add_argument("--threshold", type=float, default=0.001, metavar="FLOAT",
                        help="RMS threshold for non-silent windows (default: 0.001)")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, metavar="HZ",
                        help="Output sampling rate in Hz (default: 16000)")
    parser.add_argument("--max-clips", type=int, default=None, metavar="N",
                        help="Max clips to keep by RMS. Auto: (total//1000)*1000. Ignored with --no-quarantine")
    parser.add_argument("--no-quarantine", action="store_true",
                        help="Keep all clips for Stage6 balancing (balanced workflow)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate processing without writing audio files")
    args = parser.parse_args()

    # Ensure metadata directory exists
    Path(config.METADATA_DIR).mkdir(parents=True, exist_ok=True)

    input_root = Path(args.inroot)
    output_root = Path(args.outroot)
    sr_out = int(args.sr)
    threshold = float(args.threshold)
    csv_path = Path(args.output_csv)
    metadata_csv = args.metadata_csv
    dry_run = bool(args.dry_run)
    max_clips = int(args.max_clips) if args.max_clips is not None else None
    no_quarantine = bool(args.no_quarantine)

    if not input_root.exists():
        print(f"ERROR: Input root does not exist: {input_root}")
        sys.exit(2)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load metadata lookup from Stage4 CSV if provided
    metadata_lookup = load_metadata_lookup(metadata_csv) if metadata_csv else {}

    # Collect all species directories (exclude Stage4 quarantine folders)
    QUARANTINE_DIRS = {"perfect duplicates", "near duplicates", "perfect_duplicates", "near_duplicates"}
    species_dirs = [
        p for p in sorted(input_root.iterdir())
        if p.is_dir() and p.name not in QUARANTINE_DIRS
    ]
    total_species = len(species_dirs)

    # Determine workflow
    workflow = "Balanced (→ Stage6)" if no_quarantine else "RMS-only"

    print(f"\n{'='*60}")
    print(f"Stage 5: Extracting 3s WAV clips from FLAC files")
    print(f"{'='*60}")
    print(f"Workflow: {workflow}")
    print(f"Total species to process: {total_species}")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"RMS threshold: {threshold}")
    if not no_quarantine:
        if max_clips is not None:
            print(f"Max clips to keep: {max_clips:,}")
        else:
            print(f"Max clips to keep: Auto-calculated after extraction")
    print(f"{'='*60}\n")

    # Main processing loop with dual progress bars
    csv_records: List[dict] = []
    total_saved = 0

    # Outer progress bar for species
    species_pbar = tqdm(species_dirs, desc="Species", unit="species", position=0)

    for species_dir in species_pbar:
        species_name = species_dir.name
        species_pbar.set_postfix_str(f"Current: {species_name[:30]}")

        flac_files = sorted([p for p in species_dir.glob("*.flac")])

        # Inner progress bar for files within species
        file_pbar = tqdm(flac_files, desc=f"  Files in {species_name[:25]}", unit="file", position=1, leave=False)

        for fpath in file_pbar:
            try:
                csv_records, saved = process_file(
                    fpath, species_name, output_root, sr_out, threshold,
                    csv_records, metadata_lookup=metadata_lookup, dry_run=dry_run
                )
                total_saved += saved
            except Exception as e:
                logger.error(f"Unexpected error processing {fpath}: {e}")

        file_pbar.close()

    species_pbar.close()

    # Write all records to CSV (fresh file every run, no appending)
    if csv_records:
        df_all = pd.DataFrame(csv_records)
        df_all.to_csv(csv_path, mode="w", header=True, index=False)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"Total clips generated: {total_saved:,}")
    print(f"{'='*60}\n")

    # Calculate default max_clips if not specified: (total_clips // 1000) * 1000
    if max_clips is None:
        max_clips = (total_saved // 1000) * 1000
        print(f"Auto-calculated max_clips: {max_clips:,} (rounded down to nearest 1000)")

    # Post-processing: Select top clips by RMS (or skip if balanced workflow)
    select_top_clips_by_rms(csv_path, output_root, max_clips=max_clips, no_quarantine=no_quarantine, dry_run=dry_run)

if __name__ == "__main__":
    main()
