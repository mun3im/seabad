#!/usr/bin/env python3
"""
Stage4_deduplicate_flac.py

Detect and quarantine duplicate audio clips using FAISS-based acoustic similarity search.

Input: Stage3out_successful_conversions.csv (from config.py)
FLAC Directory: /Volumes/Evo/MYBAD2/asean-flacs/ (from config.py)

Outputs:
  - Stage4out_unique_flacs.csv (deduplicated metadata)
  - Stage4_removed_near_duplicates_metadata.csv (quarantined entries)
  - Stage4_report.txt (detailed duplicate analysis)
  - perfect_duplicates/ folder (quarantined perfect duplicates)
  - near_duplicates/ folder (quarantined near-duplicates)

Execution Modes:
  Default: Dry-run (shows what would be quarantined)
  --quarantine-perfect: Quarantine perfect duplicates only
  --quarantine-all: Quarantine both perfect + near-duplicates (recommended)

Usage:
  python Stage4_deduplicate_flac.py                  # dry-run
  python Stage4_deduplicate_flac.py --quarantine-all # execute
"""

import argparse
import platform
import sys
import shutil
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
import time
import faiss
from datetime import datetime

import numpy as np
import librosa
from tqdm import tqdm

# Import centralized config
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Please ensure config.py is in the same directory.")
    sys.exit(1)

# ------------------ CONFIG ------------------
TARGET_SR = config.TARGET_SAMPLE_RATE
TARGET_DURATION = 3.0
N_MELS = 128
HOP_LENGTH = 128
MIN_DURATION = 3.0

MEAN_SIM_THRESHOLD = 0.997
MIN_SIM_THRESHOLD = 0.985
P5_SIM_THRESHOLD = 0.992
PERFECT_DUPLICATE_THRESHOLD = 0.999

# Default paths from config
DEFAULT_FLAC_DIR = config.FLAC_OUTPUT_DIR  # Directory containing FLAC files
DEFAULT_METADATA_CSV = config.STAGE4_INPUT_CSV  # Reads Stage3 output
DEFAULT_OUTPUT_METADATA = config.STAGE4_OUTPUT_CSV
DEFAULT_REPORT = config.STAGE4_REPORT_TXT
DEFAULT_REMOVED_METADATA = config.STAGE4_REMOVED_CSV

# --------------------------------------------


class AudioEmbedder:
    """Handles audio file loading and embedding computation."""

    @staticmethod
    def compute(filepath: Path) -> Optional[np.ndarray]:
        """Compute normalized mel-spectrogram embedding for an audio file."""
        try:
            y, _ = librosa.load(
                filepath, sr=TARGET_SR, mono=True, res_type='kaiser_fast'
            )
        except Exception as e:
            print(f"Failed loading {filepath.name}: {e}", file=sys.stderr)
            return None

        if len(y) / TARGET_SR < MIN_DURATION:
            return None

        # Force exactly 3.00 seconds
        target_samples = int(TARGET_DURATION * TARGET_SR)
        y = AudioEmbedder._normalize_length(y, target_samples)

        # Compute mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_mels=N_MELS,
            hop_length=HOP_LENGTH, n_fft=512, fmax=8000
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Per-frame L2 normalization for cosine similarity
        return AudioEmbedder._normalize_frames(log_mel)

    @staticmethod
    def _normalize_length(y: np.ndarray, target_samples: int) -> np.ndarray:
        """Pad or trim audio to exact target length."""
        if len(y) > target_samples:
            return y[:target_samples]
        return np.pad(y, (0, target_samples - len(y)), mode='constant')

    @staticmethod
    def _normalize_frames(log_mel: np.ndarray) -> np.ndarray:
        """L2 normalize each frame for cosine similarity."""
        norms = np.linalg.norm(log_mel, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return (log_mel / norms).astype(np.float32)


class SimilarityCalculator:
    """Calculates frame-wise cosine similarity between embeddings."""

    @staticmethod
    def compute_metrics(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, float, float]:
        """Returns (mean_sim, min_sim, p5_sim) between two embeddings."""
        min_frames = min(emb1.shape[1], emb2.shape[1])
        if min_frames == 0:
            return 0.0, 0.0, 0.0

        sims = np.clip(
            np.sum(emb1[:, :min_frames] * emb2[:, :min_frames], axis=0),
            -1.0, 1.0
        )
        return (
            float(np.mean(sims)),
            float(np.min(sims)),
            float(np.percentile(sims, 5))
        )

    @staticmethod
    def is_similar(mean_sim: float, min_sim: float, p5_sim: float) -> bool:
        """Check if similarity metrics exceed thresholds."""
        return (
                mean_sim >= MEAN_SIM_THRESHOLD and
                min_sim >= MIN_SIM_THRESHOLD and
                p5_sim >= P5_SIM_THRESHOLD
        )


class DuplicateFinder:
    """Finds and categorizes duplicate audio files using FAISS."""

    def __init__(self, filepaths: List[Path], metadata_manager=None, top_k: int = 6):
        self.filepaths = filepaths
        self.embeddings = []       # framewise (128 x T)
        self.clip_embeddings = []  # clip-level (D,)
        self.valid_paths = []
        self.metadata_manager = metadata_manager
        self.top_k = top_k

    def compute_embeddings(self):
        print(f"\nComputing embeddings for {len(self.filepaths)} files...")
        failed_files = []

        for fp in tqdm(self.filepaths, desc="Embedding", unit="file"):
            emb = AudioEmbedder.compute(fp)
            if emb is not None:
                self.embeddings.append(emb)
                self.clip_embeddings.append(self._to_clip_embedding(emb))
                self.valid_paths.append(fp)
            else:
                failed_files.append(fp)

        self.clip_embeddings = np.vstack(self.clip_embeddings).astype("float32")
        faiss.normalize_L2(self.clip_embeddings)

        print(f"Embedded {len(self.valid_paths)}/{len(self.filepaths)} files successfully.")

        if failed_files:
            print(f"Failed to embed {len(failed_files)} file(s) (likely < 3s duration or corrupted):")
            for fp in failed_files[:10]:
                print(f"  - {fp.name}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")

    @staticmethod
    def _to_clip_embedding(emb: np.ndarray) -> np.ndarray:
        """
        Convert framewise mel embedding (128 x T)
        to clip-level descriptor using mean+std pooling.
        """
        mu = np.mean(emb, axis=1)
        std = np.std(emb, axis=1)
        return np.concatenate([mu, std])

    @staticmethod
    def _get_source_recording_id(filepath: Path) -> str:
        """
        Extract source recording identifier from filename.
        e.g., 'xc877854_A_33300.wav' -> 'xc877854_A'
        This identifies clips from the same source recording.
        """
        stem = filepath.stem  # e.g., 'xc877854_A_33300'
        parts = stem.split('_')
        if len(parts) >= 2:
            # Return 'xcNNNNNN_Q' (XC ID + quality)
            return f"{parts[0]}_{parts[1]}"
        return stem

    @staticmethod
    def _extract_xc_number(filepath: Path) -> int:
        """
        Extract XC number from filename (e.g., 'xc877854_A_33300.wav' -> 877854).
        Returns 0 if no XC number found.
        """
        try:
            name = filepath.stem.lower()
            if name.startswith('xc'):
                parts = name[2:].split('_')
                if parts:
                    return int(parts[0])
        except (ValueError, IndexError):
            pass
        return 0

    def _get_source_duration(self, filepath: Path) -> Optional[float]:
        """
        Get source recording duration from metadata.
        Returns None if not available.
        """
        if not self.metadata_manager:
            return None

        xc_id = self._extract_xc_number(filepath)
        if xc_id == 0:
            return None

        meta = self.metadata_manager.get_metadata(xc_id)
        if not meta:
            return None

        try:
            # Parse 'length' field (format: 'M:SS' or 'H:MM:SS')
            length_str = meta.get('length', '').strip()
            if not length_str:
                return None

            parts = length_str.split(':')
            if len(parts) == 2:  # M:SS
                minutes, seconds = parts
                duration = int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:  # H:MM:SS
                hours, minutes, seconds = parts
                duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            else:
                return None

            return float(duration) if duration > 0 else None
        except (ValueError, TypeError, AttributeError):
            return None

    def find_pairs(self) -> Tuple[List[Tuple[Path, Path, float]], List[Tuple[Path, Path]]]:
        if len(self.valid_paths) < 2:
            return [], []

        N, D = self.clip_embeddings.shape
        print(f"\nBuilding FAISS index (N={N}, D={D})...")

        index = faiss.IndexFlatIP(D)  # cosine after L2 norm
        index.add(self.clip_embeddings)

        print("Searching nearest neighbors...")
        S, I = index.search(self.clip_embeddings, self.top_k)

        similar_pairs = []
        perfect_duplicates = []
        visited = set()
        same_source_skipped = 0
        duration_mismatch_skipped = 0

        print("Verifying candidate pairs...")

        for i in tqdm(range(N), desc="Verifying", unit="file"):
            for r in range(1, self.top_k):  # skip self
                j = int(I[i, r])
                sim = float(S[i, r])

                if j <= i:
                    continue
                if (i, j) in visited:
                    continue
                visited.add((i, j))

                # quick rejection
                if sim < MIN_SIM_THRESHOLD:
                    continue

                # Skip pairs from the same source recording
                # (different 3s clips from same original recording)
                source_id_i = self._get_source_recording_id(self.valid_paths[i])
                source_id_j = self._get_source_recording_id(self.valid_paths[j])

                if source_id_i == source_id_j:
                    same_source_skipped += 1
                    continue

                # CRITICAL: Skip pairs with different source recording durations
                # (can't be duplicates if source recordings have different lengths)
                dur_i = self._get_source_duration(self.valid_paths[i])
                dur_j = self._get_source_duration(self.valid_paths[j])

                # If both durations are available, they MUST match (within 1 second tolerance)
                if dur_i is not None and dur_j is not None:
                    if abs(dur_i - dur_j) > 1.0:
                        duration_mismatch_skipped += 1
                        continue
                # If only one duration is available, skip (we can't verify)
                elif (dur_i is None) != (dur_j is None):
                    duration_mismatch_skipped += 1
                    continue

                # ---- Upgrade (B): instant perfect-duplicate check ----
                if np.allclose(self.clip_embeddings[i], self.clip_embeddings[j], atol=1e-7):
                    perfect_duplicates.append(
                        (self.valid_paths[i], self.valid_paths[j])
                    )
                    continue
                # ------------------------------------------------------

                mean_sim, min_sim, p5_sim = SimilarityCalculator.compute_metrics(
                    self.embeddings[i], self.embeddings[j]
                )

                if mean_sim >= PERFECT_DUPLICATE_THRESHOLD:
                    perfect_duplicates.append(
                        (self.valid_paths[i], self.valid_paths[j])
                    )

                elif SimilarityCalculator.is_similar(mean_sim, min_sim, p5_sim):
                    similar_pairs.append(
                        (self.valid_paths[i], self.valid_paths[j], mean_sim)
                    )

        if same_source_skipped > 0:
            print(f"\nSkipped {same_source_skipped} similar pairs from same source recordings (valid samples)")
        if duration_mismatch_skipped > 0:
            print(f"Skipped {duration_mismatch_skipped} similar pairs with different source durations (not duplicates)")

        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs, perfect_duplicates


class MetadataManager:
    """Manages metadata CSV operations."""

    def __init__(self, metadata_csv: Path):
        self.metadata_csv = metadata_csv
        self.metadata_dict: Dict[int, Dict] = {}
        self.removed_metadata: Dict[int, Dict] = {}  # Track removed entries separately
        self.fieldnames = []
        self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_csv.exists():
            print(f"Warning: {self.metadata_csv} not found.")
            return

        with open(self.metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames or []

            print(f"Metadata fields: {', '.join(self.fieldnames)}")
            if 'id' not in self.fieldnames:
                print("ERROR: Expected 'id' column not found in metadata CSV!", file=sys.stderr)
                return

            loaded_count = 0
            for row in reader:
                try:
                    xc_id = int(row['id'])  # now using 'id'
                    self.metadata_dict[xc_id] = row
                    loaded_count += 1
                except (ValueError, KeyError, TypeError) as e:
                    print(f"Skipping invalid row: {e}", file=sys.stderr)
                    continue

            print(f"Loaded {loaded_count} metadata entries.")

    def get_metadata(self, xc_id: int) -> Optional[Dict]:
        """Get metadata for a given XC ID."""
        return self.metadata_dict.get(xc_id)

    def remove_metadata(self, xc_ids: Set[int], track_removed: bool = False):
        """
        Remove metadata rows for given XC IDs.
        If track_removed=True, save to removed_metadata dict for later export.
        """
        for xc_id in xc_ids:
            removed = self.metadata_dict.pop(xc_id, None)
            if track_removed and removed:
                self.removed_metadata[xc_id] = removed

    def save_metadata(self, output_csv: Path):
        """Save remaining metadata to CSV."""
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata_dict.values())

    def save_removed_metadata(self, output_csv: Path):
        """Save removed metadata to separate CSV for potential restoration."""
        if not self.removed_metadata:
            return

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.removed_metadata.values())

        print(f"✓ Saved {len(self.removed_metadata)} removed near-duplicate metadata entries to {output_csv}")
        print(f"  (These can be restored if manual review justifies reinsertion)")


class DuplicateAnalyzer:
    """Analyzes differences between duplicate audio files."""

    @staticmethod
    def _extract_xc_number(filepath: Path) -> int:
        """Extract XC number from filename."""
        try:
            name = filepath.stem.lower()
            if name.startswith('xc'):
                parts = name[2:].split('_')
                if parts:
                    return int(parts[0])
        except (ValueError, IndexError):
            pass
        return 0

    @staticmethod
    def _parse_duration(length_str: str) -> Optional[float]:
        """Parse duration string (M:SS or H:MM:SS) to seconds."""
        try:
            parts = length_str.strip().split(':')
            if len(parts) == 2:  # M:SS
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:  # H:MM:SS
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    @staticmethod
    def analyze_audio_differences(file1: Path, file2: Path, metadata_manager=None) -> Dict[str, any]:
        """Analyze technical differences between two audio files using metadata."""
        differences = {}

        try:
            # Get metadata for both files
            xc1 = DuplicateAnalyzer._extract_xc_number(file1)
            xc2 = DuplicateAnalyzer._extract_xc_number(file2)

            meta1 = metadata_manager.get_metadata(xc1) if metadata_manager and xc1 else None
            meta2 = metadata_manager.get_metadata(xc2) if metadata_manager and xc2 else None

            # Get ORIGINAL sample rates from metadata (not from normalized FLACs)
            if meta1:
                try:
                    differences['file1_sr'] = int(meta1.get('smp', 0))
                except (ValueError, TypeError):
                    differences['file1_sr'] = 0
            else:
                differences['file1_sr'] = 0

            if meta2:
                try:
                    differences['file2_sr'] = int(meta2.get('smp', 0))
                except (ValueError, TypeError):
                    differences['file2_sr'] = 0
            else:
                differences['file2_sr'] = 0

            differences['sample_rate_match'] = (differences['file1_sr'] == differences['file2_sr'])

            # Get source durations from metadata
            if meta1:
                differences['file1_duration'] = DuplicateAnalyzer._parse_duration(meta1.get('length', '')) or 0
            else:
                differences['file1_duration'] = 0

            if meta2:
                differences['file2_duration'] = DuplicateAnalyzer._parse_duration(meta2.get('length', '')) or 0
            else:
                differences['file2_duration'] = 0

            # Note: We no longer load audio files for analysis
            # All info comes from metadata (original sample rate, duration, etc.)

        except Exception as e:
            differences['error'] = str(e)

        return differences


class QuarantineManager:
    """Manages duplicate folder and file movements with metadata tracking."""

    def __init__(self, root: Path, metadata_manager: Optional[MetadataManager] = None):
        self.root = root
        self.perfect_duplicates_dir = root / "perfect_duplicates"
        self.near_duplicates_dir = root / "near_duplicates"
        self.metadata_manager = metadata_manager
        self.report_entries = {
            'perfect': [],
            'near': []
        }

    @staticmethod
    def extract_xc_number(filepath: Path) -> int:
        """
        Extract XC number from filename (e.g., 'xc123456_A.flac' -> 123456).
        Returns 0 if no XC number found.
        """
        try:
            name = filepath.stem.lower()
            if name.startswith('xc'):
                # Extract digits after 'xc' and before '_'
                parts = name[2:].split('_')
                if parts:
                    return int(parts[0])
        except (ValueError, IndexError):
            pass
        return 0

    def handle_perfect_duplicates(
        self,
        perfect_duplicates: List[Tuple[Path, Path]],
        dry_run: bool = False
    ) -> Tuple[int, Set[int]]:
        """
        Handle perfect duplicate pairs based on recorder field.
        Returns (number of files moved, set of removed XC numbers).
        """
        if not perfect_duplicates:
            return 0, set()

        if not dry_run:
            self.perfect_duplicates_dir.mkdir(exist_ok=True)

        moved_count = 0
        removed_xc_numbers = set()
        moved_files = set()

        print(f"\n{'[DRY RUN] Would handle' if dry_run else 'Handling'} {len(perfect_duplicates)} perfect duplicate pair(s)...")

        for file1, file2 in perfect_duplicates:
            xc1 = self.extract_xc_number(file1)
            xc2 = self.extract_xc_number(file2)

            # Skip if either file was already moved
            if file1 in moved_files or file2 in moved_files:
                continue

            # Get metadata for both files
            meta1 = self.metadata_manager.get_metadata(xc1) if self.metadata_manager else None
            meta2 = self.metadata_manager.get_metadata(xc2) if self.metadata_manager else None

            # Debug: warn if metadata not found
            if self.metadata_manager and not meta1:
                print(f"  Warning: No metadata found for XC{xc1}", file=sys.stderr)
            if self.metadata_manager and not meta2:
                print(f"  Warning: No metadata found for XC{xc2}", file=sys.stderr)

            # Get recorder from 'rec' field with detailed debugging
            if meta1:
                rec1 = meta1.get('rec', '')
                if not rec1:
                    print(f"  DEBUG XC{xc1}: 'rec' field is empty/missing. Available fields: {list(meta1.keys())[:10]}", file=sys.stderr)
                    print(f"  DEBUG XC{xc1}: Sample values: {dict(list(meta1.items())[:5])}", file=sys.stderr)
            else:
                rec1 = ''

            if meta2:
                rec2 = meta2.get('rec', '')
                if not rec2:
                    print(f"  DEBUG XC{xc2}: 'rec' field is empty/missing. Available fields: {list(meta2.keys())[:10]}", file=sys.stderr)
                    print(f"  DEBUG XC{xc2}: Sample values: {dict(list(meta2.items())[:5])}", file=sys.stderr)
            else:
                rec2 = ''

            # Strip whitespace
            rec1 = rec1.strip() if rec1 else ''
            rec2 = rec2.strip() if rec2 else ''

            # Determine which file to remove and the reason
            if rec1 == rec2:
                # Same recorder: keep newer (higher XC number), remove older
                if xc1 > xc2:
                    target_file, keep_file = file2, file1
                    removed_xc, kept_xc = xc2, xc1
                    reason = f"Same recorder ({rec1}). Keeping newer recording XC{kept_xc}, removing older XC{removed_xc}. Recorder may have updated metadata."
                else:
                    target_file, keep_file = file1, file2
                    removed_xc, kept_xc = xc1, xc2
                    reason = f"Same recorder ({rec1}). Keeping newer recording XC{kept_xc}, removing older XC{removed_xc}. Recorder may have updated metadata."
            else:
                # Different recorders: keep older, tag newer as plagiarized
                if xc1 < xc2:
                    target_file, keep_file = file2, file1
                    removed_xc, kept_xc = xc2, xc1
                    reason = f"PLAGIARIZED UPLOAD. Different recorders: '{rec1}' (XC{xc1}, original) vs '{rec2}' (XC{xc2}, plagiarized). Keeping original, removing plagiarized newer upload."
                else:
                    target_file, keep_file = file1, file2
                    removed_xc, kept_xc = xc1, xc2
                    reason = f"PLAGIARIZED UPLOAD. Different recorders: '{rec2}' (XC{xc2}, original) vs '{rec1}' (XC{xc1}, plagiarized). Keeping original, removing plagiarized newer upload."

            # Record for report
            self.report_entries['perfect'].append({
                'file1': file1,
                'file2': file2,
                'xc1': xc1,
                'xc2': xc2,
                'rec1': rec1,
                'rec2': rec2,
                'removed_file': target_file,
                'kept_file': keep_file,
                'removed_xc': removed_xc,
                'kept_xc': kept_xc,
                'reason': reason
            })

            moved_files.add(target_file)
            removed_xc_numbers.add(removed_xc)

            # Move file
            rel_path = target_file.relative_to(self.root)
            dest = self.perfect_duplicates_dir / rel_path

            if dry_run:
                print(f"  [DRY RUN] Would move: {rel_path} (XC{removed_xc})")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(target_file), str(dest))
                    moved_count += 1
                    print(f"  Moved: {rel_path} (XC{removed_xc})")
                except Exception as e:
                    print(f"  Failed to move {rel_path}: {e}", file=sys.stderr)

        return moved_count, removed_xc_numbers

    def handle_near_duplicates(
        self,
        near_duplicates: List[Tuple[Path, Path, float]],
        dry_run: bool = False
    ) -> Tuple[int, Set[int]]:
        """
        Handle near-duplicate pairs using same logic as perfect duplicates.
        Moves only ONE file per pair, keeps the other based on recorder field.
        Returns (number of files moved, set of removed XC numbers).
        """
        if not near_duplicates:
            return 0, set()

        if not dry_run:
            self.near_duplicates_dir.mkdir(exist_ok=True)

        moved_count = 0
        removed_xc_numbers = set()
        moved_files = set()

        print(f"\n{'[DRY RUN] Would handle' if dry_run else 'Handling'} {len(near_duplicates)} near-duplicate pair(s)...")

        for file1, file2, similarity in near_duplicates:
            xc1 = self.extract_xc_number(file1)
            xc2 = self.extract_xc_number(file2)

            # Skip if either file was already moved
            if file1 in moved_files or file2 in moved_files:
                continue

            # Get metadata
            meta1 = self.metadata_manager.get_metadata(xc1) if self.metadata_manager else None
            meta2 = self.metadata_manager.get_metadata(xc2) if self.metadata_manager else None

            # Debug: warn if metadata not found
            if self.metadata_manager and not meta1:
                print(f"  Warning: No metadata found for XC{xc1}", file=sys.stderr)
            if self.metadata_manager and not meta2:
                print(f"  Warning: No metadata found for XC{xc2}", file=sys.stderr)

            # Get recorder from 'rec' field with detailed debugging
            if meta1:
                rec1 = meta1.get('rec', '')
                if not rec1:
                    print(f"  DEBUG XC{xc1}: 'rec' field is empty/missing. Available fields: {list(meta1.keys())[:10]}", file=sys.stderr)
                    print(f"  DEBUG XC{xc1}: Sample values: {dict(list(meta1.items())[:5])}", file=sys.stderr)
            else:
                rec1 = ''

            if meta2:
                rec2 = meta2.get('rec', '')
                if not rec2:
                    print(f"  DEBUG XC{xc2}: 'rec' field is empty/missing. Available fields: {list(meta2.keys())[:10]}", file=sys.stderr)
                    print(f"  DEBUG XC{xc2}: Sample values: {dict(list(meta2.items())[:5])}", file=sys.stderr)
            else:
                rec2 = ''

            # Strip whitespace
            rec1 = rec1.strip() if rec1 else ''
            rec2 = rec2.strip() if rec2 else ''

            # Analyze differences
            differences = DuplicateAnalyzer.analyze_audio_differences(file1, file2, self.metadata_manager)

            # Apply same logic as perfect duplicates
            if rec1 == rec2:
                # Same recorder: keep newer (higher XC number), remove older
                if xc1 > xc2:
                    target_file, keep_file = file2, file1
                    removed_xc, kept_xc = xc2, xc1
                    reason = f"Near-duplicate (similarity: {similarity:.4f}). Same recorder ({rec1}). Keeping newer XC{kept_xc}, removing older XC{removed_xc} for manual review."
                else:
                    target_file, keep_file = file1, file2
                    removed_xc, kept_xc = xc1, xc2
                    reason = f"Near-duplicate (similarity: {similarity:.4f}). Same recorder ({rec1}). Keeping newer XC{kept_xc}, removing older XC{removed_xc} for manual review."
            else:
                # Different recorders: keep older, move newer
                if xc1 < xc2:
                    target_file, keep_file = file2, file1
                    removed_xc, kept_xc = xc2, xc1
                    reason = f"Near-duplicate (similarity: {similarity:.4f}). Different recorders: '{rec1}' (XC{xc1}, kept) vs '{rec2}' (XC{xc2}, removed). Keeping older recording for manual review."
                else:
                    target_file, keep_file = file1, file2
                    removed_xc, kept_xc = xc1, xc2
                    reason = f"Near-duplicate (similarity: {similarity:.4f}). Different recorders: '{rec2}' (XC{xc2}, kept) vs '{rec1}' (XC{xc1}, removed). Keeping older recording for manual review."

            # Add technical details
            tech_details = []
            if 'sample_rate_match' in differences:
                if not differences['sample_rate_match']:
                    tech_details.append(f"Sample rates differ: {differences['file1_sr']} vs {differences['file2_sr']}")
            if 'rms_diff' in differences:
                tech_details.append(f"RMS difference: {differences['rms_diff']:.6f}")
            if 'waveforms_identical' in differences:
                if differences['waveforms_identical']:
                    tech_details.append("Waveforms are identical (different encoding/format)")
                else:
                    tech_details.append("Waveforms differ")
            if 'error' in differences:
                tech_details.append(f"Analysis error: {differences['error']}")

            if tech_details:
                reason += " Technical: " + "; ".join(tech_details)

            # Record for report
            self.report_entries['near'].append({
                'file1': file1,
                'file2': file2,
                'xc1': xc1,
                'xc2': xc2,
                'rec1': rec1,
                'rec2': rec2,
                'removed_file': target_file,
                'kept_file': keep_file,
                'removed_xc': removed_xc,
                'kept_xc': kept_xc,
                'similarity': similarity,
                'differences': differences,
                'reason': reason
            })

            moved_files.add(target_file)
            removed_xc_numbers.add(removed_xc)

            # Move file
            rel_path = target_file.relative_to(self.root)
            dest = self.near_duplicates_dir / rel_path

            if dry_run:
                print(f"  [DRY RUN] Would move: {rel_path} (XC{removed_xc})")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(target_file), str(dest))
                    moved_count += 1
                    print(f"  Moved: {rel_path} (XC{removed_xc})")
                except Exception as e:
                    print(f"  Failed to move {rel_path}: {e}", file=sys.stderr)

        return moved_count, removed_xc_numbers

    def generate_report(self, output_path: Path):
        """Generate detailed report of all duplicates."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 4 DUPLICATE DETECTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Perfect duplicates section
            f.write("PERFECT DUPLICATES\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total pairs: {len(self.report_entries['perfect'])}\n\n")

            for i, entry in enumerate(self.report_entries['perfect'], 1):
                f.write(f"Pair {i}:\n")
                f.write(f"  File 1: {entry['file1'].name} (XC{entry['xc1']})\n")
                f.write(f"  File 2: {entry['file2'].name} (XC{entry['xc2']})\n")
                f.write(f"  Recorder 1: {entry['rec1']}\n")
                f.write(f"  Recorder 2: {entry['rec2']}\n")
                f.write(f"  KEPT: {entry['kept_file'].name} (XC{entry['kept_xc']})\n")
                f.write(f"  REMOVED: {entry['removed_file'].name} (XC{entry['removed_xc']})\n")
                f.write(f"  Reason: {entry['reason']}\n")
                f.write("\n")

            f.write("\n" + "=" * 80 + "\n\n")

            # Near duplicates section
            f.write("NEAR DUPLICATES (For Manual Review)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total pairs: {len(self.report_entries['near'])}\n")
            f.write("NOTE: One file per pair removed for review. Metadata saved separately.\n")
            f.write("      Removed files can be restored if manual review justifies reinsertion.\n\n")

            for i, entry in enumerate(self.report_entries['near'], 1):
                f.write(f"Pair {i}:\n")
                f.write(f"  File 1: {entry['file1'].name} (XC{entry['xc1']})\n")
                f.write(f"  File 2: {entry['file2'].name} (XC{entry['xc2']})\n")
                f.write(f"  Recorder 1: {entry['rec1']}\n")
                f.write(f"  Recorder 2: {entry['rec2']}\n")
                f.write(f"  Similarity: {entry['similarity']:.6f}\n")
                f.write(f"  KEPT: {entry['kept_file'].name} (XC{entry['kept_xc']})\n")
                f.write(f"  REMOVED: {entry['removed_file'].name} (XC{entry['removed_xc']})\n")
                f.write(f"  Reason: {entry['reason']}\n")

                # Add detailed technical analysis
                diff = entry['differences']
                if diff:
                    f.write(f"  Technical Analysis:\n")
                    if 'file1_sr' in diff:
                        f.write(f"    Sample rates: {diff['file1_sr']} Hz vs {diff['file2_sr']} Hz\n")
                    if 'file1_duration' in diff:
                        f.write(f"    Durations: {diff['file1_duration']:.3f}s vs {diff['file2_duration']:.3f}s\n")
                    if 'file1_channels' in diff:
                        f.write(f"    Channels: {diff['file1_channels']} vs {diff['file2_channels']}\n")
                    if 'rms_diff' in diff:
                        f.write(f"    RMS difference: {diff['rms_diff']:.6f}\n")
                    if 'max_amplitude_diff' in diff:
                        f.write(f"    Max amplitude difference: {diff['max_amplitude_diff']:.6f}\n")
                    if 'waveforms_identical' in diff:
                        f.write(f"    Waveforms identical: {diff['waveforms_identical']}\n")

                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")


class FileCollector:
    """Collects audio files from directory structure."""

    @staticmethod
    def collect(root: Path, recursive: bool, metadata_manager=None) -> List[Path]:
        """
        Collect all .wav and .flac files.
        Default: root + immediate subfolders only
        --recursive: full recursion

        If metadata_manager is provided, only include files with metadata entries.
        """
        pattern = "**/*" if recursive else "*"
        candidates = FileCollector._find_files(root, pattern)

        # Check one level deeper if nothing found and not recursive
        if not candidates and not recursive:
            candidates = FileCollector._check_subdirs(root)

        # Filter by metadata if available
        if metadata_manager:
            candidates = FileCollector._filter_by_metadata(candidates, metadata_manager)

        return sorted(candidates)

    @staticmethod
    def _find_files(root: Path, pattern: str) -> List[Path]:
        """Find audio files matching pattern."""
        return [
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
        ]

    @staticmethod
    def _check_subdirs(root: Path) -> List[Path]:
        """Check immediate subdirectories for audio files."""
        candidates = []
        for subdir in root.iterdir():
            if subdir.is_dir():
                candidates.extend(FileCollector._find_files(subdir, "*"))
        return candidates

    @staticmethod
    def _filter_by_metadata(filepaths: List[Path], metadata_manager) -> List[Path]:
        """Filter files to only include those with matching metadata entries (ID + quality)."""
        import re
        filtered = []
        skipped_no_metadata = 0
        skipped_wrong_quality = 0

        # Pattern to extract xc{id}_{quality}.flac
        filename_pattern = re.compile(r'xc(\d+)_([A-Ea-eUun])\.flac')

        for filepath in filepaths:
            # Extract XC ID and quality from filename
            match = filename_pattern.match(filepath.name)
            if not match:
                skipped_no_metadata += 1
                continue

            xc_num = int(match.group(1))
            file_quality = match.group(2).upper()

            # Check if ID exists in metadata
            if xc_num not in metadata_manager.metadata_dict:
                skipped_no_metadata += 1
                continue

            # Check if quality matches metadata
            metadata_quality = metadata_manager.metadata_dict[xc_num].get('q', '').upper()
            # Normalize 'n' to 'U' for comparison (handles old files)
            if metadata_quality == 'N' or metadata_quality == 'NO SCORE':
                metadata_quality = 'U'
            if file_quality == 'N':
                file_quality = 'U'

            if file_quality == metadata_quality:
                filtered.append(filepath)
            else:
                skipped_wrong_quality += 1

        total_skipped = skipped_no_metadata + skipped_wrong_quality
        if total_skipped > 0:
            print(f"Skipped {total_skipped} file(s): {skipped_no_metadata} without metadata, "
                  f"{skipped_wrong_quality} with mismatched quality (from previous runs)")

        return filtered






def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Find and handle duplicate audio files with metadata tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXECUTION MODES:
  Default (no flags)        : DRY RUN - shows what would be quarantined
  --quarantine-perfect      : Quarantine perfect duplicates only, update CSV
  --quarantine-all          : Quarantine both perfect + near-duplicates, update CSV (RECOMMENDED)

EXAMPLES:
  # Dry run (default) - see what would be quarantined
  python Stage4_deduplicate_flac.py

  # Quarantine perfect duplicates only
  python Stage4_deduplicate_flac.py --quarantine-perfect

  # Quarantine all duplicates (perfect + near) - RECOMMENDED
  python Stage4_deduplicate_flac.py --quarantine-all

  # Debug mode: check metadata fields
  python Stage4_deduplicate_flac.py --debug-mode

  # Use custom FLAC directory with quarantine-all
  python Stage4_deduplicate_flac.py /custom/path/to/flacs --quarantine-all

  # With custom metadata paths
  python Stage4_deduplicate_flac.py --quarantine-all \\
    --metadata custom_metadata.csv \\
    --output-metadata clean_metadata.csv
        """
    )

    # Optional arguments (directory defaults to config)
    parser.add_argument(
        "directory",
        type=Path,
        nargs='?',
        default=DEFAULT_FLAC_DIR,
        metavar="DIR",
        help=f"Root folder containing the FLAC files (default: {DEFAULT_FLAC_DIR})"
    )

    # Processing options
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search ALL subdirectories (not just one level deep)"
    )

    # Execution modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quarantine-perfect",
        action="store_true",
        help="Quarantine perfect duplicates only and update CSV (excludes near-duplicates)"
    )
    mode_group.add_argument(
        "--quarantine-all",
        action="store_true",
        help="Quarantine both perfect and near-duplicates and update CSV (recommended)"
    )
    # Default is dry-run if neither flag is specified

    # Metadata options
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        metavar="FILE",
        help=f"Path to metadata CSV file (default: {DEFAULT_METADATA_CSV})"
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=DEFAULT_OUTPUT_METADATA,
        metavar="FILE",
        help=f"Output path for updated metadata CSV (default: {DEFAULT_OUTPUT_METADATA})"
    )
    parser.add_argument(
        "--removed-metadata",
        type=Path,
        metavar="FILE",
        help=f"Output path for removed near-duplicate metadata CSV (default: {DEFAULT_REMOVED_METADATA})"
    )

    # Report options
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        metavar="FILE",
        help=f"Output path for detailed duplicate report (default: {DEFAULT_REPORT})"
    )

    # Debug options
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Debug mode: load metadata, print first 5 records with field details, and exit"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    script_start = time.perf_counter()
    args = parse_arguments()

    # Ensure metadata directory exists
    Path(config.METADATA_DIR).mkdir(parents=True, exist_ok=True)

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Initialize metadata manager if metadata CSV provided
    metadata_manager = None
    if args.metadata:
        metadata_path = args.metadata.resolve()
        if not metadata_path.exists():
            print(f"Warning: Metadata file {metadata_path} not found. Continuing without metadata.", file=sys.stderr)
        else:
            print(f"Loading metadata from {metadata_path}...")
            metadata_manager = MetadataManager(metadata_path)
            print(f"Loaded {len(metadata_manager.metadata_dict)} metadata entries.\n")

    # Debug mode: print first 5 metadata records and exit
    if args.debug_mode:
        if not metadata_manager:
            print("Error: --debug-mode requires --metadata to be specified", file=sys.stderr)
            sys.exit(1)

        print("=" * 80)
        print("DEBUG MODE: Displaying first 5 metadata records")
        print("=" * 80)

        for i, (xc_id, record) in enumerate(list(metadata_manager.metadata_dict.items())[:5], 1):
            print(f"\nRecord {i}: XC{xc_id}")
            print(f"  Available fields ({len(record)} total): {list(record.keys())}")
            print(f"  Sample values:")
            for key, value in list(record.items())[:10]:
                print(f"    {key}: {repr(value)}")

            # Check specifically for 'rec' and 'recorder' fields
            rec_value = record.get('rec')
            recorder_value = record.get('recorder')
            print(f"  Field 'rec': {repr(rec_value)}")
            print(f"  Field 'recorder': {repr(recorder_value)}")

        print("\n" + "=" * 80)
        print("Debug mode complete. Exiting.")
        print("=" * 80)
        return

    # Collect files
    filepaths = FileCollector.collect(root, args.recursive, metadata_manager)
    if not filepaths:
        print("No WAV or FLAC files found (checked root + immediate subfolders).")
        if metadata_manager:
            print("Note: Files without metadata entries are automatically skipped.")
        print("Use --recursive if your files are nested deeper.")
        return

    print(f"Found {len(filepaths)} audio files with metadata entries.\n")

    # Find duplicates
    finder = DuplicateFinder(filepaths, metadata_manager=metadata_manager)
    compute_embeddings_start = time.perf_counter()
    finder.compute_embeddings()
    find_pairs_start = time.perf_counter()
    near_duplicates, perfect_duplicates = finder.find_pairs()
    find_pairs_end = time.perf_counter()

    print(f"\nFound {len(perfect_duplicates)} perfect duplicate pair(s)")
    print(f"Found {len(near_duplicates)} near-duplicate pair(s)\n")

    # Determine execution mode
    if args.quarantine_all:
        dry_run = False
        handle_perfect = True
        handle_near = True
        mode_desc = "QUARANTINE ALL (perfect + near-duplicates)"
    elif args.quarantine_perfect:
        dry_run = False
        handle_perfect = True
        handle_near = False
        mode_desc = "QUARANTINE PERFECT DUPLICATES ONLY"
    else:
        # Default: dry-run mode
        dry_run = True
        handle_perfect = True
        handle_near = True
        mode_desc = "DRY RUN (no changes made)"

    print(f"Mode: {mode_desc}\n")

    # Handle duplicates
    manager = QuarantineManager(root, metadata_manager)

    perfect_removed_xc = set()
    near_removed_xc = set()

    # Handle perfect duplicates
    if perfect_duplicates and handle_perfect:
        perfect_moved, perfect_removed_xc = manager.handle_perfect_duplicates(
            perfect_duplicates, dry_run=dry_run
        )
        print(f"✓ {'[DRY RUN] Would move' if dry_run else 'Moved'} {perfect_moved} file(s) to perfect_duplicates/")

    # Handle near duplicates
    if near_duplicates and handle_near:
        near_moved, near_removed_xc = manager.handle_near_duplicates(
            near_duplicates, dry_run=dry_run
        )
        print(f"✓ {'[DRY RUN] Would move' if dry_run else 'Moved'} {near_moved} file(s) to near_duplicates/")
    elif near_duplicates and not handle_near:
        print(f"  Skipping {len(near_duplicates)} near-duplicate pair(s) (--quarantine-perfect mode)")

    # Update metadata
    if metadata_manager and not dry_run:
        # Remove perfect duplicates permanently (no tracking)
        if perfect_removed_xc:
            print(f"\nRemoving {len(perfect_removed_xc)} perfect duplicate metadata entries (permanent)...")
            metadata_manager.remove_metadata(perfect_removed_xc, track_removed=False)

        # Remove near duplicates with tracking (for potential restoration)
        if near_removed_xc:
            print(f"Removing {len(near_removed_xc)} near-duplicate metadata entries (tracked for restoration)...")
            metadata_manager.remove_metadata(near_removed_xc, track_removed=True)

        # Save main updated metadata
        if perfect_removed_xc or near_removed_xc:
            output_metadata = args.output_metadata

            metadata_manager.save_metadata(output_metadata)
            print(f"✓ Updated metadata saved to {output_metadata}")
            print(f"  Remaining entries: {len(metadata_manager.metadata_dict)}")

        # Save removed near-duplicate metadata separately
        if near_removed_xc:
            if args.removed_metadata:
                removed_metadata_path = args.removed_metadata
            else:
                removed_metadata_path = Path(DEFAULT_REMOVED_METADATA)

            metadata_manager.save_removed_metadata(removed_metadata_path)

    # Generate detailed report
    if not dry_run:
        manager.generate_report(args.report)
        print(f"\n✓ Detailed report generated: {args.report.resolve()}")
    else:
        print(f"\n[DRY RUN] No report generated. Use --quarantine-all or --quarantine-perfect to execute.")

    # Print timing summary
    script_end = time.perf_counter()
    print(f"\nPlatform: {platform.platform()}")
    print(f"Time to compute embeddings: {find_pairs_start - compute_embeddings_start:.3f}s")
    print(f"Time to find pairs: {find_pairs_end - find_pairs_start:.3f}s")
    print(f"Total script time: {script_end - script_start:.3f}s\n")

if __name__ == "__main__":
    main()