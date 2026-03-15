#!/usr/bin/env python3
"""
Stage9_qa_apply_corrections.py

Apply QA corrections from qa_corrections.csv to the balanced positive dataset.

Input:
  qa_corrections.csv                   – manual QA log from Stage8 with columns:
      clip_filename                     : e.g. xc422286_A_3000.wav
      issue_type                        : wrong_onset | noise_dominated | no_bird
      corrected_onset_ms                : new start in ms (required for wrong_onset;
                                          optional for noise_dominated)
  metadata/Stage6out_balanced_clips.csv – metadata for all 25,000 balanced clips

Actions:
  wrong_onset      – re-extract 3s clip at corrected_onset_ms from source FLAC,
                     overwrite WAV in POSITIVE_FINAL_DIR
  noise_dominated  – same as wrong_onset if corrected_onset_ms provided,
                     otherwise treated as no_bird
  no_bird          – remove clip; attempt replacement from unused regions of
                     same source FLAC (highest RMS, >= 1.5s from existing clips)

Output:
  metadata/Stage9out_corrections_applied.csv  – per-clip action log
  metadata/Stage9_report.txt                  – summary report

Usage:
  python Stage9_qa_apply_corrections.py                    # Apply corrections
  python Stage9_qa_apply_corrections.py --dry-run          # Preview changes
  python Stage9_qa_apply_corrections.py --qa-csv custom.csv # Custom QA CSV
"""

import os
import re
import csv
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import pandas as pd

import config

# ── Config from config.py ─────────────────────────────────────────────────────
FLAC_OUTPUT_DIR = config.FLAC_OUTPUT_DIR
POSITIVE_FINAL_DIR = config.POSITIVE_FINAL_DIR
STAGE6_OUTPUT_CSV = config.STAGE6_OUTPUT_CSV
QA_CSV       = config.STAGE9_QA_CSV
OUTPUT_CSV   = config.STAGE9_OUTPUT_CSV
REPORT_TXT   = config.STAGE9_REPORT_TXT

WINDOW_SEC   = 3.0
MIN_SEP_SEC  = 1.5
RMS_THRESH   = 0.001
CLIP_CEIL    = 0.99
SR           = config.TARGET_SAMPLE_RATE
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Filename helpers ──────────────────────────────────────────────────────────
_FNAME_RE = re.compile(r"^xc(\d+)_([A-Ea-e])_(\d+)\.wav$")


def parse_clip_filename(fname: str) -> Tuple[str, str, int]:
    """Return (xc_id, quality, onset_ms) from 'xc422286_A_3000.wav'."""
    m = _FNAME_RE.match(Path(fname).name)
    if not m:
        raise ValueError(f"Unrecognised clip filename format: {fname}")
    return m.group(1), m.group(2).upper(), int(m.group(3))


# ── FLAC lookup ───────────────────────────────────────────────────────────────
_flac_cache: Dict[Tuple[str, str], Path] = {}


def find_flac(xc_id: str, quality: str) -> Path:
    """Locate xc{id}_{quality}.flac anywhere under FLAC_OUTPUT_DIR."""
    key = (xc_id, quality)
    if key in _flac_cache:
        return _flac_cache[key]
    pattern = f"xc{xc_id}_{quality}.flac"
    matches = list(Path(FLAC_OUTPUT_DIR).rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"FLAC not found: {pattern} under {FLAC_OUTPUT_DIR}"
        )
    _flac_cache[key] = matches[0]
    return matches[0]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def extract_clip(flac_path: Path, onset_ms: int) -> np.ndarray:
    """Load exactly WINDOW_SEC of audio from flac_path starting at onset_ms."""
    onset_s = onset_ms / 1000.0
    y, _ = librosa.load(
        str(flac_path), sr=SR, mono=True, offset=onset_s, duration=WINDOW_SEC
    )
    n_target = int(WINDOW_SEC * SR)
    if len(y) < n_target:
        y = np.pad(y, (0, n_target - len(y)))
    else:
        y = y[:n_target]
    return y


def apply_clipping_correction(y: np.ndarray) -> np.ndarray:
    """Peak-scale + soft-limit clips with peak >= 0.9999."""
    peak = float(np.max(np.abs(y)))
    if peak >= 0.9999:
        y = y / peak * CLIP_CEIL
        alpha = 5.0
        y = y / (1.0 + alpha * np.abs(y)) * (1.0 + alpha)
    return y


def save_wav(y: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, SR, subtype="PCM_16")


def find_replacement_onset(
    flac_path: Path, used_onsets_ms: List[int]
) -> Optional[int]:
    """
    Find highest-RMS 3s window in flac_path that is >= MIN_SEP_SEC away from
    every used onset. Returns onset_ms (int) or None if none found.
    """
    try:
        y_full, _ = librosa.load(str(flac_path), sr=SR, mono=True)
    except Exception as exc:
        log.warning(f"  Could not load {flac_path.name}: {exc}")
        return None

    n_win    = int(WINDOW_SEC * SR)
    hop      = int(0.1 * SR)          # 100 ms step
    min_sep  = int(MIN_SEP_SEC * SR)

    if len(y_full) < n_win:
        return None

    # All candidate windows above RMS threshold
    candidates: List[Tuple[float, int]] = []
    for start in range(0, len(y_full) - n_win + 1, hop):
        rms = float(np.sqrt(np.mean(y_full[start : start + n_win] ** 2)))
        if rms >= RMS_THRESH:
            candidates.append((rms, start))

    if not candidates:
        # Fallback: use best window regardless of threshold
        candidates = [
            (float(np.sqrt(np.mean(y_full[s : s + n_win] ** 2))), s)
            for s in range(0, len(y_full) - n_win + 1, hop)
        ]

    candidates.sort(reverse=True)
    used_samples = [int(ms / 1000.0 * SR) for ms in used_onsets_ms]

    for _, start in candidates:
        if all(abs(start - u) >= min_sep for u in used_samples):
            return int(start / SR * 1000)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply QA corrections to SEABAD positive dataset"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview actions without modifying any files",
    )
    parser.add_argument(
        "--qa-csv", default=QA_CSV,
        help=f"Path to QA corrections CSV (default: {QA_CSV})",
    )
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        log.info("DRY RUN — no files will be modified")

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.qa_csv):
        sys.exit(f"[ERROR] QA CSV not found: {args.qa_csv}")
    if not os.path.exists(STAGE6_OUTPUT_CSV):
        sys.exit(f"[ERROR] Stage 6 metadata not found: {STAGE6_OUTPUT_CSV}")

    # ── Load Stage 6 metadata → onset map ────────────────────────────────────
    meta = pd.read_csv(STAGE6_OUTPUT_CSV)
    onset_map: Dict[str, List[int]] = {}
    for clip_fname in meta["clip_filename"]:
        try:
            xc_id, _, onset_ms = parse_clip_filename(clip_fname)
            onset_map.setdefault(xc_id, []).append(onset_ms)
        except ValueError:
            pass  # skip malformed rows

    # ── Load QA CSV ───────────────────────────────────────────────────────────
    with open(args.qa_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = {"clip_filename", "issue_type"}
    if not required.issubset(fieldnames):
        sys.exit(
            f"[ERROR] {args.qa_csv} must have columns: {required}. "
            f"Found: {set(fieldnames)}"
        )

    log.info(f"Loaded {len(rows)} QA entries from {args.qa_csv}")

    # ── Process each correction ───────────────────────────────────────────────
    counts = dict(wrong_onset=0, no_bird=0, replaced=0, not_replaced=0, errors=0)
    action_log: List[dict] = []

    for row in rows:
        clip_fname   = row["clip_filename"].strip()
        issue_raw    = row["issue_type"].strip().lower()
        corr_ms_str  = row.get("corrected_onset_ms", "").strip()

        clip_path = Path(POSITIVE_FINAL_DIR) / clip_fname
        entry = {
            "clip_filename":    clip_fname,
            "issue_type":       issue_raw,
            "result":           "",
            "new_filename":     "",
            "notes":            "",
        }

        # Parse filename
        try:
            xc_id, quality, orig_onset_ms = parse_clip_filename(clip_fname)
        except ValueError as exc:
            log.error(str(exc))
            entry["result"] = "error"
            entry["notes"]  = str(exc)
            counts["errors"] += 1
            action_log.append(entry)
            continue

        # Resolve noise_dominated: treat as wrong_onset if offset given, else no_bird
        if issue_raw == "noise_dominated":
            issue = "wrong_onset" if corr_ms_str else "no_bird"
        else:
            issue = issue_raw

        # ── wrong_onset ───────────────────────────────────────────────────────
        if issue == "wrong_onset":
            if not corr_ms_str:
                log.warning(f"{clip_fname}: wrong_onset with no corrected_onset_ms — skipping")
                entry["result"] = "skipped"
                entry["notes"]  = "missing corrected_onset_ms"
                action_log.append(entry)
                continue

            corrected_ms = int(corr_ms_str)
            new_fname    = f"xc{xc_id}_{quality}_{corrected_ms}.wav"
            new_path     = Path(POSITIVE_FINAL_DIR) / new_fname
            log.info(f"  wrong_onset  {clip_fname}  →  {new_fname}")

            if not dry:
                try:
                    flac_path = find_flac(xc_id, quality)
                    y = extract_clip(flac_path, corrected_ms)
                    y = apply_clipping_correction(y)
                    save_wav(y, new_path)
                    if clip_path.exists() and clip_path.resolve() != new_path.resolve():
                        clip_path.unlink()
                except Exception as exc:
                    log.error(f"  Failed: {exc}")
                    entry["result"] = "error"
                    entry["notes"]  = str(exc)
                    counts["errors"] += 1
                    action_log.append(entry)
                    continue

            # Update onset map
            onsets = onset_map.setdefault(xc_id, [])
            if orig_onset_ms in onsets:
                onsets.remove(orig_onset_ms)
            onsets.append(corrected_ms)

            entry["result"]      = "corrected"
            entry["new_filename"] = new_fname
            counts["wrong_onset"] += 1

        # ── no_bird ───────────────────────────────────────────────────────────
        elif issue == "no_bird":
            log.info(f"  no_bird      {clip_fname}  →  removing")

            if not dry and clip_path.exists():
                clip_path.unlink()

            onsets = onset_map.get(xc_id, [])
            used_for_search = [o for o in onsets if o != orig_onset_ms]

            # Attempt replacement from same FLAC
            new_onset: Optional[int] = None
            try:
                flac_path = find_flac(xc_id, quality)
                new_onset = find_replacement_onset(flac_path, used_for_search)
            except FileNotFoundError as exc:
                log.warning(f"  {exc} — no replacement possible")

            if new_onset is not None:
                new_fname = f"xc{xc_id}_{quality}_{new_onset}.wav"
                new_path  = Path(POSITIVE_FINAL_DIR) / new_fname
                log.info(f"             replacement  →  {new_fname}")

                if not dry:
                    try:
                        y = extract_clip(flac_path, new_onset)
                        y = apply_clipping_correction(y)
                        save_wav(y, new_path)
                    except Exception as exc:
                        log.error(f"  Replacement extraction failed: {exc}")
                        entry["result"] = "removed_no_replacement"
                        entry["notes"]  = str(exc)
                        counts["not_replaced"] += 1
                        action_log.append(entry)
                        continue

                onsets = onset_map.setdefault(xc_id, [])
                if orig_onset_ms in onsets:
                    onsets.remove(orig_onset_ms)
                onsets.append(new_onset)

                entry["result"]       = "replaced"
                entry["new_filename"] = new_fname
                counts["replaced"] += 1
            else:
                log.warning(f"  No suitable replacement found for {xc_id} — net count -1")
                onsets = onset_map.get(xc_id, [])
                if orig_onset_ms in onsets:
                    onsets.remove(orig_onset_ms)
                entry["result"] = "removed_no_replacement"
                counts["not_replaced"] += 1

            counts["no_bird"] += 1

        else:
            log.warning(f"Unknown issue_type '{issue}' for {clip_fname} — skipping")
            entry["result"] = "skipped"
            entry["notes"]  = f"unknown issue_type: {issue_raw}"

        action_log.append(entry)

    # ── Write outputs ─────────────────────────────────────────────────────────
    if not dry:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["clip_filename", "issue_type", "result",
                               "new_filename", "notes"]
            )
            writer.writeheader()
            writer.writerows(action_log)
        log.info(f"Action log → {OUTPUT_CSV}")

    total      = len(rows)
    net_change = counts["replaced"] - counts["not_replaced"]
    report = "\n".join([
        "Stage 9: QA Corrections Report",
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"{'DRY RUN | ' if dry else ''}QA CSV : {args.qa_csv}",
        "",
        f"Total entries processed   : {total}",
        f"  wrong_onset corrected   : {counts['wrong_onset']}",
        f"  no_bird / noise_dom.    : {counts['no_bird']}",
        f"    → replaced            : {counts['replaced']}",
        f"    → removed (no repl.)  : {counts['not_replaced']}",
        f"  errors                  : {counts['errors']}",
        "",
        f"Net clip count change     : {net_change:+d}",
    ])

    print("\n" + report + "\n")

    if not dry:
        with open(REPORT_TXT, "w") as f:
            f.write(report + "\n")
        log.info(f"Report → {REPORT_TXT}")


if __name__ == "__main__":
    main()
