#!/usr/bin/env python3
"""
Stage3_download_and_convert.py

Downloads audio files from Xeno-Canto and immediately converts them to 16kHz mono FLAC.

Input: Stage1out_xc_bird_metadata.csv (from config.py)
Outputs:
  - FLAC files in /Volumes/Evo/MYBAD2/asean-flacs/
  - Stage3out_successful_conversions.csv (successful conversions)
  - Stage3out_failed_downloads.csv (failed downloads)
  - Stage3_download_log.csv (detailed log)

Features:
 - Downloads MP3 from Xeno-Canto
 - Immediately converts to 16kHz mono FLAC using ffmpeg
 - Deletes MP3 after successful conversion
 - Records all attempts in detailed log
 - Skips existing FLACs (idempotent)

Usage:
  python Stage3_download_and_convert.py
  python Stage3_download_and_convert.py --limit 100  # test with first 100 records
  python Stage3_download_and_convert.py --dry-run    # simulate without downloading
"""

import argparse
import os
import sys
import time
import logging
import subprocess
from typing import Optional
import requests
import pandas as pd
from urllib.parse import urlparse, unquote
import csv

# Import centralized config
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Please ensure config.py is in the same directory.")
    sys.exit(1)

# --------- Config from config.py ----------
FLAC_OUTPUT_DIR = config.FLAC_OUTPUT_DIR
STAGE3_INPUT_CSV = config.STAGE3_INPUT_CSV  # Reads Stage1 output
STAGE3_OUTPUT_CSV = config.STAGE3_OUTPUT_CSV
STAGE3_FAILED_CSV = config.STAGE3_FAILED_CSV
STAGE3_LOG_CSV = config.STAGE3_LOG_CSV

RATE_LIMIT_DELAY = config.RATE_LIMIT_DELAY
MAX_RETRIES = config.MAX_RETRIES
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
CHUNK_SIZE = config.CHUNK_SIZE
USER_AGENT = config.USER_AGENT
MIN_BYTES_ACCEPTED = config.MIN_BYTES_ACCEPTED

TARGET_SR = config.TARGET_SAMPLE_RATE
TARGET_CHANNELS = config.TARGET_CHANNELS
EXCLUDE_SPECIES = config.EXCLUDE_SPECIES
# ------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xc_downloader")


def sanitize_folder_name(name: str) -> str:
    """Make a filesystem-safe folder name from the species English name."""
    if not isinstance(name, str) or name.strip() == "":
        return "unknown_species"
    s = name.strip()
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    return s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_extension_from_url(url: str, fallback: str = ".mp3") -> str:
    """Try to extract sensible extension from the URL path. Fallback to .mp3."""
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        basename = os.path.basename(path)
        if "." in basename:
            ext = os.path.splitext(basename)[1]
            if ext:
                return ext
    except Exception:
        pass
    return fallback


def convert_to_flac(input_path: str, output_path: str) -> tuple[bool, str]:
    """
    Convert audio file to 16kHz mono FLAC using ffmpeg.
    Returns: (success, error_message)
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-compression_level", "5",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg error: {e.stderr}"
        return False, error_msg
    except FileNotFoundError:
        return False, "ffmpeg not found - please install ffmpeg"
    except Exception as e:
        return False, f"Conversion error: {e}"


def download_url_to_path(url: str, out_path: str, max_retries: int = MAX_RETRIES, timeout: int = REQUEST_TIMEOUT) -> tuple[bool, str, int, float]:
    """
    Download URL streaming to out_path with retries.
    Returns tuple: (success, error_message_or_empty, bytes_written, elapsed_seconds)
    - Writes to out_path + ".part" then renames on success.
    """
    headers = {"User-Agent": USER_AGENT}
    attempt = 0
    start_time_total = time.time()
    tmp_path = out_path + ".part"

    while attempt <= max_retries:
        try:
            if attempt > 0:
                backoff = 2 ** (attempt - 1)
                logger.info(f"Retrying after {backoff}s (attempt {attempt}/{max_retries})...")
                time.sleep(backoff)

            time.sleep(RATE_LIMIT_DELAY)

            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    msg = f"Non-200 status {r.status_code}"
                    text_sample = ""
                    try:
                        text_sample = r.text[:200]
                    except Exception:
                        pass
                    logger.warning(f"{msg} for URL {url} (resp text start: {text_sample!r})")
                    if 400 <= r.status_code < 600:
                        error_type = "client error" if r.status_code < 500 else "server error"
                        return False, f"{msg} ({error_type})", 0, time.time() - start_time_total
                    attempt += 1
                    continue

                # stream to temp file
                with open(tmp_path, "wb") as fout:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        fout.write(chunk)
                        downloaded += len(chunk)

                # basic size check
                try:
                    actual = os.path.getsize(tmp_path)
                except Exception:
                    actual = downloaded

                if actual < MIN_BYTES_ACCEPTED:
                    logger.warning(f"Downloaded tiny file {actual} bytes for URL {url} -> rejecting")
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    attempt += 1
                    continue

                # rename to final
                os.replace(tmp_path, out_path)
                elapsed = time.time() - start_time_total
                return True, "", actual, elapsed

        except requests.exceptions.RequestException as e:
            logger.warning(f"RequestException for URL {url}: {e}")
            attempt += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error while downloading {url}: {e}")
            attempt += 1
            continue

    elapsed = time.time() - start_time_total
    return False, f"Failed after {max_retries} retries", 0, elapsed


def safe_str(val) -> str:
    """Safely convert a CSV cell value to stripped string."""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    try:
        return str(val).strip()
    except Exception:
        return ""


def parse_length_to_seconds(length_str: str) -> float:
    """Parse MM:SS format to total seconds. Returns 0.0 if invalid."""
    try:
        parts = length_str.strip().split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 1:
            return float(parts[0])
    except Exception:
        pass
    return 0.0


def append_log_row(log_path: str, row: dict):
    """Append one row (dict) to download_log.csv; create header if missing."""
    ensure_dir(os.path.dirname(log_path) if os.path.dirname(log_path) else ".")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        writer = csv.DictWriter(lf, fieldnames=[
            "id", "en", "file_url", "q", "out_path", "status", "error", "bytes", "elapsed_s", "ts"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_successful_conversion(output_path: str, row: dict):
    """Append one row (dict) to successful_conversions.csv matching Stage3 format."""
    ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else ".")
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as of:
        # Format matches Stage3out_successful_conversions.csv
        writer = csv.DictWriter(of, fieldnames=[
            "id", "en", "rec", "cnt", "lat", "lon", "lic", "q", "length", "smp"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_failed_row(failed_path: str, row: dict):
    """Append one row (dict) to failed_downloads.csv; create header if missing."""
    ensure_dir(os.path.dirname(failed_path) if os.path.dirname(failed_path) else ".")
    write_header = not os.path.exists(failed_path)
    with open(failed_path, "a", newline="", encoding="utf-8") as ff:
        writer = csv.DictWriter(ff, fieldnames=[
            "id", "en", "file_url", "q", "out_path", "error", "bytes", "elapsed_s", "ts"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def process_csv_and_download(csv_path: str, out_root: str, dry_run: bool = False, limit: Optional[int] = None):
    """Read CSV and download files, converting to FLAC immediately."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(2)

    # Remove existing output CSV to regenerate from scratch
    if os.path.exists(STAGE3_OUTPUT_CSV):
        logger.info(f"Removing existing output CSV to regenerate: {STAGE3_OUTPUT_CSV}")
        os.remove(STAGE3_OUTPUT_CSV)

    # read CSV
    df = pd.read_csv(csv_path, dtype=object)

    original_count = len(df)
    logger.info(f"Loaded {original_count} rows from {csv_path}")

    # Deduplicate rows based on 'id' column
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep='first')
        deduplicated_count = len(df)
        duplicates_removed = original_count - deduplicated_count
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows (keeping first occurrence)")
            logger.info(f"Processing {deduplicated_count} unique rows")
    else:
        logger.warning("No 'id' column found for deduplication")
        deduplicated_count = original_count

    # Filter out excluded species
    if 'en' in df.columns and EXCLUDE_SPECIES:
        before_filter = len(df)
        df = df[~df['en'].isin(EXCLUDE_SPECIES)]
        after_filter = len(df)
        excluded_count = before_filter - after_filter
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} records from excluded species: {EXCLUDE_SPECIES}")
            logger.info(f"Remaining rows after species filter: {after_filter}")

    total = len(df)
    logger.info(f"Total rows to process: {total}")

    count = 0
    converted = 0
    skipped_no_url = 0
    skipped_exists = 0
    failed = 0

    for idx, row in df.iterrows():
        if limit is not None and count >= limit:
            break
        count += 1

        # safe extraction
        rec_id = safe_str(row.get("id", ""))
        en = safe_str(row.get("en", ""))
        rec = safe_str(row.get("rec", ""))
        cnt = safe_str(row.get("cnt", ""))
        file_url = safe_str(row.get("file", ""))
        q = safe_str(row.get("q", ""))
        lat = safe_str(row.get("lat", ""))
        lon = safe_str(row.get("lon", ""))
        lic = safe_str(row.get("lic", ""))
        length = safe_str(row.get("length", ""))
        smp = safe_str(row.get("smp", ""))

        if rec_id == "":
            logger.warning(f"Row {idx}: missing id - skipping")
            failed += 1
            append_log_row(STAGE3_LOG_CSV, {
                "id": "", "en": en, "file_url": file_url, "q": q, "out_path": "",
                "status": "skip", "error": "missing id", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            continue

        # skip if file URL missing
        if not file_url or file_url.upper() in ("NULL", "NONE", "NAN", "NA"):
            skipped_no_url += 1
            logger.info(f"Row id={rec_id}: no file URL (skipping)")
            append_log_row(STAGE3_LOG_CSV, {
                "id": rec_id, "en": en, "file_url": file_url, "q": q, "out_path": "",
                "status": "skip", "error": "no_url", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            continue

        # skip if recording is too short (< 3 seconds)
        duration_seconds = parse_length_to_seconds(length)
        if duration_seconds < 3.0:
            skipped_no_url += 1
            logger.info(f"Row id={rec_id}: too short ({length} = {duration_seconds}s < 3s, skipping)")
            append_log_row(STAGE3_LOG_CSV, {
                "id": rec_id, "en": en, "file_url": file_url, "q": q, "out_path": "",
                "status": "skip", "error": "too_short", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            continue

        # quality char (convert "no score" to "U" for Unknown)
        if not q or q.lower() in ("no score", "unrated", "unknown"):
            q_char = "U"
        else:
            q_char = q[0].upper()

        # id canonical digits
        try:
            id_int = int(float(rec_id))
            id_str = str(int(id_int))
        except Exception:
            digits = "".join(ch for ch in rec_id if ch.isdigit())
            id_str = digits if digits else rec_id

        species_folder = sanitize_folder_name(en) if en else "unknown_species"
        dest_folder = os.path.join(out_root, species_folder)
        ensure_dir(dest_folder)

        # Final output will be FLAC
        out_fname_flac = f"xc{id_str}_{q_char}.flac"
        out_path_flac = os.path.join(dest_folder, out_fname_flac)

        if os.path.exists(out_path_flac):
            skipped_exists += 1
            logger.info(f"Already exists: {out_path_flac} (skipping download, adding metadata)")
            append_log_row(STAGE3_LOG_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "status": "exists", "error": "", "bytes": os.path.getsize(out_path_flac), "elapsed_s": 0.0, "ts": time.time()
            })
            # Add existing file to successful conversions CSV
            append_successful_conversion(STAGE3_OUTPUT_CSV, {
                "id": id_str, "en": en, "rec": rec, "cnt": cnt, "lat": lat, "lon": lon,
                "lic": lic, "q": q_char, "length": length, "smp": smp
            })
            continue

        if dry_run:
            logger.info(f"[DRY] Would download: {file_url} -> {out_path_flac}")
            append_log_row(STAGE3_LOG_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "status": "dry", "error": "", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            converted += 1
            continue

        # Download to temporary MP3 file
        ext = get_extension_from_url(file_url, fallback=".mp3")
        tmp_mp3 = os.path.join(dest_folder, f"xc{id_str}_{q_char}_tmp{ext}")

        logger.info(f"Downloading ({count}/{total}): id={id_str} q={q_char} -> {tmp_mp3}")
        ok, err_msg, bytes_written, elapsed = download_url_to_path(file_url, tmp_mp3)

        if not ok:
            failed += 1
            logger.error(f"Failed to download id={id_str} from {file_url}: {err_msg}")
            append_log_row(STAGE3_LOG_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "status": "fail", "error": err_msg, "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })
            append_failed_row(STAGE3_FAILED_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "error": err_msg, "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })
            # Clean up any partial files
            try:
                tmp = tmp_mp3 + ".part"
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            continue

        # Successfully downloaded - now convert to FLAC
        logger.info(f"Converting to FLAC: {tmp_mp3} -> {out_path_flac}")
        conv_ok, conv_err = convert_to_flac(tmp_mp3, out_path_flac)

        if conv_ok:
            # Successful conversion - delete the MP3
            try:
                os.remove(tmp_mp3)
                logger.info(f"Deleted temporary MP3: {tmp_mp3}")
            except Exception as e:
                logger.warning(f"Could not delete temporary MP3 {tmp_mp3}: {e}")

            converted += 1
            logger.info(f"Saved FLAC: {out_path_flac} ({bytes_written} bytes downloaded)")
            append_log_row(STAGE3_LOG_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "status": "ok", "error": "", "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })
            # Add to successful conversions CSV
            append_successful_conversion(STAGE3_OUTPUT_CSV, {
                "id": id_str, "en": en, "rec": rec, "cnt": cnt, "lat": lat, "lon": lon,
                "lic": lic, "q": q_char, "length": length, "smp": smp
            })
        else:
            # Conversion failed - keep MP3 for debugging, log as failed
            failed += 1
            logger.error(f"Failed to convert id={id_str} to FLAC: {conv_err}")
            logger.info(f"Kept MP3 file for debugging: {tmp_mp3}")
            append_log_row(STAGE3_LOG_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "status": "conversion_failed", "error": conv_err, "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })
            append_failed_row(STAGE3_FAILED_CSV, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path_flac,
                "error": f"Conversion failed: {conv_err}", "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })

    total_files = converted + skipped_exists
    logger.info("=== Summary ===")
    logger.info(f"Rows scanned: {count}")
    logger.info(f"Converted (new): {converted}")
    logger.info(f"Already existed: {skipped_exists}")
    logger.info(f"TOTAL FLAC FILES: {total_files}")
    logger.info(f"Skipped (no URL/too short): {skipped_no_url}")
    logger.info(f"Failed downloads/conversions: {failed}")
    logger.info(f"Download log at: {STAGE3_LOG_CSV}")
    logger.info(f"Successful conversions CSV at: {STAGE3_OUTPUT_CSV}")
    logger.info(f"Failed downloads CSV at: {STAGE3_FAILED_CSV}")

    return {
        'original_count': original_count,
        'successful_conversions': total_files,
        'failed_downloads': failed
    }


def parse_cmdline():
    p = argparse.ArgumentParser(
        description="Stage 2: Download Xeno-Canto files and convert to 16kHz mono FLAC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Download and convert all files from Stage1 CSV
  python Stage2_download_and_convert.py

  # Dry run to test
  python Stage2_download_and_convert.py --dry-run

  # Download first 100 rows for testing
  python Stage2_download_and_convert.py --limit 100
        """
    )

    p.add_argument("--dry-run", action="store_true",
                   help="Simulate downloads without actually downloading files")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Stop after this many rows (useful for testing/resume)")
    return p.parse_args()


def main():
    args = parse_cmdline()
    dry_run = args.dry_run
    limit = args.limit

    logger.info(f"Output directory: {FLAC_OUTPUT_DIR}")
    logger.info(f"Input CSV: {STAGE3_INPUT_CSV}")
    logger.info(f"Target format: 16kHz mono FLAC")

    # Ensure directories exist
    ensure_dir(config.METADATA_DIR)
    ensure_dir(FLAC_OUTPUT_DIR)
    stats = process_csv_and_download(
        csv_path=STAGE3_INPUT_CSV,
        out_root=FLAC_OUTPUT_DIR,
        dry_run=dry_run,
        limit=limit
    )

    # Generate report file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, "Stage3_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Stage 3: Download and Conversion Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of lines in input CSV: {stats['original_count']}\n")
        f.write(f"Number of successful conversions (including already existing files): {stats['successful_conversions']}\n")
        f.write(f"Number of failed downloads/conversions: {stats['failed_downloads']}\n")

    logger.info(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
