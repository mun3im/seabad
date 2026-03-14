#!/usr/bin/env python3
"""
Stage1_xc_fetch_bird_metadata.py

Fetch ALL Xeno-Canto v3 recordings for Southeast Asian countries sharing borders with Malaysia,
filter to records whose metadata indicates they are birds, and save full metadata to CSV.

Geographic scope:
  - Malaysia (primary)
  - Singapore (shares border)
  - Indonesia (shares borders in Borneo and maritime)
  - Brunei (shares border in Borneo)
  - Thailand (shares northern border)

Requirements:
  - requests
  - pandas

Set XENO_API_KEY environment variable before running:
  export XENO_API_KEY="your_api_key_here"

Usage examples:

# Fetch single country
python Stage1_xc_fetch_bird_metadata.py --country Malaysia --output-csv Stage1out_xc_bird_metadata.csv

# Fetch all Southeast Asian countries (recommended for maximizing dataset, uses default output)
python Stage1_xc_fetch_bird_metadata.py --country all

# Fetch multiple specific countries
python Stage1_xc_fetch_bird_metadata.py --countries Malaysia Singapore Indonesia --output-csv Stage1out_xc_bird_metadata.csv

"""

import os
import sys
import time
import logging
from typing import List, Dict, Optional
import requests
import pandas as pd
import argparse

# ------------ CONFIG -------------
# Import centralized config
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Please ensure config.py is in the same directory.")
    sys.exit(1)

SUPPORTED_COUNTRIES = ["Malaysia", "Singapore", "Indonesia", "Brunei", "Thailand", # sundaland
#                       "Philippines","Vietnam","Cambodia","Laos","Myanmar","Timor-Leste"
                    ]
OUT_CSV_DEFAULT = config.STAGE1_OUTPUT_CSV
BASE_URL = config.BASE_URL
RATE_LIMIT_DELAY = 0.2
MAX_RETRIES = 4
REQUEST_TIMEOUT = 30
MAX_YEAR = config.MAX_YEAR
GROUP_FIELD_CANDIDATES = ["grp", "group", "animal", "type", "kind"]
# ----------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xc_sea_birds")


def request_with_retries(url: str, params: Dict[str, str], headers: Dict[str, str], retries: int = 0) -> Optional[requests.Response]:
    """GET with retries and exponential backoff for transient errors."""
    try:
        time.sleep(RATE_LIMIT_DELAY)
        r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r
        if 400 <= r.status_code < 500:
            logger.warning(f"Client error {r.status_code} for URL {r.url} - response start: {r.text[:200]!r}")
            return None
        if 500 <= r.status_code < 600:
            raise requests.exceptions.HTTPError(f"Server error {r.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            backoff = 2 ** retries
            logger.warning(f"Request failed (attempt {retries+1}) -> retrying in {backoff}s: {e}")
            time.sleep(backoff)
            return request_with_retries(url, params, headers, retries + 1)
        else:
            logger.error(f"Request failed after {MAX_RETRIES} retries: {e}")
            return None


def looks_like_bird(record: Dict) -> bool:
    """Heuristic to check candidate fields for bird indicators."""
    for f in GROUP_FIELD_CANDIDATES:
        if f in record and record[f] is not None:
            try:
                v = str(record[f]).strip().lower()
            except Exception:
                continue
            if v == "":
                continue
            if "bird" in v or "aves" in v or "avian" in v:
                return True
    return False


def fetch_country_birds(api_key: str, country: str) -> List[Dict]:
    """Fetch all recordings for a given country and filter bird records."""
    headers = {"User-Agent": "xc_country_birds_fetcher/1.0", "Accept": "application/json"}
    # Filter to only include metadata records from before 2026
    year_filter = f"<{MAX_YEAR + 1}"  # year:"<2026"
    query_tag = f'cnt:"{country}" year:"{year_filter}"'
    page = 1
    all_bird_records: List[Dict] = []
    total_fetched = 0

    while True:
        params = {"key": api_key, "query": query_tag, "page": page}
        logger.info(f"Requesting page {page} for country={country}")
        resp = request_with_retries(BASE_URL, params=params, headers=headers)
        if resp is None:
            logger.error(f"Network/client error fetching page {page}. Stopping.")
            break

        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON for page {page}: {e}")
            break

        recordings = data.get("recordings", [])
        if not recordings:
            logger.info(f"No recordings on page {page} -> finished paging.")
            break

        logger.info(f"Page {page}: got {len(recordings)} recordings. Filtering for birds.")
        total_fetched += len(recordings)

        # filter bird records
        for rec in recordings:
            if looks_like_bird(rec):
                for drop in ("sono", "osci"):
                    if drop in rec:
                        rec.pop(drop, None)
                all_bird_records.append(rec)

        # check pagination
        num_pages = None
        for k in ("num_pages", "numPages", "numPagesTotal", "num_pages_total"):
            if k in data:
                try:
                    num_pages = int(data[k])
                    break
                except Exception:
                    continue
        if num_pages and page >= num_pages:
            logger.info(f"Reached num_pages={num_pages}. Done.")
            break

        page += 1

    logger.info(f"Finished fetching. Total records scanned: {total_fetched}. Bird records kept: {len(all_bird_records)}.")
    return all_bird_records


def save_records(records: List[Dict], out_csv: str) -> None:
    """Save list of dicts to CSV."""
    if not records:
        logger.warning("No records to save.")
        return
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Xeno-Canto bird metadata for Southeast Asian countries",
        epilog="""
EXAMPLES:
  # Fetch all supported countries (Malaysia, Singapore, Indonesia, Brunei, Thailand)
  python Stage1_fetch_xc_bird_metadata.py --country all --output-csv Stage1out_xc_asean_bird_metadata.csv

  # Fetch single country
  python Stage1_fetch_xc_bird_metadata.py --country Malaysia --output-csv Stage1out_xc_my_bird_metadata.csv

  # Fetch specific countries
  python Stage1_fetch_xc_bird_metadata.py --countries Malaysia Indonesia Thailand --output-csv Stage1out_xc_multi_bird_metadata.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (country selection)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--country",
                      choices=SUPPORTED_COUNTRIES + ["all"],
                      metavar="COUNTRY",
                      help="Single country to fetch (or 'all' for all supported countries)")
    group.add_argument("--countries",
                      nargs='+',
                      choices=SUPPORTED_COUNTRIES,
                      metavar="COUNTRY",
                      help="Multiple countries to fetch")

    # Output options
    parser.add_argument("--output-csv", default=OUT_CSV_DEFAULT, metavar="FILE",
                        help="Output CSV filename (default: Stage1_xc_sea_birds.csv)")
    parser.add_argument("--add-country-column", action="store_true",
                        help="Add 'fetch_country' column to track which country query returned each record")

    args = parser.parse_args()

    # Ensure metadata directory exists
    os.makedirs(config.METADATA_DIR, exist_ok=True)

    api_key = os.environ.get("XENO_API_KEY")
    if not api_key:
        logger.error("No XENO_API_KEY found. Set environment variable XENO_API_KEY and retry.")
        sys.exit(2)

    # Determine which countries to fetch
    if args.country:
        countries = SUPPORTED_COUNTRIES if args.country == "all" else [args.country]
    else:
        countries = args.countries

    logger.info(f"Starting Xeno-Canto bird records fetch for: {', '.join(countries)}")

    all_records = []
    for country in countries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching records for: {country}")
        logger.info(f"{'='*60}")
        records = fetch_country_birds(api_key=api_key, country=country)

        # Optionally tag records with source country
        if args.add_country_column:
            for rec in records:
                rec['fetch_country'] = country

        all_records.extend(records)
        logger.info(f"Total records so far: {len(all_records):,}")

    # Remove duplicates based on XC ID (recordings may appear in multiple countries)
    logger.info(f"\nDeduplicating records by XC ID...")
    df = pd.DataFrame(all_records)
    if not df.empty:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        final_count = len(df)
        logger.info(f"Removed {initial_count - final_count:,} duplicate records")

        # Save deduplicated records
        df.to_csv(args.output_csv, index=False)
        logger.info(f"\nWrote {final_count:,} unique bird records to {args.output_csv}")

        # Print country breakdown
        if 'cnt' in df.columns:
            logger.info(f"\nRecords by country (based on 'cnt' field in metadata):")
            country_counts = df['cnt'].value_counts()
            for country, count in country_counts.head(10).items():
                logger.info(f"  {country}: {count:,}")
    else:
        logger.warning("No records to save.")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
