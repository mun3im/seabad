#!/usr/bin/env python3
"""
Stage1a_analyze_metadata.py

Analyzes metadata from Stage1 output CSV to provide statistics about the dataset.

Input: Stage1out_xc_bird_metadata.csv (from config.py)
Output: Prints analysis to console (no file saved)

Statistics provided:
- Number of records per country
- Number of unique species
- Top 10 species by recording count
- Bottom 10 species by recording count
- Number of unique species per country (species found only in that country)

Usage:
  python Stage1a_analyze_metadata.py
"""

import sys
import pandas as pd
from collections import Counter

# Import centralized config
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Please ensure config.py is in the same directory.")
    sys.exit(1)

STAGE2_INPUT_CSV = config.STAGE2_INPUT_CSV  # Reads Stage1 output
EXCLUDE_SPECIES = config.EXCLUDE_SPECIES


def safe_str(val) -> str:
    """Safely convert value to string."""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    try:
        return str(val).strip()
    except Exception:
        return ""


def analyze_metadata(csv_path: str):
    """Analyze metadata and print statistics."""

    print("=" * 70)
    print("STAGE 1a: METADATA ANALYSIS")
    print("=" * 70)
    print()

    # Load CSV
    try:
        df = pd.read_csv(csv_path, dtype=object)
        total_loaded = len(df)
        print(f"Loaded {total_loaded:,} records from {csv_path}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        sys.exit(1)

    # Clean data
    df['cnt'] = df['cnt'].apply(safe_str)
    df['en'] = df['en'].apply(safe_str)

    # Filter out empty country and species names
    df = df[df['cnt'] != '']
    df = df[df['en'] != '']

    # Filter out excluded species
    if EXCLUDE_SPECIES:
        before_filter = len(df)
        df = df[~df['en'].isin(EXCLUDE_SPECIES)]
        after_filter = len(df)
        excluded_count = before_filter - after_filter
        if excluded_count > 0:
            print(f"Excluded {excluded_count:,} records from species: {', '.join(EXCLUDE_SPECIES)}")
            print(f"Analyzing {after_filter:,} records with known species labels")
        print()

    print("-" * 70)
    print("1. RECORDS PER COUNTRY")
    print("-" * 70)

    country_counts = df['cnt'].value_counts()
    for country, count in country_counts.items():
        print(f"  {country:<30} {count:>8,} records")
    print()

    print("-" * 70)
    print("2. SPECIES STATISTICS")
    print("-" * 70)

    unique_species = df['en'].nunique()
    print(f"  Total unique species: {unique_species:,}")
    print()

    print("-" * 70)
    print("3. TOP 10 SPECIES (by recording count)")
    print("-" * 70)

    species_counts = df['en'].value_counts()
    top_10 = species_counts.head(10)

    for rank, (species, count) in enumerate(top_10.items(), 1):
        print(f"  {rank:2d}. {species:<45} {count:>6,} recordings")
    print()

    print("-" * 70)
    print("4. BOTTOM 10 SPECIES (by recording count)")
    print("-" * 70)

    bottom_10 = species_counts.tail(10).sort_values()

    for rank, (species, count) in enumerate(bottom_10.items(), 1):
        print(f"  {rank:2d}. {species:<45} {count:>6,} recordings")
    print()

    print("-" * 70)
    print("5. UNIQUE SPECIES PER COUNTRY")
    print("-" * 70)
    print("  (Species found only in one country)")
    print()

    # For each species, find which countries it appears in
    species_countries = {}
    for species in df['en'].unique():
        species_df = df[df['en'] == species]
        countries = set(species_df['cnt'].unique())
        species_countries[species] = countries

    # Count unique species per country
    country_unique = {}
    for country in df['cnt'].unique():
        unique_to_country = []
        for species, countries in species_countries.items():
            if len(countries) == 1 and country in countries:
                unique_to_country.append(species)
        country_unique[country] = len(unique_to_country)

    # Sort by count descending
    sorted_countries = sorted(country_unique.items(), key=lambda x: x[1], reverse=True)

    for country, unique_count in sorted_countries:
        total_species = len(df[df['cnt'] == country]['en'].unique())
        percentage = (unique_count / total_species * 100) if total_species > 0 else 0
        print(f"  {country:<30} {unique_count:>5,} unique species ({percentage:>5.1f}% of {total_species:,} total)")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    analyze_metadata(STAGE2_INPUT_CSV)


if __name__ == "__main__":
    main()
