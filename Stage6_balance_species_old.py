#!/usr/bin/env python3
"""
Stage6_balance_species.py

Apply species-level undersampling to achieve ecological diversity in the final dataset.
Ensures no single species dominates while preserving representation of rare species.

PREREQUISITE:
 - Stage5 must be run with --no-quarantine flag to keep all clips available

Key features:
 - Configurable target dataset size (default 20,000)
 - Per-species cap calculated to balance representation
 - Iteratively fills to target size while maintaining balance
 - Generates pre/post long-tail distribution plots with Gini index
 - Prioritizes higher RMS energy within each species

Strategy:
 1. Calculate base per-species cap (target_size / num_species)
 2. For species below cap: keep all samples
 3. For species above cap: keep top N by RMS energy
 4. If total < target: add more samples from species with extras (by RMS)
 5. If total > target: global RMS-based trimming

Usage example:
  # First, run Stage5 with --no-quarantine
  python Stage5_extract_3s_clips_from_flac.py \
    --inroot /path/to/flac \
    --outroot /path/to/clips \
    --output-csv Stage5_unique_3sclips.csv \
    --metadata-csv Stage4_unique_flacs.csv \
    --no-quarantine

  # Then run Stage6 (uses default target size of 20,000)
  python Stage6_balance_species.py --outroot /path/to/clips
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm
import re

# Configuration constants
TARGET_DATASET_SIZE = 20000  # Default target dataset size (can be overridden via CLI)

def extract_xc_quality(filename: str) -> str:
    """
    Extract XC quality rating from filename.
    Format: xc{id}_{quality}.flac or xc{id}_{quality}.wav
    Quality ratings: A (best) > B > C > D (worst) > U (unknown)
    Returns 'U' if not found or marked as unknown.
    """
    match = re.search(r'xc\d+_([A-DU])', filename, re.IGNORECASE)
    return match.group(1).upper() if match else 'U'


def quality_to_score(quality: str) -> int:
    """Convert quality rating to numeric score for sorting. Higher is better."""
    quality_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'U': 0}
    return quality_map.get(quality.upper(), 0)


def select_diverse_samples(group: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Select n_samples from group maximizing XC ID diversity.

    Strategy:
    1. First pass: Take top sample from each unique xc_id (sorted by quality + RMS)
    2. Second pass: If still need more, add remaining samples by quality + RMS
    """
    selected = []
    seen_xc_ids = set()

    # First pass: one sample per xc_id (prioritize diversity)
    for _, row in group.iterrows():
        xc_id = row['xc_id']
        if xc_id not in seen_xc_ids:
            selected.append(row)
            seen_xc_ids.add(xc_id)
            if len(selected) >= n_samples:
                break

    # Second pass: fill remaining slots with best quality/RMS (allow duplicate xc_ids)
    if len(selected) < n_samples:
        remaining = n_samples - len(selected)
        # Get indices of already selected rows
        selected_indices = [row.name for row in selected]
        # Get remaining rows not yet selected
        remaining_rows = group.loc[~group.index.isin(selected_indices)]
        # Add top remaining_rows by quality/RMS
        for _, row in remaining_rows.head(remaining).iterrows():
            selected.append(row)

    return pd.DataFrame(selected)


def calculate_gini(counts: np.ndarray) -> float:
    """
    Calculate Gini coefficient for species distribution.
    0 = perfect equality, 1 = perfect inequality
    """
    if len(counts) == 0:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n


def plot_species_distribution(
    pre_counts: pd.Series,
    post_counts: pd.Series,
    output_path: Path,
    pre_gini: float,
    post_gini: float,
    pre_mean: float,
    post_mean: float
):
    """
    Create side-by-side long-tail distribution plots showing species ID vs sample count.
    X-axis: Species ID (ranked by sample count, descending)
    Y-axis: Number of samples per species
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pre-undersampling long-tail plot
    ax1 = axes[0]
    pre_sorted = pre_counts.sort_values(ascending=False)
    x_pre = np.arange(len(pre_sorted))
    ax1.bar(x_pre, pre_sorted.values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(pre_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {pre_mean:.1f}')
    ax1.set_xlabel('Species ID (ranked by sample count)', fontsize=12)
    ax1.set_ylabel('Number of Samples per Species', fontsize=12)
    ax1.set_title(f'Pre-Undersampling Long-Tail Distribution\nGini: {pre_gini:.3f} | Species: {len(pre_counts)} | Total: {pre_counts.sum():,}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Post-undersampling long-tail plot
    ax2 = axes[1]
    post_sorted = post_counts.sort_values(ascending=False)
    x_post = np.arange(len(post_sorted))
    ax2.bar(x_post, post_sorted.values, color='seagreen', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(post_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {post_mean:.1f}')
    ax2.set_xlabel('Species ID (ranked by sample count)', fontsize=12)
    ax2.set_ylabel('Number of Samples per Species', fontsize=12)
    ax2.set_title(f'Post-Undersampling Long-Tail Distribution\nGini: {post_gini:.3f} | Species: {len(post_counts)} | Total: {post_counts.sum():,}',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nLong-tail distribution plot saved to: {output_path}")
    plt.close()


def balance_species(
    df: pd.DataFrame,
    target_size: int,
    num_species: int
) -> pd.DataFrame:
    """
    Apply stratified undersampling to balance species representation with diversity optimization.

    Strategy:
    1. Calculate per-species cap based on target size and number of species
    2. For species below cap: keep all samples
    3. For species above cap: prioritize diversity (unique XC IDs, quality, then RMS)
    4. If total is below target, iteratively increase cap for species with more samples

    Diversity priorities:
    - Maximize unique recording IDs (xc_id)
    - Prefer higher quality recordings (A > B > C > D)
    - Within same quality, prefer higher RMS energy
    """
    # Calculate per-species maximum
    base_per_species = target_size // num_species

    balanced_samples = []
    species_groups = []

    print(f"\nApplying diversity-aware undersampling (base cap: {base_per_species} samples/species)...")

    # First pass: apply base cap with diversity logic
    for species, group in tqdm(df.groupby('species'), desc="Processing species", unit="species"):
        group_size = len(group)

        # Add diversity columns for selection
        group = group.copy()
        # Extract quality if not already present (for backward compatibility)
        if 'quality' not in group.columns:
            group['quality'] = group['source_file'].apply(extract_xc_quality)
        group['quality_score'] = group['quality'].apply(quality_to_score)

        # Sort by: quality (desc), then RMS (desc)
        group_sorted = group.sort_values(['quality_score', 'rms_energy'], ascending=[False, False])

        if group_size <= base_per_species:
            # Keep all samples for species below cap
            selected = group_sorted
        else:
            # Undersample with XC ID diversity preference
            selected = select_diverse_samples(group_sorted, base_per_species)

        balanced_samples.append(selected)
        species_groups.append((species, group_sorted))

    result = pd.concat(balanced_samples, ignore_index=True)

    # If we're under target, gradually increase cap for species that have more samples
    if len(result) < target_size:
        needed = target_size - len(result)
        print(f"\nInitial balance yielded {len(result):,} samples. Need {needed:,} more to reach target.")

        # Calculate how many more samples each species could contribute
        current_counts = result.groupby('species').size().to_dict()

        # Build a pool of additional samples we can draw from (with diversity scores)
        additional_pool = []
        result_xc_ids = set(result['xc_id'].values)

        for species, group_sorted in tqdm(species_groups, desc="Building backfill pool", unit="species"):
            current_count = current_counts.get(species, 0)
            available = len(group_sorted)
            if available > current_count:
                # Add the next samples from this species
                extra = group_sorted.iloc[current_count:available]
                for idx, row in extra.iterrows():
                    # Bonus for new xc_ids not yet in result
                    diversity_bonus = 100.0 if row['xc_id'] not in result_xc_ids else 0.0
                    # Combined score: quality + RMS + diversity bonus
                    score = row['quality_score'] * 10.0 + row['rms_energy'] + diversity_bonus
                    additional_pool.append((species, current_count, row, score))

        # Sort additional pool by combined score (quality + RMS + diversity bonus)
        additional_pool.sort(key=lambda x: x[3], reverse=True)

        print(f"Backfilling with {min(needed, len(additional_pool)):,} samples (prioritizing new XC IDs)...")
        for species, current_count, row, score in tqdm(additional_pool[:needed], desc="Backfilling", unit="sample"):
            result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

        print(f"Added {min(needed, len(additional_pool)):,} more samples to reach closer to target")

    # If we're still over target, do global RMS-based trimming
    if len(result) > target_size:
        result = result.nlargest(target_size, 'rms_energy')

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Balance species through undersampling for ecological diversity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PREREQUISITE:
  Stage5 must be run with --no-quarantine to keep all clips available

STRATEGY:
  - Calculates per-species cap based on target size
  - Prioritizes recording diversity (unique XC IDs)
  - Prefers higher quality grades (A > B > C > D)
  - Within same quality, prefers higher RMS energy

EXAMPLES:
  # Balance with auto-calculated target (75% of clips, rounded to nearest 1000)
  python Stage6_balance_species.py --input-csv clips_log.csv --outroot clips/ \\
    --output-csv balanced_clips.csv

  # Balance with specific target size
  python Stage6_balance_species.py --input-csv clips_log.csv --outroot clips/ \\
    --output-csv balanced_clips.csv --target-size 16000 --plots balance.png
        """
    )

    # Required arguments
    parser.add_argument("--input-csv", default="Stage5_unique_3sclips.csv", metavar="FILE",
                        help="Input CSV with all clips from Stage5 (default: Stage5_unique_3sclips.csv)")
    parser.add_argument("--outroot", required=True, metavar="DIR",
                        help="Root directory containing audio clips")

    # Output options
    parser.add_argument("--output-csv", default="Stage6_balanced_clips.csv", metavar="FILE",
                        help="Output CSV with balanced dataset (default: Stage6_balanced_clips.csv)")
    parser.add_argument("--plots", default="species_balance.png", metavar="FILE",
                        help="Output PNG with distribution plots (default: species_balance.png)")

    # Processing options
    parser.add_argument("--target-size", type=int, default=TARGET_DATASET_SIZE, metavar="N",
                        help=f"Target dataset size (default: {TARGET_DATASET_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show statistics without saving output files")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    outroot = Path(args.outroot)
    output_csv = Path(args.output_csv)
    target_size = args.target_size
    plots_path = Path(args.plots)
    dry_run = args.dry_run

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Load all clips from Stage5 (should be run with --no-quarantine flag)
    df_all = pd.read_csv(csv_path)

    # Filter out skipped clips (too short)
    df_all = df_all[df_all['out_filename'] != 'SKIPPED_TOO_SHORT'].copy()

    print(f"\n{'='*60}")
    print(f"Stage 6: Species-Level Undersampling for Ecological Diversity")
    print(f"{'='*60}")
    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Target dataset size: {target_size:,}")
    print(f"{'='*60}\n")

    print(f"Loaded {len(df_all):,} valid clips from CSV")
    print(f"Note: Ensure Stage5 was run with --no-quarantine flag for balanced workflow")

    # Pre-undersampling statistics
    pre_species_counts = df_all['species'].value_counts()
    num_species = len(pre_species_counts)
    pre_gini = calculate_gini(pre_species_counts.values)
    pre_mean = pre_species_counts.mean()

    # Extract quality grades if not already present (for backward compatibility)
    if 'quality' not in df_all.columns:
        df_all['quality'] = df_all['source_file'].apply(extract_xc_quality)
    pre_quality_dist = df_all['quality'].value_counts().sort_index()

    print(f"\n{'='*60}")
    print("PRE-UNDERSAMPLING STATISTICS")
    print(f"{'='*60}")
    print(f"Total species: {num_species}")
    print(f"Total samples: {len(df_all):,}")
    print(f"Average samples per species: {pre_mean:.1f}")
    print(f"Gini coefficient: {pre_gini:.3f}")
    print(f"Min samples (species): {pre_species_counts.min()}")
    print(f"Max samples (species): {pre_species_counts.max()}")
    print(f"\nTop 5 species by sample count:")
    for species, count in pre_species_counts.head(5).items():
        print(f"  {species}: {count:,} samples")

    # Quality grade distribution across all clips
    print(f"\nQuality grade distribution (all clips):")
    for qual in ['A', 'B', 'C', 'D', 'U']:
        count = pre_quality_dist.get(qual, 0)
        pct = 100 * count / len(df_all) if len(df_all) > 0 else 0
        qual_label = 'U (unknown)' if qual == 'U' else qual
        print(f"  {qual_label}: {count:,} ({pct:.1f}%)")

    # Quality grade distribution per species (summary stats)
    quality_per_species = df_all.groupby(['species', 'quality']).size().unstack(fill_value=0)
    print(f"\nQuality distribution per species (summary):")
    print(f"  Species with grade A clips: {(quality_per_species.get('A', 0) > 0).sum()}/{num_species}")
    print(f"  Species with grade B clips: {(quality_per_species.get('B', 0) > 0).sum()}/{num_species}")
    print(f"  Species with grade C clips: {(quality_per_species.get('C', 0) > 0).sum()}/{num_species}")
    print(f"  Species with grade D clips: {(quality_per_species.get('D', 0) > 0).sum()}/{num_species}")
    print(f"  Species with grade U clips: {(quality_per_species.get('U', 0) > 0).sum()}/{num_species}")
    print(f"{'='*60}\n")

    # Apply species balancing
    print("\nApplying species-level undersampling...")
    df_balanced = balance_species(df_all, target_size, num_species)

    # Post-undersampling statistics
    post_species_counts = df_balanced['species'].value_counts()
    post_gini = calculate_gini(post_species_counts.values)
    post_mean = post_species_counts.mean()

    # Extract quality if not already present (for backward compatibility)
    if 'quality' not in df_balanced.columns:
        df_balanced['quality'] = df_balanced['source_file'].apply(extract_xc_quality)

    print(f"\n{'='*60}")
    print("POST-UNDERSAMPLING STATISTICS")
    print(f"{'='*60}")
    print(f"Total species: {len(post_species_counts)}")
    print(f"Total samples: {len(df_balanced):,}")
    print(f"Average samples per species: {post_mean:.1f}")
    print(f"Gini coefficient: {post_gini:.3f}")
    print(f"Min samples (species): {post_species_counts.min()}")
    print(f"Max samples (species): {post_species_counts.max()}")
    print(f"\nTop 5 species by sample count:")
    for species, count in post_species_counts.head(5).items():
        print(f"  {species}: {count:,} samples")

    # Diversity metrics
    unique_xc_ids = df_balanced['xc_id'].nunique()
    total_samples = len(df_balanced)
    quality_dist = df_balanced['quality'].value_counts().sort_index()

    print(f"\nDIVERSITY METRICS:")
    print(f"Unique XC recordings: {unique_xc_ids:,}")
    print(f"Samples per recording (avg): {total_samples/unique_xc_ids:.2f}")
    print(f"\nQuality distribution:")
    for qual in ['A', 'B', 'C', 'D', 'U']:
        count = quality_dist.get(qual, 0)
        pct = 100 * count / total_samples if total_samples > 0 else 0
        qual_label = 'U (unknown)' if qual == 'U' else qual
        print(f"  {qual_label}: {count:,} ({pct:.1f}%)")
    print(f"{'='*60}\n")

    # Calculate improvement
    gini_improvement = ((pre_gini - post_gini) / pre_gini) * 100 if pre_gini > 0 else 0
    print(f"Gini coefficient improvement: {gini_improvement:.1f}%")
    print(f"Samples removed: {len(df_all) - len(df_balanced):,}")

    # Save balanced dataset
    if dry_run:
        print(f"\n[DRY RUN] Would save balanced dataset to: {output_csv}")
        print(f"[DRY RUN] Would save distribution plots to: {plots_path}")
    else:
        df_balanced.to_csv(output_csv, index=False)
        print(f"\nBalanced dataset saved to: {output_csv}")

        # Generate histograms
        print("\nGenerating distribution histograms...")
        plot_species_distribution(
            pre_species_counts,
            post_species_counts,
            plots_path,
            pre_gini,
            post_gini,
            pre_mean,
            post_mean
        )

    print(f"\n{'='*60}")
    print("Species balancing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
