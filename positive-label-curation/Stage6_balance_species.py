#!/usr/bin/env python3
"""
Stage6_balance_species.py

Apply species-level undersampling with acoustic diversity optimization.
Upgrades from v1:
 - Acoustic salience instead of RMS energy
 - Acoustic diversity via within-species embedding clustering
 - Priority queue for efficient backfill

PREREQUISITE:
 - Stage5 must be run with --no-quarantine flag to keep all clips available

WORKFLOW:
 - Reads clips from positive_staging/ directory (from config.py)
 - Selects balanced subset using acoustic diversity
 - Moves selected clips to positive/ directory (final dataset)

Key features:
 - Configurable target dataset size (default 25,000 from config.py)
 - Acoustic salience: spectral contrast + centroid for foreground detection
 - Acoustic clustering: identifies call types within species
 - Priority queue: O(log n) backfill instead of O(n log n) sorting
 - Generates pre/post long-tail distribution plots with Gini index

Strategy:
 1. Compute acoustic salience for all clips
 2. Cluster embeddings within each species to identify acoustic behaviors
 3. Calculate base per-species cap (target_size / num_species)
 4. For species below cap: keep all samples
 5. For species above cap: maximize acoustic diversity (one per cluster, then best salience)
 6. If total < target: priority queue backfill with diversity bonus
 7. If total > target: global salience-based trimming

Usage example (uses config.py defaults):
  python Stage6_balance_species.py
  python Stage6_balance_species.py --target-size 20000
"""

import argparse
import sys
import heapq
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Set, Dict
from tqdm import tqdm
import re
import librosa
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config

# Configuration constants
TARGET_DATASET_SIZE = config.STAGE6_MAX_CLIPS  # Default target dataset size from config.py
N_CLUSTERS_PER_SPECIES = 5    # Number of acoustic clusters per species (configurable)
N_MELS = 128                  # Mel-spectrogram bins
HOP_LENGTH = 512              # Hop length for feature extraction
TARGET_SR = 16000             # Target sample rate
DIVERSITY_BONUS = 100.0       # Bonus for new acoustic clusters


def extract_xc_quality(filename: str) -> str:
    """
    Extract XC quality rating from filename.
    Format: xc{id}_{quality}_{start_ms}.wav
    Returns 'U' if not found.
    """
    match = re.search(r'xc\d+_([A-DU])', filename, re.IGNORECASE)
    return match.group(1).upper() if match else 'U'


def quality_to_score(quality: str) -> int:
    """Convert quality rating to numeric score. Higher is better."""
    quality_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'U': 0}
    return quality_map.get(quality.upper(), 0)


def compute_acoustic_salience(audio_path: Path) -> float:
    """
    Compute acoustic salience combining spectral contrast and centroid.
    Salience measures foreground prominence vs background noise.

    Returns normalized salience score [0, 1] where higher = more salient.
    """
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        # Spectral contrast: measures difference between peaks and valleys
        # Higher contrast = clearer foreground signal
        S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, hop_length=HOP_LENGTH)
        mean_contrast = np.mean(contrast)

        # Spectral centroid: brightness/energy distribution
        # Higher centroid = more high-frequency energy (often correlates with vocalizations)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        mean_centroid = np.mean(centroid) / sr  # Normalize by sr

        # Combine: weighted sum (tuned empirically)
        salience = 0.7 * (mean_contrast / 40.0) + 0.3 * mean_centroid

        # Clip to [0, 1]
        return float(np.clip(salience, 0, 1))

    except Exception as e:
        print(f"Warning: Failed to compute salience for {audio_path.name}: {e}", file=sys.stderr)
        return 0.0


def compute_acoustic_embedding(audio_path: Path) -> np.ndarray:
    """
    Compute compact acoustic embedding for clustering.
    Uses mel-spectrogram statistics (mean + std pooling).

    Returns: 256-dimensional embedding (128 means + 128 stds)
    """
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=512, fmax=8000
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Statistical pooling: mean + std across time
        mu = np.mean(log_mel, axis=1)
        sigma = np.std(log_mel, axis=1)

        return np.concatenate([mu, sigma]).astype(np.float32)

    except Exception as e:
        print(f"Warning: Failed to compute embedding for {audio_path.name}: {e}", file=sys.stderr)
        return np.zeros(256, dtype=np.float32)


def cluster_species_acoustics(
    species_df: pd.DataFrame,
    outroot: Path,
    n_clusters: int = N_CLUSTERS_PER_SPECIES
) -> pd.DataFrame:
    """
    Cluster samples within a species based on acoustic embeddings.
    Assigns cluster ID to identify different call types/behaviors.

    Args:
        species_df: DataFrame for a single species
        outroot: Root directory containing audio clips
        n_clusters: Number of clusters (adaptive if less samples)

    Returns:
        DataFrame with added 'cluster_id' column
    """
    if len(species_df) <= n_clusters:
        # Fewer samples than clusters: each sample is its own cluster
        species_df = species_df.copy()
        species_df['cluster_id'] = range(len(species_df))
        return species_df

    # Compute embeddings for all samples
    embeddings = []
    valid_indices = []

    for idx, row in species_df.iterrows():
        clip_path = outroot / row['clip_filename']
        if clip_path.exists():
            emb = compute_acoustic_embedding(clip_path)
            if not np.all(emb == 0):  # Skip failed embeddings
                embeddings.append(emb)
                valid_indices.append(idx)

    if len(embeddings) == 0:
        # All embeddings failed
        species_df = species_df.copy()
        species_df['cluster_id'] = 0
        return species_df

    # Cluster embeddings
    embeddings = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Use MiniBatchKMeans for speed
    actual_n_clusters = min(n_clusters, len(embeddings))
    kmeans = MiniBatchKMeans(n_clusters=actual_n_clusters, random_state=42, batch_size=256)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)

    # Assign cluster IDs
    species_df = species_df.copy()
    species_df['cluster_id'] = -1  # Default for failed embeddings
    species_df.loc[valid_indices, 'cluster_id'] = cluster_labels

    return species_df


def select_diverse_samples_v2(
    group: pd.DataFrame,
    n_samples: int,
    outroot: Path,
    n_clusters: int = N_CLUSTERS_PER_SPECIES
) -> pd.DataFrame:
    """
    Select n_samples from group maximizing ACOUSTIC diversity.

    Strategy:
    1. Cluster samples into acoustic behaviors
    2. First pass: One sample per cluster (sorted by quality + salience)
    3. Second pass: Fill remaining slots with best quality + salience
    """
    # Perform clustering
    group = cluster_species_acoustics(group, outroot, n_clusters)

    selected = []
    seen_clusters = set()

    # First pass: one sample per acoustic cluster
    for _, row in group.iterrows():
        cluster_id = row['cluster_id']
        if cluster_id not in seen_clusters and cluster_id >= 0:
            selected.append(row)
            seen_clusters.add(cluster_id)
            if len(selected) >= n_samples:
                break

    # Second pass: fill remaining slots
    if len(selected) < n_samples:
        remaining = n_samples - len(selected)
        selected_indices = [row.name for row in selected]
        remaining_rows = group.loc[~group.index.isin(selected_indices)]

        # Sort by quality + salience
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
    """Create side-by-side long-tail distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pre-balancing
    ax1 = axes[0]
    pre_sorted = pre_counts.sort_values(ascending=False)
    x_pre = np.arange(len(pre_sorted))
    ax1.bar(x_pre, pre_sorted.values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(pre_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {pre_mean:.1f}')
    ax1.set_xlabel('Species ID (ranked by sample count)', fontsize=12)
    ax1.set_ylabel('Number of Samples per Species', fontsize=12)
    ax1.set_title(f'Pre-Balancing Distribution\nGini: {pre_gini:.3f} | Species: {len(pre_counts)} | Total: {pre_counts.sum():,}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Post-balancing
    ax2 = axes[1]
    post_sorted = post_counts.sort_values(ascending=False)
    x_post = np.arange(len(post_sorted))
    ax2.bar(x_post, post_sorted.values, color='seagreen', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(post_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {post_mean:.1f}')
    ax2.set_xlabel('Species ID (ranked by sample count)', fontsize=12)
    ax2.set_ylabel('Number of Samples per Species', fontsize=12)
    ax2.set_title(f'Post-Balancing Distribution (Acoustic Diversity)\nGini: {post_gini:.3f} | Species: {len(post_counts)} | Total: {post_counts.sum():,}',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution plot saved to: {output_path}")
    plt.close()


def balance_species_v2(
    df: pd.DataFrame,
    target_size: int,
    num_species: int,
    outroot: Path,
    n_clusters: int = N_CLUSTERS_PER_SPECIES
) -> pd.DataFrame:
    """
    Apply stratified undersampling with acoustic diversity optimization.

    Strategy:
    1. Compute acoustic salience for all clips
    2. Calculate per-species cap
    3. For each species: cluster embeddings, select diverse samples
    4. Backfill using priority queue with acoustic diversity bonus
    """
    base_per_species = target_size // num_species

    print(f"\n{'='*60}")
    print("PHASE 1: Computing Acoustic Salience")
    print(f"{'='*60}")

    # Compute salience for all clips
    salience_scores = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing salience", unit="clip"):
        clip_path = outroot / row['clip_filename']
        if clip_path.exists():
            salience = compute_acoustic_salience(clip_path)
        else:
            salience = 0.0
        salience_scores.append(salience)

    df = df.copy()
    df['acoustic_salience'] = salience_scores

    print(f"Salience statistics:")
    print(f"  Mean: {np.mean(salience_scores):.3f}")
    print(f"  Median: {np.median(salience_scores):.3f}")
    print(f"  Std: {np.std(salience_scores):.3f}")

    print(f"\n{'='*60}")
    print("PHASE 2: Per-Species Capping with Acoustic Diversity")
    print(f"{'='*60}")
    print(f"Base cap: {base_per_species} samples/species")
    print(f"Clusters per species: {n_clusters}")

    balanced_samples = []
    species_groups = []

    for species, group in tqdm(df.groupby('species'), desc="Processing species", unit="species"):
        group_size = len(group)

        # Add quality scores
        group = group.copy()
        if 'quality' not in group.columns:
            # Use clip_filename instead of source_file
            filename_col = 'clip_filename' if 'clip_filename' in group.columns else 'source_file'
            group['quality'] = group[filename_col].apply(extract_xc_quality)
        if 'quality_score' not in group.columns:
            group['quality_score'] = group['quality'].apply(quality_to_score)

        # Sort by quality + salience
        group_sorted = group.sort_values(['quality_score', 'acoustic_salience'], ascending=[False, False])

        if group_size <= base_per_species:
            # Keep all for rare species
            selected = group_sorted
        else:
            # Acoustic diversity selection
            selected = select_diverse_samples_v2(group_sorted, base_per_species, outroot, n_clusters)

        balanced_samples.append(selected)
        species_groups.append((species, group_sorted))

    result = pd.concat(balanced_samples, ignore_index=True)

    print(f"\nInitial balance: {len(result):,} samples")

    # Phase 3: Priority Queue Backfill
    if len(result) < target_size:
        needed = target_size - len(result)
        print(f"\n{'='*60}")
        print("PHASE 3: Priority Queue Backfill")
        print(f"{'='*60}")
        print(f"Need {needed:,} more samples")

        # Track acoustic clusters already in result
        result_clusters = set()
        for species, group_sorted in species_groups:
            species_result = result[result['species'] == species]
            if 'cluster_id' in species_result.columns:
                for cluster_id in species_result['cluster_id'].unique():
                    if cluster_id >= 0:
                        result_clusters.add((species, cluster_id))

        # Build priority queue (use negative score for max-heap)
        priority_queue = []
        current_counts = result.groupby('species').size().to_dict()

        print("Building priority queue...")
        for species, group_sorted in tqdm(species_groups, desc="Building queue", unit="species"):
            current_count = current_counts.get(species, 0)
            available = len(group_sorted)

            if available > current_count:
                # Cluster if not already done
                if 'cluster_id' not in group_sorted.columns:
                    group_sorted = cluster_species_acoustics(group_sorted, outroot, n_clusters)

                # Add extra samples to queue
                extra = group_sorted.iloc[current_count:available]
                for idx, row in extra.iterrows():
                    # Diversity bonus for new acoustic clusters
                    cluster_key = (species, row.get('cluster_id', -1))
                    diversity_bonus = DIVERSITY_BONUS if cluster_key not in result_clusters and row.get('cluster_id', -1) >= 0 else 0.0

                    # Combined score
                    score = 10.0 * row['quality_score'] + row['acoustic_salience'] + diversity_bonus

                    # Push to heap (negate for max-heap behavior)
                    heapq.heappush(priority_queue, (-score, idx, row))

        print(f"Priority queue size: {len(priority_queue):,}")
        print(f"Extracting top {needed:,} samples...")

        # Extract top-k from priority queue
        backfill_samples = []
        for _ in tqdm(range(min(needed, len(priority_queue))), desc="Backfilling", unit="sample"):
            neg_score, idx, row = heapq.heappop(priority_queue)
            backfill_samples.append(row)

        if backfill_samples:
            result = pd.concat([result, pd.DataFrame(backfill_samples)], ignore_index=True)

        print(f"Added {len(backfill_samples):,} samples")

    # Phase 4: Global trimming if needed
    if len(result) > target_size:
        print(f"\n{'='*60}")
        print("PHASE 4: Global Trimming")
        print(f"{'='*60}")
        print(f"Trimming {len(result) - target_size:,} samples")
        result = result.nlargest(target_size, 'acoustic_salience')

    # Remove duplicates based on clip_filename (keep first occurrence)
    initial_size = len(result)
    result = result.drop_duplicates(subset='clip_filename', keep='first')
    duplicates_removed = initial_size - len(result)

    if duplicates_removed > 0:
        print(f"\n⚠ Removed {duplicates_removed:,} duplicate clip_filename entries")

        # Backfill to reach target size after deduplication
        if len(result) < target_size:
            needed_after_dedup = target_size - len(result)
            print(f"Backfilling {needed_after_dedup:,} samples after deduplication...")

            # Get set of already selected clip_filenames
            selected_filenames = set(result['clip_filename'].values)

            # Build list of remaining candidates
            remaining_candidates = []
            for species, group_sorted in species_groups:
                for idx, row in group_sorted.iterrows():
                    if row['clip_filename'] not in selected_filenames:
                        # Add quality and salience score
                        quality_score = row.get('quality_score', 0)
                        salience = row.get('acoustic_salience', 0.0)
                        score = 10.0 * quality_score + salience
                        remaining_candidates.append((score, row))

            # Sort by score and take top needed samples
            remaining_candidates.sort(key=lambda x: x[0], reverse=True)
            additional_samples = [row for score, row in remaining_candidates[:needed_after_dedup]]

            if additional_samples:
                result = pd.concat([result, pd.DataFrame(additional_samples)], ignore_index=True)
                print(f"Added {len(additional_samples):,} unique samples to reach target size")

    print(f"\nFinal dataset size: {len(result):,}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6 v2: Balance species with acoustic diversity optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPROVEMENTS over v1:
  - Acoustic salience (spectral contrast + centroid) replaces RMS energy
  - Acoustic diversity via embedding clustering (identifies call types)
  - Priority queue for O(log n) backfill scalability

STRATEGY:
  - Computes acoustic features for all clips
  - Clusters within-species to identify acoustic behaviors
  - Maximizes acoustic variety, not just source variety
  - Prefers high-quality, salient, diverse samples

EXAMPLES:
  # Balance with default settings (uses config.py for all paths)
  python Stage6_balance_species.py

  # Custom target and cluster count
  python Stage6_balance_species.py \\
    --target-size 50000 --clusters-per-species 8
        """
    )

    parser.add_argument("--input-csv", default=config.STAGE6_INPUT_CSV, metavar="FILE",
                        help=f"Input CSV from Stage5 (default: {config.STAGE6_INPUT_CSV})")
    parser.add_argument("--staging-dir", default=config.STAGE6_STAGING_DIR, metavar="DIR",
                        help=f"Staging directory containing audio clips (default: {config.STAGE6_STAGING_DIR})")
    parser.add_argument("--final-dir", default=config.STAGE6_FINAL_DIR, metavar="DIR",
                        help=f"Final output directory for balanced clips (default: {config.STAGE6_FINAL_DIR})")
    parser.add_argument("--output-csv", default=config.STAGE6_OUTPUT_CSV, metavar="FILE",
                        help=f"Output CSV (default: {config.STAGE6_OUTPUT_CSV})")
    parser.add_argument("--plots", default="species_balance.png", metavar="FILE",
                        help="Output plots (default: species_balance.png)")
    parser.add_argument("--target-size", type=int, default=TARGET_DATASET_SIZE, metavar="N",
                        help=f"Target dataset size (default: {TARGET_DATASET_SIZE})")
    parser.add_argument("--clusters-per-species", type=int, default=N_CLUSTERS_PER_SPECIES, metavar="K",
                        help=f"Acoustic clusters per species (default: {N_CLUSTERS_PER_SPECIES})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show statistics without saving output")

    args = parser.parse_args()

    # Ensure metadata directory exists
    Path(config.METADATA_DIR).mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.input_csv)
    staging_dir = Path(args.staging_dir)
    final_dir = Path(args.final_dir)
    output_csv = Path(args.output_csv)
    plots_path = Path(args.plots)

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    if not staging_dir.exists():
        print(f"ERROR: Staging directory not found: {staging_dir}")
        sys.exit(1)

    # Create final directory if it doesn't exist
    if not args.dry_run:
        final_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_all = pd.read_csv(csv_path)

    # Rename columns to match expected names
    if 'en' in df_all.columns and 'species' not in df_all.columns:
        df_all = df_all.rename(columns={'en': 'species'})

    # Filter out skipped clips if the column exists
    if 'out_filename' in df_all.columns:
        df_all = df_all[df_all['out_filename'] != 'SKIPPED_TOO_SHORT'].copy()

    # Add quality_score column if it doesn't exist (using 'q' column)
    if 'quality_score' not in df_all.columns and 'q' in df_all.columns:
        # Map quality ratings to scores: A=5, B=4, C=3, D=2, E=1
        quality_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        df_all['quality_score'] = df_all['q'].map(quality_map).fillna(3)

    print(f"\n{'='*60}")
    print("Stage 6 v2: Acoustic Diversity-Aware Species Balancing")
    print(f"{'='*60}")
    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Target size: {args.target_size:,}")
    print(f"Clusters per species: {args.clusters_per_species}")
    print(f"{'='*60}")

    print(f"\nLoaded {len(df_all):,} clips")

    # Pre-balancing statistics
    pre_species_counts = df_all['species'].value_counts()
    num_species = len(pre_species_counts)
    pre_gini = calculate_gini(pre_species_counts.values)
    pre_mean = pre_species_counts.mean()

    print(f"\nPRE-BALANCING:")
    print(f"  Species: {num_species}")
    print(f"  Total samples: {len(df_all):,}")
    print(f"  Mean per species: {pre_mean:.1f}")
    print(f"  Gini coefficient: {pre_gini:.3f}")

    # Apply balancing
    df_balanced = balance_species_v2(
        df_all,
        args.target_size,
        num_species,
        staging_dir,
        args.clusters_per_species
    )

    # Post-balancing statistics
    post_species_counts = df_balanced['species'].value_counts()
    post_gini = calculate_gini(post_species_counts.values)
    post_mean = post_species_counts.mean()

    print(f"\n{'='*60}")
    print("POST-BALANCING:")
    print(f"{'='*60}")
    print(f"  Total samples: {len(df_balanced):,}")
    print(f"  Species: {len(post_species_counts)}")
    print(f"  Mean per species: {post_mean:.1f}")
    print(f"  Gini coefficient: {post_gini:.3f}")
    print(f"  Gini reduction: {100*(pre_gini - post_gini)/pre_gini:.1f}%")

    # Diversity metrics
    if 'cluster_id' in df_balanced.columns:
        unique_clusters = df_balanced[df_balanced['cluster_id'] >= 0].groupby('species')['cluster_id'].nunique().sum()
        print(f"  Unique acoustic clusters: {unique_clusters}")

    if 'acoustic_salience' in df_balanced.columns:
        print(f"  Mean salience: {df_balanced['acoustic_salience'].mean():.3f}")

    if 'quality' in df_balanced.columns or 'source_file' in df_balanced.columns:
        if 'quality' not in df_balanced.columns:
            df_balanced['quality'] = df_balanced['source_file'].apply(extract_xc_quality)
        quality_dist = df_balanced['quality'].value_counts().sort_index()
        quality_ab = quality_dist.get('A', 0) + quality_dist.get('B', 0)
        print(f"  Quality A+B: {quality_ab} ({100*quality_ab/len(df_balanced):.1f}%)")

    print(f"{'='*60}\n")

    # Move files from staging to final directory
    if not args.dry_run:
        print(f"\nMoving {len(df_balanced):,} clips from staging to final directory...")
        moved_count = 0
        failed_count = 0

        for idx, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc="Moving files", unit="file"):
            src_path = staging_dir / row['clip_filename']
            dst_path = final_dir / row['clip_filename']

            if src_path.exists():
                try:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst_path))
                    moved_count += 1
                except Exception as e:
                    print(f"\n⚠ Failed to move {src_path}: {e}")
                    failed_count += 1
            else:
                print(f"\n⚠ Source file not found: {src_path}")
                failed_count += 1

        print(f"\nMoved: {moved_count:,} files")
        if failed_count > 0:
            print(f"Failed: {failed_count:,} files")
    else:
        print(f"\n[DRY RUN] Would move {len(df_balanced):,} clips from {staging_dir} to {final_dir}")

    # Save results
    if not args.dry_run:
        df_balanced.to_csv(output_csv, index=False)
        print(f"\nSaved balanced dataset to: {output_csv}")

        plot_species_distribution(
            pre_species_counts, post_species_counts,
            plots_path, pre_gini, post_gini, pre_mean, post_mean
        )
    else:
        print("[DRY RUN] Would save CSV to:", output_csv)

    print("\nBalancing complete!")


if __name__ == "__main__":
    main()
