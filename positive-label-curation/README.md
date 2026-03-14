# SEABAD Dataset Curation Pipeline

A complete 6-stage pipeline for creating the **Southeast Asian Bird Audio Detection (SEABAD)** dataset from Xeno-Canto recordings. Produces a balanced, deduplicated dataset optimized for bird presence-absence detection on edge devices.

## Pipeline Overview

```
Stage 1: Fetch Metadata → Stage 2: Analyze → Stage 3: Download & Convert →
Stage 4: Deduplicate → Stage 5: Extract Clips → Stage 6: Balance Species
```

**Final Output**: 25,000 balanced 3-second WAV clips (16kHz mono) spanning 1,677 Southeast Asian bird species.

---

## Quick Start

All stages use centralized configuration from `config.py`:

```bash
# Default configuration
DATASET_ROOT = "/Volumes/Evo/SEABAD"
FLAC_OUTPUT_DIR = "/Volumes/Evo/SEABAD/asean-flacs"
POSITIVE_STAGING_DIR = "/Volumes/Evo/SEABAD/positive_staging"
POSITIVE_FINAL_DIR = "/Volumes/Evo/SEABAD/positive"
```

**Run complete pipeline**:

```bash
# 1. Fetch metadata (5-10 minutes)
python Stage1_xc_fetch_bird_metadata.py --country all

# 2. Analyze metadata (optional, <1 minute)
python Stage2_analyze_metadata.py

# 3. Download and convert to FLAC (2-6 hours, 38,466 files)
python Stage3_download_and_convert.py

# 4. Deduplicate (15-30 minutes, removes 13 duplicates)
python Stage4_deduplicate_flac.py

# 5. Extract 3s clips (20-40 minutes, produces 38,453 clips)
python Stage5_extract_wav_from_flac.py --no-quarantine

# 6. Balance species (5-10 minutes, selects 25,000 clips)
python Stage6_balance_species.py
```

---

## Requirements

### Python Dependencies

```bash
pip install pandas requests librosa soundfile tqdm scikit-learn matplotlib faiss-cpu numpy
```

**Note**: For GPU acceleration in Stage 4, install `faiss-gpu` instead of `faiss-cpu`.

### System Dependencies

- **ffmpeg**: Required for audio conversion
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

### Python Version

- Python 3.8 or higher

---

## Pipeline Stages (Detailed)

### Stage 1: Fetch Metadata from Xeno-Canto

**Script**: `Stage1_xc_fetch_bird_metadata.py`

**Purpose**: Download bird recording metadata from Xeno-Canto API

**Configuration** (from `config.py`):
- `BASE_URL`: `https://xeno-canto.org/api/3/recordings`
- `MAX_YEAR`: 2025 (filters recordings from before 2026)
- `EXCLUDE_SPECIES`: `["Identity unknown", "identity unknown", "Unknown", "unknown"]`

**Countries**: Malaysia, Singapore, Indonesia, Brunei, Thailand (Sundaland region)

**Usage**:
```bash
# Fetch all Southeast Asian countries (default)
python Stage1_xc_fetch_bird_metadata.py --country all

# Fetch specific country
python Stage1_xc_fetch_bird_metadata.py --country Malaysia
```

**Output**: `Stage1out_xc_bird_metadata.csv` (~42,000 records)

**CSV Schema**:
- `id` - Xeno-Canto recording ID
- `en` - English species name
- `rec` - Recordist name
- `cnt` - Country
- `lat`, `lon` - GPS coordinates
- `file` - Download URL
- `lic` - License URL
- `q` - Quality rating (A/B/C/D)
- `length` - Recording duration (MM:SS)
- `smp` - Sample rate

**Key Features**:
- Filters recordings from before 2026 (`MAX_YEAR = 2025`)
- Excludes "Identity unknown" species
- Deduplicates by XC ID across countries

---

### Stage 2: Analyze Metadata

**Script**: `Stage2_analyze_metadata.py`

**Purpose**: Statistical analysis of metadata (species counts, distributions, quality ratings)

**Input**: `Stage1out_xc_bird_metadata.csv` (from Stage 1)

**Output**: Console output only (no CSV)

**Usage**:
```bash
python Stage2_analyze_metadata.py
```

**Displays**:
- Total recordings and species count
- Country breakdown
- Quality rating distribution
- Species counts per country
- Long-tail distribution statistics

---

### Stage 3: Download & Convert to FLAC

**Script**: `Stage3_download_and_convert.py`

**Purpose**: Download MP3s from Xeno-Canto and convert to 16kHz mono FLAC

**Input**: `Stage1out_xc_bird_metadata.csv` (from Stage 1)

**Configuration** (from `config.py`):
- `TARGET_SAMPLE_RATE`: 16000 Hz
- `TARGET_CHANNELS`: 1 (mono)
- `TARGET_FORMAT`: "flac"
- `TARGET_SUBTYPE`: "PCM_16"
- `RATE_LIMIT_DELAY`: 0.1 seconds between requests
- `MAX_RETRIES`: 4
- `REQUEST_TIMEOUT`: 30 seconds
- `MIN_BYTES_ACCEPTED`: 1024 bytes (1 KB minimum)

**Usage**:
```bash
# Download and convert all files
python Stage3_download_and_convert.py

# Dry run (simulate without downloading)
python Stage3_download_and_convert.py --dry-run

# Test with first 100 files
python Stage3_download_and_convert.py --limit 100
```

**Workflow**:
1. Download MP3 from Xeno-Canto
2. Convert to 16kHz mono FLAC using ffmpeg
3. Delete temporary MP3
4. Organize by species folders

**Outputs**:
- `Stage3out_successful_conversions.csv` (38,466 successful conversions)
- `Stage3out_failed_downloads.csv` (4 failed downloads)
- `Stage3_download_log.csv` (detailed log)
- `Stage3_report.txt` (summary report)
- FLAC files in `FLAC_OUTPUT_DIR` organized by species

**Filename format**: `xc{id}_{quality}.flac` (e.g., `xc422286_A.flac`)

**Preprocessing**:
- **NO normalization or soft-clipping** - deliberately minimal to emulate edge recording conditions
- Basic resampling and format conversion only

**Expected Results**:
- Successful: 38,466 files
- Failed: 4 files (server errors)
- Skipped: 3,791 files (<3s duration or unknown species)

**Estimated Time**: 2-6 hours (network-dependent)

---

### Stage 4: Deduplicate FLACs

**Script**: `Stage4_deduplicate_flac.py`

**Purpose**: Detect and quarantine duplicate/near-duplicate recordings using FAISS-accelerated acoustic similarity

**Input**: `Stage3out_successful_conversions.csv` (from Stage 3)

**FLAC Directory**: `FLAC_OUTPUT_DIR` from `config.py`

**Algorithm**:
1. **Mel-spectrogram Embeddings**: 128 mel bins, 0-8kHz frequency range
2. **Clip-level Descriptors**: Mean+std pooling for fast similarity search
3. **FAISS Nearest Neighbor Search**: Efficient top-k candidate retrieval (cosine similarity)
4. **Frame-wise Verification**:
   - Perfect duplicate: All embeddings identical within 10^-7 tolerance
   - Near-duplicate: Mean similarity ≥ 0.997, Min ≥ 0.985, 5th percentile ≥ 0.992

**Usage**:
```bash
# Default: dry-run mode (preview without moving files)
python Stage4_deduplicate_flac.py

# Quarantine perfect duplicates only
python Stage4_deduplicate_flac.py --quarantine-perfect

# Quarantine all duplicates (perfect + near)
python Stage4_deduplicate_flac.py --quarantine-all

# Use custom FLAC directory
python Stage4_deduplicate_flac.py /custom/path/to/flacs
```

**Execution Modes**:
- **Default (dry-run)**: Preview duplicates without moving files
- `--quarantine-perfect`: Move perfect duplicates to `perfect_duplicates/`
- `--quarantine-all`: Move both perfect and near-duplicates

**Quarantine Strategy**:
- **Same recorder**: Keep newer recording (higher XC ID), remove older
- **Different recorders**: Keep older recording for manual review

**Outputs**:
- `Stage4out_unique_flacs.csv` (38,453 unique recordings)
- `Stage4_removed_near_duplicates_metadata.csv` (13 removed entries)
- `Stage4_report.txt` (detailed duplicate pairs)

**Expected Results**:
- Perfect duplicates: 8 pairs (same recorder, metadata updates)
- Near-duplicates: 5 pairs (for manual review)
- Total removed: 13 files (0.03% of corpus)
- Final corpus: 38,453 unique recordings

**Performance**: 15-30 minutes (FAISS-accelerated, 10-100x faster than all-pairs)

---

### Stage 5: Extract 3-Second WAV Clips

**Script**: `Stage5_extract_wav_from_flac.py`

**Purpose**: Extract fixed-length 3s clips with diversity-aware selection

**Input**: `Stage4out_unique_flacs.csv` (from Stage 4)

**Configuration** (from code defaults):
- `WINDOW_SEC`: 3.0 seconds
- `STEP_SEC`: 0.1 seconds (100ms sliding step)
- `MIN_SEPARATION_SEC`: 1.5 seconds (minimum temporal separation between clips)
- `CLIPPING_CEILING`: 0.99 (target peak after correction)
- `DEFAULT_SR`: 16000 Hz

**Algorithm** (Diversity-Aware Selection):

1. **Sliding Window Analysis**:
   - 3s window with 100ms step
   - Compute RMS for each window position

2. **Candidate Selection**:
   - Threshold: RMS ≥ 0.001 (filters near-silence)
   - Rank by RMS energy (descending)

3. **Diversity Enforcement**:
   - Greedy selection with 1.5s minimum temporal separation
   - Prevents near-duplicate clips from adjacent windows
   - Guarantees at least one clip per recording (fallback to highest-energy)

4. **Per-Recording Rules**:
   - Duration < 3s: Skip (no extraction)
   - Duration 3-12s: Extract 1 clip from beginning
   - Duration > 12s: Skip first 3s (avoid voice annotations), extract 1 clip

5. **Clipping Correction**:
   - Detect: Peak ≥ 0.9999
   - Correct: Peak scaling + soft limiting (α = 5.0)

**Usage**:
```bash
# Full-retention mode (for Stage 6 balancing) - RECOMMENDED
python Stage5_extract_wav_from_flac.py --no-quarantine

# RMS-filtered mode (apply max-clips cap immediately)
python Stage5_extract_wav_from_flac.py --max-clips 25000

# Custom threshold
python Stage5_extract_wav_from_flac.py --threshold 0.002 --no-quarantine
```

**Operating Modes**:
- **Full-retention** (`--no-quarantine`): Keep all clips for Stage 6 balancing
- **RMS-filtered**: Select top N clips by RMS energy immediately

**Outputs**:
- `Stage5out_unique_3sclips.csv` (38,453 clips with metadata)
- WAV files in `POSITIVE_STAGING_DIR` (flat structure, no subdirectories)

**CSV Additions** (extends Stage 4 metadata):
- `onset_ms` - Clip start time in milliseconds
- `clip_filename` - Output filename

**Filename format**: `xc{id}_{quality}_{onset_ms}.wav` (e.g., `xc422286_A_3000.wav`)

**Expected Results**:
- Total clips: 38,453 (one per unique FLAC)
- Format: 16-bit PCM WAV, 16kHz mono
- With clipping correction where needed

**Estimated Time**: 20-40 minutes

---

### Stage 6: Balance Species with Acoustic Diversity

**Script**: `Stage6_balance_species.py`

**Purpose**: Species-level balancing using acoustic salience and within-species clustering

**Input**: `Stage5out_unique_3sclips.csv` (from Stage 5)

**Configuration** (from `config.py`):
- `STAGE6_MAX_CLIPS`: 25,000 (default target size)
- `N_CLUSTERS_PER_SPECIES`: 5 (acoustic clusters per species)
- `DIVERSITY_BONUS`: 100.0 (priority bonus for new acoustic clusters)

**Algorithm** (Acoustic Diversity-Aware Undersampling):

**Phase 1: Acoustic Salience Computation**
- Spectral contrast (foreground prominence) + spectral centroid (energy distribution)
- Formula: `salience = 0.7 × (mean_contrast / 40.0) + 0.3 × (mean_centroid / fs)`
- Prioritizes clear foreground vocalizations over noisy segments

**Phase 2: Per-Species Capping with Clustering**
- MiniBatch K-Means clustering on mel-spectrogram embeddings
- 5 clusters per species (distinct call types: songs vs. calls)
- Base per-species allocation: `n_base = ⌊target_size / num_species⌋`
- Selection strategy:
  - If species has ≤ n_base clips: Keep all
  - If species has > n_base clips: Select one per acoustic cluster (up to 5), then fill with highest-salience

**Phase 3: Priority Queue Backfilling**
- Score formula: `score = salience + quality_bonus + diversity_bonus`
- Diversity bonus (+100) rewards clips from new acoustic clusters
- Max-heap for O(log n) efficiency
- Fill to exactly 25,000 clips

**Phase 4: Global Trimming** (if total > target)
- Salience-based final adjustment

**Usage**:
```bash
# Use defaults from config.py (25,000 clips)
python Stage6_balance_species.py

# Custom target size
python Stage6_balance_species.py --target-size 50000

# Dry run (preview without moving files or saving CSV)
python Stage6_balance_species.py --dry-run

# Custom cluster count
python Stage6_balance_species.py --clusters-per-species 8
```

**Workflow**:
1. Read clips from `POSITIVE_STAGING_DIR`
2. Apply balancing algorithm
3. Move selected files to `POSITIVE_FINAL_DIR`
4. Save metadata CSV

**Outputs**:
- `Stage6out_balanced_clips.csv` (25,000 balanced clips)
- `species_balance.png` (pre/post distribution plot)
- 25,000 WAV files moved to `POSITIVE_FINAL_DIR`

**Expected Results** (from latest run):
- Input: 38,453 clips, 1,677 species
- Pre-balancing Gini: 0.601, mean 22.9 clips/species
- Post-balancing Gini: 0.519 (13.6% reduction), mean 14.9 clips/species
- Unique acoustic clusters: 3,553
- Mean salience improvement: 0.367 → 0.378 (+3%)
- Quality A+B: 92.1%

**Estimated Time**: 5-10 minutes

---

## CSV Data Flow

```
Stage 1: Stage1out_xc_bird_metadata.csv (42,000 records)
           ├─> id, en, rec, cnt, lat, lon, file, lic, q, length, smp
           ↓
Stage 2: (Analysis only, no CSV output)
           ↓
Stage 3: Stage3out_successful_conversions.csv (38,466 conversions)
           ├─> Same as Stage1 + conversion metadata
           ↓
Stage 4: Stage4out_unique_flacs.csv (38,453 unique, -13 duplicates)
           ├─> Same as Stage3 (deduplicated)
           ↓
Stage 5: Stage5out_unique_3sclips.csv (38,453 clips)
           ├─> All Stage4 fields + onset_ms, clip_filename
           ↓
Stage 6: Stage6out_balanced_clips.csv (25,000 balanced)
           └─> Same as Stage5 (balanced subset)
```

---

## Output Directory Structure

```
/Volumes/Evo/SEABAD/
├── asean-flacs/                          # Stage 3 output (38,453 FLACs)
│   ├── Species_Name_1/
│   │   ├── xc123456_A.flac
│   │   └── xc123457_B.flac
│   ├── Species_Name_2/
│   │   └── xc123458_A.flac
│   ├── perfect_duplicates/               # Stage 4 quarantine (8 files)
│   └── near_duplicates/                  # Stage 4 quarantine (5 files)
│
├── positive_staging/                     # Stage 5 output (38,453 WAVs)
│   ├── xc123456_A_0.wav
│   ├── xc123456_A_3000.wav
│   └── xc123457_B_0.wav
│
└── positive/                             # Stage 6 output (25,000 balanced WAVs)
    ├── xc123456_A_0.wav
    ├── xc123457_B_0.wav
    └── ... (25,000 files)
```

**Working Directory** (where scripts are run):
```
positive-label-curation/
├── config.py                            # Centralized configuration
├── Stage1_xc_fetch_bird_metadata.py
├── Stage2_analyze_metadata.py
├── Stage3_download_and_convert.py
├── Stage4_deduplicate_flac.py
├── Stage5_extract_wav_from_flac.py
├── Stage6_balance_species.py
├── Stage1out_xc_bird_metadata.csv
├── Stage3out_successful_conversions.csv
├── Stage3out_failed_downloads.csv
├── Stage3_download_log.csv
├── Stage3_report.txt
├── Stage4out_unique_flacs.csv
├── Stage4_removed_near_duplicates_metadata.csv
├── Stage4_report.txt
├── Stage5out_unique_3sclips.csv
├── Stage6out_balanced_clips.csv
└── species_balance.png
```

---

## Key Parameters Summary

| Parameter | Value | Stage | Purpose |
|-----------|-------|-------|---------|
| Sample Rate | 16 kHz | 3, 5 | Consistent audio processing for edge deployment |
| Clip Duration | 3.0 sec | 5 | Fixed-length segments for CNN input |
| Audio Format | FLAC (3-4), WAV (5-6) | 3-6 | Lossless compression |
| Channels | Mono | 3, 5 | Single-channel audio |
| Window Step | 100 ms | 5 | Sliding window resolution |
| Min Separation | 1.5 sec | 5 | Temporal diversity between clips |
| RMS Threshold | 0.001 | 5 | Silence filtering |
| Target Dataset | 25,000 | 6 | Balanced final dataset size |
| Clusters/Species | 5 | 6 | Acoustic diversity within species |
| Max Year | 2025 | 1 | Temporal filtering |

---

## Performance Notes

### Expected Processing Times (38K files)

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1 | 5-10 min | API-dependent, 42K records |
| Stage 2 | <1 min | Analysis only |
| Stage 3 | 2-6 hours | Network-dependent, 38.5K downloads |
| Stage 4 | 15-30 min | FAISS-accelerated |
| Stage 5 | 20-40 min | Clip extraction with diversity filtering |
| Stage 6 | 5-10 min | Acoustic clustering + balancing |

**Total pipeline runtime**: ~4-8 hours

### Disk Space Requirements

- FLACs (Stage 3): ~25-35 GB
- Staging WAVs (Stage 5): ~6-8 GB
- Final balanced WAVs (Stage 6): ~4 GB
- **Total working space**: ~35-45 GB

### File Counts (Southeast Asia)

- Stage 1: ~42,000 unique records
- Stage 3: 38,466 successful conversions, 4 failed
- Stage 4: 38,453 unique FLACs (13 duplicates removed)
- Stage 5: 38,453 clips (one per recording)
- Stage 6: 25,000 balanced clips

---

## Troubleshooting

### Stage 3: Download Failures

**Issue**: Failed downloads with 404 or 500 errors

**Solution**:
- Check `Stage3out_failed_downloads.csv` for details
- Xeno-Canto occasionally removes recordings or has server issues
- 4 failed downloads out of 38,466 is expected (~0.01%)

### Stage 4: FAISS Installation Issues

**Issue**: `ImportError: No module named 'faiss'`

**Solution**:
```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu

# macOS with Apple Silicon (use conda)
conda install -c pytorch faiss-cpu
```

### Stage 5: Insufficient Clips

**Issue**: Fewer clips than expected

**Solution**:
- Check that `--no-quarantine` flag is used (full-retention mode)
- Verify FLACs exist in `FLAC_OUTPUT_DIR`
- Many recordings produce only one clip (1 per recording is normal)

### Stage 6: Files Not Moving

**Issue**: WAVs not appearing in `positive/` directory

**Solution**:
- Check that Stage 5 output is in `positive_staging/`
- Verify `--dry-run` flag is NOT used
- Ensure sufficient disk space

---

## Configuration

All pipeline parameters are centralized in `config.py`:

### Dataset Directories

```python
DATASET_ROOT = "/Volumes/Evo/SEABAD"
FLAC_OUTPUT_DIR = os.path.join(DATASET_ROOT, "asean-flacs")
POSITIVE_STAGING_DIR = os.path.join(DATASET_ROOT, "positive_staging")
POSITIVE_FINAL_DIR = os.path.join(DATASET_ROOT, "positive")
```

### CSV Flow

```python
# Stage outputs automatically flow to next stage inputs
STAGE1_OUTPUT_CSV = "Stage1out_xc_bird_metadata.csv"
STAGE3_OUTPUT_CSV = "Stage3out_successful_conversions.csv"
STAGE4_OUTPUT_CSV = "Stage4out_unique_flacs.csv"
STAGE5_OUTPUT_CSV = "Stage5out_unique_3sclips.csv"
STAGE6_OUTPUT_CSV = "Stage6out_balanced_clips.csv"
```

### Audio Processing

```python
TARGET_SAMPLE_RATE = 16000  # 16kHz
TARGET_CHANNELS = 1         # mono
TARGET_FORMAT = "flac"
TARGET_SUBTYPE = "PCM_16"
```

### Species Filtering

```python
EXCLUDE_SPECIES = [
    "Identity unknown",
    "identity unknown",
    "Unknown",
    "unknown"
]
```

---

## Advanced Usage

### Different Geographic Regions

Edit countries in `config.py` or `Stage1_xc_fetch_bird_metadata.py`:

```python
COUNTRIES = {
    "United States": "US",
    "Canada": "CA",
    "Mexico": "MX"
}
```

### Quality Filtering

Edit `Stage3_download_and_convert.py` to only download A/B quality:

```python
# Only download A and B quality
if q_char not in ['A', 'B']:
    logger.info(f"Skipping quality {q_char}")
    continue
```

### Custom Clip Duration

Edit `Stage5_extract_wav_from_flac.py`:

```python
WINDOW_SEC = 5.0  # 5-second clips instead of 3
```

### Species-Specific Dataset

Filter `Stage1out_xc_bird_metadata.csv` before running Stage 3:

```python
import pandas as pd

df = pd.read_csv('Stage1out_xc_bird_metadata.csv')
df_filtered = df[df['en'].isin(['Oriental Magpie-Robin', 'Common Tailorbird'])]
df_filtered.to_csv('Stage1out_xc_bird_metadata.csv', index=False)
```

---

## Implementation Details for Reproducibility

### Filename Conventions

| Stage | Format | Example |
|-------|--------|---------|
| Stage 3 FLAC | `xc{id}_{quality}.flac` | `xc422286_A.flac` |
| Stage 5 WAV | `xc{id}_{quality}_{onset_ms}.wav` | `xc422286_A_3000.wav` |

### CSV Column Evolution

**Stage 1 → Stage 3**:
- Core: `id, en, rec, cnt, lat, lon, lic, q, length, smp`

**Stage 4** (adds):
- Deduplication metadata

**Stage 5** (adds):
- `onset_ms`: Clip start time in milliseconds
- `clip_filename`: Output WAV filename

**Stage 6** (adds):
- `acoustic_salience`: Computed salience score (0-1)
- `cluster_id`: Within-species acoustic cluster assignment
- `quality_score`: Numeric quality score (A=4, B=3, C=2, D=1)

### Gini Coefficient Computation

Used in Stage 6 to quantify species distribution inequality:

```
G = (2 × Σ(i × n_i)) / (S × Σn_i) - (S+1)/S
```

where:
- `n_i` = sample count for i-th species (sorted ascending)
- `S` = total number of species
- `G ∈ [0, 1]`: 0 = perfect equality, 1 = maximum inequality

**SEABAD Results**:
- Pre-balancing: G = 0.601
- Post-balancing: G = 0.519 (13.6% reduction)

---

## Citation

If you use this pipeline or SEABAD dataset, please cite:

```
SEABAD: Southeast Asian Bird Audio Detection Dataset
Xeno-canto: Bird sounds from around the world. https://www.xeno-canto.org/
```

**Dataset Paper** (in preparation):
- 50,000 clips (25,000 positive, 25,000 negative)
- 1,677 Southeast Asian bird species
- Diversity-aware curation with acoustic clustering
- Optimized for edge deployment (16kHz mono)

---

## License

This pipeline is provided for research and educational purposes. Please respect:
- Xeno-Canto's terms of service
- Individual Creative Commons licenses of each recording
- Proper attribution to recordists

All Xeno-Canto recordings retain their original licenses (CC BY, CC BY-SA, CC BY-NC, etc.).

---

## Contributing

Contributions welcome! Please open issues for:
- Bugs or errors
- Feature requests
- Documentation improvements
- Regional adaptations

---

## Repository

https://github.com/mun3im/seabad/tree/main/positive-label-curation

---

## Support

For questions or issues, please open a GitHub issue with:
- Stage where error occurred
- Error message or unexpected behavior
- Relevant CSV/log files
- System information (OS, Python version, ffmpeg version)
