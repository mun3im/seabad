# Stage4 Duplicate Detection Algorithm Documentation

## Overview

`Stage4_deduplicate_flac.py` detects near-duplicate and perfect duplicate audio clips using FAISS-accelerated acoustic similarity metrics. The script intelligently handles duplicates based on recorder metadata, moves files to separate directories, and generates comprehensive reports for manual review.

**Key Changes:**
- All metadata files (CSVs and reports) now stored in `metadata/` directory
- Uses centralized `config.py` for all paths and parameters
- Integrates with SEABAD pipeline Stage 3 → Stage 4 → Stage 5 flow

## Core Components

### 1. Audio Embedding (`AudioEmbedder`)

Converts audio files into normalized mel-spectrogram embeddings for comparison.

**Process:**
1. **Load audio**: Uses librosa with 16kHz sample rate, mono, kaiser_fast resampling
2. **Normalize length**: Pad or trim to exactly 3.0 seconds (48,000 samples)
3. **Compute mel-spectrogram**:
   - 128 mel bins
   - 512-sample FFT
   - 128-sample hop length
   - Max frequency: 8kHz
4. **Convert to log scale**: `librosa.power_to_db(mel, ref=np.max)`
5. **L2 normalize frames**: Per-frame normalization for cosine similarity

**Output:** `(128, n_frames)` array of normalized log-mel features

### 2. Similarity Calculation (`SimilarityCalculator`)

Computes frame-wise cosine similarity between two embeddings.

**Metrics:**
- **Mean similarity**: Average cosine similarity across all frames
- **Min similarity**: Minimum cosine similarity across all frames
- **5th percentile similarity**: 5th percentile of frame similarities (robust to outliers)

**Formula:**
```python
sims = np.sum(emb1[:, :min_frames] * emb2[:, :min_frames], axis=0)
# Clipped to [-1.0, 1.0] range
```

**Thresholds for "similar" classification:**
- Mean similarity ≥ 0.997
- Min similarity ≥ 0.985
- 5th percentile similarity ≥ 0.992

**Perfect duplicate threshold:** ≥ 0.999 (mean similarity)

### 3. FAISS-Accelerated Duplicate Detection (`DuplicateFinder`)

Uses FAISS IndexFlatIP for efficient nearest neighbor search instead of naive O(n²) comparison.

**Process:**
1. **Compute embeddings**: Generate framewise (128 x T) mel-spectrograms for all files
2. **Create clip-level descriptors**: Convert to fixed-size vectors using mean+std pooling
3. **Build FAISS index**: Add normalized vectors to IndexFlatIP (inner product = cosine similarity)
4. **Search k-nearest neighbors**: Find top-k most similar clips for each file
5. **Verify candidates**: Perform detailed frame-wise comparison on candidates
6. **Categorize pairs**:
   - **Perfect duplicates**:
     - Embeddings match exactly (`np.allclose` with atol=1e-7), OR
     - Mean similarity ≥ 0.999
   - **Near-duplicates**: Meets all three similarity thresholds

**Complexity:** O(n log n) with FAISS indexing (vs O(n²) for naive comparison)

**Optimization:** Two-stage approach:
1. Fast FAISS search to find candidates (top-k=6 by default)
2. Expensive frame-wise verification only on promising pairs

### 4. Metadata Management (`MetadataManager`)

Manages CSV metadata for tracking which files are removed and which are kept.

**Features:**
- Loads metadata into memory indexed by XC ID
- Removes entries for deleted files
- Tracks removed near-duplicate metadata separately for restoration
- Saves updated metadata and removed metadata to separate CSVs

**Key Methods:**
- `remove_metadata(xc_ids, track_removed=False)`: Remove entries, optionally track for restoration
- `save_metadata(output_csv)`: Save remaining metadata
- `save_removed_metadata(output_csv)`: Save tracked removals for potential restoration

### 5. Duplicate Analysis (`DuplicateAnalyzer`)

Analyzes technical differences between audio files for reporting.

**Metrics Computed:**
- Sample rate comparison
- Duration comparison
- Channel count comparison
- RMS energy difference
- Maximum amplitude difference
- Waveform identity check (detects different encodings of same content)

**Purpose:** Helps understand why files are flagged as near-duplicates and aids manual review decisions.

### 6. Intelligent Duplicate Handling (`QuarantineManager`)

Manages file movements and decision logic based on recorder metadata.

**Perfect Duplicates Logic:**

For each perfect duplicate pair:
1. **Check recorder field** from metadata
2. **If same recorder**:
   - Keep NEWER file (higher XC number)
   - Remove OLDER file
   - Reason: Recorder may have updated metadata
3. **If different recorders**:
   - Keep OLDER file (original)
   - Remove NEWER file
   - Tag as **PLAGIARIZED UPLOAD**
4. Move removed file to `perfect_duplicates/` subdirectory
5. Delete metadata entry (permanent)

**Near-Duplicates Logic:**

For each near-duplicate pair:
1. **Apply same recorder logic** as perfect duplicates
2. Move ONLY ONE file to `near_duplicates/` subdirectory
3. Keep the other file in main dataset
4. Save removed file's metadata to separate CSV for potential restoration
5. Analyze technical differences for report

**Directory Structure:**
```
project_root/
├── perfect_duplicates/
│   ├── Species_A/
│   │   └── xc12346_A.flac  (removed: newer duplicate)
│   └── Species_B/
│       └── xc67891_B.flac  (removed: plagiarized)
├── near_duplicates/
│   ├── Species_A/
│   │   └── xc12348_A.flac  (removed: for manual review)
│   └── Species_C/
│       └── xc99999_C.flac  (removed: for manual review)
├── Species_A/
│   ├── xc12345_B.flac  (kept: older original)
│   └── xc12347_A.flac  (kept: after near-dup removal)
└── Species_B/
    └── xc67890_A.flac  (kept: original)
```

### 7. Report Generation

Generates comprehensive `Stage4_report.txt` with detailed analysis.

**Report Sections:**

**A. Perfect Duplicates:**
- Lists all pairs with XC numbers
- Shows recorder information for both files
- Indicates which file was KEPT and which was REMOVED
- Explains reasoning (same recorder update OR plagiarism detection)

**B. Near Duplicates:**
- Lists all pairs with similarity scores
- Shows which file was KEPT and which was REMOVED
- Includes technical analysis:
  - Sample rate differences
  - Duration differences
  - Channel count differences
  - RMS energy differences
  - Waveform identity status
- Notes that removed files can be restored if manual review justifies it

## Algorithm Flow

```
1. Load metadata CSV (if provided)
   ↓
2. Collect audio files (.wav, .flac)
   ↓
3. Compute embeddings for all files
   ↓
4. Build FAISS index with clip-level descriptors
   ↓
5. Search k-nearest neighbors for each clip
   ↓
6. Verify candidates with frame-wise similarity
   ↓
7. Categorize pairs:
   - Perfect duplicates (≥ 0.999 or exact match)
   - Near-duplicates (≥ thresholds)
   ↓
8. Handle perfect duplicates:
   - Apply recorder-based logic
   - Move to perfect_duplicates/
   - Delete metadata permanently
   ↓
9. Handle near duplicates:
   - Apply same recorder-based logic
   - Move ONE file to near_duplicates/
   - Save metadata separately for restoration
   ↓
10. Generate detailed report (Stage4_report.txt)
    ↓
11. Save updated metadata CSV
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `TARGET_SR` | 16000 Hz | Sample rate for audio loading |
| `TARGET_DURATION` | 3.0 sec | Fixed clip duration |
| `N_MELS` | 128 | Number of mel frequency bins |
| `HOP_LENGTH` | 128 samples | Hop length for STFT |
| `MIN_DURATION` | 3.0 sec | Minimum duration filter |
| `MEAN_SIM_THRESHOLD` | 0.997 | Mean similarity for near-duplicates |
| `MIN_SIM_THRESHOLD` | 0.985 | Min similarity for near-duplicates |
| `P5_SIM_THRESHOLD` | 0.992 | 5th percentile for near-duplicates |
| `PERFECT_DUPLICATE_THRESHOLD` | 0.999 | Perfect duplicate threshold |
| `top_k` | 6 | Number of neighbors to check per file |

## Recorder-Based Decision Logic

### Same Recorder

```
XC12345 (Recorder: John Doe) ←→ XC12346 (Recorder: John Doe)
Decision: Keep XC12346 (newer), remove XC12345 (older)
Reasoning: Recorder may have uploaded updated version with better metadata
```

### Different Recorders (Plagiarism Detection)

```
XC12345 (Recorder: John Doe, 2020) ←→ XC67890 (Recorder: Jane Smith, 2023)
Decision: Keep XC12345 (older), remove XC67890 (newer)
Reasoning: Likely plagiarized upload - newer recording from different person
Tag: PLAGIARIZED UPLOAD
```

## Metadata Handling

### Perfect Duplicates
- Metadata deleted **permanently**
- Not saved separately
- Assumed to be confirmed duplicates requiring no review

### Near Duplicates
- Metadata saved to `metadata/Stage4_removed_near_duplicates_metadata.csv`
- Can be restored if manual review shows file should be kept
- Provides audit trail and recovery option

### Restoration Process

If manual review determines a near-duplicate should be restored:

1. **Restore file:**
   ```bash
   mv near_duplicates/Species_A/xc12348_A.flac Species_A/
   ```

2. **Restore metadata:**
   - Open `metadata/Stage4_removed_near_duplicates_metadata.csv`
   - Find row with xc_id=12348
   - Copy row to main metadata CSV

## Output Files

| File | Description | Can be deleted? |
|------|-------------|-----------------|
| `metadata/Stage4out_unique_flacs.csv` | Main metadata without removed files | No - this is the new master |
| `metadata/Stage4_removed_near_duplicates_metadata.csv` | Backup metadata for near-duplicates | After manual review |
| `metadata/Stage4_report.txt` | Detailed analysis report | After review, keep for records |
| `perfect_duplicates/` | Confirmed duplicate files | After verification |
| `near_duplicates/` | Files for manual review | After review decision |

**Note:** All CSV and report files are automatically saved to the `metadata/` directory per `config.py` configuration.

## Usage

**Default usage (uses config.py defaults):**
```bash
# Uses Stage3 output automatically
python Stage4_deduplicate_flac.py
```

**With quarantine options:**
```bash
# Dry run (preview changes only)
python Stage4_deduplicate_flac.py

# Quarantine only perfect duplicates
python Stage4_deduplicate_flac.py --quarantine-perfect

# Quarantine both perfect and near-duplicates (recommended)
python Stage4_deduplicate_flac.py --quarantine-all
```

**With custom paths:**
```bash
python Stage4_deduplicate_flac.py /path/to/flac/directory \
  --metadata metadata/Stage3out_successful_conversions.csv \
  --output-metadata metadata/Stage4out_unique_flacs.csv \
  --removed-metadata metadata/Stage4_removed_near_duplicates_metadata.csv \
  --report metadata/Stage4_report.txt
```

**Configuration (config.py):**
```python
FLAC_OUTPUT_DIR = "/Volumes/Evo/SEABAD/asean-flacs"
STAGE4_INPUT_CSV = "metadata/Stage3out_successful_conversions.csv"
STAGE4_OUTPUT_CSV = "metadata/Stage4out_unique_flacs.csv"
STAGE4_REMOVED_CSV = "metadata/Stage4_removed_near_duplicates_metadata.csv"
STAGE4_REPORT_TXT = "metadata/Stage4_report.txt"
```

**Options:**
- `--recursive` / `-r`: Search all subdirectories (not just one level deep)
- `--metadata`: Path to metadata CSV file (default: from config.py)
- `--output-metadata`: Custom path for updated metadata (default: `metadata/Stage4out_unique_flacs.csv`)
- `--removed-metadata`: Custom path for removed near-duplicate metadata (default: `metadata/Stage4_removed_near_duplicates_metadata.csv`)
- `--report`: Custom path for report (default: `metadata/Stage4_report.txt`)
- `--quarantine-all`: Move both perfect and near-duplicates to quarantine directories
- `--quarantine-perfect`: Move only perfect duplicates to quarantine
- `--debug-mode`: Display metadata field examples and exit (for debugging)

## Performance Characteristics

### Time Complexity
- **FAISS indexing:** O(n log n) where n = number of files
- **Candidate search:** O(n · k · log n) where k = top_k neighbors
- **Frame-wise verification:** O(c · f) where c = candidates, f = frames per file
- **Overall:** O(n log n) dominated by FAISS operations

### Space Complexity
- **Embeddings:** O(n · 128 · T) where T = number of frames (~375 for 3s audio)
- **FAISS index:** O(n · 256) for mean+std pooled descriptors
- **Metadata:** O(n · m) where m = average metadata size

### Typical Performance
- **1,000 files:** ~2-5 minutes
- **10,000 files:** ~20-40 minutes
- **100,000 files:** ~4-8 hours (estimated)

## Implementation Notes

1. **FAISS acceleration**: Dramatically faster than naive O(n²) comparison; 100x speedup for 10k+ files

2. **Two-stage verification**: Fast approximate search (FAISS) + accurate frame-wise verification (librosa)

3. **Metadata safety**: All deletions tracked separately for near-duplicates; perfect duplicates assumed safe to delete

4. **Quarantine safety**: Uses `shutil.move()` which preserves file atomicity; subfolder structure maintained

5. **Error handling**: Skips files that fail to load and logs errors to stderr

6. **Progress tracking**: Uses `tqdm` for progress bars during embedding and comparison phases

7. **Plagiarism detection**: Flags potential content theft when different recorders upload identical recordings

## Known Limitations

1. **Fixed duration**: Only works with 3-second clips (by design for this pipeline)
2. **No transposition detection**: Does not detect pitch-shifted duplicates
3. **No time-stretch detection**: Does not detect time-stretched duplicates
4. **Memory bound**: Loads all embeddings into RAM (acceptable for <100k files)
5. **Metadata dependency**: Recorder logic requires accurate metadata; missing recorder info treated as empty string

## Validation and Quality Assurance

**Before running:**
- Verify metadata CSV has `xc_id` and `recorder` columns
- Check that audio files exist at paths referenced in metadata
- Ensure sufficient disk space for duplicate subdirectories

**After running:**
- Review `Stage4_report.txt` for plagiarism flags
- Verify perfect duplicate counts seem reasonable
- Manually review a sample of near-duplicates
- Check that file counts match: `original - removed = updated_metadata_rows`

**Red flags:**
- Large number of "plagiarized" uploads from reputable recorders (may indicate metadata error)
- Near-duplicates with very high similarity (>0.998) but kept as separate
- Missing metadata for many XC IDs

## Future Improvements

1. ✅ ~~Implement FAISS for O(n log n) complexity~~ (Completed)
2. Add web interface for manual review of near-duplicates
3. Implement pitch-invariant and tempo-invariant similarity metrics
4. Add parallel processing for embedding computation (multi-GPU)
5. Add automatic restoration rules based on similarity score thresholds
6. Implement incremental updates (only process new files)
7. Add audio quality metrics to decision logic (prefer higher quality)
8. Export flagged plagiarism cases for reporting to data source

## Mathematical Details

### Clip-Level Embedding

Framewise embedding: **E** ∈ ℝ^(128 × T)

Clip descriptor: **d** = [μ, σ] where:
- μ = mean(E, axis=1) ∈ ℝ^128
- σ = std(E, axis=1) ∈ ℝ^128
- **d** ∈ ℝ^256

### Cosine Similarity

After L2 normalization:
```
similarity(d₁, d₂) = d₁ · d₂ = Σᵢ d₁ᵢ × d₂ᵢ
```

FAISS IndexFlatIP computes inner product, which equals cosine similarity after normalization.

### Frame-wise Verification

For embeddings E₁, E₂:
```
sims[t] = Σᵢ E₁[i,t] × E₂[i,t]  for t ∈ [0, min(T₁, T₂))

mean_sim = mean(sims)
min_sim = min(sims)
p5_sim = percentile(sims, 5)
```

A pair is "similar" iff:
```
mean_sim ≥ 0.997 AND min_sim ≥ 0.985 AND p5_sim ≥ 0.992
```

A pair is a "perfect duplicate" iff:
```
mean_sim ≥ 0.999 OR allclose(E₁, E₂, atol=1e-7)
```
