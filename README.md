# SEABAD: Southeast Asian Bird Audio Detection Dataset

A dataset of **50,000 automatically curated 3-second clips** spanning **1,677 Southeast Asian bird species**, standardized to 16 kHz mono for binary bird presence–absence detection on edge devices.

> 📦 **Dataset**: [zenodo.org/records/18290494](https://zenodo.org/records/18290494)
> 📄 **Paper**: *SEABAD: A Tropical Bird Audio Detection Dataset for Passive Acoustic Monitoring* (2025)
> 💻 **Code**: [github.com/mun3im/seabad](https://github.com/mun3im/seabad)

---

## Overview

Passive acoustic monitoring (PAM) enables large-scale biodiversity assessment, but most recordings contain non-informative audio. Bird audio detection (BAD)—determining bird presence without species classification—can suppress non-target recordings on-device, extending deployment and reducing annotation burden.

SEABAD addresses critical gaps in existing datasets:
- ✅ **Tropical soundscape coverage** (Southeast Asia)
- ✅ **3-second clip length** matching edge-AI inference windows
- ✅ **Diversity-aware species balancing** (1,677 species, Gini 0.519)
- ✅ **Multi-source negative curation** (environmental + field recordings)

| Property | Value |
|---|---|
| Total clips | 50,000 |
| Positive (bird present) | 25,000 |
| Negative (bird absent) | 25,000 |
| Unique bird species | 1,677 |
| Clip duration | 3 seconds |
| Sample rate | 16 kHz mono |
| Bit depth | 16-bit PCM |
| Geography | Indonesia, Malaysia, Thailand, Singapore, Brunei |
| Train / Val / Test split | 40k / 5k / 5k (80% / 10% / 10%) |

---

## 🎯 Baseline Validation Results

Standard CNN architectures (ImageNet pre-trained, fine-tuned on SEABAD) averaged across **3 random seeds** (42, 100, 786):

| Model | Params | Accuracy | AUC | Precision | Recall | F1 |
|-------|--------|----------|-----|-----------|--------|-----|
| **MobileNetV3-Small**† | **1.1M** | **99.57 ± 0.25%** | **0.9985 ± 0.0002** | 0.9956 ± 0.0012 | 0.9957 ± 0.0008 | 0.9957 ± 0.0025 |
| EfficientNetB0 | 4.4M | 99.49 ± 0.23% | 0.9991 ± 0.0004 | 0.9959 ± 0.0018 | 0.9939 ± 0.0051 | 0.9949 ± 0.0023 |
| VGG16 | 14.9M | 99.61 ± 0.03% | 0.9995 ± 0.0001 | 0.9960 ± 0.0014 | 0.9963 ± 0.0010 | 0.9961 ± 0.0025 |
| ResNet50 | 24.2M | 99.73 ± 0.02% | 0.9992 ± 0.0003 | 0.9965 ± 0.0013 | 0.9980 ± 0.0012 | 0.9973 ± 0.0019 |

**†Primary baseline for edge deployment**

### Key Findings
- ✅ All models achieve **>99.4% accuracy** with minimal variance (std <0.25%)
- ✅ **MobileNetV3-Small** is only 0.16% behind ResNet50 but has **22× fewer parameters**
- ✅ Excellent **training stability** confirmed across diverse architectures and random seeds
- ✅ High **dataset quality** validated by consistent performance

See [`validation/README.md`](validation/README.md) for detailed experimental setup and analysis.

---

## 📂 Repository Structure

```
seabad/
├── positive-label-curation/   # Stages 1–9: Xeno-Canto bird clips
│   ├── Stage1_xc_fetch_bird_metadata.py
│   ├── Stage2_analyze_metadata.py
│   ├── Stage3_download_and_convert.py
│   ├── Stage4_deduplicate_flac.py
│   ├── Stage5_extract_wav_from_flac.py
│   ├── Stage6_balance_species.py
│   ├── Stage7_qa_spectrograms.py
│   ├── Stage8_adjust_onset.py
│   └── Stage9_qa_apply_corrections.py
│
├── negative-sample-curation/  # Stages 1–6: Non-bird clips
│   ├── Stage1_extract_birdvox.py
│   ├── Stage2_extract_freefield.py
│   ├── Stage3_extract_warblr.py
│   ├── Stage4_extract_fsc22.py
│   ├── Stage5_extract_esc50.py
│   └── Stage6_extract_datasec.py
│
└── validation/                # CNN baseline training & evaluation
    ├── validate_seabad_pretrained.py
    ├── run_all_cnn.sh
    ├── utils.py
    └── README.md
```

---

## 🔧 Curation Pipeline

### Six-Stage Methodology

1. **Metadata Acquisition** — Xeno-Canto API query for Southeast Asian bird species
2. **Sound Acquisition** — Download + convert to 16 kHz mono FLAC
3. **Acoustic Deduplication** — FAISS approximate nearest-neighbor on mel embeddings
4. **Segment Extraction** — RMS-based sliding window with minimum separation
5. **Species Balancing** — Diversity-aware balancing using MiniBatch K-Means + salience ranking
6. **Quality Assurance** — Manual audit with interactive correction tool

### Positive Curation Highlights

- **Acoustic deduplication**: FAISS cosine similarity on mel-spectrogram embeddings
  - 13 near-duplicates identified and removed (0.03% of 38,481 recordings)
- **Diversity-aware segment extraction**:
  - RMS sliding window with 1.5s minimum separation
  - One representative clip per source recording
- **Species balancing**:
  - MiniBatch K-Means (5 clusters/species) + salience-ranked priority queue
  - Gini coefficient reduced from **0.601 → 0.519** (13.7% reduction)
  - Mean samples/species: 14.9 (preserves all 1,677 species)
- **Quality assurance**:
  - 1,000 clips manually audited (Cochran n=639 for 95% CI)
  - 97.8% accuracy (22 corrections: 15 onset, 6 noise, 1 false positive)
  - 92.1% rated quality A/B by Xeno-Canto community

### Negative Curation Highlights

Sources totaling **25,000 bird-absent clips**:

| Dataset | Clips | Description |
|---------|-------|-------------|
| BirdVox-DCASE-20k | 9,983 | Northeast USA field recordings |
| Freefield1010 | 5,755 | Global crowdsourced environmental audio |
| Warblr | 1,950 | UK field recordings |
| FSC-22 | 1,875 | Forest sounds (mammals, insects, weather) |
| ESC-50 | 444 | Urban/mechanical/human sounds (high RMS) |
| DataSEC | 3,597 | Mediterranean soundscapes |

All clips:
- Resampled to 16 kHz mono
- Centered 3-second extraction
- No normalization (preserves naturalistic amplitude distribution)
- Avian classes excluded from general sound datasets

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python dependencies
pip install pandas requests librosa soundfile tqdm scikit-learn matplotlib faiss-cpu numpy tensorflow

# System dependencies
# macOS: brew install ffmpeg
# Linux: apt-get install ffmpeg
```

For GPU FAISS acceleration: replace `faiss-cpu` with `faiss-gpu`.

---

### 1️⃣ Run Positive Pipeline

```bash
cd positive-label-curation

# Stage 1: Fetch metadata from Xeno-Canto API (~10 min)
python Stage1_xc_fetch_bird_metadata.py --country all

# Stage 2: Analyze metadata (optional statistics)
python Stage2_analyze_metadata.py

# Stage 3: Download and convert to FLAC (~2-6 hours)
python Stage3_download_and_convert.py

# Stage 4: Deduplicate using FAISS (~30 min)
python Stage4_deduplicate_flac.py --quarantine-all

# Stage 5: Extract 3s clips from FLAC (~1 hour)
python Stage5_extract_wav_from_flac.py --no-quarantine

# Stage 6: Balance species distribution (~15 min)
python Stage6_balance_species.py

# Stage 7: Generate QA spectrograms for manual review
python Stage7_qa_spectrograms.py

# Stage 8: Interactive onset correction tool (optional)
python Stage8_adjust_onset.py

# Stage 9: Apply corrections from QA
python Stage9_qa_apply_corrections.py
```

---

### 2️⃣ Run Negative Pipeline

```bash
cd negative-sample-curation

# Extract from DCASE 2018 datasets
python Stage1_extract_birdvox.py
python Stage2_extract_freefield.py
python Stage3_extract_warblr.py

# Extract from environmental sound datasets
python Stage4_extract_fsc22.py
python Stage5_extract_esc50.py
python Stage6_extract_datasec.py
```

**Note**: Ensure source datasets are downloaded and paths are configured in each script.

---

### 3️⃣ Run Baseline Validation

```bash
cd validation

# Train single model
python validate_seabad_pretrained.py --model mobilenetv3s --seed 42

# Train all models across all seeds
./run_all_cnn.sh

# Train specific seeds only
./run_all_cnn.sh 42 100
```

**Supported models**: `mobilenetv3s`, `resnet50`, `vgg16`, `efficientnetb0`

See [`validation/README.md`](validation/README.md) for detailed usage and configuration.

---

## 📦 Pre-Compiled Dataset (Zenodo)

Download the complete 50,000-clip dataset:
**[zenodo.org/records/18290494](https://zenodo.org/records/18290494)**

### Includes:
- ✅ 50,000 × 3-second clips (WAV, 16 kHz mono, 16-bit PCM)
- ✅ Train / validation / test CSVs (80/10/10 stratified split)
- ✅ Full provenance metadata:
  - Xeno-Canto recording IDs
  - GPS coordinates
  - Original licenses (CC BY, CC BY-NC-SA, etc.)
  - Species taxonomy (IOC World Bird List)
  - Quality ratings
  - Source dataset attribution (for negative samples)

---

## 🎛️ Audio Processing Standards

All clips standardized to:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 16,000 Hz | AudioMoth default, Nyquist covers 0-8 kHz |
| Channels | Mono | Edge deployment memory constraint |
| Duration | 3.0 seconds | BirdNET inference window |
| Format | WAV PCM_16 | Lossless, hardware-compatible |
| Normalization | **None** | Preserves naturalistic amplitude for energy-based detection |
| Source format | FLAC PCM_16 | Intermediate lossless storage |

---

## 📊 Dataset Statistics

### Geographic Distribution (Post-Balancing)

| Country | Clips | Percentage |
|---------|-------|------------|
| Indonesia | 9,155 | 36.6% |
| Malaysia | 8,400 | 33.6% |
| Thailand | 5,996 | 24.0% |
| Singapore | 1,388 | 5.6% |
| Brunei | 61 | 0.2% |

### Species Diversity

- **Total species**: 1,677
- **Mean samples per species**: 14.9
- **Gini coefficient**: 0.519 (post-balancing)
- **Acoustic clusters**: 3,553 (via K-Means)
- **Mean cluster salience**: 0.378

### Quality Metrics

- **A/B quality rating**: 92.1% (Xeno-Canto community ratings)
- **Manual QA accuracy**: 97.8% (n=1,000)
- **Deduplication rate**: 0.03% (13/38,481)

---

## 📜 License

### Curation Code
**MIT License** — Free to use, modify, and distribute with attribution.

### Audio Clips
- **Positive samples**: Inherit original Xeno-Canto Creative Commons licenses
  - CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA
  - Full attribution metadata included in dataset
  - See Zenodo for per-recording licenses
- **Negative samples**: Subject to licenses of source datasets:
  - BirdVox, Freefield1010, Warblr (DCASE 2018)
  - FSC-22, ESC-50, DataSEC
  - See source dataset documentation for terms

**Important**: When using SEABAD, you must:
1. Credit original Xeno-Canto recordists (metadata provided)
2. Respect original Creative Commons license terms
3. Cite SEABAD dataset and paper

---

## 📚 Citation

If you use SEABAD or this curation pipeline, please cite:

```bibtex
@article{seabad2025,
  title   = {{SEABAD}: A Tropical Bird Audio Detection Dataset for Passive Acoustic Monitoring},
  author  = {Author Names},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025},
  url     = {https://github.com/mun3im/seabad}
}

@dataset{seabad_zenodo2025,
  title   = {{SEABAD} Dataset: 50,000 Southeast Asian Bird Audio Clips},
  year    = {2025},
  url     = {https://zenodo.org/records/18290494},
  note    = {50,000 curated 3-second clips spanning 1,677 Southeast Asian bird species}
}
```

**Please also credit**:
- [Xeno-Canto](https://www.xeno-canto.org/) and original recordists
- Source datasets for negative samples (BirdVox, Freefield1010, Warblr, FSC-22, ESC-50, DataSEC)

---

## 🤝 Contributing

We welcome contributions to improve the curation pipeline or extend SEABAD to other regions!

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/mun3im/seabad/issues)
- **Pull requests**: Submit improvements to scripts or documentation
- **Regional adaptations**: Contact us if you're adapting the pipeline for other tropical/temperate regions

---

## 🔗 Related Resources

- **Xeno-Canto**: https://www.xeno-canto.org/
- **AudioMoth**: https://www.openacousticdevices.info/
- **BirdNET**: https://birdnet.cornell.edu/
- **DCASE 2018 Task 3**: http://dcase.community/challenge2018/task-bird-audio-detection

---

## 📧 Contact

For questions about SEABAD or the curation methodology:
- Open an issue: [github.com/mun3im/seabad/issues](https://github.com/mun3im/seabad/issues)
- Email: [contact information]

---

**Built with** ❤️ **for tropical biodiversity monitoring**
