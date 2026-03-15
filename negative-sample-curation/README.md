# SEABAD Negative Sample Curation Pipeline

Pipeline for extracting high-quality negative (non-bird) audio samples for the SEABAD bird detection dataset. All outputs are 3-second mono clips at 16 kHz in WAV format.

## Pipeline Overview

Six stages, each targeting one source dataset:

| Stage | Script | Dataset | Output subdir |
|-------|--------|---------|---------------|
| 1 | `Stage1_extract_birdvox.py` | BirdVox-DCASE-20k | `negative/bv/` |
| 2 | `Stage2_extract_freefield.py` | Freefield1010 | `negative/ff/` |
| 3 | `Stage3_extract_warblr.py` | Warblrb10k | `negative/wb/` |
| 4 | `Stage4_extract_fsc22.py` | FSC-22 | `negative/fsc/` |
| 5 | `Stage5_extract_esc50.py` | ESC-50 | `negative/esc/` |
| 6 | `Stage6_extract_datasec.py` | DataSEC | `negative/datasec/` |

All outputs land under:
```
/Volumes/Evo/SEABAD/negative/
├── bv/        # BirdVox-DCASE-20k
├── ff/        # Freefield1010
├── wb/        # Warblrb10k
├── fsc/       # FSC-22
├── esc/       # ESC-50
└── datasec/   # DataSEC
```

## Dataset Selection Rationale

| Dataset | Why chosen |
|---------|------------|
| **BirdVox-DCASE-20k** | Expert-annotated soundscapes from passive monitoring; `hasbird` labels are reliable |
| **Freefield1010** | Diverse outdoor recordings from iNaturalist; same annotation scheme as BirdVox |
| **Warblrb10k** | Crowdsourced soundscape data; adds label noise robustness |
| **FSC-22** | Forest soundscapes beyond birds — insects, wind, water, mammals |
| **ESC-50** | Curated outdoor environmental sounds; broad acoustic diversity |
| **DataSEC** | Environmental + urban + voice + music; fills acoustic gaps and stress-tests the model on hard negatives |

## Stage Details

### Stages 1–3: DCASE-family datasets

Stages 1, 2, and 3 share identical logic via `config.process_dcase_file`:

- Filter rows where `hasbird == 0`
- For each file: find the loudest 3 s window using a 100 ms sliding hop
- Reject if too short (< 3 s) or all-zero
- Run in parallel with `ProcessPoolExecutor`

```bash
python Stage1_extract_birdvox.py   # → negative/bv/bv-<id>-<start_ms>.wav
python Stage2_extract_freefield.py # → negative/ff/ff-<id>-<start_ms>.wav
python Stage3_extract_warblr.py    # → negative/wb/wb-<id>-<start_ms>.wav
```

### Stage 4: FSC-22

```bash
python Stage4_extract_fsc22.py
```

- Excludes class 23 (BirdChirping) and class 24 (WingFlapping)
- Low-amplitude clips are kept; only all-zero / missing / too-short are rejected
- Output: `fsc-<name>-<start_ms>.wav`

### Stage 5: ESC-50

```bash
python Stage5_extract_esc50.py
```

- Excludes categories: `rooster`, `hen`, `crow`, `chirping_birds`
- Loudest 3 s window per file
- Output: `esc-<name>-<start_ms>.wav`

### Stage 6: DataSEC

```bash
python Stage6_extract_datasec.py
```

Targets exactly **3 597 samples** using a priority fill strategy:

1. All core environmental clips (Group 2, ≥ 3 s)
2. Short environmental clips (Group 2, zero-padded to 3 s)
3. Voice clips (Group 3) — added if quota not yet reached
4. Music clips (Group 4, random seed 42) — last resort filler

Bird/animal folders are excluded entirely: `Birds`, `Chicken coop`, `Crows seagulls and magpies`.

Output: `datasec-<folder>-<stem>-<start_ms>.wav` (padded clips get a `_padded` suffix).

## Shared Configuration (`config.py`)

All stages import paths, constants, and utility functions from `config.py`:

```python
from config import (
    DCASE_DATASETS, STAGE1_OUTPUT_DIR,   # paths
    TARGET_SR, CLIP_DURATION,             # audio params
    extract_loudest_3s_clip,              # shared clip extractor
    process_dcase_file,                   # shared DCASE worker
)
```

To change dataset locations or the output root, edit only `config.py`.

## Audio Processing

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 000 Hz |
| Clip length | 3.0 s (48 000 samples) |
| Channels | Mono |
| Format | WAV (PCM) |
| Window hop | 100 ms |

**Rejection criteria (all stages):** file missing · clip too short (< 3 s) · clip is all-zero.
Stage 6 additionally zero-pads short clips rather than discarding them.

## Logs

Each stage writes a log file alongside the script:

| Stage | Log file |
|-------|----------|
| 1 | `Stage1_rejections.log` |
| 2 | `Stage2_rejections.log` |
| 3 | `Stage3_rejections.log` |
| 4 | `Stage4_fsc22.log` |
| 5 | `Stage5_esc50.log` |
| 6 | `Stage6_datasec.log` |

Each log records per-file rejection reasons and a final summary.

## Dependencies

```bash
pip install pandas numpy librosa soundfile tqdm
```

Python 3.8+.

## Dataset Paths

Source datasets are expected at `/Volumes/Evo/datasets/`. To use different locations, update the path constants in `config.py`.

## Citations

```bibtex
@inproceedings{lostanlen2018birdvox,
  title={BirdVox-DCASE-20k: A dataset for bird acoustic activity detection in 10-second clips},
  author={Lostanlen, Vincent and Salamon, Justin and Cartwright, Mark and McFee, Brian and Farnsworth, Andrew and Kelling, Steve and Bello, Juan Pablo},
  booktitle={DCASE Workshop},
  year={2018}
}

@inproceedings{stowell2019,
  title={Automatic acoustic detection of birds through deep learning},
  author={Stowell, Dan and Wood, Michael D and Pamuła, Hanna and Stylianou, Yannis and Glotin, Hervé},
  journal={Methods in Ecology and Evolution},
  year={2019}
}

@inproceedings{piczak2015esc,
  title={ESC: Dataset for environmental sound classification},
  author={Piczak, Karol J},
  booktitle={ACM MM},
  year={2015}
}

@article{bandara2023forest,
  title={Forest sound classification dataset: Fsc22},
  author={Bandara, Meelan and Jayasundara, Roshinie and Ariyarathne, Isuru and Meedeniya, Dulani and Perera, Charith},
  journal={Sensors},
  volume={23},
  number={4},
  pages={2032},
  year={2023},
  publisher={MDPI}
}

@article{fredianelli2025environmental,
  title={Environmental Noise Dataset for Sound Event Classification and Detection},
  author={Fredianelli, Luca and Artuso, Francesco and Pompei, Geremia and Licitra, Gaetano and Iannace, Gino and Akbaba, Andac},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1712},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Related

This pipeline is part of the SEABAD project. See [`positive-label-curation/`](../positive-label-curation/) for the positive (bird) sample pipeline.
