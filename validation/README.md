# SEABAD Baseline Validation

This directory contains scripts for training and evaluating baseline CNN models on the SEABAD dataset.

## 📊 Results Summary

Baseline validation results averaged across three random seeds (42, 100, 786) on 5,000 test samples:

| Model | Params | Accuracy | AUC | Precision | Recall | F1 |
|-------|--------|----------|-----|-----------|--------|-----|
| **MobileNetV3-Small**† | 1.1M | **99.57 ± 0.25%** | **0.9985 ± 0.0002** | 0.9956 ± 0.0012 | 0.9957 ± 0.0008 | 0.9957 ± 0.0025 |
| EfficientNetB0 | 4.4M | 99.49 ± 0.23% | 0.9991 ± 0.0004 | 0.9959 ± 0.0018 | 0.9939 ± 0.0051 | 0.9949 ± 0.0023 |
| VGG16 | 14.9M | 99.61 ± 0.03% | 0.9995 ± 0.0001 | 0.9960 ± 0.0014 | 0.9963 ± 0.0010 | 0.9961 ± 0.0025 |
| ResNet50 | 24.2M | 99.73 ± 0.02% | 0.9992 ± 0.0003 | 0.9965 ± 0.0013 | 0.9980 ± 0.0012 | 0.9973 ± 0.0019 |

**†Primary baseline for edge deployment**

### Key Findings

- **All models achieve >99.4% accuracy** with minimal variance (std <0.25%), confirming dataset quality
- **MobileNetV3-Small** is only 0.16% behind ResNet50 but has 22× fewer parameters
- **VGG16 most stable** across seeds (0.03% accuracy std)
- **Training stability confirmed** across diverse architectures and random initializations

## 🚀 Quick Start

### Run Single Model

```bash
python validate_seabad_pretrained.py --model mobilenetv3s --seed 42
```

### Run All Models (All Seeds)

```bash
./run_all_cnn.sh
```

### Run Specific Seeds Only

```bash
./run_all_cnn.sh 42 100  # Only seeds 42 and 100
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.15+
- librosa
- scikit-learn
- matplotlib
- seaborn
- tqdm

Install dependencies:
```bash
pip install tensorflow librosa scikit-learn matplotlib seaborn tqdm
```

## 🔧 Configuration

### Model Architectures

Choose from:
- `mobilenetv3s` - MobileNetV3-Small (1.1M params, **recommended**)
- `efficientnetb0` - EfficientNetB0 (4.4M params)
- `vgg16` - VGG16 (14.9M params)
- `resnet50` - ResNet50 (24.2M params)

### Command-Line Arguments

```bash
python validate_seabad_pretrained.py \
    --model mobilenetv3s \     # Model architecture
    --seed 42 \                # Random seed for reproducibility
    --dataset_dir /path/to/SEABAD/  # Dataset root directory
```

### Hyperparameters

Fixed parameters (optimized for SEABAD):
- **Input shape**: 224×224×1 mel-spectrograms
- **Mel bins**: 224
- **FFT size**: 512 (corrected from previous 1024)
- **Hop length**: 224 frames
- **Sampling rate**: 16 kHz
- **Audio duration**: 3 seconds
- **Batch size**: 32
- **Learning rate**: Cosine decay from 1e-4
- **Train/Val/Test split**: 80% / 10% / 10% (stratified)
- **Early stopping patience**: 15 epochs

## 📁 Output Structure

Each run creates a results directory:

```
results/
└── {model}_seed{seed}_{platform}/
    ├── results.txt              # Summary metrics
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── f1_score_curve.png
    ├── training_history.png
    └── {model}_best.keras      # Best model checkpoint
```

## 🧪 Experimental Setup

### Dataset Split
- **Training**: 40,000 clips (80%)
- **Validation**: 5,000 clips (10%)
- **Test**: 5,000 clips (10%)
- **Class balance**: 50% positive, 50% negative in all splits (stratified sampling)

### Training Details
- **Transfer learning** from ImageNet-pretrained weights
- **Optimizer**: Adam with cosine learning rate decay
- **Early stopping**: Monitors validation loss (patience=15)
- **Data preprocessing**:
  - Grayscale spectrograms replicated to RGB
  - Scaled to [0, 255]
  - Architecture-specific normalization
- **Hardware**: NVIDIA GTX 1080 Ti GPU
- **Framework**: TensorFlow 2.15

### Reproducibility
All experiments use fixed random seeds (42, 100, 786) for:
- NumPy random state
- TensorFlow random seed
- Python random seed
- Train/val/test split stratification

## 📈 Performance Analysis

### Dataset Quality Validation
The consistently high performance (mean accuracy 99.60%, std ≤0.25%) provides strong evidence of:
- Clean annotation labels
- Clear acoustic separability between bird-present and bird-absent classes
- Minimal systematic labeling errors
- Robust task definition

### Cross-Architecture Stability
Minimal variance across architectures (99.49–99.73% range) indicates:
- Performance differences reflect architectural capacity, not dataset biases
- Dataset does not favor specific inductive biases
- Task is learnable across diverse feature extraction paradigms

### Multi-Seed Robustness
Low standard deviation across random seeds (≤0.25%) confirms:
- Training stability
- Reproducible results
- Reliable dataset for benchmarking

## 🎯 MobileNetV3-Small as Primary Baseline

MobileNetV3-Small serves as the **primary baseline** for SEABAD because:

1. **Competitive accuracy**: 99.57 ± 0.25% (only 0.16% behind ResNet50)
2. **Edge-deployment ready**: 1.1M parameters vs 24.2M (ResNet50)
3. **Balanced metrics**: Precision 0.9956, Recall 0.9957 (no class bias)
4. **Aligned with use case**: SEABAD is designed for resource-constrained bioacoustic monitoring

### Parameter Efficiency Comparison

| Model | Accuracy | Params | Accuracy/Param |
|-------|----------|--------|----------------|
| MobileNetV3-Small | 99.57% | 1.1M | 90.5 × 10⁻⁶ |
| EfficientNetB0 | 99.49% | 4.4M | 22.6 × 10⁻⁶ |
| VGG16 | 99.61% | 14.9M | 6.7 × 10⁻⁶ |
| ResNet50 | 99.73% | 24.2M | 4.1 × 10⁻⁶ |

MobileNetV3-Small achieves **22× better parameter efficiency** than ResNet50.

## 🔬 Citation

If you use these baseline results, please cite the SEABAD paper:

```bibtex
@article{seabad2025,
  title={SEABAD: A Tropical Bird Audio Detection Dataset for Passive Acoustic Monitoring},
  author={Author Names},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📝 Notes

- **High-resolution spectrograms** (224×224) are used for validation to thoroughly assess dataset quality with standard CV architectures
- These exceed typical embedded deployment constraints but enable fair comparison across architectures
- For ultra-lightweight models optimized for microcontroller deployment, see future work
- Results demonstrate SEABAD's suitability for developing edge-optimized bird audio detectors

## 🔗 Links

- **Dataset**: https://zenodo.org/records/18290494
- **Main Repository**: https://github.com/mun3im/seabad
- **Paper**: [arXiv link coming soon]

## 📧 Contact

For questions about the validation experiments, please open an issue on GitHub.
