"""
SEABAD Dataset Training Script - Classic CNN Models
Supports: MobileNetV3Small (default), ResNet50, VGG16, EfficientNetB0
Features:
- 224x224 high-resolution spectrograms (n_mels=224, n_hops=224)
- 80:10:10 train/val/test split
- Cosine decay learning rate scheduler
- Multi-seed support for reproducibility
"""
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay
layers = keras.layers
models = keras.models
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
import os
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
from utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_f1_score_curve
)

scriptstart = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train SEABAD model with various CNN architectures.")
parser.add_argument("-m", "--model", type=str, default="mobilenetv3s",
                    choices=["mobilenetv3s", "resnet50", "vgg16", "efficientnetb0"],
                    help="Model architecture (mobilenetv3s, resnet50, vgg16, efficientnetb0)")
parser.add_argument("-s", "--seed", type=int, default=42,
                    help="Random seed for reproducibility (e.g., 42, 100, 786)")
parser.add_argument("-d", "--dataset_dir", type=str, default="/Volumes/Evo/SEABAD/",
                    help="Path to SEABAD dataset directory")
args = parser.parse_args()

# Fixed parameters for high-resolution spectrograms
SAMPLING_RATE = 16000
DATASET = "seabad"
DATASET_DIR = args.dataset_dir
PLATFORM = sys.platform
MODEL_NAME = args.model.upper()
RANDOM_SEED = args.seed
RESULTS_DIR = f"results/{args.model}_seed{RANDOM_SEED}_{PLATFORM}"

# High-resolution spectrogram parameters (224x224)
N_MELS = 224
N_HOPS = 224
TIME_STEPS = 224
INPUT_SHAPE = (224, 224, 1)
AUDIO_DURATION = 3
N_FFT = 512

# Cache directory setup
NPY_CACHE_BASE_DIR = "/Volumes/Evo/SEABAD/seabad_spectrograms_224"
NPY_CACHE_DIR = os.path.join(NPY_CACHE_BASE_DIR, f"sr_{SAMPLING_RATE}_nmels_{N_MELS}_hops_{N_HOPS}")

print("\n" * 2)
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Seed: {RANDOM_SEED}")
print(f"Sampling Rate: {SAMPLING_RATE}Hz")
print(f"Input Shape: {INPUT_SHAPE}")
print(f"N_MELS: {N_MELS}, N_HOPS: {N_HOPS}")
print(f"Results Directory: {RESULTS_DIR}")
print("=" * 80)
print("\n")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# HYPERPARAMETERS
BATCH_SIZE = 32
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
L2_REG = 1e-4
DROPOUT_RATE = 0.5

# Learning rate with cosine decay
# FIX 1: Reduced Learning Rate from 1e-3 to 1e-4 to prevent gradient explosion
INITIAL_LR = 1e-4
DECAY_STEPS = 1000  # Will be updated based on dataset size


def manage_spectrogram_cache():
    """Manage the spectrogram cache based on parameters"""
    os.makedirs(NPY_CACHE_BASE_DIR, exist_ok=True)
    os.makedirs(NPY_CACHE_DIR, exist_ok=True)
    return NPY_CACHE_DIR


class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=BATCH_SIZE, n_mels=N_MELS,
                 time_steps=TIME_STEPS, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.time_steps = time_steps
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        self.cached_npy_paths = []
        self.verify_or_create_npy_cache()
        self.on_epoch_end()

    def verify_or_create_npy_cache(self):
        """Verify NPY cache integrity or create NPY files if they don't exist"""
        print(f"Verifying/creating NPY cache for {len(self.file_paths)} files...")
        for idx in tqdm(range(len(self.file_paths)), desc="Processing NPY cache"):
            audio_path = self.file_paths[idx]
            npy_path = self.get_npy_path(audio_path)

            if os.path.exists(npy_path):
                try:
                    spectrogram = np.load(npy_path)
                    if spectrogram.shape == (self.n_mels, self.time_steps):
                        self.cached_npy_paths.append(npy_path)
                        continue
                    else:
                        os.unlink(npy_path)
                except Exception:
                    if os.path.exists(npy_path):
                        os.unlink(npy_path)

            spectrogram = self.extract_mel_spectrogram(audio_path)
            if spectrogram.max() > spectrogram.min():
                spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

            if spectrogram.shape != (self.n_mels, self.time_steps):
                spectrogram = np.resize(spectrogram, (self.n_mels, self.time_steps))

            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, spectrogram)
            self.cached_npy_paths.append(npy_path)

    def get_npy_path(self, audio_path):
        """Generate NPY cache path from audio path, using consistent hashing"""
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()
        return os.path.join(NPY_CACHE_DIR, f"{path_hash}.npy")

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_indexes), self.n_mels, self.time_steps, 1), dtype=np.float32)
        batch_y = np.zeros(len(batch_indexes), dtype=np.float32)

        for i, idx in enumerate(batch_indexes):
            npy_path = self.cached_npy_paths[idx]
            try:
                spectrogram = np.load(npy_path)
                if spectrogram.shape != (self.n_mels, self.time_steps):
                    spectrogram = np.resize(spectrogram, (self.n_mels, self.time_steps))
                batch_x[i, :, :, 0] = spectrogram
                batch_y[i] = self.labels[idx]
            except Exception as e:
                print(f"Error loading cached NPY {npy_path}: {e}")
                batch_x[i] = np.zeros((self.n_mels, self.time_steps, 1))
                batch_y[i] = 0

        # FIX 2: Ensure numerical stability in data loading (prevent NaN/Inf propagation)
        batch_x = np.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return batch_x, batch_y

    def extract_mel_spectrogram(self, file_path, n_fft=N_FFT, target_frames=TIME_STEPS, duration=AUDIO_DURATION):
        try:
            y, sr = librosa.load(file_path, sr=SAMPLING_RATE, duration=duration)
            target_length = int(SAMPLING_RATE * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode="constant")
            elif len(y) > target_length:
                y = y[:target_length]

            # Calculate hop_length to get exactly N_HOPS frames
            hop_length = max(1, int((target_length - n_fft) / (N_HOPS - 1)))

            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=SAMPLING_RATE, n_mels=self.n_mels, n_fft=n_fft, hop_length=hop_length
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            if log_mel_spec.shape[1] < target_frames:
                log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, target_frames - log_mel_spec.shape[1])),
                                      mode="constant")
            elif log_mel_spec.shape[1] > target_frames:
                log_mel_spec = log_mel_spec[:, :target_frames]

            return log_mel_spec
        except Exception as e:
            print(f"Error in extract_mel_spectrogram for {file_path}: {e}")
            return np.zeros((self.n_mels, target_frames))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def load_dataset_with_stats(dataset_dir):
    """Load dataset and generate detailed statistics"""
    all_files = []
    all_labels = []
    subdirectory_stats = {"positive": {}, "negative": {}}

    pos_dir = os.path.join(dataset_dir, "positive")
    neg_dir = os.path.join(dataset_dir, "negative")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)

    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.aiff')

    def is_valid_audio_file(filepath):
        filename = os.path.basename(filepath)
        if filename.startswith('._') or filename.startswith('.'):
            return False
        skip_patterns = ['.DS_Store', 'Thumbs.db', '.directory']
        if any(pattern in filename for pattern in skip_patterns):
            return False
        if not filename.lower().endswith(audio_extensions):
            return False
        if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
            return False
        return True

    # Process positive samples
    if os.path.exists(pos_dir):
        subdirs = [d for d in os.listdir(pos_dir) if os.path.isdir(os.path.join(pos_dir, d)) and not d.startswith('.')]
        if subdirs:
            for subdir in tqdm(subdirs, desc="Scanning positive subdirectories"):
                subdir_path = os.path.join(pos_dir, subdir)
                try:
                    valid_files = [fname for fname in os.listdir(subdir_path)
                                 if is_valid_audio_file(os.path.join(subdir_path, fname))]
                    if not valid_files:
                        continue
                    subdirectory_stats["positive"][subdir] = len(valid_files)
                    for fname in valid_files:
                        all_files.append(os.path.join(subdir_path, fname))
                        all_labels.append(1)
                except PermissionError:
                    continue
        else:
            valid_files = [fname for fname in os.listdir(pos_dir)
                         if is_valid_audio_file(os.path.join(pos_dir, fname))]
            subdirectory_stats["positive"]["root"] = len(valid_files)
            for fname in valid_files:
                all_files.append(os.path.join(pos_dir, fname))
                all_labels.append(1)

    # Process negative samples
    if os.path.exists(neg_dir):
        subdirs = [d for d in os.listdir(neg_dir) if os.path.isdir(os.path.join(neg_dir, d)) and not d.startswith('.')]
        if subdirs:
            for subdir in tqdm(subdirs, desc="Scanning negative subdirectories"):
                subdir_path = os.path.join(neg_dir, subdir)
                try:
                    valid_files = [fname for fname in os.listdir(subdir_path)
                                 if is_valid_audio_file(os.path.join(subdir_path, fname))]
                    if not valid_files:
                        continue
                    subdirectory_stats["negative"][subdir] = len(valid_files)
                    for fname in valid_files:
                        all_files.append(os.path.join(subdir_path, fname))
                        all_labels.append(0)
                except PermissionError:
                    continue
        else:
            valid_files = [fname for fname in os.listdir(neg_dir)
                         if is_valid_audio_file(os.path.join(neg_dir, fname))]
            subdirectory_stats["negative"]["root"] = len(valid_files)
            for fname in valid_files:
                all_files.append(os.path.join(neg_dir, fname))
                all_labels.append(0)

    return np.array(all_files), np.array(all_labels), subdirectory_stats


def save_dataset_stats(subdirectory_stats, results_dir):
    """Save detailed dataset statistics to stats.txt"""
    with open(os.path.join(results_dir, "stats.txt"), "w") as f:
        f.write("DATASET STATISTICS\n")
        f.write("=" * 50 + "\n\n")

        f.write("POSITIVE SAMPLES:\n")
        f.write("-" * 20 + "\n")
        total_positive = 0
        for subdir, count in subdirectory_stats["positive"].items():
            f.write(f"{subdir}: {count} files\n")
            total_positive += count
        f.write(f"Total Positive: {total_positive}\n\n")

        f.write("NEGATIVE SAMPLES:\n")
        f.write("-" * 20 + "\n")
        total_negative = 0
        for subdir, count in subdirectory_stats["negative"].items():
            f.write(f"{subdir}: {count} files\n")
            total_negative += count
        f.write(f"Total Negative: {total_negative}\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Files: {total_positive + total_negative}\n")
        f.write(f"Positive Samples: {total_positive}\n")
        f.write(f"Negative Samples: {total_negative}\n")
        if total_positive + total_negative > 0:
            pos_ratio = total_positive / (total_positive + total_negative)
            f.write(f"Positive Ratio: {pos_ratio:.3f}\n")
            f.write(f"Negative Ratio: {1 - pos_ratio:.3f}\n")


def build_model(model_name, input_shape):
    """Build model based on architecture choice"""

    if model_name == "mobilenetv3s":
        from tensorflow.keras.applications import MobileNetV3Small
        base_model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        # Prepare single-channel input
        input_layer = layers.Input(shape=input_shape)
        # FIX 2: Add BatchNormalization early to stabilize input distribution
        x = layers.BatchNormalization()(input_layer)
        x = layers.Lambda(lambda x: (x - 0.5) * 2)(x)
        x = layers.Conv2D(3, (3, 3), padding='same', use_bias=False)(x)
        # FIX 2: Add BatchNormalization before pretrained base to prevent explosion
        x = layers.BatchNormalization()(x)
        features = base_model(x)

    elif model_name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        input_layer = layers.Input(shape=input_shape)
        # FIX 2: Add BatchNormalization early to stabilize input distribution
        x = layers.BatchNormalization()(input_layer)
        x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(x)
        # FIX 2: Add BatchNormalization before pretrained base
        x = layers.BatchNormalization()(x)
        features = base_model(x)

    elif model_name == "vgg16":
        from tensorflow.keras.applications import VGG16
        base_model = VGG16(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        input_layer = layers.Input(shape=input_shape)
        # FIX 2: Add BatchNormalization early to stabilize input distribution
        x = layers.BatchNormalization()(input_layer)
        x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(x)
        # FIX 2: Add BatchNormalization before pretrained base
        x = layers.BatchNormalization()(x)
        features = base_model(x)

    elif model_name == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        input_layer = layers.Input(shape=input_shape)
        # FIX 2: Add BatchNormalization early to stabilize input distribution
        x = layers.BatchNormalization()(input_layer)
        x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(x)
        # FIX 2: Add BatchNormalization before pretrained base
        x = layers.BatchNormalization()(x)
        features = base_model(x)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Classification head
    x = layers.BatchNormalization()(features)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Initialize cache management
    global NPY_CACHE_DIR
    NPY_CACHE_DIR = manage_spectrogram_cache()
    print(f"Using spectrogram cache at: {NPY_CACHE_DIR}\n")

    # Load dataset
    all_files, all_labels, subdirectory_stats = load_dataset_with_stats(DATASET_DIR)
    save_dataset_stats(subdirectory_stats, RESULTS_DIR)

    if len(all_files) == 0:
        print(f"Error: No audio files found in dataset '{DATASET_DIR}'.")
        sys.exit(1)

    print(f"\nDataset: {DATASET}")
    print(f"Total samples: {len(all_files)}")
    print(f"Positive samples: {sum(all_labels)}")
    print(f"Negative samples: {len(all_files) - sum(all_labels)}\n")

    if sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        print("Error: Dataset must contain both positive and negative samples.")
        sys.exit(1)

    # Split the data (80:10:10)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=RANDOM_SEED, stratify=all_labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels
    )

    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}\n")

    # Initialize data generators
    train_gen = AudioDataGenerator(train_files, train_labels, batch_size=BATCH_SIZE)
    val_gen = AudioDataGenerator(val_files, val_labels, batch_size=BATCH_SIZE)
    test_gen = AudioDataGenerator(test_files, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print(f"Building {MODEL_NAME} model...")
    model = build_model(args.model, INPUT_SHAPE)

    # Setup cosine decay learning rate
    global DECAY_STEPS
    DECAY_STEPS = len(train_gen) * EPOCHS
    lr_schedule = CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=DECAY_STEPS,
        alpha=0.0
    )

    # FIX 1: Added clipnorm=1.0 to prevent gradient explosion
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=lr_schedule, clipnorm=1.0) if sys.platform == "darwin" else tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("\nModel Summary:")
    model.summary()

    # Callbacks
    checkpoint_path = os.path.join(RESULTS_DIR, "best_model.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, monitor='val_loss', verbose=1)
    ]

    # Train the model
    print(f"\nStarting training with {MODEL_NAME}, seed {RANDOM_SEED}...")
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    training_time = timedelta(seconds=end_time - start_time)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_result = model.evaluate(test_gen, verbose=1)
    test_preds = model.predict(test_gen).flatten()

    # Calculate metrics
    test_loss = test_result[0]
    test_accuracy = test_result[1]
    test_auc = test_result[2]
    precision = precision_score(test_labels, np.round(test_preds))
    recall = recall_score(test_labels, np.round(test_preds))
    f1 = f1_score(test_labels, np.round(test_preds))

    # Save plots
    plot_training_history(history, f"{DATASET}_Training_History", 1, RESULTS_DIR)
    plot_confusion_matrix(test_labels, test_preds, f"{DATASET}_Confusion_Matrix", 1, RESULTS_DIR)
    plot_roc_curve(test_labels, test_preds, f"{DATASET}_ROC_Curve", 1, RESULTS_DIR)
    plot_precision_recall_curve(test_labels, test_preds, f"{DATASET}_Precision_Recall", 1, RESULTS_DIR)
    plot_f1_score_curve(test_labels, test_preds, f"{DATASET}_F1_Score", 1, RESULTS_DIR)

    # Generate results.txt
    with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
        f.write(f"SEABAD TRAINING RESULTS\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Input Shape: {INPUT_SHAPE}\n")
        f.write(f"N_MELS: {N_MELS}, N_HOPS: {N_HOPS}\n")
        f.write(f"Sampling Rate: {SAMPLING_RATE}Hz\n")
        f.write(f"Audio Duration: {AUDIO_DURATION}s\n")
        f.write(f"Learning Rate: Cosine Decay (initial={INITIAL_LR}, steps={DECAY_STEPS})\n\n")

        f.write(f"Dataset Split:\n")
        f.write(f"Train: {len(train_files)} samples\n")
        f.write(f"Val: {len(val_files)} samples\n")
        f.write(f"Test: {len(test_files)} samples\n\n")

        f.write(f"RESULTS:\n")
        f.write(f"Training Time: {training_time}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")

        f.write(f"Classification Report:\n")
        f.write(f"{classification_report(test_labels, np.round(test_preds), digits=4)}\n")

        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        f.write(f"\nModel Parameters:\n")
        f.write(f"Total: {trainable_params + non_trainable_params:,}\n")
        f.write(f"Trainable: {trainable_params:,}\n")
        f.write(f"Non-Trainable: {non_trainable_params:,}\n")

    total_time = str(timedelta(seconds=time.time() - scriptstart))
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETED SUCCESSFULLY")
    print(f"Total time: {total_time}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Maximum Test Accuracy: {test_accuracy:.4f}")
    print(f"{'=' * 80}\n")

    with open(os.path.join(RESULTS_DIR, "results.txt"), "a") as f:
        f.write(f"\nTotal Execution Time: {total_time}\n")

    # Save final model
    model.save(os.path.join(RESULTS_DIR, "final_model.keras"))
    print(f"Final model saved to: {os.path.join(RESULTS_DIR, 'final_model.keras')}\n")


if __name__ == "__main__":
    main()