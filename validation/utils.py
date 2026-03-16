# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import os
import numpy as np
from typing import Union, Dict, List
import tensorflow as tf

# Interface documentation
"""
Plotting Function Interface
==========================
All plotting functions follow this signature:
    function(history/labels, preds, title: str, stage: int, results_dir: str) -> None
Parameters:
    history: tf.keras.callbacks.History or np.ndarray - Training history or true labels
    preds: np.ndarray - Predicted probabilities (for non-history plots)
    title: str - Plot title (e.g., 'mybad_Training_History')
    stage: int - Training stage identifier (e.g., 1 for first stage)
    results_dir: str - Directory to save plot files
"""


def plot_training_history(history: tf.keras.callbacks.History, title: str, stage: int, results_dir: str) -> None:
    """Plot training and validation accuracy and loss over epochs."""
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss", linestyle="--")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{title}_stage{stage}.png"))
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, stage: int, results_dir: str) -> None:
    """Plot confusion matrix for binary classification."""
    cm = confusion_matrix(y_true, np.round(y_pred))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{title} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(results_dir, f"{title}_stage{stage}.png"))
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, title: str, stage: int, results_dir: str) -> None:
    """Plot ROC curve and compute AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title} - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f"{title}_stage{stage}.png"))
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred: np.ndarray, title: str, stage: int,
                                results_dir: str) -> None:
    """Plot precision-recall curve and compute average precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"{title} - Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, f"{title}_stage{stage}.png"))
    plt.close()


def plot_f1_score_curve(y_true: np.ndarray, y_pred: np.ndarray, title: str, stage: int, results_dir: str) -> None:
    """Plot F1 score versus classification threshold."""
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_true, y_pred >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, color='green', lw=2)
    plt.scatter(best_threshold, best_f1, color='red',
                label=f'Best F1: {best_f1:.2f} at threshold: {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f"{title} - F1 Score vs Threshold")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{title}_stage{stage}.png"))
    plt.close()


def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
