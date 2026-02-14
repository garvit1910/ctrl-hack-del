"""
TremorTrace ML Pipeline — Evaluator

Comprehensive model evaluation suite with 7 visualization types:
1. Classification report (printed text)
2. Confusion matrix heatmap (seaborn)
3. ROC-AUC curve with shaded area
4. Prediction distribution histogram with risk tier boundaries
5. Training history plots with Phase 2 boundary marker
6. Sample predictions grid with percentages and risk tiers
7. Risk tier distribution chart (grouped by true class)

All plots are saved as high-resolution PNGs AND displayed inline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)

try:
    from src.config import RISK_TIERS
except ImportError:
    from config import RISK_TIERS


def _get_risk_tier(probability_percent):
    """
    Returns the risk tier label and color for a given probability percentage.

    Args:
        probability_percent: float, value between 0.0 and 100.0

    Returns:
        tuple: (tier_label, hex_color)
    """
    for lower, upper, label, color in RISK_TIERS:
        if lower <= probability_percent < upper:
            return label, color
    # Edge case: exactly 100%
    return RISK_TIERS[-1][2], RISK_TIERS[-1][3]


def _ensure_plot_dir(save_dir):
    """
    Creates the plots subdirectory if it doesn't exist.

    Args:
        save_dir: str, base save directory

    Returns:
        str: path to plots subdirectory
    """
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _safe_filename(name):
    """
    Converts a model name to a filesystem-safe string.

    Args:
        name: str, model display name (e.g. 'Spiral CNN')

    Returns:
        str: safe filename component (e.g. 'Spiral_CNN')
    """
    return name.replace(' ', '_')


def evaluate_model(model, test_gen, name, save_dir):
    """
    Runs full evaluation: classification report, confusion matrix, and ROC curve.

    The test generator MUST have shuffle=False so that predictions align
    with test_gen.classes (the true labels).

    Args:
        model: trained Keras model
        test_gen: test data generator (must have shuffle=False)
        name: str, model display name (e.g. 'Spiral CNN')
        save_dir: str, directory to save plot PNGs

    Returns:
        tuple: (y_pred_prob, y_true, accuracy, roc_auc)
            - y_pred_prob: np.ndarray, shape (n_samples,), raw probabilities 0–1
            - y_true: np.ndarray, shape (n_samples,), integer labels 0 or 1
            - accuracy: float, classification accuracy
            - roc_auc: float, area under ROC curve
    """
    plot_dir = _ensure_plot_dir(save_dir)
    safe_name = _safe_filename(name)

    # Reset generator to ensure we start from the beginning
    test_gen.reset()

    # Get predictions — flatten from (n, 1) to (n,)
    y_pred_prob = model.predict(test_gen, verbose=0).flatten()
    y_true = test_gen.classes
    y_pred_binary = (y_pred_prob > 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    # 1. Classification report (printed)
    print(f"\n{'='*60}")
    print(f"📋 Classification Report — {name}")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred_binary,
        target_names=['Healthy', 'Parkinson'],
        digits=4,
    ))

    # 2. Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred_binary)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Healthy', 'Parkinson'],
        yticklabels=['Healthy', 'Parkinson'],
        ax=ax,
        annot_kws={'size': 16},
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(f'{name} — Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_confusion_matrix.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()

    # 3. ROC-AUC curve with shaded area
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#E74C3C', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#E74C3C')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'{name} — ROC Curve', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_roc_curve.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()

    print(f"✅ {name} — Accuracy: {accuracy:.1%}, AUC: {roc_auc:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    return y_pred_prob, y_true, accuracy, roc_auc


def plot_training_history(history, name, save_dir):
    """
    Plots training curves for loss, accuracy, and AUC over epochs.

    A vertical dashed red line marks where Phase 2 (fine-tuning) began,
    so you can see the impact of unfreezing base model layers.

    Args:
        history: dict, merged training history containing:
            - 'loss', 'val_loss', 'accuracy', 'val_accuracy', 'auc', 'val_auc'
            - 'phase2_start': int, epoch index where Phase 2 training began
        name: str, model display name (e.g. 'Spiral CNN')
        save_dir: str, directory to save plots
    """
    plot_dir = _ensure_plot_dir(save_dir)
    safe_name = _safe_filename(name)
    phase2_start = history.get('phase2_start', None)

    metrics_to_plot = [
        ('loss', 'val_loss', 'Loss', 'Loss'),
        ('accuracy', 'val_accuracy', 'Accuracy', 'Accuracy'),
        ('auc', 'val_auc', 'AUC', 'AUC'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'{name} — Training History', fontsize=18, fontweight='bold', y=1.02)

    for ax, (train_key, val_key, ylabel, title) in zip(axes, metrics_to_plot):
        if train_key not in history:
            ax.set_visible(False)
            continue

        epochs = range(1, len(history[train_key]) + 1)
        ax.plot(epochs, history[train_key], lw=2, label='Train', color='#3498DB')
        ax.plot(epochs, history[val_key], lw=2, label='Validation', color='#E74C3C')

        # Mark Phase 2 boundary
        if phase2_start is not None and phase2_start > 0:
            ax.axvline(
                x=phase2_start + 0.5,
                color='#2ECC71', linestyle='--', lw=2, alpha=0.8,
                label='Phase 2 Start',
            )

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_training_history.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()


def show_sample_predictions(model, test_gen, name, save_dir, n=10):
    """
    Displays a 2x5 grid of test images with predicted percentages and risk tiers.

    Title color indicates correctness:
        Green = model prediction agrees with true label
        Red = model prediction disagrees with true label

    Args:
        model: trained Keras model
        test_gen: test data generator
        name: str, model display name
        save_dir: str, directory to save plots
        n: int, number of samples to display (default 10, arranged in 2x5)
    """
    plot_dir = _ensure_plot_dir(save_dir)
    safe_name = _safe_filename(name)

    # Collect images and labels from the generator
    test_gen.reset()
    images = []
    labels = []
    for batch_imgs, batch_labels in test_gen:
        for img, lbl in zip(batch_imgs, batch_labels):
            images.append(img)
            labels.append(lbl)
            if len(images) >= n:
                break
        if len(images) >= n:
            break

    images = np.array(images[:n])
    labels = np.array(labels[:n])

    # Get predictions
    predictions = model.predict(images, verbose=0).flatten()

    # Class name mapping
    class_names = {0: 'Healthy', 1: "Parkinson's"}

    # Plot 2x5 grid
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    fig.suptitle(f'{name} — Sample Predictions', fontsize=18, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis('off')
            continue

        ax.imshow(images[i])
        ax.axis('off')

        pred_prob = float(predictions[i])
        pred_percent = pred_prob * 100.0
        true_label = int(labels[i])
        pred_label = 1 if pred_prob > 0.5 else 0
        tier_label, _ = _get_risk_tier(pred_percent)

        # Green if correct, red if incorrect
        correct = (pred_label == true_label)
        title_color = '#27AE60' if correct else '#E74C3C'

        ax.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {pred_percent:.1f}% ({tier_label})",
            fontsize=10, color=title_color, fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_sample_predictions.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()


def plot_prediction_distribution(y_pred_prob, y_true, name, save_dir):
    """
    Plots overlapping histograms of prediction distributions for each true class.

    Healthy predictions are shown in blue, Parkinson's in red. Vertical colored
    lines mark risk tier boundaries at 25%, 45%, 65%, and 85%.

    Args:
        y_pred_prob: np.ndarray, predicted probabilities (0–1 scale)
        y_true: np.ndarray, true labels (0=healthy, 1=parkinson)
        name: str, model display name
        save_dir: str, directory to save plots
    """
    plot_dir = _ensure_plot_dir(save_dir)
    safe_name = _safe_filename(name)

    # Convert to percentages
    pred_percent = y_pred_prob * 100.0
    healthy_preds = pred_percent[y_true == 0]
    pd_preds = pred_percent[y_true == 1]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Overlapping histograms
    bins = np.linspace(0, 100, 21)  # 5% bins
    ax.hist(healthy_preds, bins=bins, alpha=0.55, color='#3498DB',
            label='Truly Healthy', edgecolor='white', linewidth=0.5)
    ax.hist(pd_preds, bins=bins, alpha=0.55, color='#E74C3C',
            label="Truly Parkinson's", edgecolor='white', linewidth=0.5)

    # Risk tier boundary lines
    tier_boundaries = [
        (25, '#F1C40F', 'Low→Mild (25%)'),
        (45, '#E67E22', 'Mild→Moderate (45%)'),
        (65, '#E74C3C', 'Moderate→Elevated (65%)'),
        (85, '#C0392B', 'Elevated→High (85%)'),
    ]
    for boundary_pct, color, label in tier_boundaries:
        ax.axvline(x=boundary_pct, color=color, linestyle='--', lw=2, alpha=0.7, label=label)

    ax.set_xlabel('Predicted PD Probability (%)', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title(f'{name} — Prediction Distribution by True Class', fontsize=16)
    ax.legend(fontsize=9, loc='upper center', ncol=3)
    ax.set_xlim(-2, 102)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_prediction_distribution.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()


def plot_risk_tier_distribution(y_pred_prob, y_true, name, save_dir):
    """
    Plots a grouped bar chart showing how many samples fall into each risk tier,
    split by their true class (healthy vs parkinson).

    This helps visualize whether the model's risk tiers align with actual diagnoses:
    ideally, healthy samples cluster in Low/Mild tiers and PD samples in Elevated/High.

    Args:
        y_pred_prob: np.ndarray, predicted probabilities (0–1 scale)
        y_true: np.ndarray, true labels (0=healthy, 1=parkinson)
        name: str, model display name
        save_dir: str, directory to save plots
    """
    plot_dir = _ensure_plot_dir(save_dir)
    safe_name = _safe_filename(name)

    pred_percent = y_pred_prob * 100.0

    tier_labels = [t[2] for t in RISK_TIERS]
    tier_colors = [t[3] for t in RISK_TIERS]
    healthy_counts = []
    pd_counts = []

    for lower, upper, _, _ in RISK_TIERS:
        if upper == 100:
            # Include exactly 100% in the last tier
            mask = (pred_percent >= lower) & (pred_percent <= upper)
        else:
            mask = (pred_percent >= lower) & (pred_percent < upper)
        healthy_counts.append(int(np.sum(mask & (y_true == 0))))
        pd_counts.append(int(np.sum(mask & (y_true == 1))))

    x = np.arange(len(tier_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_h = ax.bar(x - width / 2, healthy_counts, width,
                    label='Truly Healthy', color='#3498DB', alpha=0.85, edgecolor='white')
    bars_pd = ax.bar(x + width / 2, pd_counts, width,
                     label="Truly Parkinson's", color='#E74C3C', alpha=0.85, edgecolor='white')

    # Color x-tick labels by tier color for visual emphasis
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=11)
    for i, tick_label in enumerate(ax.get_xticklabels()):
        tick_label.set_color(tier_colors[i])
        tick_label.set_fontweight('bold')

    # Add count labels above bars
    for bars in [bars_h, bars_pd]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold',
                )

    ax.set_xlabel('Risk Tier', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title(f'{name} — Risk Tier Distribution', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_risk_tier_distribution.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()


def plot_model_comparison(metrics_dict, save_dir):
    """
    Plots a grouped bar chart comparing Spiral CNN vs Wave CNN across key metrics.

    Args:
        metrics_dict: dict, mapping model names to their evaluation metrics.
            Format:
            {
                'Spiral CNN': {'accuracy': 0.90, 'auc': 0.95, 'precision': 0.88, 'recall': 0.92},
                'Wave CNN':   {'accuracy': 0.87, 'auc': 0.91, 'precision': 0.85, 'recall': 0.89},
            }
        save_dir: str, directory to save plots
    """
    plot_dir = _ensure_plot_dir(save_dir)

    model_names = list(metrics_dict.keys())
    metric_keys = ['accuracy', 'auc', 'precision', 'recall']
    metric_display_names = ['Accuracy', 'AUC', 'Precision', 'Recall']

    x = np.arange(len(metric_keys))
    width = 0.3
    colors = ['#3498DB', '#E74C3C']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        values = [metrics_dict[model_name].get(m, 0.0) for m in metric_keys]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            label=model_name, color=colors[i % len(colors)], alpha=0.85,
            edgecolor='white',
        )

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_display_names, fontsize=12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Comparison — Spiral CNN vs Wave CNN', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, 'model_comparison.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()
