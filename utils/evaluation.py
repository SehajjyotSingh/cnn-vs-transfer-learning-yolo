import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES


def load_cifar10_data():
    """Load and return the CIFAR-10 test set images and labels."""
    print("\n[Data] Loading CIFAR-10 test dataset...")
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_labels = test_labels.flatten()
    print(
        f"[Data] Loaded {len(test_images)} test samples "
        f"across {len(CLASS_NAMES)} classes."
    )
    return test_images, test_labels


def _classification_metrics(model_name, model_path, model_type, test_labels,
                            predictions, total_time, num_samples):
    report = classification_report(
        test_labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    macro_avg = report['macro avg']

    print(
        f" -> {model_name} Accuracy: {report['accuracy'] * 100:.2f}% | "
        f"Macro F1: {macro_avg['f1-score'] * 100:.2f}% | Time: {total_time:.2f}s"
    )

    return {
        'name': model_name,
        'path': model_path,
        'type': model_type,
        'accuracy': report['accuracy'],
        'precision': macro_avg['precision'],
        'recall': macro_avg['recall'],
        'f1_score': macro_avg['f1-score'],
        'inference_time': total_time,
        'fps': num_samples / total_time if total_time > 0 else 0,
        'predictions': predictions,
        'confusion_matrix': confusion_matrix(test_labels, predictions)
    }


def _preprocess_keras_images(test_images, arch_type):
    images_preprocessed = test_images.astype('float32')
    arch_type = arch_type.lower()

    if arch_type == 'vgg16':
        return tf.keras.applications.vgg16.preprocess_input(images_preprocessed)
    if arch_type in ['resnet', 'resnet50']:
        return tf.keras.applications.resnet50.preprocess_input(images_preprocessed)
    return images_preprocessed / 255.0


def evaluate_keras(
    model_path,
    model_name,
    test_images,
    test_labels,
    arch_type='generic',
    batch_size=128
):
    """Evaluate a Keras model on the CIFAR-10 test set."""
    if not os.path.exists(model_path):
        print(f"[Error] Keras model path does not exist: {model_path}")
        return None

    print(f"\n[Evaluating] {model_name} (Keras {arch_type.upper()}: {model_path})...")

    try:
        model = tf.keras.models.load_model(model_path)
        images_preprocessed = _preprocess_keras_images(test_images, arch_type)

        num_samples = len(test_images)
        start_time = time.time()
        raw_preds = model.predict(
            images_preprocessed,
            batch_size=batch_size,
            verbose=0
        )
        total_time = time.time() - start_time

        if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            predictions = np.argmax(raw_preds, axis=1)
        else:
            predictions = (raw_preds > 0.5).astype(int).flatten()

        return _classification_metrics(
            model_name,
            model_path,
            f'Keras ({arch_type})',
            test_labels,
            predictions,
            total_time,
            num_samples
        )
    except Exception as e:
        print(f"[Error] Keras evaluation failed for {model_name}: {e}")
        return None


def evaluate_pytorch(
    model_path,
    model_name,
    test_images,
    test_labels,
    batch_size=128
):
    """Evaluate a PyTorch model on the CIFAR-10 test set."""
    import torch

    if not os.path.exists(model_path):
        print(f"[Error] PyTorch model path does not exist: {model_path}")
        return None

    print(f"\n[Evaluating] {model_name} (PyTorch: {model_path})...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict) and 'state_dict' in model:
            print(
                "[Warning] File contains state_dict. "
                "Full PyTorch module class definition is required."
            )
            return None

        model.eval()
    except Exception as e:
        print(f"[Error] Failed to load PyTorch model from {model_path}: {e}")
        return None

    imgs_tensor = (
        torch.tensor(test_images, dtype=torch.float32)
        .permute(0, 3, 1, 2) / 255.0
    )
    num_samples = len(test_images)
    predictions = []
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = imgs_tensor[i:i + batch_size].to(device)
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    total_time = time.time() - start_time
    predictions = np.array(predictions)

    return _classification_metrics(
        model_name,
        model_path,
        'PyTorch',
        test_labels,
        predictions,
        total_time,
        num_samples
    )


def print_summary_table(results):
    """Print a formatted comparative evaluation summary table."""
    if not results:
        print("\nNo evaluation results to display.")
        return

    print("\n" + "=" * 85)
    print("                        MODEL EVALUATION COMPARISON TABLE")
    print("=" * 85)

    header = (
        f"{'Model Name':<22} | {'Type':<12} | {'Accuracy':<10} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Time (s)':<8}"
    )
    print(header)
    print("-" * 85)

    for r in results:
        print(
            f"{r['name']:<22} | "
            f"{r['type']:<12} | "
            f"{r['accuracy'] * 100:>8.2f}% | "
            f"{r['precision'] * 100:>8.2f}% | "
            f"{r['recall'] * 100:>8.2f}% | "
            f"{r['f1_score'] * 100:>8.2f}% | "
            f"{r['inference_time']:>7.2f}s"
        )

    print("=" * 85 + "\n")


def plot_and_save_results(results, output_dir="evaluation_results"):
    """Generate and save comparison plots and confusion matrices."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    f1_scores = [r['f1_score'] * 100 for r in results]
    times = [r['inference_time'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, accuracies, width, label='Accuracy (%)', color='#2b5c8f')
    plt.bar(x + width / 2, f1_scores, width, label='Macro F1-Score (%)', color='#46a094')
    plt.xlabel('Model')
    plt.ylabel('Percentage (%)')
    plt.title('Model Performance Comparison on CIFAR-10 Test Set')
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot1_path = os.path.join(output_dir, 'accuracy_f1_comparison.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f"[Export] Saved accuracy & F1 comparison plot to: {plot1_path}")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=times, palette='viridis')
    plt.xlabel('Model')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Model Inference Time Comparison (Lower is Faster)')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot2_path = os.path.join(output_dir, 'inference_time_comparison.png')
    plt.savefig(plot2_path)
    plt.close()
    print(f"[Export] Saved inference time comparison plot to: {plot2_path}")

    for r in results:
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            r['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )
        plt.title(f"Confusion Matrix - {r['name']}")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()

        clean_name = r['name'].replace(' ', '_').lower()
        cm_path = os.path.join(output_dir, f"confusion_matrix_{clean_name}.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[Export] Saved confusion matrix for '{r['name']}' to: {cm_path}")
