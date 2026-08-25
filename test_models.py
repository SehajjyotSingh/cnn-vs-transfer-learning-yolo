import os
import time
import argparse
import sys

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Class names for CIFAR-10
CLASS_NAMES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Default Model Paths
YOLO_MODEL_PATH = r'models/yolomodel.pt'
CNN_MODEL_PATH = r'models/cnn_scratch.h5'
VGG16_MODEL_PATH = r'models/vgg16_finetuned.h5'
RESNET_MODEL_PATH = r'models/resnet50_finetuned.h5'


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


def predict_yolo(image_path, model_path=YOLO_MODEL_PATH):
    """Predict the class of a single image using a YOLO classification model."""
    from ultralytics import YOLO

    print("\n--------------------------------------------------")
    print("1. YOLO Model Prediction")
    print("--------------------------------------------------")

    if not os.path.exists(image_path):
        print(f"[YOLO] Image not found at path: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"[YOLO] Model file not found at: {model_path}")
        return None

    try:
        model = YOLO(model_path)
        results = model.predict(source=image_path, verbose=False)
        top1_idx = results[0].probs.top1
        top1_conf = results[0].probs.top1conf.item() * 100

        if hasattr(model, 'names') and top1_idx in model.names:
            predicted_class = model.names[top1_idx]
        else:
            predicted_class = (
                CLASS_NAMES[top1_idx]
                if top1_idx < len(CLASS_NAMES)
                else str(top1_idx)
            )

        print(f"Model Path      : {model_path}")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {top1_conf:.2f}%")

        return {
            'model': 'YOLO',
            'predicted_class': predicted_class,
            'confidence': top1_conf
        }
    except Exception as e:
        print(f"[Error] YOLO prediction failed: {e}")
        return None


def evaluate_yolo(model_path, model_name, test_images, test_labels, batch_size=128):
    """Evaluate a YOLO classification model on the CIFAR-10 test set."""
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        print(f"[Error] YOLO model path does not exist: {model_path}")
        print(f"[YOLO] Model file not found at: {model_path}")
        return None

    print(f"\n[Evaluating] {model_name} (YOLO: {model_path})...")
    model = YOLO(model_path)

    num_samples = len(test_images)
    predictions = []

    start_time = time.time()

    for i in range(0, num_samples, batch_size):
        batch_imgs = test_images[i:i + batch_size]
        batch_list = [img for img in batch_imgs]
        results = model.predict(source=batch_list, verbose=False)

        for res in results:
            predictions.append(res.probs.top1)

    total_time = time.time() - start_time
    predictions = np.array(predictions)

    report = classification_report(
        test_labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    acc = report['accuracy']
    prec = report['macro avg']['precision']
    rec = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    cm = confusion_matrix(test_labels, predictions)

    print(
        f" -> {model_name} Accuracy: {acc * 100:.2f}% | "
        f"Macro F1: {f1 * 100:.2f}% | Time: {total_time:.2f}s"
    )

    return {
        'name': model_name,
        'path': model_path,
        'type': 'YOLO',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'inference_time': total_time,
        'fps': num_samples / total_time if total_time > 0 else 0,
        'predictions': predictions,
        'confusion_matrix': cm
    }


def predict_cnn(image_path, model_path=CNN_MODEL_PATH):
    """Predict the class of a single image using a Keras CNN model."""
    print("\n--------------------------------------------------")
    print("2. Custom CNN Model Prediction")
    print("--------------------------------------------------")

    if not os.path.exists(image_path):
        print(f"[CNN] Image not found at path: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"[CNN] Model file not found at: {model_path}")
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[CNN] Could not read image: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (32, 32))
        img_array = np.expand_dims(
            img_resized.astype('float32') / 255.0,
            axis=0
        )

        model = tf.keras.models.load_model(model_path)
        raw_preds = model.predict(img_array, verbose=0)
        probs = (
            tf.nn.softmax(raw_preds[0]).numpy()
            if raw_preds.ndim > 1
            else raw_preds
        )
        top1_idx = np.argmax(probs)
        top1_conf = probs[top1_idx] * 100
        predicted_class = CLASS_NAMES[top1_idx]

        print(f"Model Path      : {model_path}")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {top1_conf:.2f}%")

        return {
            'model': 'Custom CNN',
            'predicted_class': predicted_class,
            'confidence': top1_conf
        }
    except Exception as e:
        print(f"[Error] CNN prediction failed: {e}")
        return None


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

        images_preprocessed = test_images.astype('float32')

        if arch_type.lower() == 'vgg16':
            images_preprocessed = (
                tf.keras.applications.vgg16.preprocess_input(images_preprocessed)
            )
        elif arch_type.lower() in ['resnet', 'resnet50']:
            images_preprocessed = (
                tf.keras.applications.resnet50.preprocess_input(images_preprocessed)
            )
        else:
            images_preprocessed = images_preprocessed / 255.0

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

        report = classification_report(
            test_labels,
            predictions,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )
        acc = report['accuracy']
        prec = report['macro avg']['precision']
        rec = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        cm = confusion_matrix(test_labels, predictions)

        print(
            f" -> {model_name} Accuracy: {acc * 100:.2f}% | "
            f"Macro F1: {f1 * 100:.2f}% | Time: {total_time:.2f}s"
        )

        return {
            'name': model_name,
            'path': model_path,
            'type': f'Keras ({arch_type})',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'inference_time': total_time,
            'fps': num_samples / total_time if total_time > 0 else 0,
            'predictions': predictions,
            'confusion_matrix': cm
        }
    except Exception as e:
        print(f"[Error] Keras evaluation failed for {model_name}: {e}")
        return None


def predict_vgg16(image_path, model_path=VGG16_MODEL_PATH):
    """Predict the class of a single image using a fine-tuned VGG16 model."""
    print("\n--------------------------------------------------")
    print("3. VGG16 Fine-Tuned Model Prediction")
    print("--------------------------------------------------")

    if not os.path.exists(image_path):
        print(f"[VGG16] Image not found at path: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"[VGG16] Model file not found at: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)

        img = cv2.imread(image_path)
        if img is None:
            print(f"[VGG16] Could not read image: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (32, 32))
        img_array = np.expand_dims(img_resized.astype('float32'), axis=0)
        img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_array)

        raw_preds = model.predict(img_preprocessed, verbose=0)
        probs = (
            tf.nn.softmax(raw_preds[0]).numpy()
            if raw_preds.ndim > 1
            else raw_preds
        )
        top1_idx = np.argmax(probs)
        top1_conf = probs[top1_idx] * 100
        predicted_class = CLASS_NAMES[top1_idx]

        print(f"Model Path      : {model_path}")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {top1_conf:.2f}%")

        return {
            'model': 'VGG16',
            'predicted_class': predicted_class,
            'confidence': top1_conf
        }
    except Exception as e:
        print(f"[Error] VGG16 prediction failed: {e}")
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

    report = classification_report(
        test_labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    acc = report['accuracy']
    prec = report['macro avg']['precision']
    rec = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    cm = confusion_matrix(test_labels, predictions)

    print(
        f" -> {model_name} Accuracy: {acc * 100:.2f}% | "
        f"Macro F1: {f1 * 100:.2f}% | Time: {total_time:.2f}s"
    )

    return {
        'name': model_name,
        'path': model_path,
        'type': 'PyTorch',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'inference_time': total_time,
        'fps': num_samples / total_time if total_time > 0 else 0,
        'predictions': predictions,
        'confusion_matrix': cm
    }


def predict_resnet(image_path, model_path=RESNET_MODEL_PATH):
    """Predict the class of a single image using a fine-tuned ResNet model."""
    print("\n--------------------------------------------------")
    print("4. ResNet Fine-Tuned Model Prediction")
    print("--------------------------------------------------")

    if not os.path.exists(image_path):
        print(f"[ResNet] Image not found at path: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"[ResNet] Model file not found at: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ResNet] Could not read image: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (32, 32))
        img_array = np.expand_dims(img_resized.astype('float32'), axis=0)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(
            img_array
        )

        raw_preds = model.predict(img_preprocessed, verbose=0)
        probs = (
            tf.nn.softmax(raw_preds[0]).numpy()
            if raw_preds.ndim > 1
            else raw_preds
        )
        top1_idx = np.argmax(probs)
        top1_conf = probs[top1_idx] * 100
        predicted_class = CLASS_NAMES[top1_idx]

        print(f"Model Path      : {model_path}")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {top1_conf:.2f}%")

        return {
            'model': 'ResNet',
            'predicted_class': predicted_class,
            'confidence': top1_conf
        }
    except Exception as e:
        print(f"[Error] ResNet prediction failed: {e}")
        return None


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
        row = (
            f"{r['name']:<22} | "
            f"{r['type']:<12} | "
            f"{r['accuracy'] * 100:>8.2f}% | "
            f"{r['precision'] * 100:>8.2f}% | "
            f"{r['recall'] * 100:>8.2f}% | "
            f"{r['f1_score'] * 100:>8.2f}% | "
            f"{r['inference_time']:>7.2f}s"
        )
        print(row)

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

    # Plot 1: Performance Metrics (Accuracy & F1-Score)
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2,
        accuracies,
        width,
        label='Accuracy (%)',
        color='#2b5c8f'
    )
    plt.bar(
        x + width / 2,
        f1_scores,
        width,
        label='Macro F1-Score (%)',
        color='#46a094'
    )
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

    # Plot 2: Inference Time Comparison
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

    # Plot 3: Confusion Matrices
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
        cm_path = os.path.join(
            output_dir,
            f"confusion_matrix_{clean_name}.png"
        )
        plt.savefig(cm_path)
        plt.close()
        print(
            f"[Export] Saved confusion matrix for "
            f"'{r['name']}' to: {cm_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Multi-Model Test & Evaluation Tool"
    )
    parser.add_argument(
        '--yolo-base',
        type=str,
        default=r'runs/yolov8n-cls.pt',
        help='Path to pretrained/base YOLO model (.pt)'
    )
    parser.add_argument(
        '--yolo-finetuned',
        type=str,
        default=r'runs/classify/YOLO_CIFAR10_Results/25_epochs_run/weights/best.pt',
        help='Path to fine-tuned YOLO classification model (.pt)'
    )
    parser.add_argument(
        '--vgg16-path',
        type=str,
        default=None,
        help='Path to fine-tuned VGG16 model (.h5, .keras, SavedModel directory, or .pt)'
    )
    parser.add_argument(
        '--resnet-path',
        type=str,
        default=None,
        help='Path to fine-tuned ResNet model (.h5, .keras, SavedModel directory, or .pt)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference (default: 128)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save comparison plots (default: evaluation_results)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable generating and saving plot images'
    )

    image_path = r"sample_test_image.jpg"

    # Allow overriding via command line argument.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    args = parser.parse_args()

    print("==================================================")
    print("       SINGLE IMAGE MULTI-MODEL CLASSIFIER        ")
    print("==================================================")
    print(f"Input Image Path: {image_path}")

    # Load Dataset
    test_images, test_labels = load_cifar10_data()
    results = []

    # Single-image predictions.
    predict_cnn(image_path)
    predict_yolo(image_path)
    predict_vgg16(image_path)
    predict_resnet(image_path)

    # 1. Evaluate Base YOLO model (if exists)
    if args.yolo_base and os.path.exists(args.yolo_base):
        res = evaluate_yolo(
            args.yolo_base,
            "YOLO Baseline",
            test_images,
            test_labels,
            batch_size=args.batch_size
        )
        if res:
            results.append(res)
    elif args.yolo_base:
        print(f"[Skip] YOLO base model path not found: {args.yolo_base}")

    print("\n==================================================")

    # 2. Evaluate Fine-Tuned YOLO model (if exists)
    if args.yolo_finetuned and os.path.exists(args.yolo_finetuned):
        res = evaluate_yolo(
            args.yolo_finetuned,
            "YOLO Fine-Tuned",
            test_images,
            test_labels,
            batch_size=args.batch_size
        )
        if res:
            results.append(res)
    elif args.yolo_finetuned:
        print(
            f"[Skip] YOLO fine-tuned model path not found: "
            f"{args.yolo_finetuned}"
        )

    # 3. Evaluate Fine-Tuned VGG16 model (if path provided)
    if args.vgg16_path:
        if not os.path.exists(args.vgg16_path):
            print(
                f"[Skip] VGG16 path provided but does not exist: "
                f"{args.vgg16_path}"
            )
        elif args.vgg16_path.endswith(('.pt', '.pth')):
            res = evaluate_pytorch(
                args.vgg16_path,
                "VGG16 Fine-Tuned",
                test_images,
                test_labels,
                batch_size=args.batch_size
            )
            if res:
                results.append(res)
        else:
            res = evaluate_keras(
                args.vgg16_path,
                "VGG16 Fine-Tuned",
                test_images,
                test_labels,
                arch_type='vgg16',
                batch_size=args.batch_size
            )
            if res:
                results.append(res)

    # 4. Evaluate Fine-Tuned ResNet model (if path provided)
    if args.resnet_path:
        if not os.path.exists(args.resnet_path):
            print(
                f"[Skip] ResNet path provided but does not exist: "
                f"{args.resnet_path}"
            )
        elif args.resnet_path.endswith(('.pt', '.pth')):
            res = evaluate_pytorch(
                args.resnet_path,
                "ResNet Fine-Tuned",
                test_images,
                test_labels,
                batch_size=args.batch_size
            )
            if res:
                results.append(res)
        else:
            res = evaluate_keras(
                args.resnet_path,
                "ResNet Fine-Tuned",
                test_images,
                test_labels,
                arch_type='resnet',
                batch_size=args.batch_size
            )
            if res:
                results.append(res)

    # Summary & Plotting
    if results:
        print_summary_table(results)
        if not args.no_plots:
            plot_and_save_results(
                results,
                output_dir=args.output_dir
            )
    else:
        print(
            "\n[Warning] No models were evaluated. "
            "Please verify the specified model paths."
        )


if __name__ == "__main__":
    main()
