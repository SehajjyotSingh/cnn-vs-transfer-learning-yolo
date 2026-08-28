import os
import time

import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    CNN_MODEL_PATH,
    RESNET_MODEL_PATH,
    VGG16_MODEL_PATH,
    YOLO_MODEL_PATH,
)
from utils.evaluation import _classification_metrics


def _print_prediction_header(number, title):
    print("\n--------------------------------------------------")
    print(f"{number}. {title}")
    print("--------------------------------------------------")


def _load_rgb_image(image_path, model_label):
    if not os.path.exists(image_path):
        print(f"[{model_label}] Image not found at path: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[{model_label}] Could not read image: {image_path}")
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _predict_keras_image(image_path, model_path, model_label, preprocess_fn=None,
                         scale=True):
    if not os.path.exists(model_path):
        print(f"[{model_label}] Model file not found at: {model_path}")
        return None

    img = _load_rgb_image(image_path, model_label)
    if img is None:
        return None

    try:
        img_resized = cv2.resize(img, (32, 32)).astype('float32')
        if scale:
            img_resized = img_resized / 255.0

        img_array = np.expand_dims(img_resized, axis=0)
        if preprocess_fn:
            img_array = preprocess_fn(img_array)

        model = tf.keras.models.load_model(model_path)
        raw_preds = model.predict(img_array, verbose=0)
        probs = tf.nn.softmax(raw_preds[0]).numpy() if raw_preds.ndim > 1 else raw_preds
        top1_idx = np.argmax(probs)
        top1_conf = probs[top1_idx] * 100
        predicted_class = CLASS_NAMES[top1_idx]

        print(f"Model Path      : {model_path}")
        print(f"Predicted Class : {predicted_class}")
        print(f"Confidence      : {top1_conf:.2f}%")

        return {
            'model': model_label,
            'predicted_class': predicted_class,
            'confidence': top1_conf
        }
    except Exception as e:
        print(f"[Error] {model_label} prediction failed: {e}")
        return None


def predict_yolo(image_path, model_path=YOLO_MODEL_PATH):
    """Predict the class of a single image using a YOLO classification model."""
    from ultralytics import YOLO

    _print_prediction_header(1, "YOLO Model Prediction")

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
            predicted_class = CLASS_NAMES[top1_idx] if top1_idx < len(CLASS_NAMES) else str(top1_idx)

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
        results = model.predict(source=[img for img in batch_imgs], verbose=False)
        predictions.extend(res.probs.top1 for res in results)

    total_time = time.time() - start_time
    predictions = np.array(predictions)

    return _classification_metrics(
        model_name,
        model_path,
        'YOLO',
        test_labels,
        predictions,
        total_time,
        num_samples
    )


def predict_cnn(image_path, model_path=CNN_MODEL_PATH):
    """Predict the class of a single image using a Keras CNN model."""
    _print_prediction_header(2, "Custom CNN Model Prediction")
    return _predict_keras_image(image_path, model_path, 'Custom CNN')


def predict_vgg16(image_path, model_path=VGG16_MODEL_PATH):
    """Predict the class of a single image using a fine-tuned VGG16 model."""
    _print_prediction_header(3, "VGG16 Fine-Tuned Model Prediction")
    return _predict_keras_image(
        image_path,
        model_path,
        'VGG16',
        preprocess_fn=tf.keras.applications.vgg16.preprocess_input,
        scale=False
    )


def predict_resnet(image_path, model_path=RESNET_MODEL_PATH):
    """Predict the class of a single image using a fine-tuned ResNet model."""
    _print_prediction_header(4, "ResNet Fine-Tuned Model Prediction")
    return _predict_keras_image(
        image_path,
        model_path,
        'ResNet',
        preprocess_fn=tf.keras.applications.resnet50.preprocess_input,
        scale=False
    )
