import os

from src.model_testing import (
    evaluate_yolo,
    predict_cnn,
    predict_resnet,
    predict_vgg16,
    predict_yolo,
)
from utils import (
    evaluate_keras,
    evaluate_pytorch,
    load_cifar10_data,
    plot_and_save_results,
    print_summary_table,
)


def _append_result(results, result):
    if result:
        results.append(result)


def _evaluate_transfer_model(model_path, model_name, arch_type, test_images,
                             test_labels, batch_size):
    if not model_path:
        return None

    model_family = model_name.split()[0]
    if not os.path.exists(model_path):
        print(f"[Skip] {model_family} path provided but does not exist: {model_path}")
        return None

    if model_path.endswith(('.pt', '.pth')):
        return evaluate_pytorch(
            model_path,
            model_name,
            test_images,
            test_labels,
            batch_size=batch_size
        )

    return evaluate_keras(
        model_path,
        model_name,
        test_images,
        test_labels,
        arch_type=arch_type,
        batch_size=batch_size
    )


def main():
    image_path = r'sample_test_image.jpg'
    yolo_base = r'runs/yolov8n-cls.pt'
    yolo_finetuned = r'runs/classify/YOLO_CIFAR10_Results/25_epochs_run/weights/best.pt'
    vgg16_path = None
    resnet_path = None
    batch_size = 128
    output_dir = 'evaluation_results'
    save_plots = True

    print("==================================================")
    print("       SINGLE IMAGE MULTI-MODEL CLASSIFIER        ")
    print("==================================================")
    print(f"Input Image Path: {image_path}")

    test_images, test_labels = load_cifar10_data()
    results = []

    predict_cnn(image_path)
    predict_yolo(image_path)
    predict_vgg16(image_path)
    predict_resnet(image_path)

    if yolo_base and os.path.exists(yolo_base):
        _append_result(results, evaluate_yolo(
            yolo_base,
            "YOLO Baseline",
            test_images,
            test_labels,
            batch_size=batch_size
        ))
    elif yolo_base:
        print(f"[Skip] YOLO base model path not found: {yolo_base}")

    print("\n==================================================")

    if yolo_finetuned and os.path.exists(yolo_finetuned):
        _append_result(results, evaluate_yolo(
            yolo_finetuned,
            "YOLO Fine-Tuned",
            test_images,
            test_labels,
            batch_size=batch_size
        ))
    elif yolo_finetuned:
        print(f"[Skip] YOLO fine-tuned model path not found: {yolo_finetuned}")

    _append_result(results, _evaluate_transfer_model(
        vgg16_path,
        "VGG16 Fine-Tuned",
        'vgg16',
        test_images,
        test_labels,
        batch_size
    ))
    _append_result(results, _evaluate_transfer_model(
        resnet_path,
        "ResNet Fine-Tuned",
        'resnet',
        test_images,
        test_labels,
        batch_size
    ))

    if results:
        print_summary_table(results)
        if save_plots:
            plot_and_save_results(results, output_dir=output_dir)
    else:
        print(
            "\n[Warning] No models were evaluated. "
            "Please verify the specified model paths."
        )


if __name__ == "__main__":
    main()
