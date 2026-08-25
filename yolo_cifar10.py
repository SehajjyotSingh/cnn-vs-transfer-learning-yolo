import os
import cv2
import numpy as np
from tensorflow.keras import datasets
from ultralytics import YOLO

def create_yolo_dataset():
    """
    YOLO requires images to be physically saved in folders on your disk.
    This function extracts CIFAR-10 from Keras and saves them as JPGs.
    """
    dataset_dir = "cifar10_yolo_data"
    
    # If the folder already exists, we assume the data is already prepared
    if os.path.exists(dataset_dir):
        print(f"Dataset folder '{dataset_dir}' already exists. Skipping extraction.")
        return dataset_dir

    print("1. Downloading CIFAR-10 and saving to disk for YOLO (this takes a moment)...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # Create directories
    for split, images, labels in [('train', train_images, train_labels), ('val', test_images, test_labels)]:
        for class_name in class_names:
            os.makedirs(os.path.join(dataset_dir, split, class_name), exist_ok=True)
            
        print(f"Saving {split} images...")
        for i, (img, label) in enumerate(zip(images, labels)):
            class_name = class_names[label[0]]
            img_path = os.path.join(dataset_dir, split, class_name, f"{i}.jpg")
            # CIFAR-10 is RGB, OpenCV expects BGR for saving
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img_bgr)
            
    print("Data extraction complete!")
    return dataset_dir

def main():
    # 1. Prepare Data
    dataset_path = create_yolo_dataset()
    
    # 2. Load the pre-trained YOLOv8 classification model
    # 'yolov8n-cls.pt' is the Nano version (fastest). 
    print("\n2. Loading YOLOv8 Classification Model...")
    model = YOLO('yolov8n-cls.pt')  

    # 3. Train the model for 25 epochs
    print("\n3. Starting YOLO Training for 25 Epochs...")
    # YOLO automatically handles timing, accuracy, and loss tracking
    results = model.train(
        data=dataset_path,  # Path to the dataset folder
        epochs=25,          # Train for 25 epochs
        imgsz=32,           # CIFAR-10 image size
        batch=128,          # Number of images per batch
        project='YOLO_CIFAR10_Results', # Folder to save results
        name='25_epochs_run'
    )
    
    print("\nTraining Complete! YOLO automatically saves all plots (Loss, Accuracy, Confusion Matrix) in the 'YOLO_CIFAR10_Results' folder.")

if __name__ == "__main__":
    main()