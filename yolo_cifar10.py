from src.yolo import create_yolo_dataset,YOLO
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