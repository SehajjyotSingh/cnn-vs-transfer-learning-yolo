import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

def main():
    print("1. Loading and normalizing CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    print("2. Building the CNN model from scratch...")
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("3. Training the model...")
    start_train = time.time()
    history = model.fit(train_images, train_labels, epochs=25, 
                        validation_data=(test_images, test_labels))
    end_train = time.time()
    training_time = end_train - start_train

    print("4. Evaluating on test set...")
    start_infer = time.time()
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    end_infer = time.time()
    inference_time = end_infer - start_infer

    print("5. Generating Predictions for Advanced Metrics...")
    raw_predictions = model.predict(test_images, verbose=0)
    probabilities = tf.nn.softmax(raw_predictions).numpy()
    predicted_classes = np.argmax(probabilities, axis=1)

    report = classification_report(test_labels, predicted_classes, target_names=class_names, output_dict=True)
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    print("\n" + "="*50)
    print("         CUSTOM CNN FROM SCRATCH SUMMARY          ")
    print("="*50)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-"*50)
    print(f"{'Total Training Time':<30} | {training_time:.2f} seconds")
    print(f"{'Total Inference Time':<30} | {inference_time:.4f} seconds")
    print(f"{'Final Training Accuracy':<30} | {history.history['accuracy'][-1]*100:.2f}%")
    print(f"{'Final Training Loss':<30} | {history.history['loss'][-1]:.4f}")
    print(f"{'Final Test/Val Accuracy':<30} | {test_acc*100:.2f}%")
    print(f"{'Final Test/Val Loss':<30} | {test_loss:.4f}")
    print(f"{'Macro Precision':<30} | {macro_precision*100:.2f}%")
    print(f"{'Macro Recall':<30} | {macro_recall*100:.2f}%")
    print(f"{'Macro F1-Score':<30} | {macro_f1*100:.2f}%")
    print("="*50 + "\n")

    # --- PLOT 1: Training vs Validation Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy (Scratch Model)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # --- PLOT 2: Confusion Matrix Heatmap ---
    cm = confusion_matrix(test_labels, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # --- PLOT 3: Precision-Recall Curves ---
    plt.figure(figsize=(10, 8))
    one_hot_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    for i in range(10):
        precision, recall, _ = precision_recall_curve(one_hot_labels[:, i], probabilities[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves per Class')
    plt.legend(loc='lower left')
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()