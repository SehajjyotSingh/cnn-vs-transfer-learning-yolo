import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications

def build_finetuned_model(base_network_name='vgg16'):
    # Input size for CIFAR-10
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Upscale images so large networks can extract high-quality features
    x = layers.UpSampling2D(size=(2,2))(inputs) # Transforms 32x32 to 64x64
    
    if base_network_name == 'vgg16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=x)
    elif base_network_name == 'resnet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    
    # Step 1: Freeze the base model layers first (Feature Extraction phase)
    base_model.trainable = False
    
    # Add custom classification head on top
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model, base_model

def train_and_evaluate_transfer(model, base_model, train_data, test_data, name="Model"):
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    # --- PHASE 1: Warmup Custom Head ---
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print(f"\n--- Training {name}: Feature Extraction Phase ---")
    model.fit(train_images, train_labels, epochs=2, batch_size=64, validation_data=(test_images, test_labels))
    
    # --- PHASE 2: Fine-Tuning ---
    # Unfreeze the last few layers of the base network
    base_model.trainable = True
    # Freeze everything EXCEPT the final block/layers
    fine_tune_at = len(base_model.layers) - 4
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a very low learning rate to safely fine-tune weights
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print(f"\n--- Fine-Tuning {name}: Adjusting Weights ---")
    start_time = time.time()
    history = model.fit(train_images, train_labels, epochs=2, batch_size=64, validation_data=(test_images, test_labels))
    end_time = time.time()
    
    # Evaluate performance
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    return {
        'time': end_time - start_time,
        'train_acc': history.history['accuracy'][-1],
        'val_acc': test_acc,
        'val_loss': test_loss
    }

def main():
    print("Loading data...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Preprocess inputs correctly according to respective model needs
    train_images_vgg = applications.vgg16.preprocess_input(train_images.astype('float32'))
    test_images_vgg = applications.vgg16.preprocess_input(test_images.astype('float32'))
    
    train_images_res = applications.resnet50.preprocess_input(train_images.astype('float32'))
    test_images_res = applications.resnet50.preprocess_input(test_images.astype('float32'))

    # Run VGG16
    vgg_model, vgg_base = build_finetuned_model('vgg16')
    vgg_results = train_and_evaluate_transfer(vgg_model, vgg_base, (train_images_vgg, train_labels), (test_images_vgg, test_labels), "VGG16")
    
    # Run ResNet50
    res_model, res_base = build_finetuned_model('resnet50')
    res_results = train_and_evaluate_transfer(res_model, res_base, (train_images_res, train_labels), (test_images_res, test_labels), "ResNet50")

    # Print Comparative Table
    print("\n" + "="*65)
    print("             PRE-TRAINED FINE-TUNING COMPARISON             ")
    print("="*65)
    print(f"{'Architecture':<15} | {'Fine-Tune Time':<15} | {'Train Acc':<12} | {'Val Acc':<12}")
    print("-"*65)
    print(f"{'VGG16':<15} | {vgg_results['time']:.2f}s | {vgg_results['train_acc']*100:.2f}% | {vgg_results['val_acc']*100:.2f}%")
    print(f"{'ResNet50':<15} | {res_results['time']:.2f}s | {res_results['train_acc']*100:.2f}% | {res_results['val_acc']*100:.2f}%")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()