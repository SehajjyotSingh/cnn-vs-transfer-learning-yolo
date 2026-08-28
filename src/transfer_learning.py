import os
import time

import tensorflow as tf
from tensorflow.keras import applications, layers, models


def build_finetuned_model(base_network_name='vgg16'):
    # Input size for CIFAR-10
    inputs = layers.Input(shape=(32, 32, 3))

    # Upscale images so large networks can extract high-quality features
    x = layers.UpSampling2D(size=(2, 2))(inputs)  # Transforms 32x32 to 64x64

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


def save_finetuned_model(model, original_name, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_name))[0]
    model_path = os.path.join(save_dir, f"{base_name}_finetune.h5")
    model.save(model_path)
    return model_path
