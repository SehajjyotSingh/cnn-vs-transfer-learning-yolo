from tensorflow.keras import applications, datasets

from src.transfer_learning import (
    build_finetuned_model,
    save_finetuned_model,
    train_and_evaluate_transfer,
)

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
    save_finetuned_model(vgg_model, 'vgg16')
    
    # Run ResNet50
    res_model, res_base = build_finetuned_model('resnet50')
    res_results = train_and_evaluate_transfer(res_model, res_base, (train_images_res, train_labels), (test_images_res, test_labels), "ResNet50")
    save_finetuned_model(res_model, 'resnet50')

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
