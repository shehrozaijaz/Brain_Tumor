import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data_preprocessing import load_and_preprocess_data
from model import create_model, create_resnet_model, train_model, save_model, visualize_training_history

def main():
    parser = argparse.ArgumentParser(description='Train a brain tumor classification model')
    parser.add_argument('--data_dir', type=str, default='C:/Users/Hp/Downloads/BT_Dataset', help='Directory containing the datasets')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=150, help='Size of input images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and results')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, labels = load_and_preprocess_data(
        base_dir=args.data_dir,
        image_size=args.image_size
    )
    
    # --- ResNet50 ---
    print("Creating ResNet50 model...")
    resnet_model = create_resnet_model(input_shape=(args.image_size, args.image_size, 3))
    resnet_model.summary()

    print("Training ResNet50 model...")
    history_resnet = train_model(
        resnet_model, X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_augmentation=not args.no_augmentation
    )

    print("Evaluating ResNet50 model...")
    test_loss_resnet, test_accuracy_resnet = resnet_model.evaluate(X_test, y_test)
    print(f"ResNet50 Test accuracy: {test_accuracy_resnet:.4f}")
    print(f"ResNet50 Test loss: {test_loss_resnet:.4f}")

    y_pred_resnet = resnet_model.predict(X_test)
    y_pred_classes_resnet = np.argmax(y_pred_resnet, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    report_resnet = classification_report(y_true_classes, y_pred_classes_resnet, target_names=labels)
    print("ResNet50 Classification Report:")
    print(report_resnet)

    with open(os.path.join(args.output_dir, 'classification_report_resnet.txt'), 'w') as f:
        f.write(report_resnet)

    cm_resnet = confusion_matrix(y_true_classes, y_pred_classes_resnet)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - ResNet50')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_resnet.png'))
    plt.close()

    acc_fig_r, loss_fig_r = visualize_training_history(history_resnet)
    acc_fig_r.savefig(os.path.join(args.output_dir, 'accuracy_plot_resnet.png'))
    loss_fig_r.savefig(os.path.join(args.output_dir, 'loss_plot_resnet.png'))
    plt.close('all')

    model_path_resnet = os.path.join(args.output_dir, 'brain_tumor_resnet.h5')
    save_model(resnet_model, model_path_resnet)

    # --- MobileNetV2 ---
    print("Creating MobileNetV2 model...")
    mobilenet_model = create_model(input_shape=(args.image_size, args.image_size, 3))
    mobilenet_model.summary()

    print("Training MobileNetV2 model...")
    history_mobilenet = train_model(
        mobilenet_model, X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_augmentation=not args.no_augmentation
    )

    print("Evaluating MobileNetV2 model...")
    test_loss_mobilenet, test_accuracy_mobilenet = mobilenet_model.evaluate(X_test, y_test)
    print(f"MobileNetV2 Test accuracy: {test_accuracy_mobilenet:.4f}")
    print(f"MobileNetV2 Test loss: {test_loss_mobilenet:.4f}")

    y_pred_mobilenet = mobilenet_model.predict(X_test)
    y_pred_classes_mobilenet = np.argmax(y_pred_mobilenet, axis=1)
    report_mobilenet = classification_report(y_true_classes, y_pred_classes_mobilenet, target_names=labels)
    print("MobileNetV2 Classification Report:")
    print(report_mobilenet)

    with open(os.path.join(args.output_dir, 'classification_report_mobilenet.txt'), 'w') as f:
        f.write(report_mobilenet)

    cm_mobilenet = confusion_matrix(y_true_classes, y_pred_classes_mobilenet)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_mobilenet, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - MobileNetV2')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_mobilenet.png'))
    plt.close()

    acc_fig, loss_fig = visualize_training_history(history_mobilenet)
    acc_fig.savefig(os.path.join(args.output_dir, 'accuracy_plot_mobilenet.png'))
    loss_fig.savefig(os.path.join(args.output_dir, 'loss_plot_mobilenet.png'))
    plt.close('all')

    model_path_mobilenet = os.path.join(args.output_dir, 'brain_tumor_mobilenet.h5')
    save_model(mobilenet_model, model_path_mobilenet)

    # Save labels (shared)
    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Training complete. Models and results saved to {args.output_dir}")

if __name__ == "__main__":
    main()