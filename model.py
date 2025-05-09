import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight

def create_model(input_shape=(150, 150, 3)):
    """
    Create a transfer learning model using MobileNetV2 as the base.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # Freeze base model for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    return model

def create_resnet_model(input_shape=(150, 150, 3)):
    """
    Create a transfer learning model using ResNet50 as the base.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # Freeze base model for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32, data_augmentation=True):
    """
    Train the model with optional data augmentation
    
    Args:
        model: Compiled model
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        epochs: Number of epochs to train
        batch_size: Batch size for training
        data_augmentation: Whether to use data augmentation
        
    Returns:
        Training history
    """
    if data_augmentation:
        # Restore to original, simpler augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            epochs=epochs,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
    return history

def save_model(model, model_path="model.h5"):
    """Save the trained model"""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path="model.h5"):
    """Load a trained model"""
    return tf.keras.models.load_model(model_path)

def visualize_training_history(history):
    """
    Visualize the training history
    
    Args:
        history: Model training history
        
    Returns:
        fig1, fig2: Figures for accuracy and loss plots
    """
    # Accuracy plot
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Loss plot
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    return fig1, fig2

def predict_image(model, img, image_size=150, labels=None):
    """
    Predict tumor type from an image
    
    Args:
        model: Trained model
        img: Input image (numpy array)
        image_size: Size to resize image to
        labels: List of class names
        
    Returns:
        class_idx: Predicted class index
        confidence: Prediction confidence
        class_name: Name of predicted class (if labels provided)
    """
    if labels is None:
        labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    # Preprocess image
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    
    return class_idx, confidence, labels[class_idx]

def fine_tune_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    """
    Fine-tune the model by unfreezing the last 30 layers of MobileNetV2 and training for a few more epochs
    
    Args:
        model: Compiled model
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        epochs: Number of epochs to train
        batch_size: Batch size for training
        
    Returns:
        Fine-tuned training history
    """
    # Unfreeze the last 30 layers of MobileNetV2
    base_model = model.layers[0]
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training for a few more epochs
    history_finetune = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    return history_finetune