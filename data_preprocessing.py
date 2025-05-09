import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_and_preprocess_data(base_dir="Datasets", image_size=150):
    """
    Load and preprocess brain tumor MRI images from Training and Testing directories
    
    Args:
        base_dir: Base directory containing Training and Testing folders
        image_size: Size to resize images to
        
    Returns:
        X_train, X_test, y_train, y_test: Train and test data and labels
        labels: List of class names
    """
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    X = []
    y = []
    
    print("Loading and preprocessing data...")
    
    # Load and preprocess Training + Testing data
    for label in labels:
        for folder in ['Training', 'Testing']:
            folder_path = os.path.join(base_dir, folder, label)
            print(f"Processing {folder_path}...")
            
            # Skip if folder doesn't exist
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist. Skipping.")
                continue
                
            for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder}/{label}"):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, (image_size, image_size))
                    img = img / 255.0  # Normalize
                    X.append(img)
                    y.append(labels.index(label))
                else:
                    print(f"Warning: Could not read image {img_path}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle data
    X, y = shuffle(X, y, random_state=42)
    
    # Convert labels to categorical
    y_categorical = tf.keras.utils.to_categorical(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape}, Test samples: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, labels