# Brain Tumor Classification Project

This project uses deep learning to classify brain tumors from MRI images into four categories: glioma, meningioma, pituitary, and no tumor.

The model and algorithm used in this project are as follows:
Model: The project uses a deep learning model based on MobileNetV2 as the base model. MobileNetV2 is a popular convolutional neural network architecture designed for efficient computation and is often used for transfer learning in image classification tasks.
Algorithm: The approach is transfer learning. The base MobileNetV2 model (pre-trained on ImageNet) is used with its weights frozen initially, and custom layers are added on top for classification. The model is trained using the Adam optimizer and categorical cross-entropy loss. There is also an option for fine-tuning, where the last 30 layers of MobileNetV2 are unfrozen and trained further on the specific dataset.
Summary:
Model: MobileNetV2 (with custom dense layers on top)
Algorithm: Transfer learning with fine-tuning, using Adam optimizer and categorical cross-entropy loss

## Project Structure

```
Brain_tumor_classification/
├── Datasets/               # Dataset directory
│   ├── Training/           # Training images
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── no_tumor/
│   │   └── pituitary_tumor/
│   └── Testing/            # Testing images
│       ├── glioma_tumor/
│       ├── meningioma_tumor/
│       ├── no_tumor/
│       └── pituitary_tumor/
├── output/                 # Output directory for model and results
│   ├── brain_tumor_model.h5
│   ├── labels.txt
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── accuracy_plot.png
│   └── loss_plot.png
├── venv/                   # Virtual environment
├── data_preprocessing.py   # Data preprocessing functions
├── model.py                # Model architecture and training functions
├── train.py                # Script to train the model
├── app.py                  # Streamlit web application
└── requirements.txt        # Required packages
```

## Installation Instructions

1. **Clone the repository or create the project structure**

2. **Create and activate a virtual environment**

   For Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   For macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**

   Ensure your dataset is organized as follows:
   ```
   Datasets/
   ├── Training/
   │   ├── glioma_tumor/
   │   ├── meningioma_tumor/
   │   ├── no_tumor/
   │   └── pituitary_tumor/
   └── Testing/
       ├── glioma_tumor/
       ├── meningioma_tumor/
       ├── no_tumor/
       └── pituitary_tumor/
   ```

   Each folder should contain MRI images corresponding to that tumor type.

## Usage

### Training the Model

```bash

python train.py

python train.py --data_dir Datasets --epochs 20 --batch_size 32
```

Options:
- `--data_dir`: Directory containing the datasets (default: 'Datasets')
- `--epochs`: Number of epochs to train (default: 20)
- `--batch_size`: Batch size for training (default: 32)
- `--image_size`: Size of input images (default: 150)
- `--output_dir`: Directory to save model and results (default: 'output')
- `--no_augmentation`: Disable data augmentation

### Running the Web Application

```bash
streamlit run app.py
```

The application will start and open in your default web browser.

### Using the Web Application

1. Navigate to the "Predict" tab
2. Upload a brain MRI image
3. The model will predict the tumor type and display the result

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture with the following layers:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers to reduce dimensionality
- Dropout layers to prevent overfitting
- Dense layers for classification with softmax activation

## Performance

The model's performance can be viewed in the "Evaluate" tab of the web application, which displays:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Training and validation accuracy/loss plots

## Acknowledgments

This project is based on a brain tumor classification tutorial and uses the Brain Tumor Classification MRI dataset.

## Requirements

- numpy
- pandas
- opencv-python
- tensorflow
- scikit-learn
- matplotlib
- seaborn
- streamlit
- pillow
- tqdm