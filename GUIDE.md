# Complete Guide to Brain Tumor Classification Project

This guide will walk you through setting up and running the Brain Tumor Classification project step by step.

## 1. Environment Setup

### Clone or Download the Project

Ensure you have all the necessary files in your project directory:
- `data_preprocessing.py`
- `model.py`
- `train.py`
- `app.py`
- `check_environment.py`
- `run.py`
- `requirements.txt`

### Create and Activate Virtual Environment

**Windows Command Prompt:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation

The project expects the dataset to be organized as follows:

```
Brain_tumor_classification/
└── Datasets/
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

Make sure you have placed the MRI images in the appropriate folders.

## 3. Check Your Environment

Before running the project, check that your environment is properly set up:

```bash
python check_environment.py
```

Or using the run script:

```bash
python run.py check
```

This will verify:
- Python version
- Required packages
- GPU availability
- Dataset structure

## 4. Train the Model

To train the model with default parameters:

```bash
python train.py
```

Or using the run script:

```bash
python run.py train
```

**Advanced Options:**

```bash
python run.py train --epochs 30 --batch_size 64 --image_size 150 --output_dir output
```

Parameters:
- `--data_dir`: Directory containing the datasets (default: 'Datasets')
- `--epochs`: Number of epochs to train (default: 20)
- `--batch_size`: Batch size for training (default: 32)
- `--image_size`: Size of input images (default: 150)
- `--output_dir`: Directory to save model and results (default: 'output')
- `--no_augmentation`: Disable data augmentation (flag)

The training process will:
1. Load and preprocess the data
2. Create and compile the model
3. Train the model
4. Evaluate performance
5. Save the model and results to the output directory

## 5. Run the Streamlit Web Application

To Run the web application:

```bash
streamlit run app.py
```

Or using the run script:

```bash

```

The application will start and automatically open in your default web browser (typically at http://localhost:8501).

## 6. Using the Web Application

The web application has three tabs:

### Description Tab
- Overview of the project
- Information about different tumor types
- Example images

### Predict Tab
- Upload an MRI image
- View the prediction results
- See prediction confidence for each class

### Evaluate Tab
- View classification report with precision, recall, and F1-score
- Examine the confusion matrix
- Review training and validation accuracy/loss plots

## 7. Model Architecture

The CNN model architecture consists of:

```
1. Conv2D(32, (3,3), activation='relu')
2. MaxPooling2D(2,2)
3. Conv2D(64, (3,3), activation='relu')
4. MaxPooling2D(2,2)
5. Conv2D(128, (3,3), activation='relu')
6. MaxPooling2D(2,2)
7. Dropout(0.4)
8. Flatten()
9. Dense(256, activation='relu')
10. Dropout(0.4)
11. Dense(128, activation='relu')
12. Dropout(0.3)
13. Dense(4, activation='softmax')
```

## 8. Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**
   - Ensure your virtual environment is activated
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. **Dataset Not Found**
   - Check that folder structure matches the expected format
   - Verify images are in the correct folders

3. **GPU Not Detected**
   - Update GPU drivers
   - Ensure CUDA and cuDNN are installed (if using NVIDIA GPU)
   - Note: The model will still work on CPU, just slower

4. **Memory Issues During Training**
   - Reduce batch size: `python run.py train --batch_size 16`
   - Reduce image size: `python run.py train --image_size 128`

5. **Low Accuracy**
   - Try increasing epochs: `python run.py train --epochs 50`
   - Ensure dataset is balanced and representative

## 9. Extending the Project

Here are some ways to extend or improve the project:

1. **Add more data augmentation techniques**
   - Modify the `ImageDataGenerator` in `model.py`

2. **Try different model architectures**
   - Experiment with deeper networks
   - Try transfer learning with pre-trained models like VGG16, ResNet, etc.

3. **Implement additional visualizations**
   - Add Grad-CAM to visualize where the model is focusing
   - Add more detailed metrics

4. **Improve the UI**
   - Add more interactive elements
   - Add batch processing capabilities

## 10. Additional Notes

- The model training time depends on your hardware. With a GPU, it should take 5-15 minutes for 20 epochs. On CPU, it might take significantly longer.
- For best results, ensure your dataset is balanced with similar numbers of images in each class.
- The default image size is 150x150 pixels, which balances accuracy and training speed.