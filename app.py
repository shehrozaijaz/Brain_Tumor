import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from sklearn.metrics import classification_report, confusion_matrix

from model import load_model, predict_image

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
)

@st.cache_resource
def get_mobilenet_model():
    if os.path.exists("output/brain_tumor_mobilenet.h5"):
        return load_model("output/brain_tumor_mobilenet.h5")
    else:
        st.error("MobileNetV2 model file not found. Please train the model first.")
        return None

@st.cache_resource
def get_resnet_model():
    if os.path.exists("output/brain_tumor_resnet.h5"):
        return load_model("output/brain_tumor_resnet.h5")
    else:
        st.error("ResNet50 model file not found. Please train the model first.")
        return None

@st.cache_data
def get_labels():
    """Load class labels"""
    if os.path.exists("output/labels.txt"):
        with open("output/labels.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        return ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def get_evaluation_results():
    """Load and display evaluation results"""
    # Load classification report
    report = ""
    if os.path.exists("output/classification_report.txt"):
        with open("output/classification_report.txt", "r") as f:
            report = f.read()
    else:
        report = "Classification report not found. Please train the model first."
    
    # Load confusion matrix
    cm_img = None
    if os.path.exists("output/confusion_matrix.png"):
        cm_img = Image.open("output/confusion_matrix.png")
    
    return report, cm_img

def load_image(uploaded_file):
    """Load an image from the uploaded file"""
    if uploaded_file is not None:
        # Read image as bytes
        image_bytes = uploaded_file.getvalue()
        
        # Convert to numpy array
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        # Convert from BGR to RGB (for display)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, image_rgb
    
    return None, None

def main():
    # Load models and labels
    mobilenet_model = get_mobilenet_model()
    resnet_model = get_resnet_model()
    labels = get_labels()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📖 Description", "🧪 Predict", "📊 Evaluate"])
    
    # Tab 1: Description
    with tab1:
        st.title("🧠 Brain Tumor Detection Project")
        st.markdown("""
        ## Project Overview
        
        This project uses a Convolutional Neural Network (CNN) to detect different types of brain tumors from MRI images.
        
        ### Tumor Types Covered:
        
        * **Glioma**: A type of tumor that occurs in the brain and spinal cord
        * **Meningioma**: A tumor that forms on membranes covering the brain and spinal cord
        * **Pituitary**: A tumor that forms in the pituitary gland
        * **No Tumor**: Normal brain MRI with no tumor
        
        ### How to Use:
        
        1. Go to the "Predict" tab
        2. Upload an MRI image
        3. The model will classify the image and provide the tumor type and confidence
        
        ### Model Evaluation:
        
        Check the "Evaluate" tab to see the model's performance metrics and confusion matrix.
        """)
        
        # Show example images if available
        st.subheader("Example Images")
        st.markdown("Here are some examples of different types of brain tumors:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            with col1:
                st.image("Datasets/Training/glioma_tumor/1.jpg", caption="Glioma Tumor")
            with col2:
                st.image("Datasets/Training/meningioma_tumor/1.jpg", caption="Meningioma Tumor")
            with col3:
                st.image("Datasets/Training/no_tumor/1.jpg", caption="No Tumor")
            with col4:
                st.image("Datasets/Training/pituitary_tumor/1.jpg", caption="Pituitary Tumor")
        except:
            st.info("Example images not found. Make sure your dataset is correctly organized.")
    
    # Tab 2: Predict
    with tab2:
        st.title("🧪 Brain Tumor Prediction")
        st.markdown("Upload a brain MRI image to predict the tumor type.")
        
        uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display the image
            image, image_rgb = load_image(uploaded_file)
            
            if image is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image_rgb, caption="Uploaded MRI Image", use_container_width=True)
                
                with col2:
                    st.info("Processing image... Please wait.")
                    
                    if mobilenet_model is not None and resnet_model is not None:
                        # Predict with both models
                        idx_m, conf_m, name_m = predict_image(mobilenet_model, image, image_size=150, labels=labels)
                        idx_r, conf_r, name_r = predict_image(resnet_model, image, image_size=150, labels=labels)
                        # Get full probabilities
                        img_resized = cv2.resize(image, (150, 150)) / 255.0
                        img_expanded = np.expand_dims(img_resized, axis=0)
                        probs_m = mobilenet_model.predict(img_expanded)[0] * 100
                        probs_r = resnet_model.predict(img_expanded)[0] * 100
                        # Choose best
                        if conf_m >= conf_r:
                            best_idx, best_conf, best_name, best_model = idx_m, conf_m, name_m, "MobileNetV2"
                            best_probs = probs_m
                        else:
                            best_idx, best_conf, best_name, best_model = idx_r, conf_r, name_r, "ResNet50"
                            best_probs = probs_r
                        st.success(f"Prediction complete! (Best: {best_model})")
                        st.markdown(f"### Detected: **{best_name.replace('_', ' ').title()}**")
                        st.markdown(f"### Confidence: **{best_conf:.2f}%**")
                        st.markdown(f"### Model: **{best_model}**")
                        st.progress(best_conf / 100)
                        st.subheader("Prediction Probabilities (Both Models)")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        y_pos = np.arange(len(labels))
                        readable_labels = [label.replace('_', ' ').title() for label in labels]
                        bars_m = ax.barh(y_pos - 0.2, probs_m, height=0.4, label='MobileNetV2', color='skyblue')
                        bars_r = ax.barh(y_pos + 0.2, probs_r, height=0.4, label='ResNet50', color='lightgreen')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(readable_labels)
                        ax.invert_yaxis()
                        ax.set_xlabel('Probability (%)')
                        ax.set_title('Prediction Probabilities')
                        for i, v in enumerate(probs_m):
                            ax.text(v + 1, i - 0.2, f'{v:.1f}%', va='center', fontsize=8, color='blue')
                        for i, v in enumerate(probs_r):
                            ax.text(v + 1, i + 0.2, f'{v:.1f}%', va='center', fontsize=8, color='green')
                        bars = bars_m if best_model == "MobileNetV2" else bars_r
                        bars[best_idx].set_color('red')
                        ax.legend()
                        st.pyplot(fig)
    
    # Tab 3: Evaluate
    with tab3:
        st.title("📊 Model Evaluation")
        st.markdown("This section shows the performance metrics of the trained model.")
        
        # Get evaluation results
        report, cm_img = get_evaluation_results()
        
        # Display classification report
        st.subheader("Classification Report")
        st.text(report)
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        if cm_img is not None:
            st.image(cm_img, use_container_width=True)
        else:
            st.warning("Confusion matrix not found. Please train the model first.")
        
        # Display training history plots
        st.subheader("Training History")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists("output/accuracy_plot.png"):
                st.image("output/accuracy_plot.png", caption="Accuracy", use_container_width=True)
            else:
                st.warning("Accuracy plot not found.")
        
        with col2:
            if os.path.exists("output/loss_plot.png"):
                st.image("output/loss_plot.png", caption="Loss", use_container_width=True)
            else:
                st.warning("Loss plot not found.")

if __name__ == "__main__":
    main()