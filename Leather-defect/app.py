import os
import random
import json
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

# Import models from the new structure
from models import PlainCNN, HybridCNNQNN

# Load environment variables
load_dotenv()

# Environment & constants
DATA_DIR = os.getenv("DATA_DIR", "Assets/Leather Defect Classification")
CHECKPOINT_PLAIN = os.getenv("CHECKPOINT_PLAIN", "plain_cnn.pth")
CHECKPOINT_HYBRID = os.getenv("CHECKPOINT_HYBRID", "hybrid_cnn.pth")

st.set_page_config(page_title="Leather Defect Detector", page_icon="🪡")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Model", "Evaluation", "Metrics", "Contact Us"])

# ------------------ PAGE 1: About ------------------
if page == "About":
    st.title("🧠 Leather Defect Classifier")
    st.subheader("Project Overview")

    st.write("""
    The **Leather Defect Classifier** is an AI-powered computer vision system built to automatically 
    **detect and classify surface defects in leather materials**. It eliminates the need for tedious 
    manual inspection by leveraging deep learning, helping manufacturers maintain consistent quality 
    while saving significant time and effort.

    ---
    ### 🧩 How It Works
    1. **Dataset Preparation:**  
       The system uses a labeled dataset of leather images, each categorized by defect types such as 
       *cracks*, *wrinkles*, *holes*, *folding marks*, and a dedicated *NotLeather* class to ensure 
       robust performance when non-leather images are uploaded.  
       Images are augmented using PyTorch’s `torchvision.transforms` (rotation, flipping, normalization) 
       to improve generalization.

    2. **Model Training:**  
       Two CNN-based architectures are trained using PyTorch:
       - **PlainCNN:** A conventional convolutional neural network designed with simplicity to serve as a baseline model.  
       - **HybridCNN-QNN:** An enhanced “hybrid” model that integrates classical CNN layers with a 
         *quantum-inspired nonlinear block* (sine-based transformation) to improve feature expressiveness and 
         generalization.

       Both models are trained, validated, and saved as PyTorch checkpoints (`.pth` files), and the training process 
       includes dynamic accuracy tracking and confusion matrix generation.

    3. **Evaluation and Metrics:**  
       After training, both models are evaluated using key metrics:
       - Accuracy  
       - Precision  
       - Recall  
       - F1-score  
       - Confusion Matrix  

       These metrics are stored in `metrics.json` and visualized interactively in the **Metrics** page of this app.

    4. **Real-Time Inference:**  
       The Streamlit interface allows users to:
       - **Upload a leather image** or pick a random dataset sample.
       - Select either **Plain CNN** or **Hybrid CNN** for prediction.
       - Get the predicted defect type with a **confidence score**.
       - Receive a warning if the uploaded image is **not leather**, using a confidence threshold or the *NotLeather* class prediction.

    ---

    ### 🧠 Why These Technologies Were Used
    - **PyTorch:** for model definition, training, and evaluation due to its flexibility and GPU support.  
    - **Torchvision:** for image augmentation and easy dataset management using `ImageFolder`.  
    - **Streamlit:** to build an interactive web-based interface that supports file uploads, visual results, and 
      real-time predictions with minimal code overhead.  
    - **Scikit-learn:** for calculating performance metrics and generating confusion matrices.  
    - **Matplotlib & Seaborn:** for high-quality visualization of evaluation metrics and confusion matrices.

    ---

    ### 🚀 Key Highlights
    - Dual-model architecture (Plain CNN and Hybrid CNN-QNN) for comparative analysis.  
    - Built-in evaluation pipeline with detailed metrics visualization.  
    - Integrated *NotLeather* detection to prevent false classifications.  
    - Real-time defect prediction and dataset sample testing.  
    - Modular design for easy extension with transfer learning or Grad-CAM explainability.

    ---
    **In short, this system demonstrates how deep learning and modern deployment tools can automate 
    the defect detection pipeline for industrial quality assurance — combining accuracy, transparency, 
    and usability in one unified platform.**
    """)


# ------------------ PAGE 2: Model ------------------
elif page == "Model":
    st.title("Leather Defect Detection")

    # Step 1: Choose Model
    model_choice = st.selectbox("Select a Model", ["Plain CNN", "Hybrid CNN"])
    st.success(f"You selected: {model_choice}")

    # Step 2: Select Image Source
    source_choice = st.radio("Select Image Source", ["Upload an Image", "Use Dataset Sample"])

    # Load image
    image = None
    if source_choice == "Upload an Image":
        uploaded_file = st.file_uploader("Upload your leather image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

    elif source_choice == "Use Dataset Sample":
        if not os.path.exists(DATA_DIR):
            st.error("Dataset not found. Please check the Assets folder.")
            st.stop()

        all_classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if all_classes:
            chosen_class = random.choice(all_classes)
            img_list = [f for f in os.listdir(os.path.join(DATA_DIR, chosen_class)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if img_list:
                sample_file = random.choice(img_list)
                sample_path = os.path.join(DATA_DIR, chosen_class, sample_file)
                image = Image.open(sample_path).convert("RGB")
                st.image(image, caption=f"Sample from {chosen_class}", use_column_width=True)
                st.info(f"True class: **{chosen_class}**")

    # Step 3: Inference
    if image is not None:
        infer_tf = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        ckpt_path = CHECKPOINT_PLAIN if model_choice == "Plain CNN" else CHECKPOINT_HYBRID
        if not os.path.exists(ckpt_path):
            st.error(f"Checkpoint {ckpt_path} not found. Please train the model first.")
            st.stop()

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        classes = ckpt.get("classes", [])
        num_classes = len(classes)

        model = PlainCNN(num_classes) if model_choice == "Plain CNN" else HybridCNNQNN(num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Prediction
        img_tensor = infer_tf(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_label = classes[pred_idx] if classes else f"class_{pred_idx}"
            pred_conf = float(probs[pred_idx])

        # Add confidence threshold for "not leather" detection
        CONF_THRESHOLD = 0.6  # tweak this based on testing

        if pred_conf < CONF_THRESHOLD:
            st.warning("⚠️ This image does not appear to be leather or contains unknown content.")
        else:
            st.success(f"Predicted Defect: **{pred_label}**  — Confidence: **{pred_conf:.2f}**")

        # Optional: show confidence chart
        st.bar_chart({classes[i]: probs[i] for i in range(len(classes))})
        
        
        
elif page == "Evaluation":
    st.title("📊 Model Evaluation")
    if os.path.exists("plain_confusion_matrix.png"):
        st.image("plain_confusion_matrix.png", caption="Plain CNN Confusion Matrix")
    if os.path.exists("hybrid_confusion_matrix.png"):
        st.image("hybrid_confusion_matrix.png", caption="Hybrid CNN Confusion Matrix")

    if os.path.exists("metrics.json"):
        metrics = json.load(open("metrics.json"))
        st.json(metrics)



# ------------------ PAGE: Metrics ------------------
elif page == "Metrics":
    import json, os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    st.title("📊 Model Performance Metrics")

    metrics_path = "metrics.json"
    if not os.path.exists(metrics_path):
        st.warning("⚠️ metrics.json not found. Please run training first.")
        st.stop()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    classes = metrics["classes"]
    plain_data = metrics["plain"]
    hybrid_data = metrics["hybrid"]

    # --- Overall Accuracy ---
    st.subheader("Model Accuracy Overview")
    acc_col1, acc_col2 = st.columns(2)
    acc_col1.metric("Plain CNN Accuracy", f"{plain_data['val_acc']*100:.2f}%")
    acc_col2.metric("Hybrid CNN Accuracy", f"{hybrid_data['val_acc']*100:.2f}%")

    st.divider()

    # --- Confusion Matrices ---
    st.subheader("Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    for col, title, data, file in [
        (cm_col1, "Plain CNN", plain_data, "plain_confusion_matrix.png"),
        (cm_col2, "Hybrid CNN", hybrid_data, "hybrid_confusion_matrix.png"),
    ]:
        cm = np.array(data["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"{title} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        col.pyplot(fig)

    st.divider()

    # --- Per-Class Metrics ---
    st.subheader("Detailed Classification Report")
    selected_model = st.selectbox("Choose Model", ["Plain CNN", "Hybrid CNN"])
    chosen_report = plain_data["report"] if selected_model == "Plain CNN" else hybrid_data["report"]

    df = pd.DataFrame(chosen_report).T
    df = df.round(3)
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))

    st.divider()

    # --- Comparison Bar Chart ---
    st.subheader("Overall Metric Comparison")

    plain_avg = plain_data["report"]["weighted avg"]
    hybrid_avg = hybrid_data["report"]["weighted avg"]
    comparison = pd.DataFrame({
        "Plain CNN": plain_avg,
        "Hybrid CNN": hybrid_avg
    }).loc[["precision", "recall", "f1-score"]]

    fig, ax = plt.subplots(figsize=(6, 3))
    comparison.T.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison (Weighted Avg)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close(fig) 

    st.caption("✅ Metrics automatically generated from the latest training run.")


# ------------------ PAGE 3: Contact ------------------
elif page == "Contact Us":
    st.title("Contact Us")
    st.subheader("Developers")
    st.write("""
    - **Aakriti Goenka (22BCE2062)**  
    - **Arnav Trivedi (22BCE2355)**  
    - **Arpit Pal (22BCE3576)**
    """)

