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
page = st.sidebar.radio("Go to", ["About", "Model", "Evaluation", "Contact Us"])

# ------------------ PAGE 1: About ------------------
if page == "About":
    st.title("Leather Defect Classifier")
    st.subheader("Abstract")
    st.write("""
    The Leather Defect Classifier is an AI-powered system designed to automatically
    detect and classify defects in leather samples. Using a **Hybrid CNN model**, 
    it enhances the quality control process in the leather industry 
    by automating defect detection with high accuracy.

    ### Key Highlights:
    - Hybrid CNN architecture trained on real-world defect data.
    - Supports both image upload and dataset sample testing.
    - Reduces manual inspection time and human error.
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


        # Optional: Confidence chart
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

# ------------------ PAGE 3: Contact ------------------
elif page == "Contact Us":
    st.title("Contact Us")
    st.subheader("Developers")
    st.write("""
    - **Aakriti Goenka (22BCE2062)**  
    - **Arnav Trivedi (22BCE2355)**  
    - **Arpit Pal (22BCE3576)**
    """)

