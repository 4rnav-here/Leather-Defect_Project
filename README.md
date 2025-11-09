# 🪡 Leather Defect Detector (Arpit bhosdiwwale bkl clone na kr diyo yeh maa chod dunga...sahil same for you)
*An AI-powered leather quality inspector that spots scratches, wrinkles, holes — and even fakes that aren’t leather.*

---

## 🧩 TL;DR — Quick Setup

```bash
git clone https://github.com/4rnav-here/Leather-Defect_Project.git
cd Leather-Defect_Project/Leather-defect

# Create a virtual environment
python -m venv .venv

# Activate it
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# (Optional) Train both models
python train.py

# Run the Streamlit app
streamlit run app.py
Then open the local URL shown in your terminal — typically http://localhost:8501.

🧠 About the Project
The Leather Defect Detector is a deep learning–based computer vision project built to detect and classify surface defects on leather automatically.

It helps manufacturers skip the slow, error-prone manual inspection process by using modern AI to analyze textures, recognize defect patterns, and even detect when an image isn’t leather at all.

⚙️ Models Inside
Two models battle it out:

PlainCNN — A clean, minimal convolutional network. Fast, reliable, straightforward.

HybridCNN-QNN — A quantum-inspired twist on CNNs that adds a sinusoidal nonlinearity layer for richer feature expression (it’s not actually quantum, but the math vibes with it).

Both models are implemented in PyTorch, trained on your leather dataset, and saved as .pth checkpoints.

The app lets you compare, evaluate, and visualize their results interactively.

🏗️ Project Structure
graphql
Copy code
Leather-defect/
├─ Assets/Leather Defect Classification/   # Dataset (ImageFolder style: subfolders = class labels)
├─ models/
│  ├─ plain_cnn.py                         # Baseline CNN
│  └─ hybrid_cnn.py                        # Fancy quantum-inspired CNN
├─ utils/
│  ├─ dataset.py                           # Data loading and augmentation
│  ├─ trainer.py                           # Training loop and checkpoint saver
│  └─ metrics.py                           # (Optional) Confusion matrix plotting helpers
├─ app.py                                  # Streamlit web app for inference and visualization
├─ train.py                                # Training and evaluation orchestrator
├─ plain_cnn.pth                           # Saved PlainCNN model (after training)
├─ hybrid_cnn.pth                          # Saved HybridCNN-QNN model
├─ metrics.json                            # Metrics and confusion matrices from training
├─ requirements.txt
└─ README.md
🧰 Requirements
Python 3.8 or newer

Git

(Optional) GPU with CUDA for faster training

Dataset organized like this:

bash
Copy code
Assets/Leather Defect Classification/
├─ crack/
├─ wrinkle/
├─ hole/
├─ fold/
└─ NotLeather/
Each folder should contain JPEG or PNG images belonging to that defect category.

🚀 Setup Guide
1️⃣ Clone the repo
bash
Copy code
git clone https://github.com/4rnav-here/Leather-Defect_Project.git
cd Leather-Defect_Project/Leather-defect
2️⃣ Create a virtual environment
bash
Copy code
python -m venv .venv
3️⃣ Activate it
macOS / Linux

bash
Copy code
source .venv/bin/activate
Windows (PowerShell)

powershell
Copy code
.venv\Scripts\Activate.ps1
Windows (CMD)

cmd
Copy code
.venv\Scripts\activate.bat
4️⃣ Install dependencies
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
If you prefer a quick install, these are the essentials:

bash
Copy code
pip install torch torchvision streamlit pillow scikit-learn matplotlib seaborn
5️⃣ (Optional) Train the models
bash
Copy code
python train.py
This will:

Load the dataset using torchvision.datasets.ImageFolder

Split into train (80%) and validation (20%)

Train both PlainCNN and HybridCNN-QNN

Save checkpoints (plain_cnn.pth, hybrid_cnn.pth)

Generate metrics.json and confusion matrix images

6️⃣ Launch the app
bash
Copy code
streamlit run app.py
7️⃣ Explore the Interface
About: Learn about the models and workflow

Model: Upload or sample an image → choose model → get prediction + confidence

Evaluation: View confusion matrices

Metrics: Explore per-class and overall performance comparisons

Contact Us: Developer info

🧬 Workflow Summary
Data Loading:
The dataset is organized in subfolders per class and loaded using torchvision.datasets.ImageFolder.

Training:
train.py calls:

PlainCNN → baseline model

HybridCNNQNN → enhanced version
Both are trained using CrossEntropyLoss and Adam optimizer with best validation checkpoint saving.

Evaluation:
Post-training, metrics (accuracy, precision, recall, F1-score, confusion matrix) are calculated and stored in metrics.json.

Deployment:
The trained models are loaded inside a Streamlit app, where users can upload or sample images and instantly view predictions and metrics interactively.

🧮 Example Checkpoint Structure
Each saved model includes:

python
Copy code
torch.save({
    "model_state_dict": model.state_dict(),
    "val_acc": best_val,
    "classes": ["crack", "wrinkle", "hole", "fold", "NotLeather"]
}, "plain_cnn.pth")
🧤 Troubleshooting Tips
Problem	Possible Fix
Checkpoint mismatch	Delete .pth and retrain.
Streamlit says checkpoint missing	Run train.py first.
Training too slow	Reduce epochs or batch size.
App not recognizing dataset	Check folder name → Assets/Leather Defect Classification/.
Metrics not showing	Ensure metrics.json exists (generated during training).
CUDA not detected	Install the correct GPU build of PyTorch.

💡 Future Improvements
Include Grad-CAM visual explanations

Add Dockerfile for one-command deployment

Implement transfer learning (ResNet / EfficientNet)

Extend dataset and improve augmentation diversity

If you clone this repo and improve it — please ⭐️ it!
If you break it — file an issue and we’ll help you debug it. 😉

🧵 Final Words
This project is more than a neural network — it’s a complete, modular AI workflow that blends practical ML, visualization, and deployment.

It’s built to be educational, reproducible, and fun to tinker with.
Train it, tweak it, break it, and most importantly — make it better.

Happy debugging and may your validation accuracy be ever in your favor. 🧠🔥