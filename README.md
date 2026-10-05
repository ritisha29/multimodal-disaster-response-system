# Multimodal Disaster Tweet Informativeness Classification & Emergency Response System

An end-to-end multimodal deep learning pipeline that classifies disaster-related social media posts (combining text and imagery) to determine **informativeness** for humanitarian relief efforts, coupled with an interactive emergency response dashboard.

Built using the **CrisisMMD v2.0** dataset, comparing single-modality baselines against a late-fusion multimodal architecture, and featuring an emergency response assistant powered by Streamlit, Named Entity Recognition (NER), Google Places API, and NDRF resource allocation.

---

## 🚀 Key Features

- **Multimodal Deep Learning Architecture:**
  - **Text Baseline:** DistilBERT sequence classifier (`distilbert-base-uncased`) fine-tuned on disaster tweets.
  - **Image Baseline:** ResNet-18 deep convolutional network fine-tuned on disaster imagery.
  - **Multimodal Fusion Model:** Concatenation late-fusion MLP combining textual embeddings (768-d) and visual representations (512-d).
- **Comprehensive Evaluation & Benchmarks:** Side-by-side comparison across Accuracy, Precision, Recall, and F1-Scores.
- **Interactive Streamlit Web Dashboard:**
  - Live side-by-side inference comparing unimodal baselines vs. multimodal fusion.
  - Second-stage disaster analysis (Disaster type classification + NER location extraction).
  - Emergency response dispatch simulation.
  - Automated resource allocation (NDRF teams, relief kits, medical supplies).
  - Nearby hospital locator integrated with Google Places API.

---

## 📊 Benchmark Results (Test Set)

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Image-Only (ResNet-18)** | 84.16% | 0.8892 | 0.8728 | 0.8809 |
| **Text-Only (DistilBERT)** | 87.42% | 0.9131 | 0.8981 | 0.9055 |
| **Multimodal Fusion (MLP)** | **90.87%** | **0.9214** | **0.9447** | **0.9329** |

---

## 📁 Repository Structure

```text
urop_project/
├── .env.example              # Template for environment configuration
├── .gitignore                # Excludes checkpoints, raw data, and credentials
├── requirements.txt          # Python dependencies
├── kaggle_notebook.py        # Kaggle/Cloud training notebook script
├── results/
│   ├── confusion_matrix.png  # Evaluation confusion matrix plot
│   ├── evaluation_metrics.json
│   ├── evaluation_summary.txt
│   ├── knn_evaluation.json
│   └── knn_summary.txt
└── src/
    ├── app.py                # Streamlit live demonstration dashboard
    ├── dataset.py            # CrisisMMD PyTorch dataset & data loaders
    ├── disaster_analysis.py  # NER location extraction & disaster categorization
    ├── evaluate.py           # Test set evaluation and metrics computation
    ├── fusion_model.py       # Multimodal fusion model definition
    ├── hospital_finder.py    # Google Places API hospital locator
    ├── image_baseline.py     # ResNet-18 baseline model
    ├── kaggle_paths.py       # Path helpers for Kaggle/local execution
    ├── knn_baseline.py       # K-Nearest Neighbors baseline
    ├── resource_matcher.py   # Disaster-to-resource recommendation logic
    └── text_baseline.py      # DistilBERT baseline model
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and provide your Google Places API Key (optional, required for nearby hospital locator):
```bash
cp .env.example .env
```

---

## 🖥️ Running the Application

### Launch Streamlit Dashboard:
```bash
streamlit run src/app.py
```

### Run Model Evaluation:
```bash
python src/evaluate.py
```

---

## 📄 License
This project is developed for academic research and demonstration purposes.
