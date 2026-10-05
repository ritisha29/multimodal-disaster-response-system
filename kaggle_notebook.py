# -*- coding: utf-8 -*-
"""
CrisisMMD Multimodal Disaster Informativeness Classification - Kaggle Notebook
================================================================================
This file contains the complete, self-contained sequence of notebook cells designed
to run on Kaggle Notebooks with GPU acceleration.

HOW TO USE IN KAGGLE:
---------------------
1. Open a new Kaggle Notebook (Set Accelerator: GPU T4 x2 or P100).
2. Add the CrisisMMD dataset to your notebook (/kaggle/input/...).
3. Copy and paste each CELL block below into a new Kaggle notebook cell in order.
4. Run each cell sequentially.

Cell Breakdown:
- CELL 1: Dependency Installation
- CELL 2: Environment Configuration, GPU Verification & Path Setup
- CELL 3: CrisisMMD Dataset Class & Split Verification
- CELL 4: Model 1 - Text-Only DistilBERT Baseline (Train & Evaluate)
- CELL 5: Model 2 - Image-Only ResNet-18 Baseline (Train & Evaluate)
- CELL 6: Model 3 - Multimodal Late Fusion Model (Train & Evaluate)
- CELL 7: Comprehensive Evaluation, Confusion Matrix & Qualitative Analysis
"""

# %% [markdown]
# # ==============================================================================
# # CELL 1: Dependency Installation
# # ==============================================================================
# # Install required Python packages not bundled with the default Kaggle environment.

# %%
# In a Kaggle Notebook cell, you can run:
# !pip install -q transformers ftfy scikit-learn matplotlib tqdm torchvision torch
import subprocess
import sys

packages = ["transformers", "ftfy", "scikit-learn", "matplotlib", "tqdm"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=False)
print("✓ Packages verified / installed successfully.")


# %% [markdown]
# # ==============================================================================
# # CELL 2: Environment Configuration, GPU Verification & Path Setup
# # ==============================================================================
# # Detect GPU acceleration, discover dataset path in /kaggle/input, and configure
# # writable directories in /kaggle/working/.

# %%
import os
import sys
import time
import csv
import warnings
from typing import Callable, Optional, Tuple, Any, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"============================================================")
print(f" Runtime Device : {device}")
if torch.cuda.is_available():
    print(f" GPU Device     : {torch.cuda.get_device_name(0)}")
    print(f" Device Count   : {torch.cuda.device_count()}")
else:
    print(f" CPU Cores      : {os.cpu_count() or 4}")
print(f"============================================================")

# Confirmed real Kaggle input paths
KAGGLE_MODELS_PATH = "/kaggle/input/datasets/ritishaburugupalli/urop-models/models/"
KAGGLE_DATA_PATH = "/kaggle/input/datasets/ritishaburugupalli/urop-dataset/data/"

# Determine environment & resolve directories
IS_KAGGLE = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def resolve_kaggle_data_dir() -> str:
    """Auto-detects the CrisisMMD dataset directory inside /kaggle/input/ or locally."""
    # 1. Environment variable override
    env_dir = os.environ.get("DATA_DIR") or os.environ.get("KAGGLE_DATA_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir

    if IS_KAGGLE:
        # Check confirmed exact path
        if os.path.exists(KAGGLE_DATA_PATH):
            return KAGGLE_DATA_PATH

        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            subdirs = [f"{kaggle_input}/{d}" for d in os.listdir(kaggle_input) if os.path.isdir(f"{kaggle_input}/{d}")]
            for path in subdirs:
                bname = os.path.basename(path).lower()
                if "crisis" in bname or "disaster" in bname or "urop" in bname:
                    return path
            for path in subdirs:
                try:
                    contents = [c.lower() for c in os.listdir(path)]
                    if any("crisismmd" in c or "data_image" in c or c.endswith(".tsv") for c in contents):
                        return path
                except Exception:
                    continue
            if len(subdirs) == 1:
                return subdirs[0]
        return KAGGLE_DATA_PATH
    else:
        # Local fallback
        for cand in ["data", "../data", "../../data"]:
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return os.path.abspath("data")

DATA_DIR = resolve_kaggle_data_dir()
MODELS_DIR = "/kaggle/working/models" if IS_KAGGLE else os.path.abspath("models")
RESULTS_DIR = "/kaggle/working/results" if IS_KAGGLE else os.path.abspath("results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f" Environment             : {'KAGGLE NOTEBOOK' if IS_KAGGLE else 'LOCAL SYSTEM'}")
print(f" Confirmed Kaggle Data   : {KAGGLE_DATA_PATH}")
print(f" Confirmed Kaggle Models : {KAGGLE_MODELS_PATH}")
print(f" Active Data Dir         : {DATA_DIR} (exists: {os.path.exists(DATA_DIR)})")
print(f" Models Working Dir      : {MODELS_DIR}")
print(f" Results Dir             : {RESULTS_DIR}")
print(f"============================================================")


# %% [markdown]
# # ==============================================================================
# # CELL 3: CrisisMMD Dataset Class & Split Verification
# # ==============================================================================
# # Defines the PyTorch Dataset for paired tweet text and disaster images.
# # Supports automatic discovery of nested TSV files and image directories.

# %%
class CrisisMMDDataset(Dataset):
    """
    PyTorch Dataset for CrisisMMD multimodal disaster classification.
    Task: Informative classification ('informative' vs 'not_informative')
    """
    LABEL_MAP = {
        'not_informative': 0,
        'informative': 1
    }

    def __init__(
        self,
        split: str = 'train',
        task: str = 'informative',
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        verify_images: bool = False
    ):
        super().__init__()
        self.split = split.lower()
        self.task = task.lower()
        self.data_dir = data_dir or DATA_DIR
        self.transform = transform

        self.tsv_path = self._resolve_tsv_path()
        self.images_root = self._resolve_images_root()
        self.samples = []
        self._load_dataset(verify_images=verify_images)

    def _resolve_tsv_path(self) -> str:
        filename = f"task_{self.task}_text_img_agreed_lab_{self.split}.tsv"
        candidates = [
            os.path.join(self.data_dir, "crisismmd_datasplit_agreed_label", "crisismmd_datasplit_agreed_label", filename),
            os.path.join(self.data_dir, "crisismmd_datasplit_agreed_label", filename),
            os.path.join(self.data_dir, filename),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand

        # Recursive search in data_dir
        for root, _, files in os.walk(self.data_dir):
            if filename in files:
                return os.path.join(root, filename)

        raise FileNotFoundError(f"Could not locate TSV '{filename}' in '{self.data_dir}'. Please verify dataset path.")

    def _resolve_images_root(self) -> str:
        candidates = [
            os.path.join(self.data_dir, "CrisisMMD_v2.0", "CrisisMMD_v2.0"),
            os.path.join(self.data_dir, "CrisisMMD_v2.0"),
            self.data_dir
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "data_image")):
                return cand

        for root, dirs, _ in os.walk(self.data_dir):
            if "data_image" in dirs:
                return root

        raise FileNotFoundError(f"Could not locate 'data_image' folder in '{self.data_dir}'.")

    def _load_dataset(self, verify_images: bool = False):
        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_idx, row in enumerate(reader):
                text = row.get('tweet_text', '').strip()
                rel_img_path = row.get('image', '').strip()
                raw_label = row.get('label', '').strip()

                if raw_label not in self.LABEL_MAP:
                    continue
                label = self.LABEL_MAP[raw_label]

                clean_rel = rel_img_path.lstrip(r"\/").replace("\\", "/")
                full_img_path = os.path.normpath(os.path.join(self.images_root, clean_rel))

                if not os.path.exists(full_img_path):
                    continue

                if verify_images:
                    try:
                        with Image.open(full_img_path) as img:
                            img.verify()
                    except Exception:
                        continue

                self.samples.append({
                    'text': text,
                    'image_path': full_img_path,
                    'label': label
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, Any, int]:
        sample = self.samples[index]
        img_path = sample['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform is not None:
            image = self.transform(image)

        return sample['text'], image, sample['label']


# Verify dataset loading
print("Verifying CrisisMMD dataset splits...")
train_dataset = CrisisMMDDataset(split='train')
dev_dataset = CrisisMMDDataset(split='dev')
test_dataset = CrisisMMDDataset(split='test')

print(f"✓ Train Split : {len(train_dataset):,} samples")
print(f"✓ Dev Split   : {len(dev_dataset):,} samples")
print(f"✓ Test Split  : {len(test_dataset):,} samples")

sample_text, sample_img, sample_lbl = train_dataset[0]
print(f"\nSample 1 Preview:")
print(f"  Text  : {sample_text[:90]}...")
print(f"  Image : {sample_img.size} RGB")
print(f"  Label : {sample_lbl} ({'Informative' if sample_lbl == 1 else 'Not Informative'})")


# %% [markdown]
# # ==============================================================================
# # CELL 4: Model 1 - Text-Only DistilBERT Baseline
# # ==============================================================================
# # Fine-tunes DistilBERT (distilbert-base-uncased) for 3 epochs with AdamW (lr=2e-5).
# # Saves checkpoint to /kaggle/working/models/text_baseline.pt.

# %%
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_BATCH_SIZE = 64
TEXT_EPOCHS = 3
TEXT_LR = 2e-5
TEXT_MAX_LEN = 64

text_save_path = os.path.join(MODELS_DIR, "text_baseline.pt")

print(f"[1/4] Loading tokenizer '{TEXT_MODEL_NAME}'...")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

def text_collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    encoding = text_tokenizer(
        texts, padding=True, truncation=True, max_length=TEXT_MAX_LEN, return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels, dtype=torch.long)
    }

train_text_loader = DataLoader(train_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=True, collate_fn=text_collate_fn)
dev_text_loader = DataLoader(dev_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=False, collate_fn=text_collate_fn)
test_text_loader = DataLoader(test_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=False, collate_fn=text_collate_fn)

print(f"[2/4] Initializing DistilBERT for binary classification...")
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, num_labels=2)
text_model.to(device)

text_optimizer = torch.optim.AdamW(text_model.parameters(), lr=TEXT_LR, weight_decay=0.01)

def evaluate_text_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), all_preds, all_labels

print(f"[3/4] Training DistilBERT for {TEXT_EPOCHS} epochs...")
best_text_loss = float('inf')

for epoch in range(1, TEXT_EPOCHS + 1):
    start_t = time.time()
    text_model.train()
    running_loss = 0.0
    for batch in tqdm(train_text_loader, desc=f"Text Epoch {epoch}/{TEXT_EPOCHS}", leave=False):
        text_optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = text_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
        text_optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_text_loader)
    dev_loss, dev_preds, dev_labels = evaluate_text_model(text_model, dev_text_loader)
    dev_acc = accuracy_score(dev_labels, dev_preds)
    elapsed = time.time() - start_t
    print(f"Epoch {epoch}/{TEXT_EPOCHS} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    if dev_loss < best_text_loss:
        best_text_loss = dev_loss
        torch.save(text_model.state_dict(), text_save_path)

print(f"[4/4] Text-Only Model Training Complete. Best weights saved to: {text_save_path}")

# Evaluate best text checkpoint on test split
text_model.load_state_dict(torch.load(text_save_path, map_location=device))
test_text_loss, test_text_preds, test_text_labels = evaluate_text_model(text_model, test_text_loader)

t_acc = accuracy_score(test_text_labels, test_text_preds)
t_prec = precision_score(test_text_labels, test_text_preds, zero_division=0)
t_rec = recall_score(test_text_labels, test_text_preds, zero_division=0)
t_f1 = f1_score(test_text_labels, test_text_preds, zero_division=0)

print("\n================ TEXT-ONLY BASELINE TEST METRICS ================")
print(f" Accuracy  : {t_acc:.4f} ({t_acc*100:.2f}%)")
print(f" Precision : {t_prec:.4f}")
print(f" Recall    : {t_rec:.4f}")
print(f" F1 Score  : {t_f1:.4f}")
print("=================================================================")


# %% [markdown]
# # ==============================================================================
# # CELL 5: Model 2 - Image-Only ResNet-18 Baseline
# # ==============================================================================
# # Fine-tunes ResNet-18 with ImageNet preprocessing for 3 epochs with AdamW (lr=1e-4).
# # Saves checkpoint to /kaggle/working/models/image_baseline.pt.

# %%
IMAGE_BATCH_SIZE = 64
IMAGE_EPOCHS = 3
IMAGE_LR = 1e-4
IMAGE_SIZE = (224, 224)

image_save_path = os.path.join(MODELS_DIR, "image_baseline.pt")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

eval_img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Create datasets with image transforms
train_img_ds = CrisisMMDDataset(split='train', transform=train_img_transform)
dev_img_ds = CrisisMMDDataset(split='dev', transform=eval_img_transform)
test_img_ds = CrisisMMDDataset(split='test', transform=eval_img_transform)

def image_collate_fn(batch):
    images = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return {'images': images, 'labels': labels}

train_img_loader = DataLoader(train_img_ds, batch_size=IMAGE_BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=image_collate_fn)
dev_img_loader = DataLoader(dev_img_ds, batch_size=IMAGE_BATCH_SIZE, shuffle=False, collate_fn=image_collate_fn)
test_img_loader = DataLoader(test_img_ds, batch_size=IMAGE_BATCH_SIZE, shuffle=False, collate_fn=image_collate_fn)

print(f"[1/4] Initializing pretrained ResNet-18...")
try:
    weights = models.ResNet18_Weights.DEFAULT
    image_model = models.resnet18(weights=weights)
except AttributeError:
    image_model = models.resnet18(pretrained=True)

# Replace final FC layer with 2-class head
image_model.fc = nn.Linear(image_model.fc.in_features, 2)
image_model.to(device)

image_criterion = nn.CrossEntropyLoss()
image_optimizer = torch.optim.AdamW(image_model.parameters(), lr=IMAGE_LR, weight_decay=0.01)

def evaluate_image_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
            loss = image_criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), all_preds, all_labels

print(f"[2/4] Training ResNet-18 for {IMAGE_EPOCHS} epochs...")
best_img_loss = float('inf')

for epoch in range(1, IMAGE_EPOCHS + 1):
    start_t = time.time()
    image_model.train()
    running_loss = 0.0
    for batch in tqdm(train_img_loader, desc=f"Image Epoch {epoch}/{IMAGE_EPOCHS}", leave=False):
        image_optimizer.zero_grad()
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        outputs = image_model(images)
        loss = image_criterion(outputs, labels)
        loss.backward()
        image_optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_img_loader)
    dev_loss, dev_preds, dev_labels = evaluate_image_model(image_model, dev_img_loader)
    dev_acc = accuracy_score(dev_labels, dev_preds)
    elapsed = time.time() - start_t
    print(f"Epoch {epoch}/{IMAGE_EPOCHS} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    if dev_loss < best_img_loss:
        best_img_loss = dev_loss
        torch.save(image_model.state_dict(), image_save_path)

print(f"[3/4] Image-Only Model Training Complete. Best weights saved to: {image_save_path}")

# Evaluate best image checkpoint on test split
image_model.load_state_dict(torch.load(image_save_path, map_location=device))
test_img_loss, test_img_preds, test_img_labels = evaluate_image_model(image_model, test_img_loader)

i_acc = accuracy_score(test_img_labels, test_img_preds)
i_prec = precision_score(test_img_labels, test_img_preds, zero_division=0)
i_rec = recall_score(test_img_labels, test_img_preds, zero_division=0)
i_f1 = f1_score(test_img_labels, test_img_preds, zero_division=0)

print("\n================ IMAGE-ONLY BASELINE TEST METRICS ================")
print(f" Accuracy  : {i_acc:.4f} ({i_acc*100:.2f}%)")
print(f" Precision : {i_prec:.4f}")
print(f" Recall    : {i_rec:.4f}")
print(f" F1 Score  : {i_f1:.4f}")
print("==================================================================")


# %% [markdown]
# # ==============================================================================
# # CELL 6: Model 3 - Multimodal Late Fusion Model (DistilBERT + ResNet-18 + MLP)
# # ==============================================================================
# # Fuses DistilBERT 768-d text representation with ResNet-18 512-d image representation
# # into a 1280-d joint vector, passed through a 3-layer MLP classification head.
# # Warm-starts backbones with baseline weights. Saves to /kaggle/working/models/fusion_model.pt.

# %%
class MultimodalFusionModel(nn.Module):
    """
    Late Fusion Architecture:
    - Text Encoder: DistilBERT [CLS] token embedding (768-dim)
    - Image Encoder: ResNet-18 penultimate features (512-dim)
    - Concatenation: 1280-dim joint multimodal representation
    - Classification Head: 3-layer MLP (1280 -> 256 -> 64 -> 2) with LayerNorm and Dropout
    """
    def __init__(self, text_model_name: str = "distilbert-base-uncased", dropout_rate: float = 0.3):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768

        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.image_encoder = models.resnet18(weights=weights)
        except AttributeError:
            self.image_encoder = models.resnet18(pretrained=True)
        image_dim = self.image_encoder.fc.in_features  # 512
        self.image_encoder.fc = nn.Identity()

        fusion_dim = text_dim + image_dim  # 1280

        self.mlp_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, images):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]  # [CLS] token
        image_feat = self.image_encoder(images)
        joint_feat = torch.cat([text_feat, image_feat], dim=1)
        logits = self.mlp_head(joint_feat)
        return logits

def fusion_collate_fn(batch):
    texts = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    encoding = text_tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'images': images,
        'labels': labels
    }

train_fusion_loader = DataLoader(train_img_ds, batch_size=64, shuffle=True, drop_last=True, collate_fn=fusion_collate_fn)
dev_fusion_loader = DataLoader(dev_img_ds, batch_size=64, shuffle=False, collate_fn=fusion_collate_fn)
test_fusion_loader = DataLoader(test_img_ds, batch_size=64, shuffle=False, collate_fn=fusion_collate_fn)

fusion_save_path = os.path.join(MODELS_DIR, "fusion_model.pt")

print(f"[1/4] Initializing Multimodal Fusion Model...")
fusion_model = MultimodalFusionModel(text_model_name=TEXT_MODEL_NAME, dropout_rate=0.3)

# Warm-start encoders from fine-tuned baselines if available
if os.path.exists(text_save_path):
    print(f"Loading pretrained text weights from: {text_save_path}")
    text_sd = torch.load(text_save_path, map_location='cpu')
    encoder_sd = {k[len("distilbert."):]: v for k, v in text_sd.items() if k.startswith("distilbert.")}
    fusion_model.text_encoder.load_state_dict(encoder_sd, strict=False)

if os.path.exists(image_save_path):
    print(f"Loading pretrained image weights from: {image_save_path}")
    img_sd = torch.load(image_save_path, map_location='cpu')
    encoder_sd = {k: v for k, v in img_sd.items() if not k.startswith("fc.")}
    fusion_model.image_encoder.load_state_dict(encoder_sd, strict=False)

fusion_model.to(device)

fusion_criterion = nn.CrossEntropyLoss()
fusion_optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=2e-5, weight_decay=0.01)

def evaluate_fusion_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask, images)
            loss = fusion_criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), all_preds, all_labels

print(f"[2/4] Training Multimodal Fusion Model for 3 epochs...")
best_fusion_loss = float('inf')

for epoch in range(1, 4):
    start_t = time.time()
    fusion_model.train()
    running_loss = 0.0
    for batch in tqdm(train_fusion_loader, desc=f"Fusion Epoch {epoch}/3", leave=False):
        fusion_optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        logits = fusion_model(input_ids, attention_mask, images)
        loss = fusion_criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
        fusion_optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_fusion_loader)
    dev_loss, dev_preds, dev_labels = evaluate_fusion_model(fusion_model, dev_fusion_loader)
    dev_acc = accuracy_score(dev_labels, dev_preds)
    elapsed = time.time() - start_t
    print(f"Epoch {epoch}/3 [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    if dev_loss < best_fusion_loss:
        best_fusion_loss = dev_loss
        torch.save(fusion_model.state_dict(), fusion_save_path)

print(f"[3/4] Multimodal Fusion Training Complete. Best weights saved to: {fusion_save_path}")


# %% [markdown]
# # ==============================================================================
# # CELL 7: Comprehensive Evaluation, Confusion Matrix & Qualitative Analysis
# # ==============================================================================
# # Evaluates all three models simultaneously, computes classification report,
# # plots & saves Confusion Matrix, compares baselines, and shows qualitative cases.

# %%
print("Loading trained checkpoints for full test evaluation...")

def get_eval_checkpoint(filename: str, session_path: str) -> str:
    # 1. Freshly trained in current notebook session
    if os.path.exists(session_path):
        return session_path
    # 2. Pre-uploaded weights in confirmed Kaggle models dataset
    kaggle_path = os.path.join(KAGGLE_MODELS_PATH, filename)
    if os.path.exists(kaggle_path):
        return kaggle_path
    return session_path

actual_fusion_path = get_eval_checkpoint("fusion_model.pt", fusion_save_path)
actual_text_path = get_eval_checkpoint("text_baseline.pt", text_save_path)
actual_image_path = get_eval_checkpoint("image_baseline.pt", image_save_path)

print(f"Loading Fusion Model from : {actual_fusion_path}")
fusion_eval_model = MultimodalFusionModel(text_model_name=TEXT_MODEL_NAME, dropout_rate=0.3)
fusion_eval_model.load_state_dict(torch.load(actual_fusion_path, map_location=device))
fusion_eval_model.to(device)
fusion_eval_model.eval()

print(f"Loading Text Model from   : {actual_text_path}")
text_eval_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, num_labels=2)
text_eval_model.load_state_dict(torch.load(actual_text_path, map_location=device))
text_eval_model.to(device)
text_eval_model.eval()

print(f"Loading Image Model from  : {actual_image_path}")
try:
    image_eval_model = models.resnet18(weights=None)
except AttributeError:
    image_eval_model = models.resnet18(pretrained=False)
image_eval_model.fc = nn.Linear(image_eval_model.fc.in_features, 2)
image_eval_model.load_state_dict(torch.load(actual_image_path, map_location=device))
image_eval_model.to(device)
image_eval_model.eval()

print("Running test set inference across all models...")
all_fusion_preds = []
all_text_preds = []
all_image_preds = []
all_ground_truth = []

with torch.no_grad():
    for batch in tqdm(test_fusion_loader, desc="Evaluating All Models", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        # Fusion inference
        fusion_logits = fusion_eval_model(input_ids, attention_mask, images)
        fusion_preds = torch.argmax(fusion_logits, dim=1).cpu().numpy()

        # Text inference
        text_logits = text_eval_model(input_ids, attention_mask).logits
        text_preds = torch.argmax(text_logits, dim=1).cpu().numpy()

        # Image inference
        img_logits = image_eval_model(images)
        img_preds = torch.argmax(img_logits, dim=1).cpu().numpy()

        all_fusion_preds.extend(fusion_preds)
        all_text_preds.extend(text_preds)
        all_image_preds.extend(img_preds)
        all_ground_truth.extend(labels.cpu().numpy())

all_fusion_preds = np.array(all_fusion_preds)
all_text_preds = np.array(all_text_preds)
all_image_preds = np.array(all_image_preds)
all_ground_truth = np.array(all_ground_truth)

# 4. Confusion Matrix Plot
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
cm = confusion_matrix(all_ground_truth, all_fusion_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Informative", "Informative"])

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
ax.set_title("CrisisMMD Multimodal Fusion Model\nConfusion Matrix (Test Split)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(cm_path, bbox_inches='tight')
plt.show()
print(f"Confusion Matrix saved to: {cm_path}")

# 5. Classification Report
print("\n=========================================================================")
print("           MULTIMODAL FUSION MODEL - CLASSIFICATION REPORT               ")
print("=========================================================================")
print(classification_report(all_ground_truth, all_fusion_preds, target_names=["Not Informative", "Informative"], digits=4))

# 6. Comparative Model Performance Table
f_acc = accuracy_score(all_ground_truth, all_fusion_preds)
f_prec = precision_score(all_ground_truth, all_fusion_preds, zero_division=0)
f_rec = recall_score(all_ground_truth, all_fusion_preds, zero_division=0)
f_f1 = f1_score(all_ground_truth, all_fusion_preds, zero_division=0)

print("===================================================================================")
print("                      MODEL PERFORMANCE COMPARISON (TEST SET)                      ")
print("===================================================================================")
print(f"{'Model Architecture':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
print("-" * 83)
print(f"{'Text-Only (DistilBERT)':<28} | {t_acc:<9.4f} | {t_prec:<9.4f} | {t_rec:<9.4f} | {t_f1:<9.4f}")
print(f"{'Image-Only (ResNet-18)':<28} | {i_acc:<9.4f} | {i_prec:<9.4f} | {i_rec:<9.4f} | {i_f1:<9.4f}")
print(f"{'Multimodal Fusion (MLP)':<28} | {f_acc:<9.4f} | {f_prec:<9.4f} | {f_rec:<9.4f} | {f_f1:<9.4f}")
print("===================================================================================")

# 7. Qualitative Error & Modality Advantage Analysis
def print_qualitative_showcase(baseline_preds, baseline_name: str, num_cases: int = 5):
    target_indices = np.where((all_fusion_preds == all_ground_truth) & (baseline_preds != all_ground_truth))[0]
    print(f"\n==========================================================================================")
    print(f" QUALITATIVE ANALYSIS: Cases Where Multimodal Fusion is CORRECT but {baseline_name} is WRONG")
    print(f" (Total instances found in test split: {len(target_indices)})")
    print(f"==========================================================================================")
    for i, idx in enumerate(target_indices[:num_cases]):
        sample = test_img_ds.samples[idx]
        gt_str = "Informative" if all_ground_truth[idx] == 1 else "Not Informative"
        base_str = "Informative" if baseline_preds[idx] == 1 else "Not Informative"
        fuse_str = "Informative" if all_fusion_preds[idx] == 1 else "Not Informative"

        print(f"Example {i+1} [Test Index: {idx}]:")
        print(f"  Tweet Text   : \"{sample['text']}\"")
        print(f"  Image Path   : {os.path.basename(sample['image_path'])}")
        print(f"  Ground Truth : {gt_str}")
        print(f"  {baseline_name:<12} : {base_str} [INCORRECT]")
        print(f"  Fusion Model : {fuse_str} [CORRECT]")
        print("-" * 90)

print_qualitative_showcase(all_text_preds, "Text Baseline", num_cases=5)
print_qualitative_showcase(all_image_preds, "Image Baseline", num_cases=5)

print("\n✓ Kaggle Notebook Pipeline Execution Completed Successfully!")
