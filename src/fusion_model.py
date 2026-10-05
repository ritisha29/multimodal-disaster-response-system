"""
Multimodal Fusion Model for CrisisMMD Disaster Informativeness Task.

This script implements a Multimodal Fusion Classifier combining text (DistilBERT)
and image (ResNet-18) features via feature-level concatenation (Late Fusion)
and a Multi-Layer Perceptron (MLP) classification head.

Requirements satisfied:
1. MultimodalFusionModel class:
   - Text encoder: DistilBERT [CLS] embedding (768-dim).
   - Image encoder: ResNet-18 penultimate features (512-dim).
   - Feature concatenation (1280-dim) -> MLP head (256 -> 64 -> 2) with Dropout.
2. Loads pretrained weights from models/text_baseline.pt and models/image_baseline.pt
   as initialization for encoders if available (falls back to HuggingFace/Torchvision pretrained).
3. Training loop using paired (text, image, label) dataset samples from CrisisMMDDataset.
4. Evaluation on test split: Accuracy, Precision, Recall, F1 + Comparison Table
   (Text-Only vs Image-Only vs Multimodal Fusion).
5. Model weight saving to models/fusion_model.pt.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Optional

try:
    from dataset import CrisisMMDDataset
    from kaggle_paths import get_data_dir, get_models_dir, resolve_checkpoint_path
except ImportError:
    from src.dataset import CrisisMMDDataset
    from src.kaggle_paths import get_data_dir, get_models_dir, resolve_checkpoint_path


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Fusion Network combining text and image modalities.

    Fusion Concept & Architecture:
    - Text Sub-network: DistilBERT transformer outputs a contextual sequence representation.
      We extract the [CLS] token embedding (768-dim), which summarizes the full tweet text.
    - Image Sub-network: ResNet-18 convolutional backbone extracts spatial visual features.
      We replace the final FC layer with Identity to retrieve the penultimate feature vector (512-dim).
    - Late Fusion (Feature Concatenation): The 768-dim text representation and 512-dim visual representation
      are concatenated into a single joint 1280-dim multimodal vector: Z = [h_text ; h_image].
    - Classification Head: A 3-layer MLP processes the joint vector with Non-linear activations (ReLU)
      and Dropout regularization (p=0.3) to output logits for binary classification (0: not_informative, 1: informative).
    """

    def __init__(self, text_model_name: str = "distilbert-base-uncased", dropout_rate: float = 0.3):
        super().__init__()

        # 1. Text Encoder (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768

        # 2. Visual Encoder (ResNet-18)
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.image_encoder = models.resnet18(weights=weights)
        except AttributeError:
            self.image_encoder = models.resnet18(pretrained=True)

        image_dim = self.image_encoder.fc.in_features  # 512
        # Replace final classification layer with Identity to extract 512-d penultimate features
        self.image_encoder.fc = nn.Identity()

        # 3. Multimodal Fusion MLP Classification Head
        fusion_dim = text_dim + image_dim  # 768 + 512 = 1280

        self.mlp_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # 2 classes: not_informative (0) vs informative (1)
        )

    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass performing modality feature extraction and late fusion.

        Args:
            input_ids (Tensor): Tokenized text input tensor [batch_size, seq_len].
            attention_mask (Tensor): Text attention mask [batch_size, seq_len].
            images (Tensor): Batch of preprocessed images [batch_size, 3, 224, 224].

        Returns:
            Tensor: Output classification logits [batch_size, 2].
        """
        # Extract text [CLS] token embedding -> shape: [batch_size, 768]
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Extract image penultimate features -> shape: [batch_size, 512]
        image_features = self.image_encoder(images)

        # Concatenate text and image features along channel dimension -> shape: [batch_size, 1280]
        multimodal_features = torch.cat((text_features, image_features), dim=1)

        # Pass concatenated features through MLP classification head -> shape: [batch_size, 2]
        logits = self.mlp_head(multimodal_features)
        return logits


def load_encoder_weights(model, text_ckpt_path: str, image_ckpt_path: str, device: torch.device):
    """
    Loads baseline fine-tuned encoder weights from text_baseline.pt and image_baseline.pt
    into the MultimodalFusionModel instance.

    If checkpoint files are not present on disk, prints an informative notice and maintains
    standard HuggingFace / Torchvision pretrained initialization.
    """
    print("\n--- Loading Pretrained Baseline Encoders ---")

    # Load Text Baseline Weights
    if os.path.exists(text_ckpt_path):
        print(f"Loading fine-tuned text encoder weights from: {text_ckpt_path}")
        text_state_dict = torch.load(text_ckpt_path, map_location=device)

        # Extract transformer backbone weights (handling prefix differences if saved from AutoModelForSequenceClassification)
        filtered_text_dict = {}
        for k, v in text_state_dict.items():
            if k.startswith("distilbert."):
                filtered_text_dict[k.replace("distilbert.", "")] = v
            elif not k.startswith("pre_classifier") and not k.startswith("classifier"):
                filtered_text_dict[k] = v

        missing, unexpected = model.text_encoder.load_state_dict(filtered_text_dict, strict=False)
        print(f"Text encoder weights loaded successfully. (Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)})")
    else:
        print(f"Notice: '{text_ckpt_path}' not found. Using default HuggingFace 'distilbert-base-uncased' weights.")

    # Load Image Baseline Weights
    if os.path.exists(image_ckpt_path):
        print(f"Loading fine-tuned image encoder weights from: {image_ckpt_path}")
        image_state_dict = torch.load(image_ckpt_path, map_location=device)

        # Load weights into image_encoder (ignoring missing fc keys since fc is nn.Identity)
        missing, unexpected = model.image_encoder.load_state_dict(image_state_dict, strict=False)
        print(f"Image encoder weights loaded successfully. (Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)})")
    else:
        print(f"Notice: '{image_ckpt_path}' not found. Using default Torchvision ImageNet ResNet-18 weights.")


def collate_fn_builder(tokenizer, max_length: int = 64):
    """
    Creates a custom collate function for DataLoader that dynamically tokenizes text batches
    and stacks corresponding image tensors and class labels.
    """
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'images': images,
            'labels': labels
        }

    return collate_fn


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one training epoch for the multimodal fusion model.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping for training stability across modalities
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the multimodal fusion model on validation or test set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def evaluate_text_baseline(test_loader, device, text_ckpt_path):
    """Evaluates text-only baseline if weights exist on disk."""
    if not os.path.exists(text_ckpt_path):
        return None
    try:
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model.load_state_dict(torch.load(text_ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        return {
            'acc': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
    except Exception:
        return None


def evaluate_image_baseline(test_loader, device, image_ckpt_path):
    """Evaluates image-only baseline if weights exist on disk."""
    if not os.path.exists(image_ckpt_path):
        return None
    try:
        try:
            model = models.resnet18(weights=None)
        except AttributeError:
            model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(image_ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        return {
            'acc': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
    except Exception:
        return None


def main(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_len: int = 64,
    image_size: tuple = (224, 224)
):
    # Seed for reproducibility
    torch.manual_seed(42)

    # Configure CPU threads for optimal execution speed
    if not torch.cuda.is_available():
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)

    # 1. Hyperparameters & Settings
    TEXT_MODEL_NAME = "distilbert-base-uncased"
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    MAX_LEN = max_len
    IMAGE_SIZE = image_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (threads: {torch.get_num_threads()})")

    # Resolve paths (auto-detects Kaggle / local)
    data_dir = get_data_dir(data_dir)
    models_dir = get_models_dir(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    text_ckpt_path = resolve_checkpoint_path("text_baseline.pt", models_dir)
    image_ckpt_path = resolve_checkpoint_path("image_baseline.pt", models_dir)
    fusion_ckpt_path = os.path.join(models_dir, "fusion_model.pt")

    print(f"Data directory:       {data_dir}")
    print(f"Models directory:     {models_dir}")
    print(f"Text checkpoint:      {text_ckpt_path} (exists: {os.path.exists(text_ckpt_path)})")
    print(f"Image checkpoint:     {image_ckpt_path} (exists: {os.path.exists(image_ckpt_path)})")
    print(f"Fusion checkpoint:    {fusion_ckpt_path}")

    # 2. Tokenizer, Image Transforms & Data Loaders
    print(f"\n[1/4] Loading tokenizer '{TEXT_MODEL_NAME}' and image transforms...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    collate_fn = collate_fn_builder(tokenizer, max_length=MAX_LEN)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    print("[2/4] Loading CrisisMMD dataset splits (train/dev/test)...")
    train_dataset = CrisisMMDDataset(split='train', data_dir=data_dir, transform=train_transform)
    dev_dataset = CrisisMMDDataset(split='dev', data_dir=data_dir, transform=eval_transform)
    test_dataset = CrisisMMDDataset(split='test', data_dir=data_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Dataset split sizes: Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")

    # 3. Instantiate Multimodal Fusion Model & Load Pretrained Weights
    print(f"\n[3/4] Initializing Multimodal Fusion Model...")
    model = MultimodalFusionModel(text_model_name=TEXT_MODEL_NAME, dropout_rate=0.3)

    # Load baseline encoder weights if available
    load_encoder_weights(model, text_ckpt_path, image_ckpt_path, device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 4. Fine-Tuning Loop
    print(f"\n[4/4] Training Multimodal Fusion Model for {EPOCHS} epochs...")
    best_dev_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_preds, dev_labels = evaluate(model, dev_loader, criterion, device)

        dev_acc = accuracy_score(dev_labels, dev_preds)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{EPOCHS} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

        # Save checkpoint if dev loss improves
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), fusion_ckpt_path)

    print(f"\nTraining complete. Saved best fusion model weights to: {fusion_ckpt_path}")

    # Load best checkpoint for final evaluation on test set
    print("\nLoading best fusion model checkpoint for test evaluation...")
    model.load_state_dict(torch.load(fusion_ckpt_path, map_location=device))
    model.to(device)

    # 5. Final Test Set Evaluation & Comparison Table
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    fusion_acc = accuracy_score(test_labels, test_preds)
    fusion_prec = precision_score(test_labels, test_preds, zero_division=0)
    fusion_rec = recall_score(test_labels, test_preds, zero_division=0)
    fusion_f1 = f1_score(test_labels, test_preds, zero_division=0)

    # Evaluate text and image baselines if trained models are present
    text_metrics = evaluate_text_baseline(test_loader, device, text_ckpt_path)
    image_metrics = evaluate_image_baseline(test_loader, device, image_ckpt_path)

    print("\n===================================================================================")
    print("                         MODEL PERFORMANCE COMPARISON (TEST SET)                   ")
    print("===================================================================================")
    print(f"{'Model Architecture':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9}")
    print("-" * 83)

    if text_metrics:
        print(f"{'Text-Only (DistilBERT)':<28} | {text_metrics['acc']:<9.4f} | {text_metrics['precision']:<9.4f} | {text_metrics['recall']:<9.4f} | {text_metrics['f1']:<9.4f}")
    else:
        print(f"{'Text-Only (DistilBERT)':<28} | {'N/A (Run text_baseline.py)':<48}")

    if image_metrics:
        print(f"{'Image-Only (ResNet-18)':<28} | {image_metrics['acc']:<9.4f} | {image_metrics['precision']:<9.4f} | {image_metrics['recall']:<9.4f} | {image_metrics['f1']:<9.4f}")
    else:
        print(f"{'Image-Only (ResNet-18)':<28} | {'N/A (Run image_baseline.py)':<48}")

    print(f"{'Multimodal Fusion (MLP)':<28} | {fusion_acc:<9.4f} | {fusion_prec:<9.4f} | {fusion_rec:<9.4f} | {fusion_f1:<9.4f}")
    print("===================================================================================")
    print(f"Test Loss (Fusion Model): {test_loss:.4f}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model on CrisisMMD")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CrisisMMD dataset root (auto-detected if None)")
    parser.add_argument("--models_dir", type=str, default=None, help="Path to models directory (auto-detected if None)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW (default: 2e-5)")
    parser.add_argument("--max_len", type=int, default=64, help="Max text sequence length (default: 64)")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_len=args.max_len
    )
