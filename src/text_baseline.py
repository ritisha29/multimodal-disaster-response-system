"""
Text-Only Baseline Classifier for CrisisMMD Disaster Informativeness Task.

This script fine-tunes a pretrained DistilBERT model to classify tweet text into:
- 0: Not Informative
- 1: Informative

Requirements satisfied:
1. Pretrained DistilBERT with sequence classification head (num_labels=2).
2. CrisisMMDDataset text wrapping with HuggingFace tokenizer and PyTorch DataLoader.
3. Training loop with 3 epochs, AdamW optimizer (lr=2e-5), train/dev loss printing.
4. Test evaluation calculating Accuracy, Precision, Recall, and F1.
5. Model weight saving to models/text_baseline.pt.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Optional

try:
    from dataset import CrisisMMDDataset
    from kaggle_paths import get_data_dir, get_models_dir
except ImportError:
    from src.dataset import CrisisMMDDataset
    from src.kaggle_paths import get_data_dir, get_models_dir


def collate_fn_builder(tokenizer, max_length: int = 64):
    """
    Creates a custom collate function for PyTorch DataLoader that tokenizes
    batches of tweet texts dynamically up to max_length and packs integer labels.

    Args:
        tokenizer: Pretrained HuggingFace tokenizer.
        max_length (int): Maximum sequence length threshold for truncation.
    """
    def collate_fn(batch):
        # Extract text strings (item[0]) and integer labels (item[2])
        texts = [item[0] for item in batch]
        labels = [item[2] for item in batch]

        # Dynamic batch padding up to the max length in batch (capped at max_length)
        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels_tensor
        }

    return collate_fn


def train_epoch(model, dataloader, optimizer, device):
    """
    Performs one epoch of fine-tuning.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass: DistilBERT calculates cross-entropy loss automatically when labels are provided
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluates model on validation or test set.

    Returns:
        tuple: (avg_loss, list of predicted class labels, list of ground truth labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def main(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_len: int = 64
):
    # Seed for reproducibility
    torch.manual_seed(42)
    
    # Configure CPU threads for optimal execution speed
    if not torch.cuda.is_available():
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)

    # 1. Hyperparameters & Settings
    MODEL_NAME = "distilbert-base-uncased"
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    MAX_LEN = max_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (threads: {torch.get_num_threads()})")

    # Resolve paths (auto-detects Kaggle / local)
    data_dir = get_data_dir(data_dir)
    models_dir = get_models_dir(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "text_baseline.pt")

    print(f"Data directory:   {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Checkpoint save:  {model_save_path}")

    # 2. Load Tokenizer and Create Data Loaders
    print(f"\n[1/4] Loading tokenizer '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate_fn = collate_fn_builder(tokenizer, max_length=MAX_LEN)

    print("[2/4] Loading CrisisMMD dataset splits (train/dev/test)...")
    train_dataset = CrisisMMDDataset(split='train', data_dir=data_dir)
    dev_dataset = CrisisMMDDataset(split='dev', data_dir=data_dir)
    test_dataset = CrisisMMDDataset(split='test', data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Dataset split sizes: Train={len(train_dataset)}, Dev={len(dev_dataset)}, Test={len(test_dataset)}")

    # 3. Instantiate Pretrained DistilBERT Model & AdamW Optimizer
    print(f"\n[3/4] Initializing pretrained DistilBERT for binary classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 4. Fine-Tuning Loop
    print(f"\n[4/4] Fine-tuning DistilBERT for {EPOCHS} epochs...")
    best_dev_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        dev_loss, dev_preds, dev_labels = evaluate(model, dev_loader, device)

        dev_acc = accuracy_score(dev_labels, dev_preds)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{EPOCHS} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

        # Save checkpoint if dev loss improves
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), model_save_path)

    print(f"\nTraining complete. Saved best model weights to: {model_save_path}")

    # Load best checkpoint for final evaluation on test set
    print("\nLoading best model checkpoint for evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    # 5. Final Test Set Evaluation & Metrics
    print("\n================ TEST SET METRICS ================")
    test_loss, test_preds, test_labels = evaluate(model, test_loader, device)

    acc = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)

    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")
    print("==================================================")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Text-Only DistilBERT Baseline on CrisisMMD")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CrisisMMD dataset root (auto-detected if None)")
    parser.add_argument("--models_dir", type=str, default=None, help="Path to models directory (auto-detected if None)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW (default: 2e-5)")
    parser.add_argument("--max_len", type=int, default=64, help="Max sequence length (default: 64)")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_len=args.max_len
    )
