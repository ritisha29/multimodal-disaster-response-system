"""
Image-Only Baseline Classifier for CrisisMMD Disaster Informativeness Task.

This script fine-tunes a pretrained ResNet-18 model to classify tweet images into:
- 0: Not Informative
- 1: Informative

Requirements satisfied:
1. Pretrained ResNet-18 with the final FC layer replaced for binary classification (num_labels=2).
2. CrisisMMDDataset image wrapping with standard ImageNet normalization and augmentations.
3. Fine-tuning training loop with AdamW optimizer and train/dev loss printed per epoch.
4. Test set evaluation printing Accuracy, Precision, Recall, and F1 (sklearn.metrics).
5. Model weight saving to models/image_baseline.pt.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Optional

try:
    from dataset import CrisisMMDDataset
    from kaggle_paths import get_data_dir, get_models_dir
except ImportError:
    from src.dataset import CrisisMMDDataset
    from src.kaggle_paths import get_data_dir, get_models_dir


def collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader that extracts image tensors
    and packs integer classification labels.

    Args:
        batch: List of tuples (text, image_tensor, label) from CrisisMMDDataset.
    """
    images = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return {
        'images': images,
        'labels': labels
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of fine-tuning for the image baseline.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the image baseline model on validation or test set.

    Returns:
        tuple: (avg_loss, list of predicted class labels, list of ground truth labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


def main(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    image_size: tuple = (224, 224)
):
    # Seed for reproducibility
    torch.manual_seed(42)

    # Configure CPU threads for optimal execution speed
    if not torch.cuda.is_available():
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)

    # 1. Hyperparameters & Settings
    MODEL_NAME = "ResNet-18"
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    IMAGE_SIZE = image_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (threads: {torch.get_num_threads()})")

    # Resolve paths (auto-detects Kaggle / local)
    data_dir = get_data_dir(data_dir)
    models_dir = get_models_dir(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "image_baseline.pt")

    print(f"Data directory:   {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Checkpoint save:  {model_save_path}")

    # 2. Define Image Transforms & Load CrisisMMD Splits
    print(f"\n[1/4] Configuring ImageNet transforms for '{MODEL_NAME}'...")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training transforms with resizing and random horizontal flip augmentation
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Evaluation transforms (no augmentations)
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

    # 3. Instantiate Pretrained ResNet-18 Model & Optimizer
    print(f"\n[3/4] Initializing pretrained ResNet-18 for binary classification...")
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    # Replace final fully-connected layer for 2 classes (not_informative vs informative)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 4. Fine-Tuning Loop
    print(f"\n[4/4] Fine-tuning ResNet-18 for {EPOCHS} epochs...")
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
            torch.save(model.state_dict(), model_save_path)

    print(f"\nTraining complete. Saved best model weights to: {model_save_path}")

    # Load best checkpoint for final evaluation on test set
    print("\nLoading best model checkpoint for evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    # 5. Final Test Set Evaluation & Metrics
    print("\n================ TEST SET METRICS ================")
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

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
    parser = argparse.ArgumentParser(description="Fine-tune Image-Only ResNet-18 Baseline on CrisisMMD")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CrisisMMD dataset root (auto-detected if None)")
    parser.add_argument("--models_dir", type=str, default=None, help="Path to models directory (auto-detected if None)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW (default: 1e-4)")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
