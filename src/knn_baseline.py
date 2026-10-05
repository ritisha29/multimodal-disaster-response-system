"""
k-Nearest Neighbors (k-NN) Baseline for CrisisMMD Disaster Tweet Informativeness Classification.

Evaluates a k-NN classifier on multimodal, text, and image representations as classical benchmarks.
Saves results to results/knn_evaluation.json.

Usage:
    python src/knn_baseline.py
    python src/knn_baseline.py --n_neighbors 5
    python src/knn_baseline.py --n_neighbors 3 --metric euclidean
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from dataset import CrisisMMDDataset
    from fusion_model import MultimodalFusionModel, collate_fn_builder
    from kaggle_paths import get_data_dir, get_models_dir, get_results_dir, resolve_checkpoint_path
except ImportError:
    from src.dataset import CrisisMMDDataset
    from src.fusion_model import MultimodalFusionModel, collate_fn_builder
    from src.kaggle_paths import get_data_dir, get_models_dir, get_results_dir, resolve_checkpoint_path
from transformers import AutoTokenizer


def load_fusion_encoder(models_dir: str, device: torch.device):
    """Loads only the Multimodal Fusion checkpoint for feature extraction."""
    ckpt_path = resolve_checkpoint_path("fusion_model.pt", models_dir)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
    
    print(f"Loading feature extractor from: {ckpt_path}")
    model = MultimodalFusionModel(text_model_name="distilbert-base-uncased", dropout_rate=0.3)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def extract_embeddings(model, dataloader, device):
    """Extracts text, visual, and multimodal feature representations along with labels."""
    all_text = []
    all_image = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            y = batch['labels'].numpy()

            # Text features via DistilBERT text_encoder [CLS] token -> shape (B, 768)
            text_out = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_vec = text_out.last_hidden_state[:, 0, :].cpu().numpy()

            # Image features via ResNet-18 image_encoder -> shape (B, 512)
            img_vec = model.image_encoder(images).cpu().numpy()

            all_text.append(text_vec)
            all_image.append(img_vec)
            all_labels.append(y)

    text_features = np.vstack(all_text)
    image_features = np.vstack(all_image)
    multimodal_features = np.hstack([text_features, image_features])
    labels = np.concatenate(all_labels)

    return text_features, image_features, multimodal_features, labels


def evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors: int, metric: str):
    """Fits k-NN and returns evaluation metrics dictionary."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "preds": preds
    }


def main():
    parser = argparse.ArgumentParser(description="Run k-Nearest Neighbors (k-NN) baseline on CrisisMMD")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors k (default: 5)")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric: cosine, euclidean, manhattan (default: cosine)")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--models_dir", type=str, default=None, help="Models directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir(args.data_dir)
    models_dir = get_models_dir(args.models_dir)
    results_dir = get_results_dir(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("=========================================================================")
    print(f"       k-NEAREST NEIGHBORS (k-NN) BASELINE EVALUATION (k={args.n_neighbors})         ")
    print("=========================================================================")
    print(f"Device: {device} | Distance Metric: '{args.metric}' | Neighbors: {args.n_neighbors}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate_fn = collate_fn_builder(tokenizer, max_length=64)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\nLoading datasets...")
    train_dataset = CrisisMMDDataset(split='dev', data_dir=data_dir, transform=eval_transform)
    test_dataset = CrisisMMDDataset(split='test', data_dir=data_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"Reference split (dev): {len(train_dataset)} samples | Evaluation split (test): {len(test_dataset)} samples.")

    fusion_model = load_fusion_encoder(models_dir, device)

    print("\nExtracting feature embeddings from reference split (dev)...")
    text_train, img_train, mm_train, y_train = extract_embeddings(fusion_model, train_loader, device)

    print("Extracting feature embeddings from test split...")
    text_test, img_test, mm_test, y_test = extract_embeddings(fusion_model, test_loader, device)

    print(f"\nEvaluating k-NN (k={args.n_neighbors}, metric='{args.metric}') across modalities...")
    res_text = evaluate_knn(text_train, y_train, text_test, y_test, args.n_neighbors, args.metric)
    res_img = evaluate_knn(img_train, y_train, img_test, y_test, args.n_neighbors, args.metric)
    res_mm = evaluate_knn(mm_train, y_train, mm_test, y_test, args.n_neighbors, args.metric)

    report_mm = classification_report(y_test, res_mm["preds"], target_names=["Not Informative", "Informative"], digits=4)

    summary = {
        "k": args.n_neighbors,
        "metric": args.metric,
        "knn_text_only": {k: v for k, v in res_text.items() if k != "preds"},
        "knn_image_only": {k: v for k, v in res_img.items() if k != "preds"},
        "knn_multimodal": {k: v for k, v in res_mm.items() if k != "preds"}
    }

    out_file = os.path.join(results_dir, "knn_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================================================================")
    print(f"                 k-NEAREST NEIGHBORS (k-NN, k={args.n_neighbors}) PERFORMANCE SCORES              ")
    print("===================================================================================")
    print(f"{'Representation / Modality':<30} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 85)
    print(f"{'k-NN (Text Features)':<30} | {res_text['accuracy']:<9.4f} | {res_text['precision']:<9.4f} | {res_text['recall']:<9.4f} | {res_text['f1']:<9.4f}")
    print(f"{'k-NN (Image Features)':<30} | {res_img['accuracy']:<9.4f} | {res_img['precision']:<9.4f} | {res_img['recall']:<9.4f} | {res_img['f1']:<9.4f}")
    print(f"{'k-NN (Multimodal Joint)':<30} | {res_mm['accuracy']:<9.4f} | {res_mm['precision']:<9.4f} | {res_mm['recall']:<9.4f} | {res_mm['f1']:<9.4f}")
    print("===================================================================================")

    print("\n=========================================================================")
    print(f"        MULTIMODAL k-NN (k={args.n_neighbors}) - CLASSIFICATION REPORT                ")
    print("=========================================================================")
    print(report_mm)
    print(f"Results saved to: {out_file}\n")


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
