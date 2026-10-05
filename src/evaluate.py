"""
Evaluation and Qualitative Analysis Script for Multimodal Disaster Informativeness Classifier.

This script loads the trained Multimodal Fusion Model along with the baseline models
(Text-Only DistilBERT and Image-Only ResNet-18) and performs:
1. Full test split evaluation of the Multimodal Fusion Model.
2. Confusion Matrix generation and visualization saved to results/confusion_matrix.png.
3. Detailed Classification Report (Precision, Recall, F1 for each class).
4. Qualitative Error & Modality Advantage Analysis:
   - 5 examples where Multimodal Fusion is CORRECT but Text-Only baseline is WRONG.
   - 5 examples where Multimodal Fusion is CORRECT but Image-Only baseline is WRONG.
"""

import sys
import os
import time
import json
import warnings
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm
from typing import Optional

# Suppress image palette and transparency warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from dataset import CrisisMMDDataset
    from fusion_model import MultimodalFusionModel, collate_fn_builder
    from kaggle_paths import get_data_dir, get_models_dir, get_results_dir, resolve_checkpoint_path
except ImportError:
    from src.dataset import CrisisMMDDataset
    from src.fusion_model import MultimodalFusionModel, collate_fn_builder
    from src.kaggle_paths import get_data_dir, get_models_dir, get_results_dir, resolve_checkpoint_path


LABEL_NAMES = {0: "Not Informative", 1: "Informative"}


def load_models(models_dir: Optional[str], device: torch.device):
    """
    Loads all trained model checkpoints (Fusion, Text Baseline, Image Baseline).
    Resolves checkpoints from models_dir or /kaggle/input if on Kaggle.
    """
    fusion_ckpt = resolve_checkpoint_path("fusion_model.pt", models_dir)
    text_ckpt = resolve_checkpoint_path("text_baseline.pt", models_dir)
    image_ckpt = resolve_checkpoint_path("image_baseline.pt", models_dir)

    for name, path in [("Fusion Model", fusion_ckpt), ("Text Baseline", text_ckpt), ("Image Baseline", image_ckpt)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint for {name} not found at '{path}'. "
                f"Please ensure models are trained (run text_baseline, image_baseline, fusion_model) "
                f"or mount pre-trained weights in Kaggle / local models/ folder."
            )

    # 1. Multimodal Fusion Model
    print(f"Loading Multimodal Fusion Model from: {fusion_ckpt}")
    fusion_model = MultimodalFusionModel(text_model_name="distilbert-base-uncased", dropout_rate=0.3)
    fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
    fusion_model.to(device)
    fusion_model.eval()

    # 2. Text-Only Baseline
    print(f"Loading Text-Only DistilBERT Model from: {text_ckpt}")
    text_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    text_model.load_state_dict(torch.load(text_ckpt, map_location=device))
    text_model.to(device)
    text_model.eval()

    # 3. Image-Only Baseline
    print(f"Loading Image-Only ResNet-18 Model from: {image_ckpt}")
    try:
        image_model = models.resnet18(weights=None)
    except AttributeError:
        image_model = models.resnet18(pretrained=False)
    image_model.fc = nn.Linear(image_model.fc.in_features, 2)
    image_model.load_state_dict(torch.load(image_ckpt, map_location=device))
    image_model.to(device)
    image_model.eval()

    return fusion_model, text_model, image_model


def run_inference(fusion_model, text_model, image_model, test_loader, device):
    """
    Runs inference across the test split for all three models and collects predictions.
    """
    print("\nRunning inference on test split for all models...")
    all_fusion_preds = []
    all_text_preds = []
    all_image_preds = []
    all_ground_truth = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            # Fusion Model prediction
            fusion_logits = fusion_model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            fusion_preds = torch.argmax(fusion_logits, dim=1).cpu().numpy()

            # Text Model prediction
            text_outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_preds = torch.argmax(text_outputs.logits, dim=1).cpu().numpy()

            # Image Model prediction
            image_logits = image_model(images)
            image_preds = torch.argmax(image_logits, dim=1).cpu().numpy()

            all_fusion_preds.extend(fusion_preds)
            all_text_preds.extend(text_preds)
            all_image_preds.extend(image_preds)
            all_ground_truth.extend(labels.cpu().numpy())

    return (
        np.array(all_fusion_preds),
        np.array(all_text_preds),
        np.array(all_image_preds),
        np.array(all_ground_truth)
    )


def plot_and_save_confusion_matrix(y_true, y_pred, save_path: str):
    """
    Computes and plots a formatted confusion matrix, saving it as a PNG file.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Informative", "Informative"])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=ax, values_format='d')
    plt.title("Multimodal Fusion Model - Test Confusion Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nSaved Confusion Matrix plot to: {save_path}")


def print_qualitative_examples(test_dataset, fusion_preds, baseline_preds, ground_truth, baseline_name: str, num_examples: int = 5):
    """
    Finds and prints examples where Multimodal Fusion is correct but the baseline model is wrong.
    """
    # Indices where Fusion is Correct AND Baseline is Wrong
    correct_fusion = (fusion_preds == ground_truth)
    wrong_baseline = (baseline_preds != ground_truth)
    target_indices = np.where(correct_fusion & wrong_baseline)[0]

    print("\n" + "=" * 90)
    print(f" QUALITATIVE ANALYSIS: Fusion Model CORRECT vs {baseline_name} WRONG ({len(target_indices)} total cases)")
    print("=" * 90)

    selected_indices = target_indices[:num_examples]

    for rank, idx in enumerate(selected_indices, 1):
        sample = test_dataset.samples[idx]
        raw_text = sample['text']
        tweet_text = raw_text.encode('ascii', errors='replace').decode('ascii')
        img_path = sample['image_path']
        rel_img_path = os.path.relpath(img_path, start=test_dataset.data_dir)

        gt_label = LABEL_NAMES[ground_truth[idx]]
        fusion_p = LABEL_NAMES[fusion_preds[idx]]
        baseline_p = LABEL_NAMES[baseline_preds[idx]]

        print(f"\n[Example {rank}/{num_examples}] (Dataset Index #{idx})")
        print(f"  Tweet Text   : \"{tweet_text}\"")
        print(f"  Image Path   : {rel_img_path}")
        print(f"  Ground Truth : {gt_label} (Label {ground_truth[idx]})")
        print(f"  {baseline_name:<13}: {baseline_p} [INCORRECT]")
        print(f"  Fusion Model : {fusion_p} [CORRECT]")
        print("-" * 90)


def main(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    batch_size: int = 64
):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on device: {device}")

    # Resolve paths (auto-detects Kaggle / local)
    data_dir = get_data_dir(data_dir)
    models_dir = get_models_dir(models_dir)
    results_dir = get_results_dir(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    cm_save_path = os.path.join(results_dir, "confusion_matrix.png")

    print(f"Data directory:    {data_dir}")
    print(f"Models directory:  {models_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Confusion Matrix:  {cm_save_path}")

    # 1. Load Test Dataset & Create DataLoader
    print("\nLoading CrisisMMD test dataset split...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate_fn = collate_fn_builder(tokenizer, max_length=64)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CrisisMMDDataset(split='test', data_dir=data_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # 2. Load Trained Models
    fusion_model, text_model, image_model = load_models(models_dir, device)

    # 3. Run Test Inference Across All Models
    fusion_preds, text_preds, image_preds, ground_truth = run_inference(
        fusion_model, text_model, image_model, test_loader, device
    )

    # 4. Generate & Save Confusion Matrix
    plot_and_save_confusion_matrix(ground_truth, fusion_preds, cm_save_path)

    # 5. Compute Comparative Performance Metrics Across All Models
    def _compute_metrics(preds, gt):
        return {
            "accuracy": float(accuracy_score(gt, preds)),
            "precision": float(precision_score(gt, preds, zero_division=0)),
            "recall": float(recall_score(gt, preds, zero_division=0)),
            "f1": float(f1_score(gt, preds, zero_division=0))
        }

    text_metrics = _compute_metrics(text_preds, ground_truth)
    image_metrics = _compute_metrics(image_preds, ground_truth)
    fusion_metrics = _compute_metrics(fusion_preds, ground_truth)

    summary_metrics = {
        "Text-Only (DistilBERT)": text_metrics,
        "Image-Only (ResNet-18)": image_metrics,
        "Multimodal Fusion (MLP)": fusion_metrics
    }

    # Save metrics JSON
    metrics_json_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Classification Report for Fusion Model
    report_str = classification_report(
        ground_truth,
        fusion_preds,
        target_names=["Not Informative", "Informative"],
        digits=4
    )

    comparison_table_str = (
        "===================================================================================\n"
        "                      MODEL PERFORMANCE COMPARISON (TEST SET)                      \n"
        "===================================================================================\n"
        f"{'Model Architecture':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}\n"
        + ("-" * 83) + "\n"
        f"{'Text-Only (DistilBERT)':<28} | {text_metrics['accuracy']:<9.4f} | {text_metrics['precision']:<9.4f} | {text_metrics['recall']:<9.4f} | {text_metrics['f1']:<9.4f}\n"
        f"{'Image-Only (ResNet-18)':<28} | {image_metrics['accuracy']:<9.4f} | {image_metrics['precision']:<9.4f} | {image_metrics['recall']:<9.4f} | {image_metrics['f1']:<9.4f}\n"
        f"{'Multimodal Fusion (MLP)':<28} | {fusion_metrics['accuracy']:<9.4f} | {fusion_metrics['precision']:<9.4f} | {fusion_metrics['recall']:<9.4f} | {fusion_metrics['f1']:<9.4f}\n"
        "==================================================================================="
    )

    # Save summary TXT
    summary_txt_path = os.path.join(results_dir, "evaluation_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(comparison_table_str)
        f.write("\n\n=========================================================================\n")
        f.write("           MULTIMODAL FUSION MODEL - CLASSIFICATION REPORT               \n")
        f.write("=========================================================================\n")
        f.write(report_str)

    # 6. Qualitative Analysis Demonstrating Multimodal Value
    print_qualitative_examples(
        test_dataset, fusion_preds, text_preds, ground_truth,
        baseline_name="Text Baseline", num_examples=5
    )

    print_qualitative_examples(
        test_dataset, fusion_preds, image_preds, ground_truth,
        baseline_name="Image Baseline", num_examples=5
    )

    # 7. Print Prominent Final Summary & Comparative Table
    print("\n" + "=" * 80)
    print("           MULTIMODAL FUSION MODEL - CLASSIFICATION REPORT               ")
    print("=" * 80)
    print(report_str)

    print("\n" + comparison_table_str)
    print(f"\n✓ Saved evaluation metrics JSON to: {metrics_json_path}")
    print(f"✓ Saved full evaluation summary to:  {summary_txt_path}")
    print(f"✓ Saved confusion matrix plot to:    {cm_save_path}")
    print("\nEvaluation and qualitative report generation completed successfully!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Multimodal & Baseline Models on CrisisMMD")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CrisisMMD dataset root (auto-detected if None)")
    parser.add_argument("--models_dir", type=str, default=None, help="Path to models directory (auto-detected if None)")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to results directory (auto-detected if None)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader (default: 64)")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size
    )
