import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from omegaconf import OmegaConf
from models import BaselineResNet, SEResNet # Import models from the models package
from data_loader import get_data_loaders # Import data loading function
from models.baseline_model import baseline_resnet34 # Import the correct helper function
from models.se_resnet_model import se_resnet34 # Import the correct helper function (after renaming)
from utils import set_seed, ensure_directories
import argparse # Import argparse here

def evaluate_model(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Set seed for reproducibility during evaluation
    set_seed(config.training.seed)

    # Ensure output directories exist
    ensure_directories(config)

    # Load the model architecture
    if config.model.name == "baseline_resnet":
        model = baseline_resnet34(num_classes=config.model.num_classes) # Call the helper function
    elif config.model.name == "se_resnet":
        model = se_resnet34(num_classes=config.model.num_classes, reduction_ratio=config.model.get('reduction_ratio', 16)) # Call the helper function (after renaming se_resnet18 -> se_resnet34)
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

    # Load the trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Load Test Data ---
    # Define paths to your test CSV file
    # Placeholder path - REPLACE WITH YOUR ACTUAL PATH
    csv_path_test = os.path.join(config.data.data_path, "meta", "test.csv") # Example path
    _, _, test_loader = get_data_loaders(config, None, None, csv_path_test) # Pass None for train/val paths if only loading test

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    all_probs = [] # Store probabilities for AUC calculation if needed

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1) # Get probabilities
            pred = output.argmax(dim=1, keepdim=True)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Calculate Metrics ---
    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = 100. * correct / total

    # Classification Report (Precision, Recall, F1 per class and macro avg)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {config.model.name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(config.paths.figures_dir, f'confusion_matrix_{config.model.name}.png')
    plt.savefig(cm_path)
    plt.close() # Close the plot to free memory
    print(f"Confusion matrix saved to {cm_path}")

    # --- Save Results ---
    results = {
        "model_name": config.model.name,
        "checkpoint_path": checkpoint_path,
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix_path": cm_path
    }

    results_path = os.path.join(config.paths.results_dir, f'evaluation_results_{config.model.name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation results saved to {results_path}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio CNN Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint file (.pth)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    evaluate_model(config, args.checkpoint)

if __name__ == "__main__":
    main()