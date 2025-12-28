# -*- coding: utf-8 -*-
# trainer.py

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from omegaconf import OmegaConf # Import OmegaConf to resolve config

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Collect predictions and labels for metrics
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        if batch_idx % 10 == 0: # Log every 10 batches
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    # Calculate metrics using sklearn
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    avg_precision = report['macro avg']['precision']
    avg_recall = report['macro avg']['recall']
    avg_f1 = report['macro avg']['f1-score']

    # Log metrics to TensorBoard and W&B
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
    writer.add_scalar('Precision/Train', avg_precision, epoch)
    writer.add_scalar('Recall/Train', avg_recall, epoch)
    writer.add_scalar('F1-Score/Train', avg_f1, epoch)

    wandb.log({
        "epoch": epoch,
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "train_precision": avg_precision,
        "train_recall": avg_recall,
        "train_f1": avg_f1,
    })

    print(f'Epoch [{epoch}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')

def validate_one_epoch(model, dataloader, criterion, device, epoch, writer, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    # Calculate metrics using sklearn
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    avg_precision = report['macro avg']['precision']
    avg_recall = report['macro avg']['recall']
    avg_f1 = report['macro avg']['f1-score']

    # Log metrics to TensorBoard and W&B
    writer.add_scalar('Loss/Validation', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Validation', epoch_acc, epoch)
    writer.add_scalar('Precision/Validation', avg_precision, epoch)
    writer.add_scalar('Recall/Validation', avg_recall, epoch)
    writer.add_scalar('F1-Score/Validation', avg_f1, epoch)

    wandb.log({
        "epoch": epoch,
        "val_loss": epoch_loss,
        "val_acc": epoch_acc,
        "val_precision": avg_precision,
        "val_recall": avg_recall,
        "val_f1": avg_f1,
    })

    print(f'Epoch [{epoch}] - Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%')
    return epoch_acc, avg_f1 # Return metrics for potential best model saving

def train_model(config):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    from utils import set_seed
    set_seed(config.training.seed)

    # Ensure output directories exist
    from utils import ensure_directories
    ensure_directories(config)

    # Initialize W&B
    wandb.init(
        project=config.logging.project_name,
        entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True) # Log the full config
    )

    # Initialize TensorBoard writer
    log_dir = os.path.join(config.logging.log_dir, f"run_{wandb.run.name}")
    writer = SummaryWriter(log_dir=log_dir)

    # --- Data Loading ---
    # You need to define paths to your train/val CSV files
    # Example: csv_path_train = os.path.join(config.data.data_path, "meta", "train.csv")
    # Example: csv_path_val = os.path.join(config.data.data_path, "meta", "val.csv")
    # Adjust these paths according to your dataset structure
    # For ESC-50, you might use folds for train/val splits
    # e.g., train_fold_files = [f for f in os.listdir(os.path.join(config.data.data_path, "audio")) if "fold1" in f or "fold2" in f ...]
    # and create CSVs accordingly.
    # Placeholder paths - REPLACE WITH YOUR ACTUAL PATHS
    csv_path_train = os.path.join(config.data.data_path, "meta", "train.csv") # Example path
    csv_path_val = os.path.join(config.data.data_path, "meta", "val.csv")     # Example path

    train_loader, val_loader = get_data_loaders(config, csv_path_train, csv_path_val)

    # --- Model Definition ---
    # Import the specific helper functions needed based on the config
    if config.model.name == "baseline_resnet":
        from models.baseline_model import baseline_resnet34 # Import the correct helper function
        model = baseline_resnet34(num_classes=config.model.num_classes) # Call the helper function
    elif config.model.name == "se_resnet":
        from models.se_resnet_model import se_resnet34 # Import the correct helper function (after renaming in se_resnet_model.py)
        model = se_resnet34(num_classes=config.model.num_classes, reduction_ratio=config.model.get('reduction_ratio', 16)) # Call the helper function
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

    model = model.to(device)

    # --- Optimizer and Scheduler ---
    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    elif config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    elif config.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.lr, momentum=0.9, weight_decay=config.training.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    if config.training.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.training.lr, total_steps=len(train_loader) * config.training.epochs)
    elif config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    else:
        scheduler = None # No scheduler

    criterion = torch.nn.CrossEntropyLoss() # Standard loss for classification

    # --- Training Loop ---
    best_val_acc = 0.0
    best_val_f1 = 0.0
    for epoch in range(1, config.training.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config)
        val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device, epoch, writer, config)

        # Save best model based on validation accuracy or F1-score
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(config.paths.model_save_dir, f"best_model_{config.model.name}_acc.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best accuracy! Saved model to {checkpoint_path}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = os.path.join(config.paths.model_save_dir, f"best_model_{config.model.name}_f1.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best F1-score! Saved model to {checkpoint_path}")

        if scheduler:
            scheduler.step()

    # Close TensorBoard writer and finish W&B run
    writer.close()
    wandb.finish()

    print(f"Training finished. Best Val Acc: {best_val_acc:.2f}%, Best Val F1: {best_val_f1:.4f}")

# Note: You need to import get_data_loaders from data_loader.py
from data_loader import get_data_loaders
# Note: The helper function imports (baseline_resnet34, se_resnet34) are now done inside the train_model function where they are used.