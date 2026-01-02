import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from early_stopping import EarlyStopping
from loss import contrastive_loss_func
from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import load_config, plot_convergence, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def validate(model, val_loader, device):
    """
    Standardized Validation Function:
    Calculates performance metrics using joint loss (CE + CL).
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0.0

    criterion_ce = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward Pass
            logits, current_scale = model(images, input_ids, attention_mask)

            # Joint Loss Calculation (CE + CL)
            loss_ce = criterion_ce(logits, labels)
            v_f = model.get_vision_features(images)
            loss_cl = contrastive_loss_func(v_f, labels, current_scale)

            loss = loss_ce + loss_cl
            total_val_loss += loss.item()

            # Predictions and Labels
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Standardized Performance Metrics
    metrics = {
        "loss": total_val_loss / len(val_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    return metrics


def train(
    model, model_type, train_loader, val_loader, config, device, early_stopping=None
):
    """
    Standardized Training Loop:
    Implements joint optimization using Cross-Entropy and Contrastive Loss.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion_ce = nn.CrossEntropyLoss()

    # temperature = 0.07  # Temperature for Contrastive Loss
    epochs = config["train"][model_type]["epochs"]

    # History dictionary for convergence plot
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0
    model_path = Path(__file__).parent.parent / "saved_models"
    model_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0.0

        # Warm-up strategy for epoch 1
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-5

        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, current_scale = model(images, input_ids, attention_mask)

            # Joint optimization: CE + CL
            loss_ce = criterion_ce(logits, labels)
            v_f = model.get_vision_features(images)
            loss_cl = contrastive_loss_func(v_f, labels, current_scale)

            loss = loss_ce + loss_cl
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        if epoch > 0:
            scheduler.step()

        # Validation Phase
        val_metrics = validate(model, val_loader, device)

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_pre = val_metrics["precision"]
        val_re = val_metrics["recall"]
        val_f1 = val_metrics["f1_macro"]

        history["train_loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Check early stopping condition
        early_stopping(val_f1)
        if early_stopping.stop_training:
            logging.info(
                f"Early stopping triggered at epoch {epoch+1}. Training terminated."
            )
            break

        # Save the best model based on Macro F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path / f"best_{model_type}_model.pt")
            logging.info(f"Best model saved with F1: {val_f1:.4f}")

    plot_convergence(history, model_type)


if __name__ == "__main__":

    # load config
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Parameter Extraction
        SEMANTIC_PROMPTS = config["prompts"]
        template = "A network traffic grey photo of {}"
        ORIGINAL_PROMPTS = {
            label: template.format(label) for label in SEMANTIC_PROMPTS.keys()
        }
        NPZ_PATH = config["paths"]["output_data_file"]
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = 42
        BATCH_SIZE = config["preprocess"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set seed for reproducibility
    set_seed(42)

    # create dataloaders for TrafficClip
    try:
        train_loader_original, val_loader_original, _ = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            prompts=ORIGINAL_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
        )

        logging.info("DataLoaders created successfully for TrafficClip.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip: {e}")
        raise

    # initialize original trafficclip model
    traffic_clip = TrafficCLIP()
    traffic_clip.to(device)

    # train original traffic clip model
    patience = config["early_stopping"]["patience"]
    delta = config["early_stopping"]["delta"]
    early_stopping_traffic_clip = EarlyStopping(
        patience=patience, delta=delta, verbose=True, mode="max"
    )
    logging.info("Starting training for TrafficCLIP")
    try:
        train(
            model=traffic_clip,
            model_type="traffic_clip",
            train_loader=train_loader_original,
            val_loader=val_loader_original,
            config=config,
            device=device,
            early_stopping=early_stopping_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training TrafficCLIP: {e}")
        raise

    # create dataloaders for TrafficClip Optimized
    try:
        train_loader_optimized, val_loader_optimized, _ = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            prompts=SEMANTIC_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
        )

        logging.info("DataLoaders created successfully for TrafficClip Optimized.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip Optimized: {e}")
        raise

    # initialize optimized trafficclip model
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
    optimized_traffic_clip = OptimizedTrafficCLIP(num_classes=num_classes)
    optimized_traffic_clip.to(device)
    # train optimized traffic clip model
    early_stopping_optimized_traffic_clip = EarlyStopping(
        patience=patience, delta=delta, verbose=True, mode="max"
    )
    logging.info("Starting training for OptimizedTrafficCLIP")
    try:
        train(
            model=optimized_traffic_clip,
            model_type="optimized_traffic_clip",
            train_loader=train_loader_optimized,
            val_loader=val_loader_optimized,
            config=config,
            device=device,
            early_stopping=early_stopping_optimized_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training OptimizedTrafficCLIP: {e}")
        raise
