import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from early_stopping import EarlyStopping
from loss import contrastive_loss_func
from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import load_config, plot_convergence

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (important!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, val_loader, device):
    """
    Standardized Validation Function:
    Calculates Accuracy and Macro F1-score as used in TrafficCLIP benchmarks.
    """
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Get model predictions
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            # Convert logits to class indices
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics according to the paper's standards
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = val_loss / len(val_loader)

    return avg_loss, acc, f1


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

    temperature = 0.07  # Temperature for Contrastive Loss
    epochs = config["train"][model_type]["epochs"]

    # History dictionary for convergence plot
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0

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
            logits = model(images, input_ids, attention_mask)

            # Joint optimization: CE + CL
            loss_ce = criterion_ce(logits, labels)
            v_f = model.get_vision_features(images)
            loss_cl = contrastive_loss_func(v_f, labels, temperature)

            loss = loss_ce + loss_cl
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        if epoch > 0:
            scheduler.step()

        # Validation Phase
        val_loss, val_acc, val_f1 = validate(model, val_loader, device)

        history["train_loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Check early stopping condition
        early_stopping.check_early_stop(val_loss)

        # Save the best model based on Macro F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"saved_models/best_{model_type}_model.pt")
            logging.info(f"--> Best model saved with F1: {val_f1:.4f}")

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
    # set seed for reproducibility
    set_seed(42)

    try:
        # Parameter Extraction
        SEMANTIC_PROMPTS = config["prompts"]
        NPZ_PATH = config["paths"]["output_data_file"]
        TENSOR_DIR = Path(config["paths"]["tensors_dir"])
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = 42
        BATCH_SIZE = config["preprocess"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # create dataloaders
    try:
        train_loader, val_loader, test_loader = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            prompts=SEMANTIC_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
        )

        logging.info("DataLoaders created successfully.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders: {e}")
        raise

    # initialize original trafficclip model
    traffic_clip = TrafficCLIP()
    traffic_clip.to(device)

    # train original traffic clip model
    patience = config["early_stopping"]["patience"]
    delta = config["early_stopping"]["delta"]
    early_stopping_traffic_clip = EarlyStopping(
        patience=patience, delta=delta, verbose=True
    )
    logging.info("Starting training for TrafficCLIP")
    try:
        train(
            model=traffic_clip,
            model_type="traffic_clip",
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            early_stopping=early_stopping_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training TrafficCLIP: {e}")
        raise

    # initialize optimized trafficclip model
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
    optimized_traffic_clip = OptimizedTrafficCLIP(num_classes=num_classes)
    optimized_traffic_clip.to(device)
    # train optimized traffic clip model
    early_stopping_optimized_traffic_clip = EarlyStopping(
        patience=patience, delta=delta, verbose=True
    )
    logging.info("Starting training for OptimizedTrafficCLIP")
    try:
        train(
            model=optimized_traffic_clip,
            model_type="optimized_traffic_clip",
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            early_stopping=early_stopping_optimized_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training OptimizedTrafficCLIP: {e}")
        raise
