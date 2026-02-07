import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from early_stopping import EarlyStopping
from loss import contrastive_loss_func
from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import (
    get_original_descriptor_bank,
    load_config,
    plot_convergence,
    save_metrics,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def validate(
    model,
    model_version,
    args,
    val_loader,
    device,
    config,
    lambda_cl=5.0,
):
    """
    Standardized Validation Function:
    Calculates performance metrics using joint loss (CE + CL).
    """
    model.eval()
    all_preds = []  # Total predictions across all batches
    all_labels = []  # Total true labels across all batches
    total_val_loss = 0.0

    criterion_ce = torch.nn.CrossEntropyLoss()

    # Load configuration
    NPZ_PATH = config["paths"]["output_data_file"]
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    MAX_LENGTH = config["test"]["max_length"]
    BATCH_SIZE = config["test"]["batch_size"]

    # val_loader = get_dataloader(
    #     npz_path=NPZ_PATH,
    #     tokenizer=TOKENIZER_NAME,
    #     batch_size=BATCH_SIZE,
    #     max_length=MAX_LENGTH,
    #     use_dynamic_prompts=use_dynamic_prompts,
    # )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    class_names = val_loader.dataset.dataset.class_names

    # PRE-ENCODE static bank for Original variant
    if not args.use_stats_prompts:
        static_descriptor_bank = get_original_descriptor_bank(
            model, tokenizer, class_names, MAX_LENGTH, device
        )

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            stats = batch["stats"].to(device)

            # LOSS CALCULATION
            if model_version == "original":
                logits_gt, current_scale = model(images, input_ids, attention_mask)
            else:
                logits_gt, current_scale = model(
                    images, input_ids, attention_mask, stats
                )

            loss_ce = criterion_ce(logits_gt, labels)
            v_f = model.get_vision_features(images)
            loss_cl = contrastive_loss_func(v_f, labels, current_scale)

            loss = loss_ce + (lambda_cl * loss_cl)
            total_val_loss += loss.item()

            # PREDICTION STEP
            # List for this specific batch's predictions
            current_batch_preds = []

            # Variant 1: Original Prompts (Global Static Matching)
            if not args.use_stats_prompts:
                # Vision features normalized for similarity
                v_e = model.get_vision_features(images)
                # Match against all K classes in the static bank
                logits = (
                    torch.matmul(v_e, static_descriptor_bank.T)
                    * model.logit_scale.exp()
                )
                preds = torch.argmax(logits, dim=1)

            # Variant 2: Statistical Prompts (Global Dynamic Matching)
            else:

                v_e = model.get_vision_features(images)
                s_e = None
                if args.use_stats:
                    s_e = model.stats_proj(stats)  # [Batch, 512]
                    s_e = torch.nn.functional.normalize(s_e, p=2, dim=-1)

                all_preds = []

                # Must iterate because descriptions depend on specific sample statistics
                for i in range(len(images)):
                    if not args.use_stats_prompts:
                        # Use the pre-encoded bank you made at the start of the 'run'
                        t_e_all = static_descriptor_bank  # [K, 1024]
                    else:
                        m_iat, m_jitter, m_entropy = stats[i].cpu().numpy()

                        # Generate K descriptions for THIS specific image's behavior
                        sample_prompts = [
                            f"A network traffic gray photo of class {name} with "
                            f"{m_iat:.2f}ms mean IAT, {m_jitter:.2f}ms jitter, "
                            f"and {m_entropy:.2f} byte entropy."
                            for name in class_names
                        ]

                        encoded = tokenizer(
                            sample_prompts,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                            max_length=MAX_LENGTH,
                        ).to(device)
                    t_e_all = model.get_text_features(
                        encoded["input_ids"], encoded["attention_mask"]
                    )  # [K, 1024]

                    # Expand current image and stats to match K class hypotheses
                    num_classes = len(class_names)
                    v_e_hyp = v_e[i : i + 1].expand(num_classes, -1)  # [K, 1024]

                    if s_e is not None:
                        s_e_hyp = s_e[i : i + 1].expand(num_classes, -1)  # [K, 512]
                        combined = torch.cat(
                            (v_e_hyp, t_e_all, s_e_hyp), dim=1
                        )  # [K, 2560]
                    else:
                        combined = torch.cat((v_e_hyp, t_e_all), dim=1)  # [K, 2048]

                        # Pass all K hypotheses through the MLP at once
                        logits = model.fusion_head(
                            combined
                        )  # [K_hypotheses, K_classes]

                        confidences = torch.diag(logits)
                        # The predicted class is the one with the highest confidence (diagonal element)
                        current_batch_preds.append(torch.argmax(confidences).item())

        all_preds.extend(current_batch_preds)
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


# def train(
#     model,
#     model_version,
#     model_type,
#     train_loader,
#     val_loader,
#     config,
#     device,
#     lambda_cl,
#     early_stopping=None,
# ):
#     """
#     Standardized Training Loop:
#     Implements joint optimization using Cross-Entropy and Contrastive Loss.
#     """

#     epochs = config["train"][model_version]["epochs"]
#     optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#     criterion_ce = nn.CrossEntropyLoss()

#     # History dictionary for convergence plot
#     history = {"train_loss": [], "val_loss": [], "val_f1": []}
#     best_f1 = 0.0
#     model_path = (
#         Path(__file__).parent.parent
#         / Path("experiments")
#         / Path(model_version)
#         # / f"{model_type}_L{lambda_cl}"
#         / model_type
#     )
#     model_path.mkdir(parents=True, exist_ok=True)

#     run_metrics = []
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         total_train_loss = 0.0

#         # Warm-up strategy for epoch 1
#         if epoch == 0:
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = 1e-5

#         for batch in train_loader:
#             images = batch["image"].to(device)
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["label"].to(device)
#             # Extract Physics Modality (IAT, Jitter, Entropy)
#             stats_vector = batch["stats"].to(device)

#             optimizer.zero_grad()
#             if model_version == "original":
#                 logits, current_scale = model(images, input_ids, attention_mask)
#             else:
#                 logits, current_scale = model(
#                     images, input_ids, attention_mask, stats_vector
#                 )

#             # Joint optimization: CE + CL
#             loss_ce = criterion_ce(logits, labels)
#             v_f = model.get_vision_features(images)
#             loss_cl = contrastive_loss_func(v_f, labels, current_scale)

#             loss = loss_ce + (lambda_cl * loss_cl)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         if epoch > 0:
#             scheduler.step()

#         # Validation Phase
#         val_metrics = validate(model, model_version, val_loader, device, lambda_cl)

#         val_loss = val_metrics["loss"]
#         val_acc = val_metrics["accuracy"]
#         val_pre = val_metrics["precision"]
#         val_re = val_metrics["recall"]
#         val_f1 = val_metrics["f1_macro"]

#         history["train_loss"].append(total_train_loss / len(train_loader))
#         history["val_loss"].append(val_loss)
#         history["val_f1"].append(val_f1)

#         logging.info(
#             f"Epoch {epoch+1}/{epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | "
#             f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
#         )

#         # Check early stopping condition
#         early_stopping(val_f1)
#         if early_stopping.stop_training:
#             logging.info(
#                 f"Early stopping triggered at epoch {epoch+1}. Training terminated."
#             )
#             break

#         # Save the best model based on Macro F1 score
#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             logging.info(model_path)
#             torch.save(model.state_dict(), model_path / "best_model.pt")
#             logging.info(f"Best model saved with F1: {val_f1:.4f}")

#     run_metrics.append([val_acc, val_pre, val_re, val_f1])
#     save_metrics(run_metrics, model_type, model_version)
#     plot_convergence(history, model_type, save_path=model_path)


def train(
    model,
    model_version,
    model_type,
    train_loader,
    val_loader,
    args,
    config,
    device,
    lambda_cl,
    early_stopping=None,
):
    """
    Optimized Tri-modal Training Loop :
    - AdamW Optimizer for multimodal gradient stability.
    - 5-Epoch Linear LR Warm-up.
    - Scheduled Lambda increase (1.0 -> Target) over 20 epochs.
    """

    epochs = config["train"][model_version]["epochs"]
    warmup_epochs = 5
    start_lambda = 1.0

    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce = nn.CrossEntropyLoss()

    # History dictionary for convergence plot
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0
    model_path = (
        Path(__file__).parent.parent
        / Path("experiments")
        / Path(model_version)
        # / f"{model_type}_L{lambda_cl}"
        / model_type
    )
    model_path.mkdir(parents=True, exist_ok=True)

    run_metrics = []
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0.0

        # # --- 1. Dynamic Lambda Scheduling ---
        # # Linearly scale lambda from 1.0 to  chosen lambda_cl over 20 epochs
        # if epoch < 20:
        #     current_lambda = start_lambda + (lambda_cl - start_lambda) * (epoch / 20)
        # else:
        #     current_lambda = lambda_cl

        # # --- 2. Learning Rate Warm-up (Linear) ---
        # if epoch < warmup_epochs:
        #     lr_scale = (epoch + 1) / warmup_epochs
        #     for pg in optimizer.param_groups:
        #         pg["lr"] = 1e-4 * lr_scale

        # Warm-up strategy for epoch 1
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-5

        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Extract Physics Modality (IAT, Jitter, Entropy)
            stats_vector = batch["stats"].to(device)

            optimizer.zero_grad()
            if model_version == "original":
                logits, current_scale = model(images, input_ids, attention_mask)
            else:
                logits, current_scale = model(
                    images, input_ids, attention_mask, stats_vector
                )

            # Joint optimization: CE + CL
            loss_ce = criterion_ce(logits, labels)
            v_f = model.get_vision_features(images)
            loss_cl = contrastive_loss_func(v_f, labels, current_scale)

            # loss = loss_ce + (current_lambda * loss_cl)
            loss = loss_ce + (lambda_cl * loss_cl)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        if epoch > 0:
            scheduler.step()

        # Validation Phase
        if model_version == "original":
            val_metrics = validate(
                model,
                model_version,
                device=device,
                args=args,
                val_loader=val_loader,
                config=config,
                lambda_cl=lambda_cl,
            )
        else:
            val_metrics = validate(
                model,
                model_version,
                args=args,
                device=device,
                val_loader=val_loader,
                config=config,
                lambda_cl=lambda_cl,
            )

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
            logging.info(model_path)
            torch.save(model.state_dict(), model_path / "best_model.pt")
            logging.info(f"Best model saved with F1: {val_f1:.4f}")

    run_metrics.append([val_acc, val_pre, val_re, val_f1])
    save_metrics(run_metrics, model_type, model_version)
    plot_convergence(history, model_type, save_path=model_path)


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
        NPZ_PATH = config["paths"]["output_data_file"]
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = config["train"]["seed"]
        BATCH_SIZE = config["preprocess"]["batch_size"]
        LAMBDA_CL_ORIGINAL = config["train"]["traffic_clip"]["lambda_cl"]
        LAMBDA_CL_OPTIMIZED = config["train"]["optimized_traffic_clip"]["lambda_cl"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set seed for reproducibility
    set_seed(SEED)

    # create dataloaders for TrafficClip
    try:
        train_loader_original, val_loader_original, _ = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            # prompts=ORIGINAL_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
            use_dynamic_prompts=False,
        )

        logging.info("DataLoaders created successfully for TrafficClip.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip: {e}")
        raise

    # initialize original trafficclip model
    traffic_clip = TrafficCLIP()
    traffic_clip.to(device)

    # train original traffic clip model
    patience_traffic_clip = config["early_stopping"]["traffic_clip"]["patience"]
    delta_traffic_clip = config["early_stopping"]["traffic_clip"]["delta"]
    early_stopping_traffic_clip = EarlyStopping(
        patience=patience_traffic_clip,
        delta=delta_traffic_clip,
        verbose=True,
        mode="max",
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
            lambda_cl=LAMBDA_CL_ORIGINAL,
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
            # prompts=SEMANTIC_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
            use_dynamic_prompts=True,
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

    patience_traffic_clip_optimized = config["early_stopping"][
        "optimized_traffic_clip"
    ]["patience"]
    delta_traffic_clip_optimized = config["early_stopping"]["optimized_traffic_clip"][
        "delta"
    ]

    # train optimized traffic clip model
    early_stopping_optimized_traffic_clip = EarlyStopping(
        patience=patience_traffic_clip_optimized,
        delta=delta_traffic_clip_optimized,
        verbose=True,
        mode="max",
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
            lambda_cl=LAMBDA_CL_OPTIMIZED,
            early_stopping=early_stopping_optimized_traffic_clip,
        )
    except Exception as e:
        logging.error(f"Error during training OptimizedTrafficCLIP: {e}")
        raise
