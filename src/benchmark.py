import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import load_config, plot_confusion_matrix, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def test_and_evaluate(model, test_loader, device, model_type, num_runs=3):
    """
    Standardized Testing:
    1. Performs inference 3 times.
    2. Averages Accuracy (AC), Precision (PR), Recall (RC), and Macro F1.
    3. Identifies the best F1 run and generates a Confusion Matrix.
    """
    model.eval()
    run_metrics = []
    best_f1 = -1.0
    best_preds = None
    all_labels = None

    logging.info(f"Starting testing on the test set ({num_runs} inference passes)...")

    for run in range(num_runs):
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits, _ = model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        # Calculate metrics for this pass
        acc = accuracy_score(labels_list, preds_list)
        pr = precision_score(
            labels_list, preds_list, average="macro", zero_division=0.0
        )
        rc = recall_score(labels_list, preds_list, average="macro", zero_division=0.0)
        f1 = f1_score(labels_list, preds_list, average="macro", zero_division=0.0)

        run_metrics.append([acc, pr, rc, f1])

        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds_list
            all_labels = labels_list

        logging.info(f"Pass {run+1}: Accuracy={acc:.4f}, F1={f1:.4f}")

    # Average the metrics
    avg_metrics = np.mean(run_metrics, axis=0)

    logging.info(f"Average Accuracy (AC):  {avg_metrics[0]:.4f}")
    logging.info(f"Average Precision (PR): {avg_metrics[1]:.4f}")
    logging.info(f"Average Recall (RC):    {avg_metrics[2]:.4f}")
    logging.info(f"Average Macro F1 Score: {avg_metrics[3]:.4f}")
    logging.info(f"Highest F1 Score Found: {best_f1:.4f}")

    # Confusion Matrix for similar classes
    plot_confusion_matrix(
        all_labels,
        best_preds,
        test_loader.dataset.dataset.class_names,
        model_type=model_type,
    )


if __name__ == "__main__":

    # load config
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set seed for reproducibility
    set_seed(42)
    try:
        # Parameter Extraction
        SEMANTIC_PROMPTS = config["prompts"]
        template = "A network traffic grey photo of {}"
        ORIGINAL_PROMPTS = {
            label: template.format(label) for label in SEMANTIC_PROMPTS.keys()
        }
        NPZ_PATH = config["paths"]["output_data_file"]
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["test"]["max_length"]
        SEED = 42
        BATCH_SIZE = config["test"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloaders for TrafficClip
    try:
        _, _, test_loader_original = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            prompts=ORIGINAL_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
        )
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip: {e}")
        raise

    # initialize trafficclip model
    traffic_clip = TrafficCLIP()

    try:
        check_point_path_original = (
            Path(__file__).parent.parent / "saved_models" / "best_traffic_clip_model.pt"
        )
        # load weights for trafficclip model
        traffic_clip.load_state_dict(
            torch.load(check_point_path_original, map_location=device)
        )
        traffic_clip.to(device)
    except Exception as e:
        logging.error(f"Error loading TrafficClip model weights: {e}")
        raise

    try:
        logging.info("Testing TrafficClip model")
        test_and_evaluate(
            traffic_clip, test_loader_original, device, model_type="TrafficClip"
        )
    except Exception as e:
        logging.error(f"Error during testing TrafficClip model: {e}")
        raise

    # create dataloaders for TrafficClip Optimized
    try:
        _, _, test_loader_optimized = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            prompts=ORIGINAL_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
        )
    except Exception as e:
        logging.error(f"Error creating DataLoaders for TrafficClip: {e}")
        raise

    # initialize trafficclip optimized model
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
    traffic_clip_optimized = OptimizedTrafficCLIP(num_classes=num_classes)

    try:
        check_point_path_optimized = (
            Path(__file__).parent.parent
            / "saved_models"
            / "best_traffic_clip_optimized_model.pt"
        )
        # load weights for trafficclip optimized model
        traffic_clip_optimized.load_state_dict(
            torch.load(check_point_path_optimized, map_location=device)
        )
        traffic_clip_optimized.to(device)
    except Exception as e:
        logging.error(f"Error loading TrafficClip Optimized model weights: {e}")
        raise
    try:
        logging.info("Testing TrafficClip Optimized model")
        test_and_evaluate(
            traffic_clip_optimized,
            test_loader_optimized,
            device,
            model_type="TrafficClip_Optimized",
        )
    except Exception as e:
        logging.error(f"Error during testing TrafficClip Optimized model: {e}")
        raise
