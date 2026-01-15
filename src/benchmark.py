import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import load_config, plot_confusion_matrix, save_metrics, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def test_and_evaluate(
    model,
    device,
    model_type,
    model_version,
    config,
    use_dynamic_prompts,
    num_runs=3,
    seed=42,
):
    """
    Standardized Testing for M.Tech Thesis:
    1. Regenerates a stratified test_loader for each pass using different seeds.
    2. Averages AC, PR, RC, and Macro F1 across all runs.
    3. Identifies the highest F1 run for the Confusion Matrix.
    """
    model.eval()
    run_metrics = []
    best_f1 = -1.0
    best_preds = None
    all_labels = None

    NPZ_PATH = config["paths"]["mini_output_data_file"]
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    MAX_LENGTH = config["test"]["max_length"]
    BATCH_SIZE = config["test"]["batch_size"]

    logging.info(f"Starting evaluation for {model_type} ({num_runs} passes)")

    for run in range(num_runs):
        # Generate a unique seed for this specific run
        current_seed = seed + run
        set_seed(current_seed)

        # Recreate dataloader with the new seed to get a different stratified test split
        _, _, test_loader = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            # prompts=ORIGINAL_PROMPTS,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=current_seed,
            use_dynamic_prompts=use_dynamic_prompts,
        )

        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Unpack tuple: (logits, logit_scale)
                logits, _ = model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        # Calculate metrics with zero_division safety
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
            class_names = test_loader.dataset.dataset.class_names

        logging.info(
            f"Pass {run+1} (Seed {current_seed}): Accuracy={acc:.4f}, Macro F1={f1:.4f}"
        )

    # Average the metrics for the final report
    avg_metrics = np.mean(run_metrics, axis=0)
    std_metrics = np.std(run_metrics, axis=0)

    logging.info(f"Final Results for {model_type}")
    logging.info(f"Avg Accuracy (AC):  {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    logging.info(f"Avg Precision (PR): {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    logging.info(f"Avg Recall (RC):    {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    logging.info(f"Avg Macro F1 Score: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")

    # save run metrics to pandas dataframe
    # results_path = Path(__file__).parent.parent / "results" / "metrics" / model_version
    # results_path.mkdir(parents=True, exist_ok=True)
    # results_file = results_path / f"{model_type}_test_results.csv"
    # df = pd.DataFrame(
    #     run_metrics, columns=["Accuracy", "Precision", "Recall", "Macro F1"]
    # )
    # df.to_csv(results_file, index_label="Run")
    # logging.info(f"Saved detailed run metrics to {results_file}")
    run_metrics_array = np.array(run_metrics)
    save_metrics(run_metrics_array, model_type, model_version)

    plot_confusion_matrix(
        all_labels,
        best_preds,
        class_names,
        model_type=model_type,
        model_version=model_version,
    )


if __name__ == "__main__":
    # Load config and set initial environment
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate TrafficClip
    traffic_clip = TrafficCLIP().to(device)
    path_orig = (
        Path(__file__).parent.parent / "saved_models" / "best_traffic_clip_model.pt"
    )
    traffic_clip.load_state_dict(torch.load(path_orig, map_location=device))
    test_seed = config["test"].get("seed", 42)
    test_and_evaluate(
        traffic_clip,
        device,
        "TrafficClip",
        config,
        use_dynamic_prompts=False,
        seed=test_seed,
    )

    # Evaluate TrafficClip Optimized
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(c) for c in traffic_cfg.values())
    opt_model = OptimizedTrafficCLIP(num_classes=num_classes).to(device)
    path_opt = (
        Path(__file__).parent.parent
        / "saved_models"
        / "best_optimized_traffic_clip_model.pt"
    )
    opt_model.load_state_dict(torch.load(path_opt, map_location=device))

    test_and_evaluate(
        opt_model,
        device,
        "TrafficClip_Optimized",
        config,
        use_dynamic_prompts=True,
        seed=test_seed,
    )
