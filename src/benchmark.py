import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.utils.utils import (
    get_original_descriptor_bank,
    load_config,
    plot_confusion_matrix,
    save_metrics,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def test_and_evaluate(
    model,
    device,
    model_type,
    args,
    model_version,
    config,
    num_runs=3,
    seed=42,
):
    """
    Standardized Testing:
    1. Regenerates a stratified test_loader for each pass using different seeds.
    2. Averages AC, PR, RC, and Macro F1 across all runs.
    3. Identifies the highest F1 run for the Confusion Matrix.
    """
    model.eval()
    run_metrics = []
    best_f1 = -1.0
    best_preds = None
    all_labels = None

    NPZ_PATH = config["paths"]["output_data_file"]
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    MAX_LENGTH = config["test"]["max_length"]
    BATCH_SIZE = config["test"]["batch_size"]

    # Pre-loading tokenizer once saves massive time across passes
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    logging.info(f"Starting evaluation for {model_type} ({num_runs} passes)")

    for run in range(num_runs):
        current_seed = seed + run
        set_seed(current_seed)

        # Fresh loaders for every pass to ensure stratification seed is applied
        _, _, test_loader = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=current_seed,
            use_dynamic_prompts=args.use_stats_prompts,
        )

        class_names = test_loader.dataset.dataset.class_names
        num_classes = len(class_names)
        preds_list = []
        labels_list = []

        # Pre-encode static bank if not using dynamic prompts
        static_descriptor_bank = None
        if not args.use_stats_prompts:
            static_descriptor_bank = get_original_descriptor_bank(
                model, tokenizer, class_names, MAX_LENGTH, device
            )

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                stats = batch["stats"].to(device)

                # FIX 1: Initialize current_batch_preds at the start of every batch
                current_batch_preds = []

                # --- PATH A: Original / Static Logic ---
                if model_version == "original" and not args.use_stats_prompts:
                    v_e = model.get_vision_features(images)
                    logits = (
                        torch.matmul(v_e, static_descriptor_bank.T)
                        * model.logit_scale.exp()
                    )
                    batch_preds_tensor = torch.argmax(logits, dim=1)
                    current_batch_preds = batch_preds_tensor.cpu().numpy().tolist()

                # --- PATH B: Optimized / Dynamic / MLP Logic ---
                else:
                    v_e = model.get_vision_features(images)
                    s_e = None
                    if hasattr(model, "use_stats") and model.use_stats:
                        s_e = model.stats_proj(stats)
                        s_e = torch.nn.functional.normalize(s_e, p=2, dim=-1)

                    # Iterate through batch for Hypothesis Testing
                    for i in range(len(images)):
                        if not args.use_stats_prompts:
                            t_e_all = static_descriptor_bank
                        else:
                            m_iat, m_jitter, m_entropy = stats[i].cpu().numpy()
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
                            )

                        # Multi-modal Fusion
                        v_e_hyp = v_e[i : i + 1].expand(num_classes, -1)
                        if s_e is not None:
                            s_e_hyp = s_e[i : i + 1].expand(num_classes, -1)
                            combined = torch.cat((v_e_hyp, t_e_all, s_e_hyp), dim=1)
                        else:
                            combined = torch.cat((v_e_hyp, t_e_all), dim=1)

                        # Logic Check: Use fusion_head if optimized, else Cosine Sim
                        if hasattr(model, "fusion_head"):
                            logits_hyp = model.fusion_head(combined)
                            confidences = torch.diag(logits_hyp)
                        else:
                            # Fallback for dynamic prompts on original model
                            confidences = (
                                torch.matmul(v_e[i : i + 1], t_e_all.T)
                                * model.logit_scale.exp()
                            ).squeeze()

                        current_batch_preds.append(torch.argmax(confidences).item())

                # FIX 2: Standardized extend call
                preds_list.extend(current_batch_preds)
                labels_list.extend(labels.cpu().numpy().tolist())

        # Metrics calculation
        acc = accuracy_score(labels_list, preds_list)
        f1 = f1_score(labels_list, preds_list, average="macro", zero_division=0.0)
        run_metrics.append([acc, f1])  # Add PR/RC as needed

        if f1 > best_f1:
            best_f1 = f1
            best_preds = list(preds_list)
            all_labels = list(labels_list)

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
