import argparse
import logging

# import numpy as np
import torch

from early_stopping import EarlyStopping
from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.train import train
from src.utils.utils import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_experiment(args, config, device):
    """
    Orchestrates a single ablation run based on command line arguments.
    """
    # Set global seed for reproducibility
    set_seed(args.seed)
    train_loader, val_loader, _ = get_dataloader(
        npz_path=config["paths"]["output_data_file"],
        tokenizer=config["preprocess"]["tokenizer"],
        batch_size=config["preprocess"]["batch_size"],
        max_length=config["preprocess"]["max_length"],
        seed=args.seed,
        use_dynamic_prompts=args.use_stats_prompts,  # Phase 2 Toggle
    )

    # Initialize Model
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(cl) for cl in traffic_cfg.values())

    if args.model_version == "optimized":
        model = OptimizedTrafficCLIP(num_classes=num_classes).to(device)
        p_cfg = config["early_stopping"]["optimized"]
    else:
        model = TrafficCLIP().to(device)
        p_cfg = config["early_stopping"]["original"]

    # Setup Early Stopping
    early_stopping = EarlyStopping(
        patience=p_cfg["patience"], delta=p_cfg["delta"], mode="max"
    )

    logging.info(
        f"Early Stopping Config: Patience={p_cfg['patience']}, Delta={p_cfg['delta']}"
    )

    logging.info(
        f"Running: {args.model_version} | Lambda: {args.lambda_cl} | Stats: {args.use_stats_prompts}"
    )

    # Create a unique tag for the experiment
    unique_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}"

    train(
        model=model,
        model_version=args.model_version,
        model_type=unique_tag,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        lambda_cl=args.lambda_cl,
        early_stopping=early_stopping,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficCLIP Ablation Runner")
    parser.add_argument(
        "--lambda_cl", type=float, default=2.0, help="Weight for Contrastive Loss"
    )
    parser.add_argument(
        "--use_stats_prompts",
        action="store_true",
        help="Toggle dynamic physics prompts",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["original", "optimized"],
        default="optimized",
    )
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_experiment(args, config, device)
