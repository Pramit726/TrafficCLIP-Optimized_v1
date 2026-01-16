import argparse
import logging
from pathlib import Path

import torch

from src.explainability import debug_misclassifications
from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_gradcam_diagnostic(args, config, target_conflicts):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths based on the Training structure
    exp_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}"

    model_path = (
        Path(__file__).parent.parent
        / Path("experiments")
        / Path(args.model_version)
        / Path(exp_tag)
    )
    model_path = model_path / "best_model.pt"
    exp_dir = model_path.parent / Path("gradcam_debug")

    if not model_path.exists():
        logging.error(f"Model checkpoint not found at {model_path}")
        return

    # Initialize model architecture
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(c) for c in traffic_cfg.values())

    if args.model_version == "optimized":
        model = OptimizedTrafficCLIP(num_classes=num_classes).to(device)
    else:
        model = TrafficCLIP().to(device)

    if not model_path.exists():
        logging.error(f"Weights not found at {model_path}")
        return

    # Load the Weights
    logging.info(f"Loading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Execute Debugging
    debug_misclassifications(model, args, config, device, exp_dir, target_conflicts)
    logging.info(f"Grad-CAM heatmaps saved to {exp_dir}")


if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(description="TrafficCLIP Grad-CAM Runner")
    parser.add_argument("--lambda_cl", type=float, required=True)
    parser.add_argument("--use_stats_prompts", action="store_true")
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["original", "optimized"],
        default="optimized",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conflicts",
        nargs="+",
        help="Pairs of True and Predicted labels to debug (e.g., True1 Pred1 True2 Pred2)",
    )

    args = parser.parse_args()
    conflict_list = []
    if args.conflicts:
        conflict_list = list(zip(args.conflicts[0::2], args.conflicts[1::2]))
    run_gradcam_diagnostic(args, config, conflict_list)
