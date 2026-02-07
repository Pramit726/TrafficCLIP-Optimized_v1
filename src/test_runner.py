import argparse
import logging
from pathlib import Path

import torch

from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.benchmark import test_and_evaluate
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_test_experiment(args):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths based on the Training structure
    if args.use_stats:
        exp_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}_stats_data{args.use_stats}"
    else:
        exp_tag = (
            f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}"
        )
    # if args.use_stats_prompts:
    #     exp_tag += "_statsTrue"
    # else:
    #     exp_tag += "_statsFalse"

    results_path = (
        Path(__file__).parent.parent
        / Path("experiments")
        / Path(args.model_version)
        / Path(exp_tag)
    )
    model_path = results_path / "best_model.pt"

    if not model_path.exists():
        logging.error(f"Model checkpoint not found at {model_path}")
        return

    # Initialize model architecture
    traffic_cfg = config["dataset"]["traffic"]["classes"]
    num_classes = sum(len(c) for c in traffic_cfg.values())

    if args.model_version == "optimized":
        if args.use_stats:
            model = OptimizedTrafficCLIP(
                num_classes=num_classes,
                use_stats=True,
                stats_input_dim=args.stats_input_dim,
            ).to(device)
        else:
            model = OptimizedTrafficCLIP(num_classes=num_classes, use_stats=False).to(
                device
            )
    else:
        model = TrafficCLIP().to(device)

    # Load the Weights
    logging.info(f"Loading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_and_evaluate(
        model=model,
        device=device,
        model_version=args.model_version,
        model_type=exp_tag,
        args=args,
        config=config,
        num_runs=args.num_runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficCLIP Ablation Test Runner")
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--lambda_cl", type=float, required=True)
    parser.add_argument("--use_stats_prompts", action="store_true")
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["original", "optimized"],
        default="optimized",
    )
    parser.add_argument(
        "--use_stats", action="store_true", help="Toggle use of statistical features"
    )
    parser.add_argument(
        "--stats_input_dim", type=int, default=3, help="Number of statistical features"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_test_experiment(args)
