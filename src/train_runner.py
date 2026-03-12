import argparse
import logging

import dagshub
import mlflow

# import numpy as np
import torch

from early_stopping import EarlyStopping
from models.opt_traffic_clip import OptimizedTrafficCLIP
from models.traffic_clip import TrafficCLIP
from src.dataset import get_dataloader
from src.train import train
from src.utils.utils import load_config, set_seed

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_experiment(args, config, device, is_final=False, is_tune=False):
    """
    Orchestrates a single ablation run based on command line arguments.
    """
    # Set global seed for reproducibility
    set_seed(args.seed)
    train_loader, val_loader, _ = get_dataloader(
        # npz_path=config["paths"]["output_data_file"],
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
        # use stats flag to toggle statistical features
        if args.use_stats:
            model = OptimizedTrafficCLIP(
                num_classes=num_classes,
                use_stats=True,
                stats_input_dim=args.stats_input_dim,
            ).to(device)
            p_cfg = config["early_stopping"]["optimized"]
        else:
            model = OptimizedTrafficCLIP(num_classes=num_classes, use_stats=False).to(
                device
            )
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
        f"Running: {args.model_version} | Lambda: {args.lambda_cl} | Stats Prompts: {args.use_stats_prompts} | Stats Data: {args.use_stats}"
    )

    # Create a unique tag for the experiment
    if args.use_stats:
        unique_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}_stats_data{args.use_stats}"
    else:
        unique_tag = (
            f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}"
        )

    from contextlib import nullcontext

    # --- MLflow Logging ---
    if not is_tune:
        mlflow.set_experiment("TrafficCLIP_Ablation_Study")
        run_ctx = mlflow.start_run(run_name=unique_tag)
    else:
        run_ctx = nullcontext()

    with run_ctx:
        if not is_tune:
            mlflow.log_params(vars(args))

        if is_final:
            class_names = val_loader.dataset.dataset.class_names
            combined_dataset = torch.utils.data.ConcatDataset(
                [train_loader.dataset, val_loader.dataset]
            )

            # Create new split for early stopping
            total_size = len(combined_dataset)
            val_size = int(0.1 * total_size)  # 10% for validation
            train_size = total_size - val_size

            new_train_dataset, new_val_dataset = torch.utils.data.random_split(
                combined_dataset, [train_size, val_size]
            )

            train_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=config["preprocess"]["batch_size"],
                shuffle=True,
                num_workers=4,
            )

            val_loader = torch.utils.data.DataLoader(
                new_val_dataset,
                batch_size=config["preprocess"]["batch_size"],
                shuffle=False,
                num_workers=4,
            )

        best_f1 = train(
            model=model,
            args=args,
            model_version=args.model_version,
            model_type=unique_tag,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            lambda_cl=args.lambda_cl,
            early_stopping=early_stopping,
            optimizer=None,
            scheduler=None,
            is_tune=is_tune,
            is_final=is_final,
            class_names=class_names if is_final else None,
        )

        # else:
        #     best_f1 = train(
        #         model=model,
        #         args=args,
        #         model_version=args.model_version,
        #         model_type=unique_tag,
        #         train_loader=train_loader,
        #         val_loader=val_loader,
        #         config=config,
        #         device=device,
        #         lambda_cl=args.lambda_cl,
        #         early_stopping=early_stopping,
        #         optimizer=None,
        #         scheduler=None,
        #     )

    return best_f1


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
    parser.add_argument(
        "--use_stats", action="store_true", help="Toggle use of statistical features"
    )
    parser.add_argument(
        "--stats_input_dim", type=int, default=3, help="Number of statistical features"
    )

    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    args = parser.parse_args()

    config = load_config()
    USER_NAME = config["user"]["name"]
    REPO_NAME = config["user"]["repo"]
    dagshub.init(
        repo_owner=USER_NAME,
        repo_name=REPO_NAME,
        mlflow=True,
    )

    import logging

    # from importlib import reload
    # reload(logging)
    # # Updated logging setup to save to a file
    # log_file_name = "training.log"
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[
    #         logging.StreamHandler(),  # Keep terminal logs
    #         logging.FileHandler(log_file_name),  # Save to this file
    #     ],
    # )
    logger = logging.getLogger("TrafficCLIP")  # Named logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Stream (Terminal)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # File
        fh = logging.FileHandler("training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_experiment(args, config, device)
