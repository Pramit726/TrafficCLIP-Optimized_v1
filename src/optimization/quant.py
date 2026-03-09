import argparse
import logging
import sys
from pathlib import Path

import dagshub
import mlflow
import torch

from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.optimization.calibration import create_calibration_dataloader
from src.utils.utils import load_config

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# def quantize_traffic_model(model, calibration_loader, backend="fbgemm"):
#     """
#     Performs Post-Training Static Quantization (PTQ) on the OptimizedTrafficCLIP model.

#     Args:
#         model: The trained OptimizedTrafficCLIP model (float32).
#         calibration_loader: Dataloader with 200-300 balanced validation samples.
#         backend: 'fbgemm' for x86 CPUs (Intel/AMD) or 'qnnpack' for ARM.
#     """
#     model.eval()

#     # 1. Module Fusion: Merges Linear + ReLU to reduce memory round-trips
#     logging("Fusing modules")
#     modules_to_fuse = [
#         ["stats_proj.net.0", "stats_proj.net.2"],  # Linear + ReLU (Stats Head)
#         ["fusion_head.mlp.0", "fusion_head.mlp.2"],  # Linear + ReLU (Fusion Head)
#     ]

#     # We use a try-except because if layers were already fused, this might error.
#     try:
#         fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
#     except Exception as e:
#         logging(f"Fusion skipped or failed: {e}")
#         fused_model = model

#     # 2. Assign Quantization Configuration
#     fused_model.qconfig = torch.quantization.get_default_qconfig(backend)

#     # 3.Inserts 'observers' to track the min/max ranges of your data
#     torch.quantization.prepare(fused_model, inplace=True)

#     # 4. Calibration
#     logging(f"Calibrating on {len(calibration_loader.dataset)} samples")
#     with torch.no_grad():
#         for images, stats, input_ids, attn_mask, _ in calibration_loader:
#             fused_model(images, input_ids, attn_mask, stats)

#     # 5. Convert: Actually transforms the weights from Float32 to Int8
#     logging("Converting model to INT8")
#     quantized_model = torch.quantization.convert(fused_model, inplace=False)

#     return quantized_model


# if __name__ == "__main__":
#     try:
#         config = load_config()
#         traffic_cfg = config["dataset"]["traffic"]["classes"]

#         num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
#         model = OptimizedTrafficCLIP(
#             num_classes=num_classes, stats_input_dim=3, use_stats=True
#         )  # IAT, Jitter, Entropy
#         model.load_state_dict(
#             torch.load("best_trafficclip_model.pt", map_location="cpu")
#         )
#         logging.info("Model loaded successfully.")
#     except Exception as e:
#         logging.error(f"Error initializing OptimizedTrafficCLIP model: {e}")
#         raise

#     calibration_dataloader = create_calibration_dataloader(num_samples=200)
#     quantized_traffic_model = quantize_traffic_model(model, calibration_dataloader)


# def quantize_traffic_model(model, calibration_loader, backend="fbgemm"):
#     """
#     Performs Post-Training Static Quantization (PTQ) on OptimizedTrafficCLIP.
#     Skips Embedding layers to prevent 'float_qparams_weight_only_qconfig' errors.
#     """
#     model.eval()
#     model.to("cpu")  # PTQ Eager mode requires CPU for calibration/conversion

#     # --- 1. Module Fusion ---
#     logger.info("Fusing modules for optimized INT8 execution...")
#     modules_to_fuse = [
#         ["stats_proj.net.0", "stats_proj.net.2"],
#         ["fusion_head.mlp.0", "fusion_head.mlp.2"],
#     ]

#     try:
#         fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
#     except Exception as e:
#         logger.warning(f"Fusion skipped: {e}")
#         fused_model = model

#     # --- 2. Configuration & Layer Filtering ---
#     # Set the default global qconfig
#     fused_model.qconfig = torch.quantization.get_default_qconfig(backend)

#     # Manually disable quantization for Embedding layers
#     for name, module in fused_model.named_modules():
#         if isinstance(module, torch.nn.Embedding):
#             logger.info(f"Skipping quantization for embedding layer: {name}")
#             module.qconfig = None

#     # Prepare inserts 'observers' into the non-skipped layers
#     torch.quantization.prepare(fused_model, inplace=True)

#     # Log quantization parameters to MLflow
#     mlflow.log_params(
#         {
#             "quant_backend": backend,
#             "quant_calibration_samples": len(calibration_loader.dataset),
#             "quant_method": "Static PTQ (Hybrid)",
#         }
#     )

#     # --- 3. Calibration ---
#     logger.info(f"Calibrating on {len(calibration_loader.dataset)} samples...")
#     with torch.no_grad():
#         for batch in calibration_loader:
#             # Explicitly move tensors to CPU to match the model
#             images = batch["image"].to("cpu")
#             stats = batch["stats"].to("cpu")
#             input_ids = batch["input_ids"].to("cpu")
#             attn_mask = batch["attention_mask"].to("cpu")
#             fused_model(images, input_ids, attn_mask, stats)

#     # --- 4. Conversion ---
#     logger.info("Converting model weights from Float32 to Int8...")
#     # This will now skip the Embedding layers and convert the rest
#     quantized_model = torch.quantization.convert(fused_model, inplace=False)

#     # Tag this run in MLflow
#     mlflow.set_tag("model_precision", "INT8-Hybrid")

#     return quantized_model


# def quantize_traffic_model(model, calibration_loader, backend="fbgemm"):
#     """
#     Performs Post-Training Static Quantization (PTQ) on OptimizedTrafficCLIP.
#     Logs quantization metadata and final model size to MLflow.
#     """
#     model.eval()
#     model.to("cpu")

#     # --- 1. Module Fusion ---
#     # Merges Linear + ReLU into a single operator to speed up inference
#     logger.info("Fusing modules for optimized INT8 execution...")
#     modules_to_fuse = [
#         ["stats_proj.net.0", "stats_proj.net.2"],
#         ["fusion_head.mlp.0", "fusion_head.mlp.2"],
#     ]

#     try:
#         fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
#     except Exception as e:
#         logger.warning(f"Fusion skipped: {e}")
#         fused_model = model

#     # --- 2. Configuration & Observers ---
#     # Log quantization parameters to MLflow
#     mlflow.log_params(
#         {
#             "quant_backend": backend,
#             "quant_calibration_samples": len(calibration_loader.dataset),
#             "quant_method": "Static PTQ",
#         }
#     )

#     fused_model.qconfig = torch.quantization.get_default_qconfig(backend)
#     torch.quantization.prepare(fused_model, inplace=True)

#     # --- 3. Calibration ---
#     # We pass data through the model to let 'observers' calculate min/max ranges
#     logger.info(f"Calibrating on {len(calibration_loader.dataset)} samples...")
#     with torch.no_grad():
#         for batch in calibration_loader:
#             # Using dictionary keys from your dataloader
#             images = batch["image"]
#             stats = batch["stats"]
#             input_ids = batch["input_ids"]
#             attn_mask = batch["attention_mask"]
#             fused_model(images, input_ids, attn_mask, stats)

#     # --- 4. Conversion ---
#     logging.info("Converting model weights from Float32 to Int8...")
#     quantized_model = torch.quantization.convert(fused_model, inplace=False)

#     # Tag this run in MLflow as a quantization experiment
#     mlflow.set_tag("model_precision", "INT8")

#     return quantized_model

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# def quantize_traffic_model(model, calibration_loader, backend="fbgemm"):
#     model.eval()
#     model.to("cpu")

#     # --- 1. Module Fusion ---
#     modules_to_fuse = [
#         ["stats_proj.net.0", "stats_proj.net.2"],
#         ["fusion_head.mlp.0", "fusion_head.mlp.2"],
#     ]
#     try:
#         fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
#     except Exception as e:
#         logging.warning(f"Fusion skipped: {e}")
#         fused_model = model

#     # --- 2. Configuration & Skip Embeddings ---
#     fused_model.qconfig = torch.quantization.get_default_qconfig(backend)
#     for name, module in fused_model.named_modules():
#         if isinstance(module, torch.nn.Embedding):
#             module.qconfig = None  # Static PTQ doesn't support BERT embeddings

#     torch.quantization.prepare(fused_model, inplace=True)

#     # --- 3. Calibration ---
#     with torch.no_grad():
#         for batch in calibration_loader:
#             fused_model(
#                 batch["image"].to("cpu"),
#                 batch["input_ids"].to("cpu"),
#                 batch["attention_mask"].to("cpu"),
#                 batch["stats"].to("cpu"),
#             )

#     # --- 4. Conversion ---
#     quantized_model = torch.quantization.convert(fused_model, inplace=False)
#     return quantized_model


def quantize_traffic_model(model):
    """
    Simplified Dynamic Quantization:
    - Works out of the box with BERT/Residuals.
    - No calibration loop required.
    """
    logging.info("Performing Dynamic Quantization on Linear layers...")
    model.eval()
    model.to("cpu")

    # Dynamic quantization only targets Linear and RNN layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize the heavy MLP/Linear layers
        dtype=torch.qint8,
    )

    # Tag this run in MLflow
    mlflow.set_tag("model_precision", "INT8-Dynamic")

    return quantized_model


if __name__ == "__main__":
    # --- 1. Path & Environment Setup ---
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    config = load_config()
    logger = logging.getLogger("TrafficCLIP")
    logger.setLevel(logging.INFO)

    import torch.serialization

    from models.opt_traffic_clip import OptimizedTrafficCLIP

    torch.serialization.add_safe_globals([OptimizedTrafficCLIP])

    parser = argparse.ArgumentParser(description="Quantize OptimizedTrafficCLIP Model")
    parser.add_argument("--lambda_cl", type=float, required=True)
    parser.add_argument("--use_stats_prompts", action="store_true")
    parser.add_argument("--model_version", type=str, default="optimized")
    parser.add_argument(
        "--use_stats", action="store_true", help="Toggle use of statistical features"
    )
    parser.add_argument(
        "--stats_input_dim", type=int, default=3, help="Number of statistical features"
    )

    args = parser.parse_args()

    dagshub.init(
        repo_owner=config["user"]["name"], repo_name=config["user"]["repo"], mlflow=True
    )

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        fh = logging.FileHandler("quant.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    mlflow.set_experiment("TrafficCLIP_Quantization")

    with mlflow.start_run(run_name="Quantization_Run"):
        try:
            traffic_cfg = config["dataset"]["traffic"]["classes"]
            num_classes = sum(len(c) for c in traffic_cfg.values())
            exp_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}_stats_data{args.use_stats}"

            # STEP 1: Initialize Clean Architecture
            model = OptimizedTrafficCLIP(
                num_classes=num_classes,
                use_stats=args.use_stats,
                stats_input_dim=args.stats_input_dim,
            ).to("cpu")

            # STEP 2: Download and Handle Flexible Loading
            model_uri = f"models:/{exp_tag}/latest"
            logging.info(f"Downloading artifact folder from {model_uri}")
            local_dir = mlflow.artifacts.download_artifacts(model_uri)
            state_dict_path = Path(local_dir) / "data" / "model.pth"

            # Load the file (could be a dict or a full object)
            checkpoint = torch.load(
                state_dict_path, map_location="cpu", weights_only=False
            )

            # FLEXIBLE EXTRACTION LOGIC
            if isinstance(checkpoint, torch.nn.Module):
                logging.info("Detected full model object. Extracting state_dict...")
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict):
                # Handle cases where state_dict is nested
                state_dict = checkpoint.get(
                    "model", checkpoint.get("state_dict", checkpoint)
                )
                logging.info("Detected dictionary-based checkpoint.")
            else:
                state_dict = checkpoint
                logging.info("Loaded raw state_dict.")

            model.load_state_dict(state_dict)
            logging.info("Weights successfully loaded into architecture.")

            # STEP 3: Perform Simplified Dynamic Quantization
            # Since we are using Dynamic, no calibration_dataloader is strictly needed
            logging.info("Starting Dynamic Quantization...")
            quant_model = quantize_traffic_model(model)

            # STEP 4: Save and Log
            save_path = "quantized_traffic_model.pt"
            torch.save(quant_model.state_dict(), save_path)
            mlflow.pytorch.log_model(quant_model, name="quantized_model")

            logging.info("Quantization complete and model artifact logged to DagsHub.")

        except Exception as e:
            logging.error(f"Quantization failed: {e}")
            import traceback

            logging.error(traceback.format_exc())
            mlflow.set_tag("status", "failed")
