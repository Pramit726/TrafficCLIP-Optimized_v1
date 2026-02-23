import logging
from xml.parsers.expat import model

import torch

from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.optimization.calibration import create_calibration_dataloader
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def quantize_traffic_model(model, calibration_loader, backend="fbgemm"):
    """
    Performs Post-Training Static Quantization (PTQ) on the OptimizedTrafficCLIP model.

    Args:
        model: The trained OptimizedTrafficCLIP model (float32).
        calibration_loader: Dataloader with 200-300 balanced validation samples.
        backend: 'fbgemm' for x86 CPUs (Intel/AMD) or 'qnnpack' for ARM.
    """
    model.eval()

    # 1. Module Fusion: Merges Linear + ReLU to reduce memory round-trips
    logging("Fusing modules")
    modules_to_fuse = [
        ["stats_proj.net.0", "stats_proj.net.2"],  # Linear + ReLU (Stats Head)
        ["fusion_head.mlp.0", "fusion_head.mlp.2"],  # Linear + ReLU (Fusion Head)
    ]

    # We use a try-except because if layers were already fused, this might error.
    try:
        fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
    except Exception as e:
        logging(f"Fusion skipped or failed: {e}")
        fused_model = model

    # 2. Assign Quantization Configuration
    fused_model.qconfig = torch.quantization.get_default_qconfig(backend)

    # 3.Inserts 'observers' to track the min/max ranges of your data
    torch.quantization.prepare(fused_model, inplace=True)

    # 4. Calibration
    logging(f"Calibrating on {len(calibration_loader.dataset)} samples")
    with torch.no_grad():
        for images, stats, input_ids, attn_mask, _ in calibration_loader:
            fused_model(images, input_ids, attn_mask, stats)

    # 5. Convert: Actually transforms the weights from Float32 to Int8
    logging("Converting model to INT8")
    quantized_model = torch.quantization.convert(fused_model, inplace=False)

    return quantized_model


if __name__ == "__main__":
    try:
        config = load_config()
        traffic_cfg = config["dataset"]["traffic"]["classes"]

        num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
        model = OptimizedTrafficCLIP(
            num_classes=num_classes, stats_input_dim=3, use_stats=True
        )  # IAT, Jitter, Entropy
        model.load_state_dict(
            torch.load("best_trafficclip_model.pt", map_location="cpu")
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error initializing OptimizedTrafficCLIP model: {e}")
        raise

    calibration_dataloader = create_calibration_dataloader(num_samples=200)
    quantized_traffic_model = quantize_traffic_model(model, calibration_dataloader)
