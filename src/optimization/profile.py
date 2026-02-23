import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# def profile_model(model, num_samples=100, warm_up=10):
#     """
#     Measures Inference Latency and Throughput on CPU.
#     """
#     # Ensure the model is in evaluation mode and on CPU
#     model.eval()
#     model.to("cpu")

#     # Create dummy inputs for Tri-modal shapes
#     # (Batch, Channel, H, W) -> (1, 1, 28, 28) for images
#     dummy_img = torch.randn(1, 1, 28, 28)
#     # (Batch, Stats) -> (1, 3) for [Mean IAT, Jitter, Entropy]
#     dummy_stats = torch.randn(1, 3)
#     # For text, we simulate a single encoded prompt vector
#     dummy_text = torch.randn(1, 1024)

#     logging.info("Starting Profiling")

#     # 1. Warm-up Phase: "Wakes up" the CPU and fills the cache
#     with torch.no_grad():
#         for _ in range(warm_up):
#             _ = model(dummy_img, dummy_stats, dummy_text)

#     # 2. Measurement Phase
#     latencies = []
#     with torch.no_grad():
#         for i in range(num_samples):
#             start_time = time.perf_counter()
#             _ = model(dummy_img, dummy_stats, dummy_text)
#             end_time = time.perf_counter()

#             # Convert to milliseconds
#             latencies.append((end_time - start_time) * 1000)

#     # 3. Calculate Metrics
#     avg_latency = np.mean(latencies)
#     p99_latency = np.percentile(latencies, 99)  # Tail latency
#     throughput = 1000 / avg_latency  # Flows per second

#     logging.info(f"Average Latency: {avg_latency:.2f} ms")
#     logging.info(f"P99 (Tail) Latency: {p99_latency:.2f} ms")
#     logging.info(f"Throughput: {throughput:.2f} flows/sec")
#     logging.info(f"---------------------------------")

#     return avg_latency, throughput


# if __name__ == "__main__":
#     from src.models.opt_traffic_clip import OptimizedTrafficCLIP

#     config = load_config()
#     traffic_cfg = config["dataset"]["traffic"]["classes"]

#     num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
#     model = OptimizedTrafficCLIP(num_classes=num_classes, use_stats=False)
#     model.load_state_dict(torch.load("best_trafficclip_model.pt", map_location="cpu"))

#     avg_latency, throughput = profile_model(model)


def evaluate_quantized_system(
    original_model, quantized_model, test_loader, original_path, quantized_path
):
    """
    Enhanced evaluation that merges accuracy testing with professional
    profiling (warm-up, P99, and throughput).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def run_comprehensive_bench(model, loader, model_name):
        model.eval()
        model.to("cpu")

        all_preds = []
        all_labels = []
        latencies = []

        # 1. Warm-up Phase (Critical for realistic CPU metrics)
        # We process a small batch to wake up the CPU and fill the L3 cache
        warm_up_batches = 5
        with torch.no_grad():
            for i, (img, stats, ids, mask, _) in enumerate(loader):
                if i >= warm_up_batches:
                    break
                _ = model(img, ids, mask, stats)

        # 2. Combined Accuracy & Latency Measurement
        logging.info(f"Benchmarking {model_name}...")
        with torch.no_grad():
            for img, stats, ids, mask, labels in loader:
                # We measure latency per batch to get a distribution for P99
                start = time.perf_counter()
                logits, _ = model(img, ids, mask, stats)
                end = time.perf_counter()

                # Per-sample latency in this batch
                batch_latency = ((end - start) * 1000) / img.size(0)
                latencies.append(batch_latency)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        # 3. Calculate Metrics
        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)
        throughput = 1000 / avg_lat  # Flows per second per core

        return f1, avg_lat, p99_lat, throughput

    # Run benchmarks
    orig_f1, orig_lat, orig_p99, orig_tp = run_comprehensive_bench(
        original_model, test_loader, "Float32 Model"
    )
    quant_f1, quant_lat, quant_p99, quant_tp = run_comprehensive_bench(
        quantized_model, test_loader, "Int8 Model"
    )

    # 4. Model Size Calculation
    orig_size = os.path.getsize(original_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)

    # --- FINAL APPLIED SCIENCE REPORT ---
    print("\n" + "=" * 50)
    print("      TRAFFIC-CLIP PRODUCTION REPORT")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Original (FP32)':<12} | {'Quantized (INT8)':<12}")
    print(f"{'-'*20}-|-{'-'*12}-|-{'-'*12}")
    print(f"{'Model Size (MB)':<20} | {orig_size:<12.2f} | {quant_size:<12.2f}")
    print(f"{'Macro F1 Score':<20} | {orig_f1:<12.4f} | {quant_f1:<12.4f}")
    print(f"{'Avg Latency (ms)':<20} | {orig_lat:<12.2f} | {quant_lat:<12.2f}")
    print(f"{'P99 Latency (ms)':<20} | {orig_p99:<12.2f} | {quant_p99:<12.2f}")
    print(f"{'Throughput (fps)':<20} | {orig_tp:<12.1f} | {quant_tp:<12.1f}")
    print(f"{'-'*50}")
    print(
        f"RESULTS: {orig_lat/quant_lat:.1f}x Speedup | {orig_size/quant_size:.1f}x Compression"
    )
    print("=" * 50)

    return quant_f1, quant_lat
