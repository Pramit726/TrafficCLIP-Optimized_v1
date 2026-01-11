import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.traffic_clip import TrafficCLIP
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class NonLinearFusionHead(nn.Module):
    """
    Mod 2: Replaces static Cosine Similarity with a 2-layer MLP.
    Captures non-linear correlations between visual byte patterns
    and textual behavioral anchors.
    """

    def __init__(self, num_classes, input_dim=2048, hidden_dim=512):
        """
        Phase 3 Architecture:
        - LayerNorm for multimodal stability
        - Dropout (0.4) for regularization against session-level overfitting
        """
        super().__init__()
        self.mlp = nn.Sequential(
            # Layer 1: Expansion and Non-linearity
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            # Layer 2: Classification Logits
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, combined_features):
        return self.mlp(combined_features)


class OptimizedTrafficCLIP(TrafficCLIP):
    def __init__(self, num_classes, vision_dim=1024, text_dim=1024):
        super().__init__(num_classes)

        # Override the similarity logic with the new Fusion Head
        self.fusion_head = NonLinearFusionHead(
            input_dim=vision_dim + text_dim, num_classes=num_classes
        )

    def forward(self, images, input_ids, attention_mask):
        """
        Modified forward pass:
        1. Extract vision features (Detail + Semantics + Adapter)
        2. Extract text features (BERT Behavioral Anchors)
        3. Concatenate and pass through MLP Fusion Head
        """

        # Vision Modality Representation Learning
        v_e = self.get_vision_features(images)  # [Batch, 1024]

        # Textual Modality Representation Learning
        t_e = self.get_text_features(input_ids, attention_mask)  # [Batch, 1024]

        # Mod 2 Fusion
        # Concatenate features into a single 2048-dim vector
        combined = torch.cat((v_e, t_e), dim=1)  # [Batch, 2048]
        logits = self.fusion_head(combined)

        # Prevents scaling the logits by more than 100 (exp(4.6052) ≈ 100)
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        return logits, self.logit_scale.exp()


if __name__ == "__main__":
    try:
        config = load_config()
        traffic_cfg = config["dataset"]["traffic"]["classes"]

        num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
        model = OptimizedTrafficCLIP(num_classes=num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(
            f"Total Parameters in TrafficCLIP Optimized: {total_params/1e6:.2f} Million"
        )
        logging.info(
            f"Trainable Parameters in TrafficCLIP Optimized: {trainable_params/1e6:.2f} Million"
        )
        logging.info(f"Trainable Ratio: {trainable_params/total_params:.2f}")

    except Exception as e:
        logging.error(f"Error initializing OptimizedTrafficCLIP model: {e}")
