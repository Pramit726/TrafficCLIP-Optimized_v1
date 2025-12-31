import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class TrafficAdapter(nn.Module):
    """
    Modifies the pre-trained model by appending a few trainable layers
    including a down-projection, non-linear activation, and up-projection[cite: 286, 288].
    """

    def __init__(self, embed_dim, bottleneck_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

    def forward(self, x):
        return self.adapter(x)


class TrafficCLIP(nn.Module):
    """
    Original TrafficCLIP Model with Detail and Semantics-Aware Visual Encoder,
    Traffic Visual Adapter, and BERT Text Encoder.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha  # Residual rate for adapter

        # Detail-Aware Visual Encoder
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512),  # Feature vector f1
        )

        # Semantics-Aware Visual Encoder (Pre-trained Resnet-50 Backbone)
        logging.info("Loading pre-trained ResNet-50 for Semantics-Aware Encoder")
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify first layer to accept grayscale images
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the classification head
        self.semantics_backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze the Semantics-Aware Backbone
        for param in self.semantics_backbone.parameters():
            param.requires_grad = False
        self.semantics_proj = nn.Linear(2048, 512)  # Feature vector f2

        # Traffic Visual Adapter
        self.visual_adapter = TrafficAdapter(embed_dim=512)

        logging.info("Loading pre-trained BERT for Text Encoder")
        # Text Encoder (Pre-trained BERT Backbone)
        self.text_encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased")

        # The BERT encoder is frozen
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_proj = nn.Linear(768, 1024)

        # Temperature parameter for similarity scaling in contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_vision_features(self, images):
        # f1: Detail-aware features
        f1 = self.detail_encoder(images)

        # f2: Semantics-aware features
        with torch.no_grad():
            f2_raw = self.semantics_backbone(images).squeeze()
        f2 = self.semantics_proj(f2_raw)

        # Apply Traffic Adapter with residual connection
        # f2* = a * Adapter(f2) + (1-a) * f2
        f2_star = self.alpha * self.visual_adapter(f2) + (1 - self.alpha) * f2

        # Multi-level Vision Fusion
        v_f = torch.cat([f1, f2_star], dim=-1)  # Shape: [Batch, 1024]
        return F.normalize(v_f, p=2, dim=-1)  # L2 Normalization

    def get_text_features(self, input_ids, attention_mask):
        # Text-modality Representation Learning
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_f = self.text_proj(outputs.pooler_output)  # Use [CLS] pooler output
        return F.normalize(t_f, p=2, dim=-1)  # L2 Normalization

    def forward(self, images, input_ids, attention_mask):
        # Extract normalized features
        v_e = self.get_vision_features(images)
        t_e = self.get_text_features(input_ids, attention_mask)

        # Cross-modality Representation Fusion via Cosine Similarity
        # logits = np.dot(V_e, T_e.T) * np.exp(t)
        t = self.logit_scale.exp()
        logits = t * torch.matmul(v_e, t_e.t())

        return logits


if __name__ == "__main__":
    try:
        model = TrafficCLIP()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total Parameters in TrafficCLIP: {total_params/1e6:.2f} Million")
        logging.info(
            f"Trainable Parameters in TrafficCLIP: {trainable_params/1e6:.2f} Million"
        )
        logging.info(f"Trainable Ratio: {trainable_params/total_params:.2f}")
    except Exception as e:
        logging.error(f"Error initializing TrafficCLIP model: {e}")
