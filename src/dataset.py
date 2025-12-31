import logging
import pathlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class TrafficDataset(Dataset):
    def __init__(
        self,
        npz_path,
        prompts,
        tokenizer_name="google/bert-base-uncased",
        max_length=64,
    ):
        data = np.load(npz_path, allow_pickle=True)

        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Image Tensors: (N, 1, 28, 28)
        self.images = torch.from_numpy(data["x"]).float()
        self.labels = torch.from_numpy(data["y"]).long()
        self.class_names = data["labels"].tolist()

        # Pre-calculate prompts
        self.raw_prompts = [prompts[name] for name in self.class_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        text_description = self.raw_prompts[label]

        # Tokenize prompt
        tokens = self.tokenizer(
            text_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,  # Input for Vision Encoder
            "input_ids": tokens["input_ids"].squeeze(0),  # Input for BERT Encoder
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": label,
        }


def get_dataloader(npz_path, tokenizer, prompts, batch_size=64, max_length=64, seed=42):
    """
    Creates Train, Validation, and Test loaders for the TrafficCLIP pipeline.
    Splits: 70% Train, 15% Val, 15% Test
    """
    full_dataset = TrafficDataset(
        npz_path,
        prompts=prompts,
        tokenizer_name=tokenizer,
        max_length=max_length,
    )
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    #  Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logging.info(f"Total Samples: {total_size}")
    logging.info(f"Training:      {len(train_set)} samples")
    logging.info(f"Validation:    {len(val_set)} samples")
    logging.info(f"Testing:       {len(test_set)} samples")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = load_config()
    try:
        # Parameter Extraction
        SEMANTIC_PROMPTS = config["prompts"]
        NPZ_PATH = config["paths"]["output_data_file"]
        TENSOR_DIR = Path(config["paths"]["tensors_dir"])
        TOKENIZER_NAME = config["preprocess"]["tokenizer"]
        MAX_LENGTH = config["preprocess"]["max_length"]
        SEED = 42
        BATCH_SIZE = config["preprocess"]["batch_size"]
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    try:
        train_loader, val_loader, test_loader = get_dataloader(
            npz_path=NPZ_PATH,
            tokenizer=TOKENIZER_NAME,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            seed=SEED,
            prompts=SEMANTIC_PROMPTS,
        )

        logging.info("DataLoaders created successfully.")
    except Exception as e:
        logging.error(f"Error creating DataLoaders: {e}")
        raise
