import logging
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(path="config/config.yaml"):
    """
    Load YAML config file from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def unzip_and_extract(zip_file_path_str: str, extraction_dir_str: str) -> bool:
    """
    Unzips a file to a specified directory.
    """
    zip_path = Path(zip_file_path_str)
    extraction_dir = Path(extraction_dir_str)

    # Check if the source zip file exists
    if not zip_path.is_file():
        logging.error(f"Source file not found: {zip_path}")
        return False

    # Ensure the extraction directory exists
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # Extract the contents
    try:
        logging.info(f"Extracting '{zip_path.name}' to '{extraction_dir}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)
        logging.info("Extraction successful.")
        return True
    except zipfile.BadZipFile:
        logging.error(f"Error: '{zip_path}' is not a valid zip file.")
        return False
    except Exception:
        logging.exception(f"An unexpected error occurred during extraction.")
        return False


def export_data(dataframe: pd.DataFrame, export_path: str, name: str) -> None:
    """
    Exports a DataFrame to a CSV file.
    """
    file_path = Path(export_path) / f"{name}.csv"

    try:
        dataframe.to_csv(file_path, index=False)
        print(f"Data exported to: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to export data to {file_path}") from e


def get_metrics(true, pred):
    """Calculates common regression metrics between true and predicted values.
    Parameters:
        true (np.ndarray): Array of true target values.
        pred (np.ndarray): Array of predicted values.
    Returns:
        tuple: A tuple containing the following metrics:
            - mae (float): Mean Absolute Error.
            - mse (float): Mean Squared Error.
            - rmse (float): Root Mean Squared Error.
            - mape (float): Mean Absolute Percentage Error (computed only where true > 0).
    Notes:
        - MAPE is calculated only for elements where the true value is greater than zero to avoid division by zero.
        - All metrics are computed using numpy operations.
    """

    # Mean Absolute Error
    mae = np.mean(np.abs(true - pred))

    # Mean Squared Error
    mse = np.mean((true - pred) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (Handling Zeros)
    # np.where to avoid division by zero
    # Calculates MAPE only for cases where true demand > 0
    mask = true > 0
    if np.any(mask):
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    else:
        mape = 0.0

    return mae, mse, rmse, mape


def plot_convergence(history, model_type):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title(f"{model_type} Loss Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss (CE + CL)")
    plt.legend()
    plt.grid(True)

    # F1 Score Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], "g-", label="Val Macro F1")
    plt.title(f"{model_type} F1 Score Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # save figure
    figure_path = Path(__file__).parent.parent.parent / "results" / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path / f"{model_type}_convergence.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_type):
    target_labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=target_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",  # Changed from 'd' to '.1f' to handle float values
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        f"Confusion Matrix: Performance on Similar Traffic Classes - {model_type}"
    )
    # save figure
    figure_path = Path(__file__).parent.parent.parent / "results" / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path / f"{model_type}_confusion_matrix.png")
    plt.show()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    load_config()
