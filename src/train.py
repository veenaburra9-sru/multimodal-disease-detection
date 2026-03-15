"""
Main Training Script
Usage:
    python src/train.py --config configs/train_config.yaml
    python src/train.py --modalities imaging ehr ecg --epochs 100
"""

import argparse
import yaml
import torch
import numpy as np
import random
import os

from models.multimodal_model import MultimodalDiseaseDetector
from utils.losses import MultimodalLoss
from utils.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Disease Detection Model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--modalities", nargs="+", default=["imaging", "ehr", "ecg"],
                        choices=["imaging", "ehr", "ecg"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--modality_dropout", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict):
    """
    Build train/val dataloaders.
    Replace this stub with actual data loading for ChestX-ray14, MIMIC-III, PhysioNet ECG.
    """
    from data.preprocess import MultimodalMedicalDataset, collate_multimodal, get_image_transforms

    # ── Stub: replace with real dataset loading ──
    # Example for real usage:
    #   image_paths_train = load_chestxray14_paths(split='train')
    #   ecg_train = load_physionet_ecg(split='train')
    #   ehr_train = load_mimic3_features(split='train')
    #   labels_train = load_labels(split='train')
    n_train, n_val = 800, 200
    ehr_dim = config["model"]["ehr_input_dim"]
    ecg_T   = config["data"]["ecg_window_samples"]

    train_dataset = MultimodalMedicalDataset(
        labels=np.random.randint(0, 2, n_train),
        ecg_data=np.random.randn(n_train, ecg_T, 1).astype(np.float32),
        ehr_data=np.random.randn(n_train, ehr_dim).astype(np.float32),
    )
    val_dataset = MultimodalMedicalDataset(
        labels=np.random.randint(0, 2, n_val),
        ecg_data=np.random.randn(n_val, ecg_T, 1).astype(np.float32),
        ehr_data=np.random.randn(n_val, ehr_dim).astype(np.float32),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=collate_multimodal
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=collate_multimodal
    )

    return train_loader, val_loader


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.modality_dropout:
        config["model"]["modality_dropout_prob"] = args.modality_dropout

    print("=" * 60)
    print("  Multimodal Disease Detection — Training")
    print("=" * 60)
    print(f"  Modalities: {args.modalities}")
    print(f"  Epochs:     {config['training']['epochs']}")
    print(f"  LR:         {config['training']['lr']}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # Build model
    model = MultimodalDiseaseDetector(
        num_classes=config["model"]["num_classes"],
        latent_dim=config["model"]["latent_dim"],
        img_backbone=config["model"]["img_backbone"],
        ecg_input_dim=config["model"]["ecg_input_dim"],
        ehr_input_dim=config["model"]["ehr_input_dim"],
        modality_dropout_prob=config["model"]["modality_dropout_prob"],
        pretrained_cnn=config["model"].get("pretrained_cnn", True)
    )

    # Resume from checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from checkpoint: {args.checkpoint}")

    # Loss function
    loss_fn = MultimodalLoss(
        lambda_reg=config["training"]["lambda_reg"],
        mu_align=config["training"]["mu_align"],
        num_classes=config["model"]["num_classes"]
    )

    # Data
    train_loader, val_loader = build_dataloaders(config)

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["patience"],
        save_dir=config["training"]["save_dir"],
        device=args.device
    )

    # Train
    history = trainer.fit(num_epochs=config["training"]["epochs"])

    print("\nTraining complete. Best model saved to:", config["training"]["save_dir"])


if __name__ == "__main__":
    main()
