"""
Evaluation Script — Loads a trained checkpoint and evaluates on test set.
Usage:
    python src/evaluate.py --checkpoint results/checkpoints/best_model.pth
"""

import argparse
import torch
import numpy as np
import yaml

from models.multimodal_model import MultimodalDiseaseDetector
from utils.metrics import MetricTracker, print_metrics
from data.preprocess import MultimodalMedicalDataset, collate_multimodal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--missing_modality", type=str, default=None,
                        choices=["imaging", "ecg", "ehr"],
                        help="Simulate missing modality for robustness testing")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, missing_modality=None):
    model.eval()
    tracker = MetricTracker()
    all_attn_weights = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        images = batch.get("image", None) if missing_modality != "imaging" else None
        ecg    = batch.get("ecg", None)   if missing_modality != "ecg"     else None
        ehr    = batch.get("ehr", None)   if missing_modality != "ehr"     else None
        labels = batch["label"]

        output = model(images=images, ecg=ecg, ehr=ehr)
        tracker.update(labels, output["probabilities"])
        all_attn_weights.append(output["attention_weights"].cpu().numpy())

    metrics = tracker.compute()
    attn_weights = np.concatenate(all_attn_weights, axis=0)
    return metrics, attn_weights


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = MultimodalDiseaseDetector(
        num_classes=config["model"]["num_classes"],
        latent_dim=config["model"]["latent_dim"],
        img_backbone=config["model"]["img_backbone"],
        ecg_input_dim=config["model"]["ecg_input_dim"],
        ehr_input_dim=config["model"]["ehr_input_dim"],
        pretrained_cnn=False
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', 'N/A')})")

    # Stub test data — replace with real test set loading
    n_test = 200
    test_dataset = MultimodalMedicalDataset(
        labels=np.random.randint(0, 2, n_test),
        ecg_data=np.random.randn(n_test, config["data"]["ecg_window_samples"], 1).astype(np.float32),
        ehr_data=np.random.randn(n_test, config["model"]["ehr_input_dim"]).astype(np.float32)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_multimodal
    )

    # Evaluate
    missing = args.missing_modality
    if missing:
        print(f"\nEvaluating with MISSING modality: {missing}")
    else:
        print("\nEvaluating with ALL modalities")

    metrics, attn_weights = evaluate(model, test_loader, device, missing_modality=missing)
    print_metrics(metrics, prefix=f"Test Metrics (missing={missing})")

    print(f"\nMean attention weights (imaging | ECG | EHR):")
    mean_w = attn_weights.mean(axis=0)
    print(f"  {mean_w[0]:.3f} | {mean_w[1]:.3f} | {mean_w[2]:.3f}")


if __name__ == "__main__":
    main()
