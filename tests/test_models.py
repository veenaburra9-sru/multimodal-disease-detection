"""
Unit Tests for Multimodal Disease Detection Models
Run with: python -m pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from models.cnn_encoder import CNNEncoder
from models.lstm_encoder import LSTMEncoder
from models.mlp_encoder import MLPEncoder, EHREmbeddingEncoder
from models.cross_modal_attention import CrossModalAttention, ModalityDropout
from models.multimodal_model import MultimodalDiseaseDetector
from utils.losses import MultimodalLoss
from utils.metrics import compute_metrics, MetricTracker


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def latent_dim():
    return 256

@pytest.fixture
def dummy_images(batch_size):
    return torch.randn(batch_size, 3, 224, 224)

@pytest.fixture
def dummy_ecg(batch_size):
    return torch.randn(batch_size, 1500, 1)

@pytest.fixture
def dummy_ehr(batch_size):
    return torch.randn(batch_size, 64)

@pytest.fixture
def dummy_labels(batch_size):
    return torch.randint(0, 2, (batch_size,)).float()

@pytest.fixture
def full_model():
    return MultimodalDiseaseDetector(
        num_classes=1,
        latent_dim=256,
        img_backbone="resnet18",   # Faster for testing
        ecg_input_dim=1,
        ehr_input_dim=64,
        modality_dropout_prob=0.3,
        pretrained_cnn=False
    )


# ── CNN Encoder Tests ─────────────────────────────────────────

class TestCNNEncoder:

    @pytest.mark.parametrize("backbone", ["resnet18", "resnet50", "efficientnet_b0"])
    def test_output_shape(self, backbone, batch_size, latent_dim, dummy_images):
        encoder = CNNEncoder(backbone=backbone, latent_dim=latent_dim, pretrained=False)
        out = encoder(dummy_images)
        assert out.shape == (batch_size, latent_dim), f"Expected ({batch_size}, {latent_dim}), got {out.shape}"

    def test_invalid_backbone(self):
        with pytest.raises(ValueError):
            CNNEncoder(backbone="vgg16", pretrained=False)

    def test_frozen_backbone(self, dummy_images, latent_dim):
        encoder = CNNEncoder(backbone="resnet18", latent_dim=latent_dim, pretrained=False, freeze_backbone=True)
        for name, param in encoder.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

    def test_gradient_flow(self, dummy_images, latent_dim):
        encoder = CNNEncoder(backbone="resnet18", latent_dim=latent_dim, pretrained=False)
        out = encoder(dummy_images)
        loss = out.sum()
        loss.backward()
        # Projection layers should have gradients
        for p in encoder.projection.parameters():
            assert p.grad is not None


# ── LSTM Encoder Tests ────────────────────────────────────────

class TestLSTMEncoder:

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_output_shape(self, rnn_type, bidirectional, batch_size, latent_dim, dummy_ecg):
        encoder = LSTMEncoder(
            input_dim=1, hidden_dim=64, num_layers=2,
            latent_dim=latent_dim, rnn_type=rnn_type, bidirectional=bidirectional
        )
        out = encoder(dummy_ecg)
        assert out.shape == (batch_size, latent_dim)

    def test_variable_length_input(self, batch_size, latent_dim):
        encoder = LSTMEncoder(input_dim=1, hidden_dim=64, latent_dim=latent_dim)
        # Variable-length ECG with padding
        lengths = torch.randint(500, 1500, (batch_size,))
        ecg = torch.randn(batch_size, 1500, 1)
        out = encoder(ecg, lengths=lengths)
        assert out.shape == (batch_size, latent_dim)

    def test_no_nan_in_output(self, dummy_ecg, latent_dim):
        encoder = LSTMEncoder(input_dim=1, hidden_dim=64, latent_dim=latent_dim)
        out = encoder(dummy_ecg)
        assert not torch.isnan(out).any(), "NaN detected in LSTM output"


# ── MLP Encoder Tests ─────────────────────────────────────────

class TestMLPEncoder:

    def test_output_shape(self, batch_size, latent_dim, dummy_ehr):
        encoder = MLPEncoder(input_dim=64, hidden_dims=[256, 256], latent_dim=latent_dim)
        out = encoder(dummy_ehr)
        assert out.shape == (batch_size, latent_dim)

    def test_embedding_encoder(self, batch_size, latent_dim):
        encoder = EHREmbeddingEncoder(
            continuous_dim=40,
            categorical_vocab_sizes=[100, 50],
            latent_dim=latent_dim
        )
        cont = torch.randn(batch_size, 40)
        cats = torch.randint(0, 50, (batch_size, 2))
        out = encoder(cont, cats)
        assert out.shape == (batch_size, latent_dim)

    def test_no_categorical(self, batch_size, latent_dim):
        encoder = EHREmbeddingEncoder(
            continuous_dim=64,
            categorical_vocab_sizes=None,
            latent_dim=latent_dim
        )
        cont = torch.randn(batch_size, 64)
        out = encoder(cont)
        assert out.shape == (batch_size, latent_dim)


# ── Cross-Modal Attention Tests ───────────────────────────────

class TestCrossModalAttention:

    def test_output_shape(self, batch_size, latent_dim):
        attn = CrossModalAttention(latent_dim=latent_dim, num_modalities=3, num_heads=4)
        feats = [torch.randn(batch_size, latent_dim) for _ in range(3)]
        result = attn(feats)
        assert result["fused"].shape == (batch_size, latent_dim)
        assert result["weights"].shape == (batch_size, 3)

    def test_weights_sum_to_one(self, batch_size, latent_dim):
        attn = CrossModalAttention(latent_dim=latent_dim, num_modalities=3, num_heads=4)
        feats = [torch.randn(batch_size, latent_dim) for _ in range(3)]
        result = attn(feats)
        weight_sums = result["weights"].sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-4), \
            f"Weights should sum to 1, got: {weight_sums}"

    def test_missing_modality_mask(self, batch_size, latent_dim):
        attn = CrossModalAttention(latent_dim=latent_dim, num_modalities=3, num_heads=4)
        feats = [torch.randn(batch_size, latent_dim) for _ in range(3)]
        # Mark imaging as absent for all samples
        mask = torch.zeros(batch_size, 3, dtype=torch.bool)
        mask[:, 0] = True
        result = attn(feats, modality_mask=mask)
        # Imaging weight should be 0
        assert (result["weights"][:, 0] < 1e-6).all(), "Masked modality should have ~0 weight"


class TestModalityDropout:

    def test_at_least_one_active_train(self, batch_size, latent_dim):
        dropper = ModalityDropout(num_modalities=3, dropout_prob=0.9, min_active=1)
        dropper.train()
        feats = [torch.randn(batch_size, latent_dim) for _ in range(3)]
        _, mask = dropper(feats)
        active = (~mask).sum(dim=1)
        assert (active >= 1).all(), "Each sample must have at least 1 active modality"

    def test_no_dropout_at_eval(self, batch_size, latent_dim):
        dropper = ModalityDropout(num_modalities=3, dropout_prob=1.0, min_active=1)
        dropper.eval()
        feats = [torch.randn(batch_size, latent_dim) for _ in range(3)]
        dropped, mask = dropper(feats)
        assert not mask.any(), "No dropout should happen at eval time"


# ── Full Model Tests ──────────────────────────────────────────

class TestMultimodalModel:

    def test_trimodal_forward(self, full_model, dummy_images, dummy_ecg, dummy_ehr, batch_size):
        full_model.eval()
        with torch.no_grad():
            out = full_model(images=dummy_images, ecg=dummy_ecg, ehr=dummy_ehr)
        assert out["logits"].shape == (batch_size, 1)
        assert out["probabilities"].shape == (batch_size, 1)
        assert out["attention_weights"].shape == (batch_size, 3)
        assert out["fused_features"].shape == (batch_size, 256)

    def test_missing_imaging(self, full_model, dummy_ecg, dummy_ehr, batch_size):
        full_model.eval()
        with torch.no_grad():
            out = full_model(ecg=dummy_ecg, ehr=dummy_ehr)
        assert out["probabilities"].shape == (batch_size, 1)
        assert not torch.isnan(out["probabilities"]).any()

    def test_missing_ecg(self, full_model, dummy_images, dummy_ehr, batch_size):
        full_model.eval()
        with torch.no_grad():
            out = full_model(images=dummy_images, ehr=dummy_ehr)
        assert out["probabilities"].shape == (batch_size, 1)

    def test_missing_ehr(self, full_model, dummy_images, dummy_ecg, batch_size):
        full_model.eval()
        with torch.no_grad():
            out = full_model(images=dummy_images, ecg=dummy_ecg)
        assert out["probabilities"].shape == (batch_size, 1)

    def test_probabilities_in_range(self, full_model, dummy_images, dummy_ecg, dummy_ehr):
        full_model.eval()
        with torch.no_grad():
            out = full_model(images=dummy_images, ecg=dummy_ecg, ehr=dummy_ehr)
        probs = out["probabilities"]
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"

    def test_backward_pass(self, full_model, dummy_images, dummy_ecg, dummy_ehr, dummy_labels):
        full_model.train()
        out = full_model(images=dummy_images, ecg=dummy_ecg, ehr=dummy_ehr)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["logits"].squeeze(), dummy_labels
        )
        loss.backward()
        # Check that some gradients exist
        grads = [p.grad for p in full_model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"

    def test_parameter_count(self, full_model):
        counts = full_model.count_parameters()
        assert counts["total"] > 0
        assert counts["cnn_encoder"] > 0
        assert counts["lstm_encoder"] > 0
        assert counts["mlp_encoder"] > 0


# ── Loss Function Tests ───────────────────────────────────────

class TestMultimodalLoss:

    def test_loss_is_scalar(self, dummy_labels, latent_dim, batch_size):
        loss_fn = MultimodalLoss(num_classes=1)
        logits = torch.randn(batch_size, 1)
        losses = loss_fn(logits, dummy_labels)
        assert losses["total"].shape == torch.Size([])

    def test_all_loss_components(self, dummy_labels, latent_dim, batch_size):
        loss_fn = MultimodalLoss(lambda_reg=1e-4, mu_align=0.1, num_classes=1)
        logits = torch.randn(batch_size, 1)
        feats = [torch.randn(batch_size, latent_dim) for _ in range(3)]
        losses = loss_fn(logits, dummy_labels, modality_features=feats)
        for key in ["total", "classification", "regularization", "alignment"]:
            assert key in losses
            assert not torch.isnan(losses[key]), f"NaN in {key} loss"

    def test_loss_decreases_with_correct_predictions(self):
        loss_fn = MultimodalLoss(num_classes=1)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # Good predictions (high confidence and correct)
        good_logits = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
        # Bad predictions
        bad_logits = torch.tensor([[-5.0], [5.0], [-5.0], [5.0]])
        good_loss = loss_fn(good_logits, labels)["classification"]
        bad_loss  = loss_fn(bad_logits, labels)["classification"]
        assert good_loss < bad_loss


# ── Metrics Tests ─────────────────────────────────────────────

class TestMetrics:

    def test_perfect_prediction(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.05, 0.95, 0.90, 0.05, 0.85])
        m = compute_metrics(y_true, y_prob)
        assert m["auc"] > 0.99
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_random_prediction(self):
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.rand(200)
        m = compute_metrics(y_true, y_prob)
        assert 0.3 < m["auc"] < 0.7  # Should be near 0.5

    def test_metric_tracker(self):
        tracker = MetricTracker()
        for _ in range(5):
            tracker.update(
                torch.randint(0, 2, (16,)),
                torch.rand(16)
            )
        m = tracker.compute()
        assert "auc" in m
        assert "accuracy" in m
        assert "f1" in m


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
