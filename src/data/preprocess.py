"""
Data Preprocessing Pipeline for All Three Modalities
Paper: "Optimizing Multimodal Deep Learning Architectures for Early Disease Detection"

Handles:
  - Imaging: normalization, augmentation, resize to 224x224
  - ECG/Time-series: windowing (60s), denoising, normalization
  - EHR: imputation, one-hot encoding, normalization
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, List, Dict, Tuple
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────
# 1. IMAGING PREPROCESSING
# ─────────────────────────────────────────────────────────────

def get_image_transforms(mode: str = "train") -> transforms.Compose:
    """
    Standard image augmentation pipeline for medical X-rays (Section 4.1).

    Train: random rotate ±15°, horizontal flip, contrast jitter, resize 224x224
    Val/Test: only resize and normalize
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])


# ─────────────────────────────────────────────────────────────
# 2. ECG / TIME-SERIES PREPROCESSING
# ─────────────────────────────────────────────────────────────

class ECGPreprocessor:
    """
    Preprocessing for ECG time-series data (Section 4.1).

    Steps:
      1. Bandpass filter (0.5 - 40 Hz) to remove noise artifacts
      2. Z-score normalization per signal
      3. Fixed-length windowing (default: 60 seconds at sampling_rate Hz)
    """

    def __init__(
        self,
        sampling_rate: int = 500,
        window_seconds: int = 60,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        filter_order: int = 4
    ):
        self.sampling_rate = sampling_rate
        self.window_length = window_seconds * sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyq = 0.5 * self.sampling_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = scipy_signal.butter(self.filter_order, [low, high], btype="band")
        return scipy_signal.filtfilt(b, a, signal, axis=0)

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalization per channel."""
        mean = signal.mean(axis=0, keepdims=True)
        std  = signal.std(axis=0, keepdims=True) + 1e-8
        return (signal - mean) / std

    def window(self, signal: np.ndarray, start: int = 0) -> np.ndarray:
        """Extract fixed-length window starting at `start`."""
        end = start + self.window_length
        if end > len(signal):
            # Pad if signal is shorter than window
            pad = np.zeros((end - len(signal), signal.shape[1] if signal.ndim > 1 else 1))
            signal = np.concatenate([signal, pad], axis=0)
        return signal[start:end]

    def preprocess(self, signal: np.ndarray, random_crop: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            signal: (T, C) numpy array — time steps × channels
            random_crop: If True, random window start; else use start=0

        Returns:
            (window_length, C) preprocessed segment
        """
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        filtered = self.bandpass_filter(signal)
        normalized = self.normalize(filtered)

        if random_crop and len(normalized) > self.window_length:
            max_start = len(normalized) - self.window_length
            start = np.random.randint(0, max_start)
        else:
            start = 0

        windowed = self.window(normalized, start)
        return windowed.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 3. EHR / STRUCTURED DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────

class EHRPreprocessor:
    """
    Preprocessing for structured EHR/tabular data (Section 4.1).

    Steps:
      1. Impute missing continuous values (mean/median)
      2. One-hot encode categorical features
      3. Z-score normalize continuous features
    """

    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        impute_strategy: str = "median"
    ):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols

        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in categorical_cols}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit preprocessors on training data."""
        cont_data = df[self.continuous_cols].values
        self.imputer.fit(cont_data)
        cont_imputed = self.imputer.transform(cont_data)
        self.scaler.fit(cont_imputed)

        for col in self.categorical_cols:
            self.label_encoders[col].fit(df[col].astype(str).fillna("missing"))

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dataframe to feature matrix.

        Returns:
            np.ndarray of shape (N, num_features)
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        # Continuous features: impute + scale
        cont_data = df[self.continuous_cols].values
        cont_imputed = self.imputer.transform(cont_data)
        cont_scaled = self.scaler.transform(cont_imputed)

        # Categorical features: label encode + one-hot
        cat_parts = []
        for col in self.categorical_cols:
            encoded = self.label_encoders[col].transform(
                df[col].astype(str).fillna("missing")
            )
            n_classes = len(self.label_encoders[col].classes_)
            one_hot = np.eye(n_classes)[encoded]
            cat_parts.append(one_hot)

        if cat_parts:
            cat_data = np.concatenate(cat_parts, axis=1)
            return np.concatenate([cont_scaled, cat_data], axis=1).astype(np.float32)
        else:
            return cont_scaled.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def output_dim(self) -> int:
        """Total number of output features."""
        cont_dim = len(self.continuous_cols)
        cat_dim = sum(
            len(enc.classes_) for enc in self.label_encoders.values()
        )
        return cont_dim + cat_dim


# ─────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────

class MultimodalMedicalDataset(Dataset):
    """
    PyTorch Dataset that yields (image, ecg, ehr, label) triplets.
    Any modality can be None if not available for a sample.

    Args:
        image_paths (List[str]): Paths to X-ray images
        ecg_data (np.ndarray or None): (N, T, C) ECG array
        ehr_data (np.ndarray or None): (N, D) EHR feature matrix
        labels (np.ndarray): (N,) binary/multi-class labels
        image_transform: torchvision transform pipeline
    """

    def __init__(
        self,
        labels: np.ndarray,
        image_paths: Optional[List[str]] = None,
        ecg_data: Optional[np.ndarray] = None,
        ehr_data: Optional[np.ndarray] = None,
        image_transform=None
    ):
        self.labels = labels
        self.image_paths = image_paths
        self.ecg_data = ecg_data
        self.ehr_data = ehr_data
        self.image_transform = image_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Dict:
        sample = {"label": torch.tensor(self.labels[idx], dtype=torch.float32)}

        # Image
        if self.image_paths is not None:
            from PIL import Image
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.image_transform:
                img = self.image_transform(img)
            sample["image"] = img

        # ECG
        if self.ecg_data is not None:
            sample["ecg"] = torch.tensor(self.ecg_data[idx], dtype=torch.float32)

        # EHR
        if self.ehr_data is not None:
            sample["ehr"] = torch.tensor(self.ehr_data[idx], dtype=torch.float32)

        return sample


def collate_multimodal(batch: List[Dict]) -> Dict:
    """Custom collate_fn to handle optional modalities."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key in ("image", "ecg", "ehr", "label"):
            tensors = [item[key] for item in batch if key in item]
            if tensors:
                collated[key] = torch.stack(tensors)
    return collated


if __name__ == "__main__":
    # Quick smoke test of ECG preprocessor
    ecg_proc = ECGPreprocessor(sampling_rate=500, window_seconds=10)
    fake_ecg = np.random.randn(6000, 1)  # 12 seconds
    processed = ecg_proc.preprocess(fake_ecg)
    print(f"ECG preprocessed shape: {processed.shape}")  # (5000, 1)

    # EHR preprocessor test
    import pandas as pd
    df = pd.DataFrame({
        "age": [45, 67, np.nan, 55],
        "crp": [12.0, np.nan, 5.0, 8.0],
        "sex": ["M", "F", "M", np.nan]
    })
    ehr_proc = EHRPreprocessor(
        continuous_cols=["age", "crp"],
        categorical_cols=["sex"]
    )
    features = ehr_proc.fit_transform(df)
    print(f"EHR preprocessed shape: {features.shape}")
    print(f"EHR output dim: {ehr_proc.output_dim}")
