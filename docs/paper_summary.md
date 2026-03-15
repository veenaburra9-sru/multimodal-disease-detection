# Paper Summary

## Optimizing Multimodal Deep Learning Architectures for Early Disease Detection

**Authors**: Burra Veena, Dr. Suresh Kumar Mandala  
**Institution**: SR University, Warangal, Telangana  

---

## Problem Statement

Traditional diagnostic systems are **unimodal** (imaging alone, or EHR alone), missing complementary signals across data types. The challenge is to fuse heterogeneous medical data (images, structured records, time-series) into a unified model that is:
- More accurate than any single modality
- Robust to missing data (common in clinical settings)
- Interpretable (clinician-trusted)

---

## Key Equations

| Equation | Formula | Description |
|---|---|---|
| Eq. 1 | `y = f(X₁, X₂, X₃; Θ)` | Multimodal prediction |
| Eq. 2 | `F_img = f_CNN(X₁; θ₁)` | Image feature extraction |
| Eq. 3 | `F_time = f_RNN(X₂; θ₂)` | ECG feature extraction |
| Eq. 4 | `F_struct = f_MLP(X₃; θ₃)` | EHR feature extraction |
| Eq. 5 | `F_fusion = Σ αₘ · Fₘ` | Cross-modal fusion |
| Eq. 6 | `αₘ = softmax(scoreₘ)` | Attention weights |
| Eq. 7 | `F_fusion = Σ_{m∈M} αₘ · Fₘ` | Fusion with modality dropout |
| Eq. 8 | `y = σ(W_pred · F_fusion + b)` | Classification head |
| Eq. 9 | `L = L_cls + λL_reg + μL_align` | Combined loss |
| Eq. 11 | `θ* = argmin_θ L(θ)` | Optimization objective |

---

## Architecture

```
Input Modalities:
  X₁: Chest X-ray image     ──► ResNet-50 / EfficientNet (CNN)
  X₂: ECG time-series       ──► Bi-LSTM with attention pooling
  X₃: EHR structured data   ──► MLP with BatchNorm + Dropout

Projection: Each encoder → shared latent space (dim=256)

Fusion:
  - Modality Dropout (p=0.3, training only)
  - Cross-Modal Attention (4 heads)
  - Weighted sum: F_fusion = Σ αₘ · Fₘ

Head:
  - FC(256→128) → BN → ReLU → Dropout → FC(128→1) → Sigmoid
```

---

## Datasets

| Dataset | Modality | Samples | Positive Prevalence |
|---|---|---|---|
| ChestX-ray14 | X-ray | 112,120 | 22.1% |
| MIMIC-III | EHR | 58,976 | 19.3% |
| PhysioNet ECG | Time-series | 10,000 | 26.7% |

**Split**: 70% train / 20% test / 10% val — patient-level stratified

---

## Key Results

### Performance (Table 4)
| Configuration | AUC | F1 |
|---|---|---|
| Imaging only | 0.85 | 0.81 |
| EHR only | 0.83 | 0.78 |
| ECG only | 0.80 | 0.76 |
| Imaging + EHR | 0.92 | 0.87 |
| **Tri-modal (Ours)** | **0.94** | **0.89** |

### Fusion Strategy (Table 5)
| Fusion | AUC |
|---|---|
| Early | 0.88 |
| Late | 0.89 |
| **Intermediate (Ours)** | **0.94** |

### Ablation (Table 6)
- Removing EHR: **-5% AUC**
- Removing ECG: **-4% AUC**
- Removing Imaging: **-6% AUC**

### SHAP Top Features (Table 9)
| Feature | SHAP | Modality |
|---|---|---|
| Troponin levels | 0.48 | EHR |
| Lesion intensity | 0.44 | Imaging |
| HR variability | 0.39 | ECG |
| Age | 0.31 | EHR |
| CRP level | 0.27 | EHR |

---

## Clinical Impact

- **Early pneumonia**: Recall improved from 78% → **88%** vs imaging-only
- **Missing imaging**: AUC remains at **0.88** (vs 0.94 full) — still outperforms unimodal baselines
- **Cross-hospital**: AUC 0.86 on PadChest (trained on ChestX-ray14), 0.85 on eICU (trained on MIMIC-III)

---

## Limitations & Future Work

- Variation in performance across disease subtypes and demographics
- Computational cost for edge deployment
- Federated learning not empirically evaluated (planned future work)
- Future: self-supervised pretraining, genomics modality, wearables integration
