# Model Training — Subham Pal
### Week 4 Final — EfficientNet-B3 | IEEE EMBS Pune Chapter Student Internship 2026

---

## 📁 Folder Contents

| File | Description |
|---|---|
| `Subham_Training_FINAL.ipynb` | Week 4 training notebook — EfficientNet-B3 with full pipeline |
| `our_results.json` | Final evaluation metrics (7-class + binary) |

---

## 🧠 Model — EfficientNet-B3

- **Backbone:** EfficientNet-B3 (pretrained on ImageNet1K)
- **Classifier Head:** Dropout(0.2) → Linear(1536, 7)
- **Weight Initialisation:** Orthogonal seeding (gain=1.2)
- **Input Size:** 224×224 px
- **Classes:** 7 (akiec, bcc, bkl, df, mel, nv, vasc)

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 7×10⁻⁴ |
| Weight Decay | 1×10⁻⁵ |
| Batch Size | 32 |
| Scheduler | CosineAnnealingLR |
| Precision | Mixed (AMP Autocast) |
| Augmentations | RandomHorizontalFlip, RandomVerticalFlip |
| Class Weighting | Logarithmic smoothing — log(1+w) |

---

## 📊 Results

### 7-Class Classification

| Split | Accuracy | Weighted F1 |
|---|---|---|
| Train | 99.24% | 0.9919 |
| Validation | 99.21% | 0.9913 |
| **Test** | **99.21%** | **0.9911** |

### Binary Malignant / Benign

| Metric | Score |
|---|---|
| Binary Accuracy | 99.21% |
| Weighted F1 | 0.9911 |
| AUC (ROC) | 0.9997 |

---

## 🔍 Explainability

We implement **Grad-CAM** across all 7 classes to visualise which skin regions the model attends to during classification. This provides clinical interpretability beyond raw predictions.

---

## 📌 Evaluation Notes

- Split method: `GroupShuffleSplit` on `lesion_id` — **leak-free**, prevents same-lesion photos spanning train and test
- This is a stricter protocol than most published HAM10000 papers which use random image-level splits
- Our 99.21% test accuracy outperforms several published works including Lee et al. (87.91%), PLOS ONE 2023 (94.96%), and DSCC_Net (94.17%)

---

## 📦 Model Weights

Model weights (`skin_lesion_deploy_model.pth`) are not stored in this repository due to file size limits.

> **Download:** _[https://drive.google.com/file/d/1QbflE7bjeQUmTzSZSoHmCH_jljjtuNaH/view?usp=drive_link]_

---

*IEEE EMBS Pune Chapter Student Internship 2026 | Team CodeCrafters | Team ID: AI ML-21*
