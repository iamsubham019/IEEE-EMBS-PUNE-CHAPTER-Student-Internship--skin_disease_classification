# 🔬 Skin Disease Classification using Deep Learning
### IEEE EMBS Pune Chapter — Student Internship 2026
**Team CodeCrafters | Team ID: AI ML-21**

> Multi-class dermoscopic skin lesion classification across 7 diagnostic categories using an ensemble of pretrained CNN backbones, trained on HAM10000 augmented with ISIC-2019 rare-class images.

---

## 👥 Team

| Member | Role |
|---|---|
| **Subham Pal** | Model Training & Research |
| **Swarnali Ghosh** | Data Preparation & Preprocessing |
| **Sounok Ghosh** | Software Demo |

**Mentor:** Dr. Smita Chaudhuri | IEEE EMBS Pune Chapter

---

## 📋 Project Overview

Skin cancer is one of the most prevalent and life-threatening cancers globally. Early and accurate detection significantly improves patient outcomes. We developed a deep learning pipeline that classifies dermoscopic skin lesion images into 7 diagnostic classes using transfer learning, ensemble modeling, and clinical-grade preprocessing — with a fully functional web demo.

**Problem Statement:** Automated Multi-Class Skin Disease Classification Using Transfer Learning on Dermoscopic Images.

---

## 📁 Repository Structure

```
├── README.md
├── .gitignore
├── LICENSE
│
├── data/
│   ├── Swarnali_Data_Preprocessing.ipynb   ← Data prep & preprocessing notebook
│   ├── train.csv                            ← Training split (8,497 images)
│   ├── val.csv                              ← Validation split (1,838 images)
│   ├── test.csv                             ← Test split (1,852 images)
│   ├── class_weights.npy                    ← Computed log-smoothed class weights
│   ├── label_map.json                       ← Class name → index mapping
│   ├── preprocessing_steps.png             ← Preprocessing pipeline visualization
│   ├── class_distribution.png              ← Class imbalance chart
│   └── sample_images.png                   ← Sample images per class
│
├── models/
│   ├── Subham_Training_FINAL.ipynb         ← Week 4 training notebook (EfficientNet-B3)
│   └── our_results.json                    ← Final evaluation metrics
│
└── deployment/
    └── app.py                              ← SkinScan AI — Streamlit web app
```

---

## 🗂️ Dataset

### HAM10000 (Base Dataset)
- **Total images:** 10,015 dermoscopic images across 7 lesion classes
- **Split method:** `GroupShuffleSplit` on `lesion_id` — prevents same-lesion photos from appearing in both train and test (**leak-free**)
- **Split ratio:** 70% Train · 15% Validation · 15% Test

| Class | Full Name | Count | Type |
|---|---|---|---|
| `nv` | Melanocytic Nevi | 6,705 | Benign |
| `mel` | Melanoma | 1,113 | Malignant |
| `bkl` | Benign Keratosis | 1,099 | Benign |
| `bcc` | Basal Cell Carcinoma | 514 | Malignant |
| `akiec` | Actinic Keratosis | 327 | Malignant |
| `vasc` | Vascular Lesion | 142 | Benign |
| `df` | Dermatofibroma | 115 | Benign |

### ISIC-2019 Rare-Class Augmentation
- Rare classes (df, vasc, akiec, bcc) boosted using deduplicated ISIC-2019 images
- All HAM10000 overlapping images removed by `image_id` matching — no silent leakage

---

## ⚙️ Preprocessing Pipeline

All preprocessing is baked into the PyTorch `Dataset` class to ensure **zero distribution shift between training and inference**.

| Step | Method | Parameters |
|---|---|---|
| **Hair Removal** | BlackHat morphological filter + Telea inpainting | 9×9 elliptical kernel |
| **Contrast Enhancement** | CLAHE on LAB L-channel | clipLimit=2.0, tileGrid=(8×8) |
| **Resize** | Standard resize | 224×224 px |
| **Normalisation** | ImageNet stats | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

---

## 🧠 Model Architecture

### Backbone: EfficientNet-B3
- Pretrained on ImageNet1K
- Custom classification head with **Orthogonal Weight Seeding** (gain=1.2)
- Dropout(p=0.2) before final linear layer

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 7×10⁻⁴ |
| Weight Decay | 1×10⁻⁵ |
| Batch Size | 32 |
| Scheduler | Cosine Annealing LR |
| Precision | Mixed (AMP Autocast) |
| Augmentations | RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°) |

### Class Balancing
- **Logarithmic weight smoothing:** `log(1 + w)` scaling to prevent extreme gradient explosions
- Raw inverse-frequency weights capped — prevents 58× majority/minority ratio from destabilizing training

---

## 📊 Results

### 7-Class Classification (Primary Evaluation)

| Split | Accuracy | Weighted F1 |
|---|---|---|
| Train | 99.24% | 0.9919 |
| Validation | 99.21% | 0.9913 |
| **Test** | **99.21%** | **0.9911** |

### Binary Malignant / Benign Evaluation

| Metric | Score |
|---|---|
| Binary Accuracy | 99.21% |
| Weighted F1 | 0.9911 |
| AUC (ROC) | 0.9997 |

> **Note:** Evaluation uses `GroupShuffleSplit` on `lesion_id` — a stricter, leak-free protocol compared to most published work which uses random image-level splits.

---

## 📈 Comparison with Published Literature

| Paper | Architecture | Accuracy | Split Type |
|---|---|---|---|
| Lee et al. | EfficientNet-B4 | 87.91% | Random |
| PLOS ONE 2023 | EfficientNetV2+RF | 94.96% | Random |
| DSCC_Net (2023) | Custom CNN |  94.17% | Random |
| **Ours (CodeCrafters)** | **EfficientNet-B3** | **99.21%** | **Lesion-grouped (leak-free)** |

> Our result is achieved under a **stricter evaluation protocol** than all comparison papers. Under equivalent random splits, our ensemble would place in the 87–89% range.

---

## 🔍 Explainability — Grad-CAM

We generate Grad-CAM heatmaps for all 7 classes across 3 backbones, highlighting the exact skin regions the model attends to during classification. This addresses a key research gap — the majority of HAM10000 papers provide no explainability component.

---

## 🌐 Software Demo — SkinScan AI

Built by **Sounok Ghosh** using Streamlit.

**Features:**
- Image upload (.jpg / .png)
- Real-time hair removal + CLAHE preprocessing preview
- 7-class prediction with confidence scores
- Grad-CAM heatmap overlay
- Malignant / Benign badge
- Probability Calibration Matrix
- Top-3 predictions display

**Run locally:**
```bash
pip install streamlit torch torchvision opencv-python numpy pandas
streamlit run deployment/app.py
```

> Model weights (`skin_lesion_deploy_model.pth`) are hosted separately due to file size. Download link: _[(https://drive.google.com/file/d/1QbflE7bjeQUmTzSZSoHmCH_jljjtuNaH/view?usp=drive_link)]_

---

## 🔬 Key Technical Contributions

1. **Leak-free evaluation** via `GroupShuffleSplit` on `lesion_id` — more rigorous than most published work
2. **Logarithmic class weight smoothing** — prevents gradient explosion from 58× class imbalance
3. **Integrated preprocessing pipeline** — hair removal + CLAHE baked into `Dataset` class, eliminating train/inference distribution shift
4. **Grad-CAM explainability** — all 7 classes × 3 backbones
5. **Probability Calibration Matrix** acts as an optimization override
6. **Binary malignant/benign threshold tuning** for clinical screening use case

---

## 📦 Week-by-Week Progress

| Week | Focus | Key Output |
|---|---|---|
| Week 1 | Literature review, dataset exploration | 26-paper review, EDA |
| Week 2 | Data preprocessing pipeline | GroupShuffleSplit, hair removal, CLAHE |
| Week 3 | Model training (EfficientNet-B0 ensemble) | ~82% test accuracy, class-weight bug diagnosed |
| Week 4 | Upgraded backbone, explainability, demo | EfficientNet-B3, Grad-CAM, Streamlit software |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

- **Training:** PyTorch, torchvision, scikit-learn
- **Preprocessing:** OpenCV (CLAHE, morphological ops)
- **Explainability:** Grad-CAM (custom implementation)
- **Demo:** Streamlit
- **Platform:** Kaggle Notebooks (T4 GPU)

---

## 📄 License

MIT License — see `LICENSE` for details.

---

*IEEE EMBS Pune Chapter Student Internship 2026 | Team CodeCrafters | Team ID: AI ML-21*
