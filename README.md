# 🔬 Skin Disease Classification Using Transfer Learning

![IEEE EMBS](https://img.shields.io/badge/IEEE%20EMBS-Pune%20Chapter-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Accuracy](https://img.shields.io/badge/Accuracy-85.30%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

-----

## 📌 Project Overview

Automated multi-class skin disease classification system using **Transfer Learning on Dermoscopic Images**.

We fine-tuned **EfficientNet-B0** (pretrained on ImageNet) on the **HAM10000 dataset** to classify **7 types of skin diseases** — with weighted loss to handle class imbalance and a Streamlit web demo for real-world deployment.

-----

## 🏆 Results

|Metric               |Score                      |
|---------------------|---------------------------|
|**Test Accuracy**    |**85.30%**                 |
|**Weighted F1 Score**|**0.8548**                 |
|**Macro F1 Score**   |**0.7648**                 |
|**Epochs**           |20                         |
|**Hardware**         |Tesla T4 GPU (Google Colab)|

-----

## 👥 Team — CodeCrafters (AI ML-21)

|Member            |Role                            |
|------------------|--------------------------------|
|**Subham Pal**    |Model Training & Research       |
|**Swarnali Ghosh**|Data Preparation & Preprocessing|
|**Sounok Ghosh**  |Deployment & Streamlit Demo     |

**IEEE EMBS Pune Chapter — Student Internship 2026**
**Duration:** June 1 – June 30, 2026 | **Mode:** Online

-----

## 📁 Repository Structure

```
├── README.md
├── .gitignore
│
├── data/
│   ├── Swarnali_Data_Preprocessing.ipynb  ← Data prep notebook
│   ├── train.csv                          ← Training split (7010 images)
│   ├── val.csv                            ← Validation split (1502 images)
│   ├── test.csv                           ← Test split (1503 images)
│   ├── class_weights.npy                  ← Computed class weights
│   ├── label_map.json                     ← Class name ↔ number mapping
│   ├── preprocessing_steps.png            ← Preprocessing visualization
│   ├── class_distribution.png             ← Class imbalance chart
│   └── sample_images.png                  ← Sample images per class
│
├── models/
│   ├── Subham_Training_FINAL.ipynb        ← Model training notebook
│   ├── training_curves.png                ← Loss & accuracy curves
│   ├── confusion_matrix.png               ← Confusion matrix
│   └── our_results.json                   ← Final evaluation results
│
└── deployment/                            ← Sounok (Week 3)
    └── app.py                             ← Streamlit web app (coming soon)
```

-----

## 🗂️ Dataset

**HAM10000 — Human Against Machine with 10000 Training Images**

|Property    |Value                         |
|------------|------------------------------|
|Total Images|10,015                        |
|Classes     |7 skin disease types          |
|Source      |ISIC Archive                  |
|Split       |70% Train / 15% Val / 15% Test|

**7 Disease Classes:**

|Code |Disease             |
|-----|--------------------|
|nv   |Melanocytic Nevi    |
|mel  |Melanoma            |
|bkl  |Benign Keratosis    |
|bcc  |Basal Cell Carcinoma|
|akiec|Actinic Keratosis   |
|vasc |Vascular Lesion     |
|df   |Dermatofibroma      |

📥 **Download Dataset:** [Kaggle — HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

-----

## ⚙️ Preprocessing Pipeline

```
Raw Image
    → Hair Removal (BlackHat filter + Inpainting)
    → CLAHE Contrast Enhancement
    → Resize to 224 × 224
    → Normalize (Mean=[0.485,0.456,0.406], Std=[0.229,0.224,0.225])
```

-----

## 🧠 Model Architecture

- **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
- **Final Layer:** Linear(1280, 7) — replaced for 7-class task
- **Loss Function:** Weighted Cross-Entropy Loss
- **Optimizer:** Adam (lr = 1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Parameters:** 4,016,515 (all trainable)

-----

## 🚀 How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/iamsubham019/IEEE-EMBS-PUNE-CHAPTER-Student-Internship--skin_disease_classification.git
cd IEEE-EMBS-PUNE-CHAPTER-Student-Internship--skin_disease_classification
```

### Step 2 — Install dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn opencv-python pandas numpy
```

### Step 3 — Download dataset

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ./dataset --unzip
```

### Step 4 — Run preprocessing (Swarnali’s notebook)

```
Open data/Swarnali_Data_Preprocessing.ipynb in Google Colab
Run all cells
```

### Step 5 — Run training (Subham’s notebook)

```
Open models/Subham_Training_FINAL.ipynb in Google Colab
Run all cells
```

-----

## 📊 Model Comparison

|Model                     |Year    |Accuracy  |Parameters|Deployable|
|--------------------------|--------|----------|----------|----------|
|VGG-16                    |2019    |~81%      |138M      |✗         |
|ResNet-50                 |2019    |~83%      |25M       |Limited   |
|Inception-v3              |2017    |~85%      |23M       |Limited   |
|**Our EfficientNet-B0**   |**2026**|**85.30%**|**4M**    |**✓**     |
|EfficientNet-B0 (AIP 2024)|2024    |~89%      |4M        |✓         |
|MDPI EfficientNet 2025    |2025    |97.15%    |4M        |✓         |

-----

## 📄 References

1. Esteva et al. (2017). Dermatologist-level classification of skin cancer. *Nature*, 542, 115–118.
1. Tschandl et al. (2018). The HAM10000 dataset. *Scientific Data*, 5:180161.
1. MDPI (2025). Skin Cancer Detection Using EfficientNet. *MDPI*.
1. Frederich et al. (2024). Skin Lesion Classification Using EfficientNet-B0. *AIP Conference Proceedings*.

-----

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

-----

*IEEE EMBS Pune Chapter — Student Internship 2026 | Team CodeCrafters (AI ML-21)*