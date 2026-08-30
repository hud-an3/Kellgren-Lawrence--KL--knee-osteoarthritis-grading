

# Knee Osteoarthritis Severity Classification (KL Grades)

Automated classification of **Kellgren–Lawrence (KL) grades** for knee osteoarthritis using **classical machine learning and deep learning** on knee X-ray images.
The project incorporates **ordinal classification**, evaluates ensemble strategies and compares performance of Classical Machine Learning models with Deep Learning. Performance is evaluated using Accuracy, Macro F1-score, and Cohen’s Kappa,
along with qualitative analysis from confusion matrices.

---

##  Overview

Knee osteoarthritis (KOA) is a degenerative joint disease commonly assessed using X-ray imaging and the **Kellgren–Lawrence grading system (KL-0 to KL-4)**. Manual grading is subjective and inconsistent across observers.

This project investigates automated KL grade classification by:

* Comparing **classical ML vs deep learning**
* Addressing **class imbalance**
* Modeling the **ordinal nature** of KL grades
* Evaluating clinically meaningful metrics

---

##  Dataset

* **Source:** Mendeley Knee X-ray Dataset
* **Subset:** `ClsKLData`
* **Classes:** 5 (KL-0 to KL-4)
* **Image size:** 224 × 224
* **Splits:** Train / Validation / Test

The dataset exhibits **class imbalance**, with mild and moderate grades occurring more frequently.
<img width="1561" height="1156" alt="{FC85AD4E-1170-41F8-AB6D-9E31F32EA082}" src="https://github.com/user-attachments/assets/c050eb85-32ed-4500-9b43-f22a8860a35b" />
Sample images per class in training dataset: <img width="1116" height="1433" alt="{ACB915AA-EA57-4F93-9801-D6E57322D260}" src="https://github.com/user-attachments/assets/43a6df10-4303-4835-b0cb-2f271fe4459b" />


---
## Models Compared

### Classical ML 

- **Feature extraction:** Gray-Level Co-occurrence Matrix (GLCM)
  - Contrast, Correlation, Energy, Homogeneity
  - Computed across multiple distances and angles
- **Classifier:** Random Forest (500 trees)
- **Class imbalance handling:** `class_weight="balanced"`

### Deep Learning 

1. **ResNet-18** (Transfer Learning)
   - ImageNet pretrained
   - End-to-end CNN feature learning

2. **EfficientNet-B0** (Transfer Learning)
   - ImageNet pretrained
   - Partial fine-tuning of higher feature blocks
   - `WeightedRandomSampler` used to address class imbalance

##  Methodology

### Classical Machine Learning

* Grayscale preprocessing and normalization
* **GLCM texture feature extraction**
* **Random Forest classifier** with class balancing

### Deep Learning

* Transfer learning using pretrained CNNs
* Limited fine-tuning of feature extractors
* **Weighted sampling** to handle imbalance
* Learning rate scheduling and early stopping

### Ordinal Classification

* Modified learning objective to respect KL grade ordering

### Ensemble Learning

* Probability-level averaging across the models

---

## 🧠 Models

* Random Forest (GLCM features)
* EfficientNet-B0 (transfer learning)
* ResNet-18 (transfer learning)
* Ordinal EfficientNet-B0
* Probability-based ensemble

---

## Evaluation

Models are evaluated on a held-out test set using:

* **Accuracy**
* **Macro F1-score**
* **Cohen’s Kappa**
* **Confusion Matrix**

Grad-CAM is used to visualize salient regions influencing CNN predictions.
<img width="1377" height="1155" alt="{54917C2C-1528-431C-A8E7-31826A987954}" src="https://github.com/user-attachments/assets/cc95214d-3a89-4f30-9a8c-4ba819c22f09" />

<img width="1103" height="311" alt="image" src="https://github.com/user-attachments/assets/1ca78030-780a-43a1-8cd4-e415e34c5f89" />


---

## Tech Stack

**Language:**

* Python 3

**Frameworks & Libraries:**

* PyTorch, Torchvision
* Scikit-learn
* NumPy

**Image Processing:**

* Pillow (PIL)
* Gray-Level Co-occurrence Matrix (GLCM)

**Training & Optimization:**

* AdamW
* WeightedRandomSampler
* ReduceLROnPlateau
* Early Stopping

**Visualization & Analysis:**

* Matplotlib
* Grad-CAM

**Hardware:**

* CUDA-enabled GPU

---

## Setup & Usage

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Run the relevant scripts for:

* Classical ML
* CNN training
* Ordinal classification
* Ensemble evaluation

---

## Key Findings

* Deep learning models significantly outperform classical ML
* Classical ML with GLCM features is insufficient for capturing complexity of OA severity.
* ResNet-18 achieves the best overall performance
* EfficientNet-B0 gives decent performance but requires careful tuning and more data.
* Ordinal classification improves agreement with true labels
* Ensemble learning provides stable but marginal gains

---

## Limitations & Future Work

* Limited dataset size restricts generalization
* Single-modal imaging (X-ray only)
* Future work may explore:

  * Larger datasets
  * Advanced ordinal losses
  * Attention-based architectures
  * Multimodal medical data




