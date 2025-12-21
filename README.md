

# Knee Osteoarthritis Severity Classification (KL Grades)

Automated classification of **Kellgren–Lawrence (KL) grades** for knee osteoarthritis using **classical machine learning and deep learning** on knee X-ray images.
The project compares handcrafted texture features with transfer-learning–based CNNs, incorporates **ordinal classification**, and evaluates ensemble strategies.

---

## 📌 Overview

Knee osteoarthritis (KOA) is a degenerative joint disease commonly assessed using X-ray imaging and the **Kellgren–Lawrence grading system (KL-0 to KL-4)**. Manual grading is subjective and inconsistent across observers.

This project investigates automated KL grade classification by:

* Comparing **classical ML vs deep learning**
* Addressing **class imbalance**
* Modeling the **ordinal nature** of KL grades
* Evaluating clinically meaningful metrics

---

## 🗂 Dataset

* **Source:** Mendeley Knee X-ray Dataset
* **Subset:** `ClsKLData`
* **Classes:** 5 (KL-0 to KL-4)
* **Image size:** 224 × 224
* **Splits:** Train / Validation / Test

The dataset exhibits **class imbalance**, with mild and moderate grades occurring more frequently.

---

## ⚙️ Methodology

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

* Probability-level averaging across multiple models

---

## 🧠 Models

* Random Forest (GLCM features)
* EfficientNet-B0 (transfer learning)
* ResNet-18 (transfer learning)
* Ordinal EfficientNet-B0
* Probability-based ensemble

---

## 📊 Evaluation

Models are evaluated on a held-out test set using:

* **Accuracy**
* **Macro F1-score**
* **Cohen’s Kappa**
* **Confusion Matrix**

Grad-CAM is used to visualize salient regions influencing CNN predictions.

---

## 🧰 Tech Stack

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

## 🚀 Setup & Usage

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

## 📈 Key Findings

* Deep learning models significantly outperform classical ML
* ResNet-18 achieves the best overall performance
* Ordinal classification improves agreement with true labels
* Ensemble learning provides stable but marginal gains

---

## ⚠️ Limitations & Future Work

* Limited dataset size restricts generalization
* Single-modal imaging (X-ray only)
* Future work may explore:

  * Larger datasets
  * Advanced ordinal losses
  * Attention-based architectures
  * Multimodal medical data

---

## 📜 License

This project is intended for **academic and research purposes**.


