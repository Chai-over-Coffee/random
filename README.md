# 🧠 Brain Tumor Detection Using Deep Learning (VGG16)

## 📌 Project Overview  
This project presents a **deep learning-based automated brain tumor classification system** using **MRI images**.  
The model is capable of classifying brain MRIs into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

This project leverages **transfer learning with VGG16** to achieve high accuracy with limited medical data and can assist radiologists in rapid screening and early diagnosis.

---

## 🎯 Aim  
To develop a reliable, fast, and accurate **brain tumor detection system** that can:
- Automatically analyze MRI scans.
- Classify tumor types with confidence scores.
- Reduce manual effort in early-stage screening and aid doctors in decision-making.

---

## 🏗️ Detailed Approach  

### 1️⃣ Data Collection & Preprocessing  
- **Dataset:** Publicly available brain MRI dataset organized into 4 categories.  
- **Preprocessing Steps:**
  - Converted all images to **128×128 resolution** for uniformity.
  - Normalized pixel values to **[0, 1] range** for stable training.
  - Performed **data augmentation**:
    - Rotation (to account for different scan angles)
    - Horizontal/Vertical flips
    - Zoom & Shift (to simulate variability in MRI positioning)
  - Split dataset into:
    - **Training set** (for learning)
    - **Validation set** (for tuning hyperparameters)
    - **Test set** (for final evaluation)

---

### 2️⃣ Model Selection & Why VGG16  
The core of this project is based on **VGG16**, a popular Convolutional Neural Network (CNN) architecture trained on **ImageNet**.  

**Why VGG16?**
- **Deep architecture (16 layers)**: Extracts hierarchical features — edges, textures, shapes, and high-level tumor patterns.
- **Proven Transfer Learning Performance:** Known to generalize well on medical images even with smaller datasets.
- **Simplicity & Interpretability:** Easier to fine-tune and understand compared to very large models like ResNet or Inception.

---

### 3️⃣ Model Architecture  
Our final model consists of:

- **Base Model:**  
  - Pretrained **VGG16** (`include_top=False`, weights="imagenet").
  - Used as a **feature extractor**.
  - Most convolutional layers **frozen** to preserve generic visual features.
  - Final few convolutional layers **fine-tuned** for MRI-specific feature adaptation.

- **Custom Classification Head:**
  - `Flatten()` – Converts extracted features into a 1D vector.
  - `Dense(128, activation='relu')` – Learns task-specific tumor patterns.
  - `Dropout(0.5)` – Prevents overfitting by randomly dropping 50% of neurons during training.
  - `Dense(4, activation='softmax')` – Outputs probabilities for each tumor category.

---

### 4️⃣ Model Compilation & Training  
- **Optimizer:** Adam (adaptive gradient descent for faster convergence)
- **Loss Function:** Sparse Categorical Crossentropy (ideal for integer-labeled multi-class problems)
- **Metrics:** Accuracy, Precision, Recall
- **Training Strategy:**
  - Mini-batch training with augmented images.
  - Early stopping applied to prevent overfitting.
  - Fine-tuned for multiple epochs until validation loss stabilized.

---

### 5️⃣ Model Evaluation  
- **Confusion Matrix:** To visualize class-wise predictions.
- **Classification Report:** Precision, Recall, F1-Score for each tumor type.
- **ROC-AUC Curves:** To measure model’s ability to distinguish between classes.
- **Accuracy Curves:** Training vs validation accuracy to check overfitting.

---

### 6️⃣ Prediction Pipeline  
- Accepts a single MRI image (JPG/PNG).
- Preprocesses (resize → normalize → expand dimensions).
- Predicts probabilities for all classes.
- Returns **tumor type** and **confidence score**.

---

## 📥 Input & 📤 Output  

| **Input** | **Output** |
|-----------|-----------|
| MRI image (128x128, grayscale/RGB) | Class label: `Glioma / Meningioma / Pituitary / No Tumor` + confidence |

---

## 🌟 Key Features  
- ✅ **Transfer Learning:** Saves training time and improves accuracy with small datasets.  
- ✅ **High Generalization:** Data augmentation prevents overfitting.  
- ✅ **Scalable:** Can be deployed in clinics for real-time diagnosis assistance.  
- ✅ **Explainable:** Confusion matrix and classification reports help understand performance.  

---


## 🚀 Future Scope  
- 🏗 **Web Application:** Currently in development.  
  - Users will be able to upload MRI images and receive predictions via a simple interface.  
  - Will integrate Flask/Streamlit for real-time inference.  
- 🔎 **Grad-CAM Visualization:** To highlight regions of interest where the tumor is located.  
- 📈 **Model Optimization:** Experimenting with ResNet50 / EfficientNet for performance comparison.


