
# Sleep Position Classifier

A machine learning-based Sleep Position Classifier designed to identify one of three sleep postures: **Left**, **Supine**, and **Right**. The system leverages classical machine learning (SVM, KNN) and deep learning (CNN) models, using both handcrafted HOG features and raw image inputs. Additionally, a novel **Neutrosophic conversion** step is applied for enhanced preprocessing using **Falsity**, **Indeterminacy**, and **Truth** maps.

---

## Project Overview

This project aims to classify sleep positions based on image data. The complete pipeline includes image preprocessing using **Neutrosophy**, feature extraction using **HOG (Histogram of Oriented Gradients)**, and classification using:

- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Convolutional Neural Network (CNN)**

---

## 📁 Dataset

- Images of individuals in three sleep positions: **Left**, **Supine**, and **Right**
- Stored in structured class-wise directories
- Images are initially processed for Neutrosophic mapping to reduce uncertainty
  
---

## 🧠 Preprocessing

### Neutrosophic Image Conversion
All input images undergo transformation using the **Neutrosophic theory**, resulting in three grayscale components:
- **Falsity**
- **Indeterminacy**
- **Truth**

These components are used in downstream model training and evaluation.

### HOG Feature Extraction
- Images are transformed into HOG features with **1764 feature dimensions**
- Used as input for traditional ML models like SVM and KNN

---

## ⚙️ Models and Methods

| Model | Input Type            | Description                                           | Accuracy |
|-------|------------------------|-------------------------------------------------------|----------|
| **SVM** | HOG Features (1764D)     | Linear classification on handcrafted features        | **100%** |
| **KNN** | HOG Features (1764D)     | K=7 neighbors for classification                     | **99.67%** |
| **CNN** | Raw Images              | End-to-end deep learning with ConvNet architecture   | **100%** |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV** - Image processing
- **scikit-image** - HOG feature extraction
- **scikit-learn** - SVM, KNN models
- **TensorFlow / Keras** - CNN model building
- **NumPy, Matplotlib** - Data handling and visualization
- **Custom Neutrosophic Module** - Preprocessing logic
- **HOG** - Algorithm

---


## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/karthiksuki/Sleep-Position-Classification
   cd sleep-position-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Neutrosophic Preprocessing**
   ```bash
   python neutroscophy_conversion/neutroscopic_conversion.py
   ```

4. **Train Models**
   ```bash
   python models/model_svm.py
   python models/model_knn.py
   python models/model_cnn.py
   ```

5. **Make Predictions**
   ```bash
   python Prediction_Model.py
   ```

---

## 📊 Results

- **SVM**: 100% Accuracy
- **KNN**: 98.7% Accuracy 
- **CNN**: 100% Accuracy 

---

## 🧪 Future Improvements

- Integration with real-time video-based prediction
- Expand dataset with various body types and sleepwear
- Include edge-case postures (e.g., fetal, prone)

---

## 👨‍💻 Author

- **Dr. Nagarajan Deivanayagampillai, Post-Doctorate, Ph.D** 
- **Karthikeyan Ganesh** - [Linkedln](https://www.linkedin.com/in/karthikeyan-g7/)

