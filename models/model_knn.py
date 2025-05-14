import os
import cv2
import numpy as np
import pandas as pd
import pickle

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix

df = pd.read_csv("sleep_pattern_neutrosophic_all.csv")
X = df.drop(columns=["label"])
y = df["label"]

# LabelEncoder (FEATURES => LEFT, RIGHT, SUPINE)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test.to_numpy())

accuracy = accuracy_score(y_test, y_pred_knn)
mse = mean_squared_error(y_test, y_pred_knn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_knn)

print(f"R² Score: {r2:.4f}, Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

def classify_image(image_path):
    features = extract_features(image_path)
    features = np.array(features).reshape(1, -1)
    prediction = knn_model.predict(features)
    predicted_label = le.inverse_transform([prediction[0]])[0]
    return predicted_label

# - - - - - - - - - -  PATH TO IMAGE - - - - - - - - - - - - - -
print("-------------------- KNN PREDICTION --------------------------")
image_path_1 = "all/left/658.png"
predicted_class = classify_image(image_path_1)
print(f"Predicted Sleep Posture: {predicted_class}")

image_path_2 = "all/right/87.png"
predicted_class = classify_image(image_path_2)
print(f"Predicted Sleep Posture: {predicted_class}")
