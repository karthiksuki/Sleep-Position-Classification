import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Resize for consistency
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))  # HOG feature extraction
    return features

data = []
labels = []
base_dir = ""

for label in ["Truth", "Indeterminacy", "Falsity"]:
    class_dir = os.path.join(base_dir, label)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        features = extract_features(img_path)
        data.append(features)
        labels.append(f"Supine_{label}")

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("sleep_pattern_supine.csv", index=False)
