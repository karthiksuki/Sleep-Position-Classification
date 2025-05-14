from skimage.feature import hog
from model_svm import *

# PICKLE MODEL DUMP
with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

def classify_image(image_path):
    features = extract_features(image_path)
    features = np.array(features).reshape(1, -1)
    prediction = svm_model.predict(features)
    predicted_label = le.inverse_transform([prediction[0]])[0]
    return predicted_label

# - - - - - - - - - -  PATH TO IMAGE - - - - - - - - - - - - - -
print("-------------------- PREDICTION --------------------------")
image_path = ""
predicted_class = classify_image(image_path)
print(f"Predicted Sleep Posture: {predicted_class}")
