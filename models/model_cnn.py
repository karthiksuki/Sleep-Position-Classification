import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

input_dirs = {
    # Left Position
    "Left_truth": r"",
    "Left_indeterminacy": r"",
    "Left_falsity": r"",
    # Right Position
    "Right_truth" : r"",
    "Right_indeterminacy": r"",
    "Right_falsity": r"",
    # Supine Position
    "Supine_truth": r"",
    "Supine_indeterminacy": r"",
    "Supine_falsity": r""

}

def load_data(input_dirs, img_size=(64, 64)):
    images = []
    labels = []

    for label, path in input_dirs.items():
        image_dir = path
        print(f"Loading images from {image_dir}...")
        for file in os.listdir(image_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, img_size)
                image = image.flatten().astype(np.float32)
                images.append(image)
                labels.append(label)

    print(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels)


X, y = load_data(input_dirs)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_reshaped = X_train.reshape(-1, 64, 64, 1)
X_test_reshaped = X_test.reshape(-1, 64, 64, 1)
X_train_reshaped = X_train_reshaped / 255.0
X_test_reshaped = X_test_reshaped / 255.0

# CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test))

cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_reshaped, y_test)
print(f"CNN Accuracy: {cnn_accuracy:.4f}")

y_pred_cnn = cnn_model.predict(X_test_reshaped)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)

accuracy = accuracy_score(y_test, y_pred_cnn)
mse = mean_squared_error(y_test, y_pred_cnn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_cnn)
print(f"Accuracy: {accuracy:.4f}, R2: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred_cnn, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_cnn))

cnn_model.save('sleep_pattern_cnn_model.h5')
print("Model saved as 'sleep_pattern_cnn_model.h5'")

def predict_new_image(image_path, model='sleep_pattern_cnn_model.h5', img_size=(64, 64)):
    model = tf.keras.models.load_model(model)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)
    image = image.flatten().astype(np.float32)
    image = image.reshape(-1, 64, 64, 1)
    image = image / 255.0

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)

    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]


new_image_1 = r""
predicted_label = predict_new_image(new_image_1)
print(f"Predicted label for the new image: {predicted_label}")

new_image_2 = r""
predicted_label = predict_new_image(new_image_2)
print(f"Predicted label for the new image: {predicted_label}")
