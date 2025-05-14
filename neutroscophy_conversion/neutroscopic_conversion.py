import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def neutrosophic_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = cv2.GaussianBlur(gray, (5,5), 0)  # True component (smoothing)
    I = cv2.absdiff(gray, T)  # Indeterminacy component (difference with smoothed)
    F = 255 - gray  # False component (negative image)
    return T, I, F

input_dir = r""
output_dir = r""

true_dir = os.path.join(output_dir, "Truth")
indeterminacy_dir = os.path.join(output_dir, "Indeterminacy")
false_dir = os.path.join(output_dir, "Falsity")

os.makedirs(true_dir, exist_ok=True)
os.makedirs(indeterminacy_dir, exist_ok=True)
os.makedirs(false_dir, exist_ok=True)

# Get list of image files in the directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No images found in the directory.")
else:
    print(f"Processing {len(image_files)} images...")

for file in image_files:
    image_path = os.path.join(input_dir, file)  # Correct file path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Unable to load {image_path}. Skipping...")
        continue  # Skip unreadable images

    T, I, F = neutrosophic_transform(image)

    t_path = os.path.join(true_dir, f"T_{file}")
    i_path = os.path.join(indeterminacy_dir, f"I_{file}")
    f_path = os.path.join(false_dir, f"F_{file}")

    cv2.imwrite(t_path, T)
    cv2.imwrite(i_path, I)
    cv2.imwrite(f_path, F)

    print(f"Saved: {t_path}, {i_path}, {f_path}")

    # Display results
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Original: {file}")
    axs[1].imshow(T, cmap="gray")
    axs[1].set_title("True (T)")
    axs[2].imshow(I, cmap="gray")
    axs[2].set_title("Indeterminacy (I)")
    axs[3].imshow(F, cmap="gray")
    axs[3].set_title("False (F)")

    for ax in axs:
        ax.axis("off")

    plt.show()