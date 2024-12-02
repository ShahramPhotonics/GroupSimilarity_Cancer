import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
from skimage.feature import greycomatrix, greycoprops

# Feature extraction
def extract_features(image, subsection_size=(50, 50)):
    """
    Extract 5 features (Intensity, Contrast, Entropy, Energy, Homogeneity)
    for each subsection of the image.
    """
    rows, cols = image.shape
    sub_h, sub_w = subsection_size
    all_features = []

    for i in range(0, rows, sub_h):
        for j in range(0, cols, sub_w):
            subsection = image[i:i+sub_h, j:j+sub_w]
            features = []

            # 1. Intensity (Mean)
            intensity = np.mean(subsection)
            features.append(intensity)

            # 2. Contrast (Standard Deviation)
            contrast = np.std(subsection)
            features.append(contrast)

            # 3. Entropy
            hist, _ = np.histogram(subsection, bins=256, range=(0, 256), density=True)
            hist = hist + 1e-9  # Avoid log(0)
            entropy = -np.sum(hist * np.log2(hist))
            features.append(entropy)

            # 4. Energy (Sum of squared pixel values normalized)
            energy = np.sum((subsection / 255.0) ** 2)
            features.append(energy)

            # 5. Homogeneity (GLCM property)
            glcm = greycomatrix(subsection.astype(np.uint8), [1], [0], symmetric=True, normed=True)
            homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            features.append(homogeneity)

            all_features.append(features)

    return np.array(all_features)

# Group subsections
def group_subsections(features, num_groups):
    """
    Group subsections for each feature into specified ranges.
    """
    grouped_features = []
    for feature_idx in range(features.shape[1]):
        feature_column = features[:, feature_idx]
        min_val, max_val = np.min(feature_column), np.max(feature_column)
        thresholds = np.linspace(min_val, max_val, num_groups + 1)
        groups = [[] for _ in range(num_groups)]

        for feature in feature_column:
            for idx in range(num_groups):
                if thresholds[idx] <= feature < thresholds[idx + 1]:
                    groups[idx].append(feature)
                    break
        grouped_features.append([np.mean(group) if group else 0 for group in groups])
    return np.array(grouped_features)

# Load images
def load_images(folder, prefix, count, size=(500, 500)):
    images = []
    for i in range(count + 1):  # from c00/h00 to c10/h10
        filepath = os.path.join(folder, f"{prefix}{i:02}.png")
        if os.path.exists(filepath):
            image = np.array(Image.open(filepath).convert("L").resize(size))
            images.append(image)
        else:
            print(f"Image not found: {filepath}")
    return images

# Main analysis
if __name__ == "__main__":
    # Paths for cancerous and healthy images
    cancerous_folder = "C:/Users/shahr/Desktop/Douplik/Python/cancerous"
    healthy_folder = "C:/Users/shahr/Desktop/Douplik/Python/healthy"

    # Load all cancerous and healthy images
    cancerous_images = load_images(cancerous_folder, "c", 10)
    healthy_images = load_images(healthy_folder, "h", 10)

    if not cancerous_images or not healthy_images:
        print("Error: Ensure that images are placed in the specified directories.")
        exit()

    subsection_size = (50, 50)
    num_groups = 10

    # Extract features and group for all cancerous and healthy images
    cancerous_features = [extract_features(img, subsection_size) for img in cancerous_images]
    healthy_features = [extract_features(img, subsection_size) for img in healthy_images]

    cancerous_groups = [group_subsections(features, num_groups) for features in cancerous_features]
    healthy_groups = [group_subsections(features, num_groups) for features in healthy_features]

    # Combine cancerous and healthy groups into a single dataset
    X = np.array([group.flatten() for group in cancerous_groups + healthy_groups])  # Flatten grouped features
    y = np.array([1] * len(cancerous_groups) + [0] * len(healthy_groups))  # Labels: 1 = cancerous, 0 = healthy

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a neural network
    clf = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Predict on a new image
    new_image_path = "C:/Users/shahr/Desktop/Douplik/Python/new_image.png"
    if os.path.exists(new_image_path):
        new_image = np.array(Image.open(new_image_path).convert("L").resize((500, 500)))
        new_features = extract_features(new_image, subsection_size)
        new_group = group_subsections(new_features, num_groups).flatten()
        prediction = clf.predict([new_group])
        print(f"Prediction for new image (1 = Cancerous, 0 = Healthy): {prediction[0]}")
    else:
        print("New image for prediction not found.")
