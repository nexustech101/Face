import logging
import joblib
import cv2
import glob
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_image(image):
    """
        Preprocess a single image by converting it to grayscale, resizing, flattening, and normalizing.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The preprocessed image.
    """
    # Apply histogram equalization
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to the desired size
    resized_image = cv2.resize(gray_image, (100, 100))
    # Flatten the image
    flattened_image = resized_image.flatten()
    # Normalize pixel values to [0, 1]
    normalized_image = flattened_image / 255.0

    logging.info("Image preprocessed.")
    return normalized_image


def load_data(image_dir):
    """
        Load and preprocess image data from a directory.

        Parameters:
            image_dir (str): Path to the image directory.

        Returns:
            tuple: Numpy arrays of the data and labels.
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    data = []
    labels = []

    for path in image_paths:
        image = cv2.imread(path)
        preprocessed_image = preprocess_image(image)
        data.append(preprocessed_image)
        # Assume label is the first part of filename
        label = os.path.basename(path).split("_")[0]
        labels.append(label)

    logging.info(f"{len(data)} images loaded from {image_dir}.")
    return np.array(data), np.array(labels)


def filter_minority_classes(data, labels, min_samples_per_class=2):
    """
        Remove classes with fewer samples than min_samples_per_class.

        Parameters:
            data (numpy.ndarray): Image data.
            labels (numpy.ndarray): Class labels.
            min_samples_per_class (int): Minimum number of samples per class.

        Returns:
            tuple: Filtered data and labels.
    """
    counter = Counter(labels)
    valid_labels = {label for label,
                    count in counter.items() if count >= min_samples_per_class}
    valid_indices = [i for i, label in enumerate(
        labels) if label in valid_labels]

    return data[valid_indices], labels[valid_indices]


def train_evaluate_model(train_data, train_labels, test_data, test_labels, le):
    """
        Train and evaluate a SVM model.

        Parameters:
            train_data (numpy.ndarray): Training data.
            train_labels (numpy.ndarray): Training labels.
            test_data (numpy.ndarray): Test data.
            test_labels (numpy.ndarray): Test labels.
            le (LabelEncoder): Fitted label encoder.

        Returns:
            sklearn.svm.SVC: The trained model.
    """
    model = SVC(kernel="linear", probability=True)
    logging.info("Training the model...")
    model.fit(train_data, train_labels)

    logging.info("Making predictions...")
    predictions = model.predict(test_data)

    logging.info("Evaluating the model...")
    unique_labels = np.unique(test_labels)
    target_names = le.inverse_transform(unique_labels)

    print("\nModel Evaluation:")
    print(classification_report(test_labels, predictions,
          labels=unique_labels, target_names=target_names))

    return model


# File paths
model_path = 'svm_model.pkl'
label_encoder_path = 'label_encoder.pkl'

# Check if model and label encoder are saved
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    logging.info("Loading model and label encoder from disk...")
    try:
        model = joblib.load(model_path)
        le = joblib.load(label_encoder_path)
    except Exception as e:
        logging.error(f"Error loading model or label encoder: {e}")
else:
    # Load and preprocess data
    image_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Python\\data\\csv\\archive (3)\\images"  # Replace with your image directory
    try:
        data, labels = load_data(image_dir)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        # Handle error (e.g., exit script) to prevent further execution

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Filter minority classes
    filtered_data, filtered_labels = filter_minority_classes(
        data, encoded_labels)

    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(
        filtered_data, filtered_labels, test_size=0.2, stratify=filtered_labels, random_state=42
    )

    # Train and evaluate the model
    model = train_evaluate_model(
        train_data, train_labels, test_data, test_labels, le
    )

    # Save the trained model and label encoder
    try:
        joblib.dump(model, model_path)
        joblib.dump(le, label_encoder_path)
        logging.info("Model and label encoder saved to disk.")
        
    except Exception as e:
        logging.error(f"Error saving model or label encoder: {e}")
