import pandas as pd
import argparse
import logging
import joblib
import cv2
import requests
import numpy as np
import os
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from model import preprocess_image  # Ensure model.py is accessible

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_image_from_path(image_path):
    """
        Load an image from a file path.
        
        Parameters:
            image_path (str): Path to the image.
        
        Returns:
            numpy.ndarray: The loaded image.
        
        Raises:
            FileNotFoundError: If no file is found at the specified path.

        Exception:
            FileNotFoundError: If the file is corrupt or not a valid image file.
        
        Exception:
            FileNotFoundError: You will run into errors if you are trying to 
            access an image from a cloud service that is not fully synced to you device.

    """
    # Check if file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"No file found at {image_path}. Ensure the file is synced from the cloud if using a cloud storage service such as Microsoft OneDrive.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Failed to load image at {image_path}. The file might be corrupt or not a valid image file.")
    
    return image



def load_image_from_url(image_url):
    """
        Load an image from a URL.
        
        Parameters:
            image_url (str): URL of the image.
        
        Returns:
            numpy.ndarray: The loaded image or None if loading failed.
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Check if requests were successful

        # Open the image
        image = Image.open(BytesIO(response.content))
        return np.array(image)

    except requests.RequestException as e:
        logging.error(f"Failed to fetch image from {image_url}: {str(e)}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    return None


def extract_exif_data(image_path):
    """
        Extract and display EXIF data from an image.

        Parameters:
            image_path (str): Path to the image.

        Returns:
            pd.DataFrame: DataFrame containing the EXIF data.
    """
    def format_gps(gps_info):
        """Format GPS information."""
        gps_data = {}
        for t in gps_info:
            sub_label = GPSTAGS.get(t, t)
            gps_data[sub_label] = gps_info[t]
        # If GPSInfo is well-structured, extract and format it.
        # Otherwise, log the raw data and return None.
        try:
            lat_data = gps_data["GPSLatitude"]
            lon_data = gps_data["GPSLongitude"]
            # Convert the GPS coordinates to a human-readable format
            lat = f"{lat_data[0][0] / lat_data[0][1]}°{lat_data[1][0] / lat_data[1][1]}'{lat_data[2][0] / lat_data[2][1]}\"{'N' if gps_data['GPSLatitudeRef'] == 'N' else 'S'}"
            lon = f"{lon_data[0][0] / lon_data[0][1]}°{lon_data[1][0] / lon_data[1][1]}'{lon_data[2][0] / lon_data[2][1]}\"{'E' if gps_data['GPSLongitudeRef'] == 'E' else 'W'}"
            return f"{lat}, {lon}"
        except Exception as e:
            print(f"Failed to parse GPSInfo: {e}. Raw data: {gps_info}")
            return None

    try:
        # Open image
        image = Image.open(image_path)

        # Extract EXIF data
        exif_data = image._getexif()
        if exif_data is None:
            print(f"No EXIF data found for {image_path}.")
            return None

        # Translate EXIF data to labeled data
        labeled_exif_data = {}
        for key, val in exif_data.items():
            label = TAGS.get(key, key)
            if label == "GPSInfo":
                labeled_exif_data[label] = format_gps(val)
            else:
                labeled_exif_data[label] = val

        # Convert to DataFrame for pretty printing
        df = pd.DataFrame.from_dict(
            labeled_exif_data, orient='index', columns=['Value'])

        # Display EXIF data
        print(f"\nEXIF Data for {image_path}:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df)

        return df

    except Exception as e:
        print(f"Failed to extract EXIF data from {image_path}: {str(e)}")
        return None


def main(args):
    """
        Load images, make predictions, and compare them.
        
        Parameters:
            args (Namespace): Parsed command-line arguments.
    """
    # Load trained model and label encoder
    try:
        model = joblib.load('svm_model.pkl')
        le = joblib.load('label_encoder.pkl')
    except Exception as e:
        logging.error(f"Error loading model or label encoder: {str(e)}")
        return

    # Load images
    if args.local:
        try:
            image1 = load_image_from_path(args.image1)
            image2 = load_image_from_path(args.image2)
        except FileNotFoundError as e:
            logging.error(str(e))
            return
    else:
        image1 = load_image_from_url(args.image1)
        image2 = load_image_from_url(args.image2)

        if image1 is None or image2 is None:
            logging.error("Failed to load one or both images. Exiting.")
            return

    # Preprocess images
    preprocessed_image1 = preprocess_image(image1)
    preprocessed_image2 = preprocess_image(image2)

    # Make predictions
    label1 = model.predict([preprocessed_image1])[0]
    label2 = model.predict([preprocessed_image2])[0]

    # Get probability estimates
    probabilities1 = model.predict_proba([preprocessed_image1])
    probabilities2 = model.predict_proba([preprocessed_image2])

    # Decode labels
    label1_name = le.inverse_transform([label1])[0]
    label2_name = le.inverse_transform([label2])[0]

    # Find max probability
    max_prob1 = np.max(probabilities1) * 100  # as a percentage
    max_prob2 = np.max(probabilities2) * 100  # as a percentage

    # Log results
    logging.info(
        f"Image 1 is predicted as: {label1_name} with {max_prob1:.2f}% confidence")
    logging.info(
        f"Image 2 is predicted as: {label2_name} with {max_prob2:.2f}% confidence")

    # Compare the labels directly
    if label1 == label2:
        logging.info("The images are predicted to be the same person.")
    else:
        logging.info("The images are predicted to be different people.")

    # Check if EXIF data should be printed
    if args.exif:
        exif_data_image1 = extract_exif_data(args.image1)
        exif_data_image2 = extract_exif_data(args.image2)

        print(f"\nEXIF Data for {args.image1}:")
        print(exif_data_image1)

        print(f"\nEXIF Data for {args.image2}:")
        print(exif_data_image2)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compare two images using a trained model.')
    parser.add_argument('image1', help='Path or URL of the first image.')
    parser.add_argument('image2', help='Path or URL of the second image.')
    parser.add_argument('--local', action='store_true',
                        help='Flag indicating whether the images are local file paths.')
    parser.add_argument('--exif', action='store_true',
                        help='Flag indicating whether to print EXIF data of the images.')

    args = parser.parse_args()

    # Load trained model and label encoder
    model = joblib.load('svm_model.pkl')
    le = joblib.load('label_encoder.pkl')

    # Compare images
    main(args)
