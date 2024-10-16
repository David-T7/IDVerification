import datetime
import os
import sys
import re
import cv2
import numpy as np
import csv
import binascii
import pytesseract as tess
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt
# ------------------------------ Utility Functions ------------------------------

def load_models(forgery_model_path, autoencoder_model_path):
    """Load pre-trained deep learning models."""
    try:
        model_forgery = tf.keras.models.load_model(forgery_model_path)
        autoencoder = tf.keras.models.load_model(autoencoder_model_path)
        print("Models loaded successfully.")
        return model_forgery, autoencoder
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

def parse_mrz_data(mrz_data):
    """Parse MRZ data and extract relevant fields."""
    try:
        nationality = mrz_data.get('nationality', 'N/A')
        name = mrz_data.get('names', 'N/A').strip()
        surname = mrz_data.get('surname', 'N/A').strip()
        type_passport = mrz_data.get('type', 'N/A').strip()
        country = mrz_data.get('country', 'N/A').strip()
        birth = mrz_data.get('date_of_birth', 'N/A').strip()
        id_num = mrz_data.get('personal_number', 'N/A').strip()
        pass_num = mrz_data.get('number', 'N/A').strip()
        sex = mrz_data.get('sex', 'N/A').strip()
        exp_date = mrz_data.get('expiration_date', 'N/A').strip()
        raw_text = mrz_data.get('raw_text', 'N/A')

        # Parse dates
        birth_date = parse_mrz_date(birth)
        expiration_date = parse_mrz_date(exp_date)

        parsed_data = {
            'Nationality': nationality,
            'Names': name,
            'Surname': surname,
            'Passport Type': type_passport,
            'Country Code': country,
            'Date of Birth': birth_date,
            'ID Number': id_num,
            'Passport Number': pass_num,
            'Gender': sex,
            'Expiration Date': expiration_date,
            'Raw Text': raw_text
        }

        return parsed_data
    except Exception as e:
        print(f"Error parsing MRZ data: {e}")
        return {}

def parse_mrz_date(date_str):
    """Convert YYMMDD to a readable date format."""
    try:
        return datetime.strptime(date_str, '%y%m%d').date()
    except ValueError:
        return "Invalid Date"

def save_to_csv(data_dict, csv_path, header=None):
    """Save extracted data to a CSV file."""
    try:
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            if header is None:
                header = list(data_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)
        print(f"Data saved to {csv_path}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def display_passport_info(parsed_data):
    """Print passport information in a structured format."""
    print("----- Passport Information -----")
    for key, value in parsed_data.items():
        print(f"{key:<20}: {value}")
    print("--------------------------------")

# ------------------------------ Forgery Detection Functions ------------------------------

def perform_ela(image_path, output_path='ela_image.png', quality=90, threshold=10):
    """
    Perform Error Level Analysis (ELA) on an image to detect tampering.
    """
    try:
        original = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return False, 0

    # Save the image with a specific quality to create a temp file
    temp_path = 'temp.jpg'
    original.save(temp_path, 'JPEG', quality=quality)
    temp = Image.open(temp_path)

    # Perform ELA by comparing the original and the temp image
    ela = ImageChops.difference(original, temp)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    # Enhance the ELA image for better visibility (optional)
    if max_diff > 0:
        scale = 255.0 / max_diff
    else:
        scale = 1
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    ela.save(output_path)

    # Convert ELA image to numpy array for metric calculation
    ela_array = np.array(ela)
    ela_score = np.mean(ela_array)  # Example metric

    # Determine if tampering is detected based on threshold
    is_tampered = ela_score > threshold

    return is_tampered, ela_score

def clone_detection(image_path, visualize=False, distance_threshold=30, area_threshold=50):
    """
    Detect cloned regions in an image using ORB feature matching.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return False, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 2:
        print("Not enough descriptors for clone detection.")
        return False, 0

    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Find all matches
    matches = bf.match(descriptors, descriptors)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Analyze matches to find duplicates
    # Remove self-matches
    good_matches = [m for m in matches if m.queryIdx != m.trainIdx]

    # Find matches below the distance threshold
    cloned_matches = [m for m in good_matches if m.distance < distance_threshold]

    # Create a mask to identify cloned regions
    mask = np.zeros_like(gray, dtype=np.uint8)
    for match in cloned_matches:
        kp1 = keypoints[match.queryIdx].pt
        kp2 = keypoints[match.trainIdx].pt
        cv2.circle(mask, (int(kp1[0]), int(kp1[1])), 5, 255, -1)
        cv2.circle(mask, (int(kp2[0]), int(kp2[1])), 5, 255, -1)

    # Find connected components in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Count cloned regions exceeding the area threshold
    clone_count = 0
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        if area > area_threshold:
            clone_count += 1

    is_cloned = clone_count > 0  # Adjust based on desired sensitivity

    # Visualize cloned regions
    if visualize and clone_count > 0:
        output_image = image.copy()
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > area_threshold:
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert BGR to RGB for plotting
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image)
        plt.title('Clone Detection')
        plt.axis('off')
        plt.show()

    return is_cloned, clone_count

def predict_forgery(model, img_path):
    """
    Predict if an image is forged using a pre-trained model.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        is_forged = prediction[0][0] >= 0.5
        confidence = prediction[0][0] if is_forged else 1 - prediction[0][0]
        return is_forged, confidence
    except Exception as e:
        print(f"Error in forgery prediction: {e}")
        return False, 0.0

def detect_anomaly(autoencoder, img_path, threshold=0.05):
    """
    Detect anomalies in an image using a pre-trained autoencoder.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        reconstructed = autoencoder.predict(img_array)
        mse = np.mean(np.power(img_array - reconstructed, 2))
        is_anomaly = mse > threshold
        return is_anomaly, mse
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return False, 0.0

def detect_template(image_path, template_path, threshold=0.8):
    """
    Detect a specific template (e.g., hologram) in an image.
    """
    try:
        image = cv2.imread(image_path, 0)
        template = cv2.imread(template_path, 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        detected = False
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            detected = True
        if detected:
            cv2.imshow('Detected Template', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return detected
    except Exception as e:
        print(f"Error in template detection: {e}")
        return False