from datetime import datetime
import re
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image as PIL
import zxingcpp
import os
import face_recognition

import os
os.environ['DEEPFACE_HOME'] = '/vol/web/deepface'


from deepface import DeepFace
import dlib
from imutils import face_utils

def detect_template(image_path, template_path, threshold=0.8):
    image = cv2.imread(image_path, 0)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return res >= threshold



def clone_detection(image_path, visualize=False, distance_threshold=30, area_threshold=50):
    """
    Detect cloned regions in an image using ORB feature matching.

    Parameters:
    - image_path: Path to the image to be analyzed.
    - visualize: Boolean indicating whether to visualize cloned regions.
    - distance_threshold: Distance threshold for feature matching.
    - area_threshold: Minimum area to consider a region as cloned.

    Returns:
    - is_cloned: Boolean indicating if cloning is detected.
    - clone_count: Number of cloned regions detected.
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
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    is_forged = prediction[0][0] >= 0.5
    confidence = prediction[0][0] if is_forged else 1 - prediction[0][0]
    return is_forged, confidence


def detect_anomaly(autoencoder, img_path, threshold=0.05):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    reconstructed = autoencoder.predict(img_array)
    mse = np.mean(np.power(img_array - reconstructed, 2))
    is_anomaly = mse > threshold
    return is_anomaly, mse



def perform_ela(image_path, output_path='ela_image.png', quality=90, threshold=10):
    """
    Perform Error Level Analysis (ELA) on an image to detect tampering.

    Parameters:
    - image_path: Path to the original image.
    - output_path: Path to save the ELA image.
    - quality: JPEG quality for saving.
    - threshold: ELA score threshold to flag tampering.

    Returns:
    - is_tampered: Boolean indicating if tampering is detected.
    - ela_score: Computed ELA metric.
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
    os.remove(temp_path)
    os.remove(output_path)
    return is_tampered, ela_score
def preprocess_image(img_path, output_path=None, target_size=(224, 224)):
    """
    Preprocess the ID image by resizing, enhancing, and noise reduction.

    Args:
        img_path (str): Path to the original image.
        output_path (str, optional): Path to save the preprocessed image. 
                                     If None, it overwrites the original image.
        target_size (tuple, optional): Desired image size (width, height). Defaults to (224, 224).

    Returns:
        None
    """
    # Load the image using PIL
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        raise ValueError(f"Error loading image {img_path}: {e}")

    # Enhance brightness and contrast
    try:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)  # Adjust brightness factor as needed

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # Adjust contrast factor as needed
    except Exception as e:
        print(f"Error enhancing image {img_path}: {e}")
        raise ValueError(f"Error enhancing image {img_path}: {e}")

    # Resize the image
    try:
        img = img.resize(target_size)
    except Exception as e:
        print(f"Error resizing image {img_path}: {e}")
        raise ValueError(f"Error resizing image {img_path}: {e}")

    # Convert to OpenCV format for further processing
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error converting image {img_path} to OpenCV format: {e}")
        raise ValueError(f"Error converting image {img_path} to OpenCV format: {e}")

    # Apply Gaussian Blur to reduce noise
    try:
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    except Exception as e:
        print(f"Error applying Gaussian Blur to image {img_path}: {e}")
        raise ValueError(f"Error applying Gaussian Blur to image {img_path}: {e}")

    # Optionally, convert to grayscale if needed
    # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Save the preprocessed image
    save_path = output_path if output_path else img_path
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Verify that save_path has a valid image extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        _, ext = os.path.splitext(save_path)
        if ext.lower() not in valid_extensions:
            raise ValueError(f"Invalid image extension '{ext}' for save_path '{save_path}'. Supported extensions are: {valid_extensions}")

        success = cv2.imwrite(save_path, img_cv)
        if not success:
            raise ValueError(f"Unable to save processed image at {save_path}")
        print(f"Preprocessed image saved at: {save_path}")
    except Exception as e:
        print(f"Error saving image {save_path}: {e}")
        raise ValueError(f"Error saving image {save_path}: {e}")


# Assuming you have the following functions defined:
# - detect_template
# - perform_ela
# - clone_detection
# - predict_forgery
# - detect_anomaly

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import base64
import binascii

def log_decoded_data(data, barcode_type):
    """Log decoded data and type."""
    print(f"Decoded Data: {data}, Type: {barcode_type}")


def clean_hex_string(hex_string):
    """
    Removes any non-hexadecimal characters from the string.
    """
    cleaned = re.sub(r'[^0-9a-fA-F]', '', hex_string)
    return cleaned

def hex_to_binary(hex_string):
    """
    Cleans and converts a hexadecimal string to binary data.
    
    Parameters:
    - hex_string: The hexadecimal string representation of the barcode data.
    
    Returns:
    - binary_data: The binary data decoded from the hex string, or None if decoding fails.
    """
    # Remove surrounding quotes if present
    hex_string = hex_string.strip('"').strip("'")
    
    # Remove any non-hex characters
    hex_string_cleaned = clean_hex_string(hex_string)
    
    # Ensure even length
    if len(hex_string_cleaned) % 2 != 0:
        hex_string_cleaned = '0' + hex_string_cleaned
    
    try:
        binary_data = binascii.unhexlify(hex_string_cleaned)
        return binary_data
    except binascii.Error as e:
        print(f"Error decoding hex string: {e}")
        return None

def parse_aamva(binary_data):
    """
    Parses AAMVA-compliant binary data from a PDF417 barcode.

    Parameters:
    - binary_data: The binary data extracted from the barcode.

    Returns:
    - data_dict: Dictionary containing the extracted personal information.
    """
    # Convert binary data to string using ANSI encoding
    try:
        # AAMVA uses ASCII or ANSI encoding; 'latin1' can be a good fallback
        data_str = binary_data.decode('latin1')
    except UnicodeDecodeError as e:
        print(f"Error decoding binary data: {e}")
        return {}

    # Split the data into lines
    lines = data_str.split('\n')

    # Initialize a dictionary to hold the extracted data
    data_dict = {}

    # Define regex patterns for identifiers
    # AAMVA identifiers are typically 3 characters
    pattern = re.compile(r'^([A-Z]{3})(.*)')

    for line in lines:
        match = pattern.match(line)
        if match:
            identifier = match.group(1)
            value = match.group(2).strip()

            # Map identifiers to human-readable field names
            field_name = map_identifier_to_field(identifier)

            if field_name:
                data_dict[field_name] = value

    return data_dict

def map_identifier_to_field(identifier):
    """
    Maps AAMVA identifiers to human-readable field names.

    Parameters:
    - identifier: The 3-character AAMVA identifier.

    Returns:
    - field_name: The corresponding human-readable field name.
    """
    # A subset of AAMVA identifiers
    mapping = {
        'DCS': 'Customer Family Name',
        'DCT': 'Customer First Name',
        'DAC': 'Customer Middle Name',
        'DBA': 'Document Expiration Date',
        'DBD': 'Document Issue Date',
        'DBB': 'Date of Birth',
        'DBC': 'Sex',
        'DAY': 'Eye Color',
        'DAQ': 'License or ID Number',
        'DCG': 'Country Identification',
        'DDF': 'Jurisdiction-Specific Vehicle Classification',
        'DDG': 'Jurisdiction-Specific Restriction Codes',
        'DDH': 'Jurisdiction-Specific Endorsement Codes',
        'DEL': 'Jurisdiction-Specific Code',
        'DDA': 'License/ID Document Discriminator',
        'DCU': 'Customer Suffix Name',
        'DBA': 'Document Expiration Date',
        # Add more mappings as needed
    }

    return mapping.get(identifier, None)



def read_barcodes(image_path, visualize=False):
    """
    Detect and decode barcodes in an image and extract personal information.

    Parameters:
    - image_path: Path to the image containing the barcode.
    - visualize: Boolean indicating whether to visualize detected barcodes.

    Returns:
    - barcodes: List of decoded barcode data with extracted personal information.
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at path '{image_path}'.")
        return []
    
    # Detect barcodes using zxingcpp
    results = zxingcpp.read_barcodes(img)
    
    if len(results) == 0:
        print("Could not find any barcode.")
        return []
    
    extracted_barcodes = []
    
    for result in results:
        print("Found barcode:")
        print(f' Text:   {result.text}')
        print(f' Format:   {result.format}')
        print(f' Content:  {result.content_type}')
        print(f' Position: {result.position}\n')

    return results

def extract_text_from_image(image_path, preprocess=True):
    """
    Extract text from an image using Tesseract OCR.
    
    Parameters:
    - image_path: Path to the image.
    - preprocess: Boolean indicating whether to preprocess the image for better OCR accuracy.
    
    Returns:
    - extracted_text: String containing the extracted text.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return ""
    
    if preprocess:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get binary image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Remove noise
        gray = cv2.medianBlur(gray, 3)
    else:
        gray = image
    
    # Perform OCR
    print("extracting.....")
    extracted_text = pytesseract.image_to_string(gray)
    
    return extracted_text


def parse_barcode_text(barcode_text):
    """
    Parses the barcode text to retrieve personal information.
    
    Parameters:
    - barcode_text: String containing the barcode data, separated by '>'.
    
    Returns:
    - info_dict: Dictionary containing extracted personal information.
    """
    info_dict = {}
    
    # Split the barcode text by '>'
    segments = barcode_text.split('>')
    
    # Ensure there are enough segments
    if len(segments) < 11:
        print("Warning: Barcode data has fewer segments than expected.")
    
    # Map each segment to its corresponding field
    try:
        info_dict['ID_Number'] = segments[0].strip()
        info_dict['Reg_No'] = segments[1].strip()
        info_dict['Full_Name'] = segments[2].strip()
        info_dict['Date_of_Birth'] = segments[3].strip()
        info_dict['Sex'] = segments[4].strip()
        info_dict['Blood_Group'] = segments[5].strip()
        info_dict['Address'] = segments[6].strip()
        info_dict['Phone_Number_1'] = segments[7].strip()
        info_dict['Phone_Number_2'] = segments[8].strip()
        info_dict['Issue_Date'] = segments[9].strip()
        info_dict['Expiry_Date'] = segments[10].strip()
    except IndexError as e:
        print(f"Error parsing barcode segments: {e}")
    
    return info_dict

def extract_first_and_middle_name(full_name):
    """
    Extract only the first and last name from the full name.
    
    Parameters:
    - full_name: A string containing the full name.
    
    Returns:
    - first_last_name: A string containing only the first and last name.
    """
    name_parts = full_name.split()
    if len(name_parts) < 2:
        return full_name.strip()  # If only one name is present, return as is.
    return f"{name_parts[0]} {name_parts[1]}"  # Return first and last name


def parse_extracted_text(extracted_text):
    """
    Parse the extracted OCR text to retrieve personal information.
    
    Parameters:
    - extracted_text: String containing the OCR-extracted text.
    
    Returns:
    - info_dict: Dictionary containing extracted personal information.
    """
    info_dict = {}
    
    # Define regex patterns tailored to the OCR text structure
    patterns = {
        'Full_Name': r'Full Name\s+([A-Z\s]+)',  # e.g., Full Name ABERESH MULUGETA KASSA
        'Date_of_Birth': r'DOB\s*[:\-]?\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4})',  # e.g., DOB MAY 05, 1976
        'Sex': r'Sex\s+([MF])',  # e.g., Sex F
        'Blood_Group': r'Blood Group\s+([A-Z]+)',  # e.g., Blood Group NA
        'ID_Number': r'IDNo\s+([A-Za-z0-9\/]+)',  # e.g., IDNo ID/AR/W06/00762618
        'Reg_No': r'Reg\.? No\s+([A-Za-z0-9]+)',  # e.g., Reg. No AA0001286032
        'Issue_Date': r'Issue Dt\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4})',  # e.g., Issue Dt Nov 13,2022
        'Expiry_Date': r'Expiry Dt\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4})',  # e.g., Expiry Dt Nov 13,2026
        'Address': r'Address\s+([A-Za-z\s\/]+)',  # e.g., Address Arada/Woreda 6
        # Add more patterns as needed based on OCR output
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            # Clean the extracted value
            value = match.group(1).strip()
            # Replace multiple spaces with single space
            value = re.sub(r'\s+', ' ', value)
            info_dict[key] = value
    
    return info_dict

def match_personal_info(barcode_data_dict, ocr_data_dict):
    """
    Compare barcode data with OCR-extracted personal information.
    
    Parameters:
    - barcode_data_dict: Dictionary containing data extracted from the barcode.
    - ocr_data_dict: Dictionary containing data extracted via OCR.
    
    Returns:
    - match_score: Integer indicating the number of matching fields.
    - total_fields: Total number of fields compared.
    - mismatches: Dictionary detailing mismatched fields.
    """
    match_score = 0
    total_fields = 0
    mismatches = {}
    
    # Define the fields to compare
    fields = ['Full_Name', 'Date_of_Birth', 'Sex', 'ID_Number', 'Reg_No', 'Issue_Date', 'Expiry_Date', 'Address']
    
    for field in fields:
        barcode_value = barcode_data_dict.get(field, '').lower()
        ocr_value = ocr_data_dict.get(field, '').lower()
        
        if barcode_value and ocr_value:
            total_fields += 1
            if barcode_value == ocr_value:
                match_score += 1
            else:
                mismatches[field] = {
                    'Barcode': barcode_data_dict.get(field),
                    'OCR': ocr_data_dict.get(field)
                }
    
    return match_score, total_fields, mismatches


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
    

def load_face_encoding(image_path):
    """
    Loads an image and returns the face encoding.
    """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
        raise ValueError(f"No faces found in the image: {image_path}")
    
    return encodings[0]


def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    """
    Compares two face encodings and returns whether they match and the distance.
    """
    results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance)
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    return results[0], distance

def detect_smile(image_path):
    """
    Analyzes the image and returns the dominant emotion.
    """
    try:
        # Perform the emotion analysis
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)
        
        # Check if the result is a list (multiple faces detected)
        if isinstance(analysis, list):
            # Assuming you want to process the first face found
            analysis = analysis[0]  

        # Return the dominant emotion
        return analysis['dominant_emotion']
    
    except Exception as e:
        raise ValueError(f"Error in emotion analysis: {str(e)}")



def get_right_head_pose(image, predictor_path, mmod_model_path):
    detector = dlib.cnn_face_detection_model_v1(mmod_model_path)
    predictor = dlib.shape_predictor(predictor_path)

    if image is None:
        raise ValueError("Input image is None or could not be read.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(image, 1)
    
    print(f"Number of faces detected: {len(detections)}")

    if len(detections) == 0:
        raise ValueError("No faces detected for head pose estimation.")

    rect = detections[0].rect
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    model_points = np.array([
    (0.0, 0.0, 0.0),              # Nose tip
    (0.0, -300.0, -65.0),         # Chin (adjusted slightly)
    (-175.0, 150.0, -135.0),      # Left eye left corner (adjusted for symmetry)
    (175.0, 150.0, -135.0),       # Right eye right corner
    (-125.0, -125.0, -125.0),     # Left Mouth corner
    (125.0, -125.0, -125.0)       # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype=np.float64)

    size = image.shape
    focal_length = size[1] * 1.2  # Fine-tuned focal length
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("Head pose estimation failed.")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

    yaw = euler_angles[1][0]
    pitch = euler_angles[0][0]
    roll = euler_angles[2][0]    
    return yaw, pitch, roll



def get_left_head_pose(image, predictor_path, mmod_model_path):
    detector = dlib.cnn_face_detection_model_v1(mmod_model_path)
    predictor = dlib.shape_predictor(predictor_path)

    if image is None:
        raise ValueError("Input image is None or could not be read.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(image, 1)
    
    print(f"Number of faces detected: {len(detections)}")

    if len(detections) == 0:
        raise ValueError("No faces detected for head pose estimation.")

    rect = detections[0].rect
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    model_points = np.array([
        (0.0, 0.0, 0.0),              # Nose tip
        (0.0, -330.0, -65.0),         # Chin
        (-200.0, 170.0, -135.0),      # Left eye left corner (symmetrically adjusted)
        (200.0, 170.0, -135.0),       # Right eye right corner
        (-200.0, -170.0, -65.0),     # Left Mouth corner
        (150.0, -150.0, -125.0)       # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype=np.float64)

    size = image.shape
    focal_length = size[1] * 1.3  # Fine-tuned focal length
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("Head pose estimation failed.")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

    yaw = euler_angles[1][0]
    pitch = euler_angles[0][0]
    roll = euler_angles[2][0]
    
    return yaw, pitch, roll

from django.conf import settings


def detect_left_head_rotation(image_path, predictor_path="shape_predictor_68_face_landmarks.dat", yaw_threshold=20):
    """
    Detects if the head is rotated to the left beyond the specified yaw threshold.
    Returns a tuple indicating (rotated_left: bool).
    """
    try:
        
        predictor_path = os.path.join(settings.BASE_DIR, 'files', 'shape_predictor_68_face_landmarks.dat')
        mmod_model_path = os.path.join(settings.BASE_DIR, 'files', 'mmod_human_face_detector.dat')
        
        # Read the image from the InMemoryUploadedFile
        img_array = np.frombuffer(image_path.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Error in image decoding. Please check the uploaded image.")

        # Resize the image to reduce resource usage
        max_height = 480  # Adjust based on your requirements
        max_width = 640   # Adjust based on your requirements

        height, width = image.shape[:2]
        if height > max_height or width > max_width:
            scale = min(max_width / width, max_height / height)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        yaw, pitch, roll = get_left_head_pose(image, predictor_path, mmod_model_path)
        print(f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")
        
        rotated_left = yaw < -yaw_threshold
        
        return rotated_left
    except Exception as e:
        raise ValueError(f"Error in head rotation detection: {str(e)}")

def detect_right_head_rotation(image_path, predictor_path="shape_predictor_68_face_landmarks.dat", yaw_threshold=20):
    """
    Detects if the head is rotated to the right beyond the specified yaw threshold.
    Returns a tuple indicating (rotated_right: bool).
    """
    try:
        predictor_path = os.path.join(settings.BASE_DIR, 'files', 'shape_predictor_68_face_landmarks.dat')
        mmod_model_path = os.path.join(settings.BASE_DIR, 'files', 'mmod_human_face_detector.dat')
        
        # Read the image from the InMemoryUploadedFile
        img_array = np.frombuffer(image_path.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Error in image decoding. Please check the uploaded image.")

        # Resize the image to reduce resource usage
        max_height = 480  # Adjust based on your requirements
        max_width = 640   # Adjust based on your requirements

        height, width = image.shape[:2]
        if height > max_height or width > max_width:
            scale = min(max_width / width, max_height / height)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        yaw, pitch, roll = get_right_head_pose(image, predictor_path, mmod_model_path)
        print(f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")

        rotated_right = yaw > yaw_threshold
        
        return rotated_right
    except Exception as e:
        raise ValueError(f"Error in head rotation detection: {str(e)}")
