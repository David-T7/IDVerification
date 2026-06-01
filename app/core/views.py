# app/verification/views.py

from io import BytesIO
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import IDUploadSerializer , PassportUploadSerializer , BiometricImageUploadSerializer
from django.conf import settings
import os
from passporteye import read_mrz
from .models import UserImage  # Ensure this import is correct
import cv2
import numpy as np
from django.core.exceptions import ObjectDoesNotExist
import difflib
from PIL import Image
from django.shortcuts import get_object_or_404

from .utils import (
    preprocess_image,
    perform_ela,
    clone_detection,
    predict_forgery,
    detect_anomaly,
    read_barcodes,
    parse_barcode_text,
    extract_text_from_image,
    parse_extracted_text,
    match_personal_info,
    detect_template ,
    parse_mrz_data ,
    load_face_encoding, compare_faces, 
    detect_smile, 
    detect_left_head_rotation ,
    detect_right_head_rotation ,
    extract_first_and_middle_name
)

import tensorflow as tf
from core.authentication import CustomJWTAuthentication, TokenPayloadPermission

class VerifyIDView(APIView):
    """
    API endpoint that accepts a passport image and returns its authenticity score.
    """
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]
    
    def post(self, request, format=None):
        serializer = IDUploadSerializer(data=request.data)
        if serializer.is_valid():
            id_front_image = serializer.validated_data['front_id_image']
            id_back_image = serializer.validated_data['back_id_image']
            freelancer_id = request.data.get('freelancer_id')  # Get freelancer UUID from the request
            full_name = request.data.get('full_name')  # Get freelancer UUID from the request

            # Define paths and ensure directories exist
            id_front_upload_path = os.path.join(settings.MEDIA_ROOT, 'images/government_id', f"front_{id_front_image.name}")
            id_back_upload_path = os.path.join(settings.MEDIA_ROOT, 'images/government_id', f"back_{id_back_image.name}")

            os.makedirs(os.path.dirname(id_front_upload_path), exist_ok=True)
            os.makedirs(os.path.dirname(id_back_upload_path), exist_ok=True)

            # Save uploaded images
            with open(id_front_upload_path, 'wb+') as destination:
                for chunk in id_front_image.chunks():
                    destination.write(chunk)

            with open(id_back_upload_path, 'wb+') as destination:
                for chunk in id_back_image.chunks():
                    destination.write(chunk)
            
            # Load models
            model_forgery_path = os.path.join(settings.MODELS_ROOT, 'forgery_detection_model.keras')
            autoencoder_path = os.path.join(settings.MODELS_ROOT, 'autoencoder_model.keras')

            try:
                model_forgery = tf.keras.models.load_model(model_forgery_path)
                autoencoder = tf.keras.models.load_model(autoencoder_path)
            except Exception as e:
                return Response({"error": f"Error loading models: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Perform Verification Steps
            try:
                # Error Level Analysis (ELA)
                ela_tampered, ela_score = perform_ela(id_front_upload_path, threshold=10)

                # Clone Detection
                clone_detected, clone_count = clone_detection(id_front_upload_path, visualize=False, distance_threshold=30, area_threshold=50)

                # Forgery Detection
                is_forged, confidence_forgery = predict_forgery(model_forgery, id_front_upload_path)

                # Anomaly Detection
                is_anomaly, mse = detect_anomaly(autoencoder, id_front_upload_path)

                # Barcode Reading and Parsing
                barcodes = read_barcodes(id_back_upload_path, visualize=False)
                if barcodes:
                    barcode_text = barcodes[0].text
                    barcode_data_dict = parse_barcode_text(barcode_text)
                else:
                    barcode_data_dict = {}

                # OCR Text Extraction and Parsing
                ocr_text = extract_text_from_image(id_front_upload_path, preprocess=True)
                ocr_data_dict = parse_extracted_text(ocr_text)
                ocr_first_middle = extract_first_and_middle_name(ocr_data_dict["Full_Name"])
                request_first_middle = extract_first_and_middle_name(full_name)

                # Normalize by removing spaces and converting to lowercase
                normalized_ocr_name = ocr_first_middle.replace(" ", "").strip().lower()
                normalized_request_name = request_first_middle.replace(" ", "").strip().lower()

                # Calculate the similarity ratio using difflib
                similarity_score = difflib.SequenceMatcher(None, normalized_ocr_name, normalized_request_name).ratio() * 100
                name_match_threshold = 85  # Define the acceptable threshold for name matching
                print("name similarity score is",similarity_score)
                if similarity_score < name_match_threshold:
                    return Response({"error": "Full name does not much with the provided document!."}, status=status.HTTP_400_BAD_REQUEST)

                # Personal Information Matching
                match_score, total_fields, mismatches = match_personal_info(barcode_data_dict, ocr_data_dict)
                match_percentage = (match_score / total_fields) * 100 if total_fields > 0 else 0
                is_match = match_percentage >= 75  # Define threshold as needed
                
                # Composite Scoring
                authenticity_score = 0
                total_features = 5  # Adjust based on implemented features
                
                # Image Forensics
                if not ela_tampered:
                    authenticity_score += 1
                if not clone_detected:
                    authenticity_score += 1
                
                # Deep Learning Models
                if not is_forged:
                    authenticity_score += 1
                if not is_anomaly:
                    authenticity_score += 1
                
                # Personal Info Matching
                if is_match:
                    authenticity_score += 1
                
                authenticity_percentage = (authenticity_score / total_features) * 100
                print("Government id authenticity percnetage ",authenticity_percentage)

                # Check if authenticity percentage is above the threshold
                if authenticity_percentage >= 80:
                    # Store the user image and ID image in the database
                    user_image_instance = UserImage.objects.create(
                        freelancer_id=freelancer_id,
                        id_image=id_front_image,  # Store back ID image too, if required
                        id_type='Government ID',  # Adjust as necessary
                    )
                    return Response({
                    "status": "success", 
                    "message": "ID Verification successful.",
                }, status=status.HTTP_200_OK)
                
                else:
                    return Response({
                    "status": "failure", 
                    "message": "ID Verification not successful.",
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": f"Error during verification: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class VerifyPassportView(APIView):
    """
    API endpoint that accepts a passport image and returns its authenticity score.
    """
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]
    
    def post(self, request, format=None):
        serializer = PassportUploadSerializer(data=request.data)
        if serializer.is_valid():
            passport_image = serializer.validated_data['passport_image']
            freelancer_id = request.data.get('freelancer_id')  # Get freelancer UUID from the request
            full_name = request.data.get('full_name')  # Get freelancer UUID from the request
            # Define paths and ensure directories exist
            passport_image_upload_path = os.path.join(settings.MEDIA_ROOT, 'images/passport', f"{passport_image.name}")
            passport_template_path = os.path.join(settings.MEDIA_ROOT, 'images/passport',"passport_template.jpg")
            
            os.makedirs(os.path.dirname(passport_template_path), exist_ok=True)
            os.makedirs(os.path.dirname(passport_image_upload_path), exist_ok=True)

            # Save uploaded images
            with open(passport_image_upload_path, 'wb+') as destination:
                for chunk in passport_image.chunks():
                    destination.write(chunk)            
            
            # Define paths to models and templates
            model_forgery_path = os.path.join(settings.MODELS_ROOT, 'forgery_detection_model.keras')
            autoencoder_path = os.path.join(settings.MODELS_ROOT, 'autoencoder_model.keras')
            # template_path = os.path.join(settings.TEMPLATES_ROOT, 'hologram_template.jpg')  # Ensure this exists
            
            # Load models
            try:
                model_forgery = tf.keras.models.load_model(model_forgery_path)
                autoencoder = tf.keras.models.load_model(autoencoder_path)
            except Exception as e:
                return Response({"error": f"Error loading models: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Perform Verification Steps
            try:
                authenticity_score = 0
                total_features = 6  # Adjust based on implemented features

                # Step 1: Extract MRZ Data
                mrz = read_mrz(passport_image_upload_path)
                if mrz is None:
                    print("MRZ could not be read from the image.")
                    return
                mrz_data = mrz.to_dict()
                
                # Validate MRZ check digits
                if not (mrz_data.get('valid_number', False) and
                        mrz_data.get('valid_date_of_birth', False) and
                        mrz_data.get('valid_expiration_date', False) and
                        mrz_data.get('valid_composite', False) and
                        mrz_data.get('valid_personal_number', False)):
                    print("Warning: MRZ data failed validation checks.")
                else:
                    authenticity_score += 1  # MRZ validation passed

                # Parse MRZ data
                parsed_mrz = parse_mrz_data(mrz_data)
                # Step 2: Preprocess Image (optional based on requirements)
                # Implement if needed
                if not full_name:
                    return Response({"error": "Full name is required for comparison."}, status=status.HTTP_400_BAD_REQUEST)
                parsed_first_middle_name = extract_first_and_middle_name(parsed_mrz['Names'])
                request_first_middle = extract_first_and_middle_name(full_name)
                # Normalize names by removing all spaces and converting to lowercase
                normalized_full_name_request = request_first_middle.replace(" ", "").strip().lower()
                normalized_mrz_names = parsed_first_middle_name.replace(" ", "").strip().lower()

                # Step 4: Fuzzy name comparison using difflib's SequenceMatcher
                similarity_score = difflib.SequenceMatcher(None, normalized_full_name_request, normalized_mrz_names).ratio() * 100
                name_match_threshold = 85  # Define the acceptable threshold for name matching
                print("name similarity score is",similarity_score)
                if similarity_score <  name_match_threshold:
                    return Response({"error": "Full name does not much with the provided document!."}, status=status.HTTP_400_BAD_REQUEST)

                # 3a. Error Level Analysis (ELA)
                ela_tampered, ela_score = perform_ela(passport_image_upload_path, threshold=10)
                if not ela_tampered:
                    authenticity_score += 1
                print(f"ELA Tampered: {ela_tampered}, ELA Score: {ela_score:.2f}")

                # 3b. Clone Detection
                clone_detected, clone_count = clone_detection(passport_image_upload_path, visualize=False, distance_threshold=30, area_threshold=50)
                if not clone_detected:
                    authenticity_score += 1
                print(f"Clone Detected: {clone_detected}, Clone Count: {clone_count}")

                # 3c. Deep Learning-Based Forgery Detection
                is_forged, confidence_forgery = predict_forgery(model_forgery, passport_image_upload_path)
                if not is_forged:
                    authenticity_score += 1
                else:
                    authenticity_score += confidence_forgery
                print(f"Forgery Detected: {is_forged}, Confidence: {confidence_forgery:.2f}")

                # 3d. Anomaly Detection with Autoencoder
                is_anomaly, mse = detect_anomaly(autoencoder, passport_image_upload_path)
                if not is_anomaly:
                    authenticity_score += 1
                print(f"Anomaly Detected: {is_anomaly}, MSE: {mse:.4f}")

                # Step 4: Barcode Reading and Verification
                # Step 5: Template Detection (e.g., Hologram)
                hologram_detected = detect_template(passport_image_upload_path, passport_template_path, threshold=0.8)
                if hologram_detected:
                    authenticity_score += 1
                print(f"Hologram Detected: {hologram_detected}")

                # Step 6: Composite Scoring
                authenticity_percentage = (authenticity_score / total_features) * 100
                print("passport authenticity percnetage ",authenticity_percentage)
                # Check if authenticity percentage is above the threshold
                if authenticity_percentage >= 80:
                    # Store the user image and ID image in the database
                    user_image_instance = UserImage.objects.create(
                        freelancer_id=freelancer_id,
                        id_image=passport_image,  # Store back ID image too, if required
                        id_type='Passport',  # Adjust as necessary
                    )
                    return Response({
                    "status": "success", 
                    "message": "ID Verification successful.",
                }, status=status.HTTP_200_OK)
                
                else:
                    return Response({
                    "status": "failure", 
                    "message": "ID Verification not successful.",
                }, status=status.HTTP_200_OK)
            
            except Exception as e:
                return Response({"error": f"Error during verification: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class FaceMatchingView(APIView):
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]
   
    def post(self, request, *args, **kwargs):
            user_image = request.FILES.get('user_image')
            freelancer_id = request.data.get('freelancer_id')  # Get freelancer UUID from the request
            print("freelancer image is ",user_image)
            print("freelance id is",freelancer_id)

            if not freelancer_id:
                return Response({"error": "Freelancer UUID is required."}, status=status.HTTP_400_BAD_REQUEST)
            elif not user_image:
                return Response({"error": "User image is required."}, status=status.HTTP_400_BAD_REQUEST)
            try:
                user_image_instance = UserImage.objects.filter(freelancer_id=freelancer_id).last()
                # Convert `user_image` to a byte stream for `load_face_encoding`
                new_user_image = BytesIO()
                img = Image.open(user_image)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(new_user_image, format='JPEG')
                new_user_image.seek(0)
                # Load and compare face encodings
                print("user image object found in the database")
                known_encoding = load_face_encoding(user_image_instance.id_image)
                print("face encoding for known worked....")
                unknown_encoding = load_face_encoding(new_user_image)
                print("face encoding for unknown worked....")
              
                print("mtaching....")
                match, distance = compare_faces(known_encoding, unknown_encoding)

                if not match:
                    return Response({"status": "failed", "message": f"Face mismatch. Distance: {distance:.4f}"}, status=status.HTTP_200_OK)
                 
                return Response({
                    "status": "success", 
                    "message": "Face match successful.",
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

def match_faces(freelancer_id, new_image):
    """
    Matches a face from an a new user image with the face stored in the UserImage model.

    Parameters:
    - freelancer_id: UUID of the freelancer to retrieve the user image.
    - new_image: The uploaded image file.

    Returns:
    - A dictionary with the status and message.
    """
    try:
        # Retrieve the user image from the database based on the freelancer_id
        user_image_instance = UserImage.objects.get(freelancer_uuid=freelancer_id)
        user_image = user_image_instance.user_image  # Get the user image from the instance

        # Load and compare face encodings
        known_encoding = load_face_encoding(new_image)  # ID image to match against
        unknown_encoding = load_face_encoding(user_image)  # User image from the model

        match, distance = compare_faces(known_encoding, unknown_encoding)
        return match
    except ObjectDoesNotExist:
        return {"status": "failed", "message": "Freelancer UUID does not exist."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class SmileDetectionView(APIView):
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]

    def post(self, request, *args, **kwargs):
            user_image = request.FILES.get('user_image')
            freelancer_id = request.data.get('freelancer_id')  # Get freelancer UUID from the request

            if not freelancer_id:
                return Response({"error": "Freelancer UUID is required."}, status=status.HTTP_400_BAD_REQUEST)
            elif not user_image:
                return Response({"error": "User image is required."}, status=status.HTTP_400_BAD_REQUEST)
            new_user_image = BytesIO()
            img = Image.open(user_image)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(new_user_image, format='JPEG')
            new_user_image.seek(0)

            match = match_faces(freelancer_id , new_user_image)
            if not match:
                return Response({"status": "failed", "message": "Image does not match with the previous id image uploaded"}, status=status.HTTP_200_OK)

            try:
                # Convert the InMemoryUploadedFile to a NumPy array
                img_array = np.frombuffer(new_user_image.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    return Response({"error": "Invalid image format."}, status=status.HTTP_400_BAD_REQUEST)

                # Detect smile in the user's image
                print(f"Image shape: {img.shape}")  # Log the shape of the image
                emotion = detect_smile(img)  # Pass the NumPy array to detect_smile

                if emotion.lower() != 'happy':
                    return Response({"status": "failed", "message": f"Required emotion 'happy' not detected. Detected emotion: '{emotion}'."}, status=status.HTTP_200_OK)

                return Response({"status": "success", "message": "Smile detected successfully."}, status=status.HTTP_200_OK)

            except Exception as e:
                print(f"Error during smile detection: {e}")
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    


class HeadRotationRightView(APIView):
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]

    def post(self, request, *args, **kwargs):
            user_image = request.FILES.get('user_image')
            freelancer_id = request.data.get('freelancer_id')  # Get freelancer UUID from the request
            if not freelancer_id:
                return Response({"error": "Freelancer UUID is required."}, status=status.HTTP_400_BAD_REQUEST)
            elif not user_image:
                return Response({"error": "User image is required."}, status=status.HTTP_400_BAD_REQUEST)
            new_user_image = BytesIO()
            img = Image.open(user_image)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(new_user_image, format='JPEG')
            new_user_image.seek(0)
            match = match_faces(freelancer_id , new_user_image)
            if not match:
                return Response({"status": "failed", "message": "Image does not match with the previous id image uploaded"}, status=status.HTTP_200_OK)
        
            yaw_threshold = request.data.get('yaw_threshold', 15)
            print(f"Type of user_image: {type(new_user_image)}")
            print(f"Contents of user_image: {new_user_image}")

            try:
                # Detect head rotation in the user's image
                rotated_right = detect_right_head_rotation(new_user_image, yaw_threshold=float(yaw_threshold))

                if not rotated_right:
                    return Response({"status": "failed", "message": "Head rotation requirement not met. Please rotate your head to the right."}, status=status.HTTP_200_OK)

                return Response({"status": "success", "message": "Head rotation detected successfully."}, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class HeadRotationLeftView(APIView):
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]

    def post(self, request, *args, **kwargs):
            user_image = request.FILES.get('user_image')
            freelancer_id = request.data.get('freelancer_id')  # Get freelancer UUID from the request
            if not freelancer_id:
                return Response({"error": "Freelancer UUID is required."}, status=status.HTTP_400_BAD_REQUEST)
            elif not user_image:
                return Response({"error": "User image is required."}, status=status.HTTP_400_BAD_REQUEST)
            new_user_image = BytesIO()
            img = Image.open(user_image)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(new_user_image, format='JPEG')
            new_user_image.seek(0)
            match = match_faces(freelancer_id , new_user_image)
            if not match:
                return Response({"status": "failed", "message": "Image does not match with the previous id image uploaded"}, status=status.HTTP_200_OK)
            yaw_threshold = request.data.get('yaw_threshold', 15)
            print(f"Type of user_image: {type(new_user_image)}")
            print(f"Contents of user_image: {new_user_image}")


            try:
                # Detect head rotation in the user's image
                rotated_left = detect_left_head_rotation(new_user_image, yaw_threshold=float(yaw_threshold))

                if not rotated_left:
                    return Response({"status": "failed", "message": "Head rotation requirement not met. Please rotate your head to the left."}, status=status.HTTP_200_OK)

                return Response({"status": "success", "message": "Head rotation detected successfully."}, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        


class UpdateUserImageView(APIView):
    permission_classes = [TokenPayloadPermission]
    authentication_classes = [CustomJWTAuthentication]

    def patch(self, request, *args, **kwargs):
        freelancer_id = request.data.get("freelancer_id")
        user_image = request.FILES.get("user_image")

        if not freelancer_id:
            return Response({"error": "Freelancer UUID is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not user_image:
            return Response({"error": "User image is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch the freelancer's existing image record
        user_image_instance = get_object_or_404(UserImage, freelancer_id=freelancer_id)

        # Update the user image
        user_image_instance.user_image = user_image
        user_image_instance.save()

        return Response(
            {"status": "success", "message": "Profile picture updated successfully."},
            status=status.HTTP_200_OK,
        )