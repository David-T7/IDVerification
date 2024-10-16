# app/verification/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import IDUploadSerializer , PassportUploadSerializer
from django.conf import settings
import os
from passporteye import read_mrz

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
    parse_mrz_data
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

            # Define paths and ensure directories exist
            id_front_upload_path = os.path.join(settings.MEDIA_ROOT, 'images/government_id', f"front_{id_front_image.name}")
            id_back_upload_path = os.path.join(settings.MEDIA_ROOT, 'images/government_id', f"back_{id_back_image.name}")
            # id_front_process_path = os.path.join(settings.MEDIA_ROOT, 'images',f"processed_{id_front_image.name}")
            
            os.makedirs(os.path.dirname(id_front_upload_path), exist_ok=True)
            os.makedirs(os.path.dirname(id_back_upload_path), exist_ok=True)
            # os.makedirs(os.path.dirname(id_front_process_path), exist_ok=True)

            # Save uploaded images
            with open(id_front_upload_path, 'wb+') as destination:
                for chunk in id_front_image.chunks():
                    destination.write(chunk)

            with open(id_back_upload_path, 'wb+') as destination:
                for chunk in id_back_image.chunks():
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
                # Preprocess Image
                # preprocess_image(id_front_upload_path , id_front_process_path)  # Overwrite the original image
                
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
                    barcode_text = barcodes[0].text  # Assuming the first barcode is relevant
                    barcode_data_dict = parse_barcode_text(barcode_text)
                else:
                    barcode_data_dict = {}
                print("in ocr")
                # OCR Text Extraction and Parsing
                ocr_text = extract_text_from_image(id_front_upload_path, preprocess=True)
                print("ocr extraction finished")
                ocr_data_dict = parse_extracted_text(ocr_text)
                print("ocr parsing finished")

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
                
                # Prepare Response Data
                response_data = {
                    "authenticity_score": authenticity_percentage,
                    "details": {
                        "ELA_Tampered": ela_tampered,
                        "ELA_Score": ela_score,
                        "Clone_Detected": clone_detected,
                        "Clone_Count": clone_count,
                        "Forgery_Detected": is_forged,
                        "Forgery_Confidence": confidence_forgery,
                        "Anomaly_Detected": is_anomaly,
                        "MSE": mse,
                        "Personal_Info_Match": {
                            "Is_Match": is_match,
                            "Match_Percentage": match_percentage,
                            "Match_Score": match_score,
                            "Total_Fields": total_fields,
                            "Mismatches": mismatches
                        }
                    }
                }
                
                return Response(response_data, status=status.HTTP_200_OK)
            
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

                # Step 3: Perform Forgery Detection Techniques

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
                
                # Prepare Response Data
                response_data = {
                    "authenticity_score": authenticity_percentage,
                    "details": {
                        "ELA_Tampered": ela_tampered,
                        "ELA_Score": ela_score,
                        "Clone_Detected": clone_detected,
                        "Clone_Count": clone_count,
                        "Forgery_Detected": is_forged,
                        "Forgery_Confidence": confidence_forgery,
                        "Anomaly_Detected": is_anomaly,
                        "MSE": mse,
                    }
                }
                
                return Response(response_data, status=status.HTTP_200_OK)
            
            except Exception as e:
                return Response({"error": f"Error during verification: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


